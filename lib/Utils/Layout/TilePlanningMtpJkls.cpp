#include "lib/Utils/Layout/TilePlanningMtpJkls.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "lib/Utils/Layout/Utils.h"
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/Support/LLVM.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

namespace mlir {
namespace heir {

namespace {

// Valid (unpadded) extent of tile `tile` along a dimension of total size
// `dimSize`, tiled by `tileSize`: min(tileSize, dimSize - tile*tileSize).
// Safe against overflow: for any `tile` in [0, ceilDivPositive(dimSize,
// tileSize)), `tile*tileSize < dimSize` always holds (by the definition of
// ceiling division), so the subtraction never underflows and the product
// never exceeds the already-valid `dimSize`.
int64_t validExtent(int64_t dimSize, int64_t tile, int64_t tileSize) {
  int64_t remaining = dimSize - tile * tileSize;
  return remaining < tileSize ? remaining : tileSize;
}

}  // namespace

FailureOr<GemmTilePlan> computeGemmTilePlan(int64_t M, int64_t K, int64_t N,
                                            int64_t mu, int64_t minSlotCount) {
  if (M <= 0 || K <= 0 || N <= 0 || mu <= 0 || minSlotCount <= 0) {
    return failure();
  }

  int64_t I = ceilDivPositive(M, mu);
  int64_t Q = ceilDivPositive(K, mu);
  int64_t J = ceilDivPositive(N, mu);

  // Overflow-safe taskCount = I*J and totalTasks = taskCount*Q: check the
  // divisor bound before multiplying, matching getBalancedMtpPacking's own
  // discipline, never compute a product speculatively and check it after.
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (I > kMax / J) {
    return failure();
  }
  int64_t taskCount = I * J;
  if (taskCount > kMax / Q) {
    return failure();
  }
  int64_t totalTasks = taskCount * Q;

  // Destination tiles (taskCount of them) are the physical batch positions
  // packed across ciphertexts -- identical to how Phase 7 packs independent
  // batch elements, reused here rather than reimplemented. Propagates
  // failure for mu*mu exceeding minSlotCount (no tile fits at all).
  FailureOr<BalancedMtpPacking> packing =
      getBalancedMtpPacking(taskCount, mu, mu, minSlotCount);
  if (failed(packing)) {
    return failure();
  }

  GemmTilePlan plan;
  plan.M = M;
  plan.K = K;
  plan.N = N;
  plan.mu = mu;
  plan.minSlotCount = minSlotCount;
  plan.I = I;
  plan.Q = Q;
  plan.J = J;
  plan.taskCount = taskCount;
  plan.numCiphertexts = packing->numCiphertexts;
  plan.tilesPerCiphertext = packing->tilesPerCiphertext;
  plan.tasks.reserve(static_cast<size_t>(totalTasks));

  // Canonical order: outer q, inner destTileId (itself row-major (i,j)).
  for (int64_t q = 0; q < Q; ++q) {
    for (int64_t destTileId = 0; destTileId < taskCount; ++destTileId) {
      int64_t i = destTileId / J;
      int64_t j = destTileId % J;

      GemmTask task;
      task.i = i;
      task.j = j;
      task.q = q;
      task.validRows = validExtent(M, i, mu);
      task.validContraction = validExtent(K, q, mu);
      task.validColumns = validExtent(N, j, mu);
      task.lhsTileId = i * Q + q;
      task.rhsTileId = q * J + j;
      task.destTileId = destTileId;
      task.group = destTileId / plan.tilesPerCiphertext;
      task.position = destTileId % plan.tilesPerCiphertext;
      plan.tasks.push_back(task);
    }
  }

  return plan;
}

std::vector<GemmTask> GemmTilePlan::reductionChainForDest(int64_t i,
                                                          int64_t j) const {
  std::vector<GemmTask> chain;
  if (i < 0 || i >= I || j < 0 || j >= J) {
    return chain;
  }
  int64_t destTileId = i * J + j;
  chain.reserve(static_cast<size_t>(Q));
  for (int64_t q = 0; q < Q; ++q) {
    // Canonical order is outer q, inner destTileId, so this task's index in
    // `tasks` is q*taskCount + destTileId -- but guard against an
    // arbitrarily-constructed (e.g. test-malformed) plan whose `tasks`
    // vector does not actually have that many entries, rather than
    // indexing out of bounds.
    int64_t idx = q * taskCount + destTileId;
    if (idx < 0 || static_cast<size_t>(idx) >= tasks.size()) {
      return {};
    }
    chain.push_back(tasks[static_cast<size_t>(idx)]);
  }
  return chain;
}

LogicalResult verifyGemmTilePlan(const GemmTilePlan& plan) {
  if (plan.M <= 0 || plan.K <= 0 || plan.N <= 0 || plan.mu <= 0 ||
      plan.minSlotCount <= 0) {
    return failure();
  }
  if (plan.I != ceilDivPositive(plan.M, plan.mu) ||
      plan.Q != ceilDivPositive(plan.K, plan.mu) ||
      plan.J != ceilDivPositive(plan.N, plan.mu)) {
    return failure();
  }
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (plan.I > kMax / plan.J) {
    return failure();
  }
  if (plan.taskCount != plan.I * plan.J) {
    return failure();
  }
  if (plan.taskCount > kMax / plan.Q) {
    return failure();
  }

  FailureOr<BalancedMtpPacking> packing = getBalancedMtpPacking(
      plan.taskCount, plan.mu, plan.mu, plan.minSlotCount);
  if (failed(packing) || packing->numCiphertexts != plan.numCiphertexts ||
      packing->tilesPerCiphertext != plan.tilesPerCiphertext) {
    return failure();
  }

  if (static_cast<int64_t>(plan.tasks.size()) != plan.taskCount * plan.Q) {
    return failure();
  }

  // The canonical order places the task for (q, destTileId) at array
  // position idx = q*taskCount+destTileId, and idx <-> (q, destTileId) is a
  // bijection over [0, taskCount*Q) (standard mixed-radix encoding). So
  // checking, independently at every array position, that the task actually
  // stored there has exactly the (q, destTileId) that position requires
  // proves three properties simultaneously: full coverage (every (q,
  // destTileId) -- equivalently every (i,j,q) -- appears, since every idx is
  // checked), uniqueness (each (q, destTileId) is pinned to exactly one
  // idx, so it cannot also appear at another), and canonical ordering
  // (idx's own bijection is exactly the order this checks). No separate
  // "seen" bookkeeping is needed.
  for (size_t idx = 0; idx < plan.tasks.size(); ++idx) {
    const GemmTask& task = plan.tasks[idx];

    int64_t expectedQ = static_cast<int64_t>(idx) / plan.taskCount;
    int64_t expectedDestTileId = static_cast<int64_t>(idx) % plan.taskCount;
    if (task.q != expectedQ || task.destTileId != expectedDestTileId) {
      return failure();
    }
    int64_t expectedI = expectedDestTileId / plan.J;
    int64_t expectedJ = expectedDestTileId % plan.J;
    if (task.i != expectedI || task.j != expectedJ) {
      return failure();
    }

    if (task.validRows != validExtent(plan.M, task.i, plan.mu) ||
        task.validContraction != validExtent(plan.K, task.q, plan.mu) ||
        task.validColumns != validExtent(plan.N, task.j, plan.mu)) {
      return failure();
    }
    if (task.validRows <= 0 || task.validRows > plan.mu ||
        task.validContraction <= 0 || task.validContraction > plan.mu ||
        task.validColumns <= 0 || task.validColumns > plan.mu) {
      return failure();
    }

    if (task.lhsTileId != task.i * plan.Q + task.q ||
        task.rhsTileId != task.q * plan.J + task.j) {
      return failure();
    }

    if (task.group != task.destTileId / plan.tilesPerCiphertext ||
        task.position != task.destTileId % plan.tilesPerCiphertext) {
      return failure();
    }
  }

  return success();
}

}  // namespace heir
}  // namespace mlir
