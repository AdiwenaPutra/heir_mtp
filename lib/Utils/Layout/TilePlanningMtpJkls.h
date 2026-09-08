#ifndef LIB_UTILS_LAYOUT_TILEPLANNINGMTPJKLS_H_
#define LIB_UTILS_LAYOUT_TILEPLANNINGMTPJKLS_H_

#include <cstdint>
#include <vector>

#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {

// One task in a general outer-tiled GEMM C[M,N] = A[M,K] * B[K,N], tiled by
// a primitive square tile size `mu`: the product of logical lhs tile (i,q)
// and logical rhs tile (q,j), contributing to logical destination tile
// (i,j). This is a pure data description used by computeGemmTilePlan below
// -- it names coordinates, extents, reuse identities, and physical grouping,
// but performs no MLIR construction, requires no encryption or backend, and
// does not itself decide how a future pass would materialize it.
struct GemmTask {
  // Logical destination row tile, column tile, and contraction tile
  // indices, in [0,I), [0,J), [0,Q) respectively.
  int64_t i = 0;
  int64_t j = 0;
  int64_t q = 0;

  // Valid (unpadded) extents at this task's position: rows actually present
  // in the lhs/destination row tile, contraction depth actually present in
  // this tile, and columns actually present in the rhs/destination column
  // tile. Each is in [1, mu]; strictly less than mu exactly at a boundary
  // tile (M, K, or N not evenly divisible by mu).
  int64_t validRows = 0;
  int64_t validContraction = 0;
  int64_t validColumns = 0;

  // Flat logical identity of this task's lhs and rhs source tiles.
  // lhsTileId = i*Q+q is shared by every task with the same (i,q) -- the
  // logical lhs tile A[i,q] is reused across all J destination-column
  // tiles. rhsTileId = q*J+j is shared by every task with the same (q,j) --
  // B[q,j] is reused across all I destination-row tiles. A future
  // memory-aware scheduler groups tasks by these ids to find candidates for
  // cache-vs-recompute of the processed (rotated) ciphertext variants each
  // task's gather would otherwise redo independently.
  int64_t lhsTileId = 0;
  int64_t rhsTileId = 0;

  // Flat logical identity of this task's destination tile: destTileId =
  // i*J+j. Exactly Q tasks (one per q, see GemmTilePlan::tasks' canonical
  // order) share a given destTileId and jointly accumulate into it.
  int64_t destTileId = 0;

  // Physical ciphertext-group index and position within that ciphertext
  // that this task's destination tile occupies. Computed once, from
  // getBalancedMtpPacking(I*J, mu, mu, minSlotCount) -- identical to Phase
  // 7's balanced packing formula and to the recovery formula
  // ConvertLinalgBatchMatmul::mtpJklsKernel independently applies, so a
  // task's packing here is always consistent with the existing
  // materializer's own recovery of it from a physical shape.
  int64_t group = 0;
  int64_t position = 0;
};

// The complete, deterministic tile plan for one general GEMM
// C[M,N] = A[M,K] * B[K,N], tiled by primitive square tile size mu, with
// physical grouping computed for a target minSlotCount slots per
// ciphertext.
struct GemmTilePlan {
  int64_t M = 0, K = 0, N = 0, mu = 0, minSlotCount = 0;
  int64_t I = 0, Q = 0, J = 0;  // ceil(M/mu), ceil(K/mu), ceil(N/mu)
  int64_t taskCount = 0;        // I*J: destination tiles / physical batch positions
  int64_t numCiphertexts = 0;
  int64_t tilesPerCiphertext = 0;

  // All I*J*Q tasks, in the plan's canonical deterministic order: outer q
  // (0..Q-1), inner destTileId (0..taskCount-1, itself in row-major (i,j)
  // order). Every destination tile's Q tasks are therefore NOT contiguous
  // in this list -- they are Q entries spaced taskCount apart, one per
  // q-pass -- matching the shape of a materializer that builds one shared
  // [taskCount, mu, mu] physical batch per q-step and chains the batched
  // matmul's `outs` across q (as validated for this exact case in the
  // Phase 8 B2/B3/B6 feasibility spike). Use reductionChainForDest below
  // instead of scanning this list directly when what is wanted is one
  // destination tile's accumulation order.
  std::vector<GemmTask> tasks;

  // Returns the ordered list (q=0,1,...,Q-1) of tasks that accumulate into
  // destination tile (i,j) (i.e. destTileId = i*J+j), in accumulation
  // order. Per linalg.matmul's own `result = init + sum_q product(i,j,q)`
  // semantics, the FIRST entry (q=0) is the one a materializer must combine
  // with the destination's real, possibly-nonzero `init` operand -- this
  // plan does not perform that combination itself (it is backend-neutral
  // and does not touch MLIR), but the order it fixes here is what a later
  // materialization pass relies on to preserve init correctly. Every
  // subsequent q accumulates onto the prior q's partial result for the same
  // destination tile: this is a linear, not tree-shaped, reduction order.
  // Returns an empty vector if (i,j) is out of [0,I)x[0,J) bounds.
  std::vector<GemmTask> reductionChainForDest(int64_t i, int64_t j) const;
};

// Constructs the complete, deterministic tile plan for
// C[M,N] = A[M,K] * B[K,N], tiled by primitive square tile size mu, with
// destination tiles packed across ciphertexts of capacity minSlotCount via
// getBalancedMtpPacking (reused, not reimplemented, so this planner always
// agrees with Phase 7's balanced-packing formula and its materializer-side
// recovery).
//
// Every arithmetic step (I, Q, J, taskCount, and the total task count) is
// checked for overflow and positivity before use. Returns failure --
// without partially constructing a plan, crashing, or asserting -- for any
// non-positive M/K/N/mu/minSlotCount, for mu*mu exceeding minSlotCount (no
// tile fits at all, delegated to getBalancedMtpPacking), or for any
// intermediate arithmetic overflow. mu larger than a matrix dimension is
// explicitly NOT rejected: e.g. mu=4, M=3 legally gives I=1 with a single
// boundary tile of valid extent 3.
FailureOr<GemmTilePlan> computeGemmTilePlan(int64_t M, int64_t K, int64_t N,
                                            int64_t mu, int64_t minSlotCount);

// Independently verifies the structural invariants of a tile plan already
// produced by computeGemmTilePlan, rather than merely trusting it: exactly
// I*J*Q tasks; every (i,j,q) in [0,I)x[0,J)x[0,Q) present exactly once (full
// coverage, no duplicates); every task's valid extents match the boundary
// arithmetic implied by its own coordinates and the plan's M/K/N/mu; every
// task's lhsTileId/rhsTileId/destTileId/group/position are consistent with
// its coordinates and with getBalancedMtpPacking on the plan's own
// parameters; and the task list is in the canonical order documented on
// GemmTilePlan::tasks. Returns failure (not a crash) on any violation --
// intended both as a test oracle and as a sanity check a future
// materialization pass can run on a plan before consuming it.
LogicalResult verifyGemmTilePlan(const GemmTilePlan& plan);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_TILEPLANNINGMTPJKLS_H_
