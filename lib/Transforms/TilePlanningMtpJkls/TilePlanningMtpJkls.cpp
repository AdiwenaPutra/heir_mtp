#include "lib/Transforms/TilePlanningMtpJkls/TilePlanningMtpJkls.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
#include "lib/Dialect/Secret/IR/SecretOps.h"
#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "lib/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "lib/Dialect/TensorExt/IR/TensorExtOps.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/Layout/TilePlanningMtpJkls.h"
#include "lib/Utils/Layout/Utils.h"
#include "llvm/include/llvm/ADT/DenseMap.h"                 // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"                // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlow/Utils.h"     // from @llvm-project
#include "mlir/include/mlir/Analysis/DataFlowFramework.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"      // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"     // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"    // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"    // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"                 // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"        // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"                // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"       // from @llvm-project

#define DEBUG_TYPE "tile-planning-mtp-jkls-gemm"

namespace mlir {
namespace heir {

using tensor_ext::AssignLayoutOp;
using tensor_ext::LayoutAttr;
using tensor_ext::TensorExtDialect;

#define GEN_PASS_DEF_TILEPLANNINGMTPJKLS
#include "lib/Transforms/TilePlanningMtpJkls/TilePlanningMtpJkls.h.inc"

struct TilePlanningMtpJkls
    : impl::TilePlanningMtpJklsBase<TilePlanningMtpJkls> {
  using TilePlanningMtpJklsBase::TilePlanningMtpJklsBase;

  void runOnOperation() override;
};

namespace {

// Extracts and memoizes (keyed by `tileId`, one of GemmTask's
// lhsTileId/rhsTileId/destTileId) the mu x mu tile of `source` located at
// logical tile-row `tileRow`, tile-column `tileCol` (offset (tileRow*mu,
// tileCol*mu)). Multiple tasks that share the same tileId -- e.g. P8.3's
// fixed lhs, which has the SAME lhsTileId for every destination-column task
// sharing that row tile -- reuse the exact same extracted SSA value rather
// than re-extracting it redundantly. This is also what lets a downstream
// test directly prove tile reuse (or, for distinct tileIds, tile
// distinctness) by operand identity rather than by structural shape alone.
Value getOrExtractTile(ImplicitLocOpBuilder& b, Value source,
                       llvm::DenseMap<int64_t, Value>& cache, int64_t tileId,
                       int64_t tileRow, int64_t tileCol, int64_t mu) {
  auto it = cache.find(tileId);
  if (it != cache.end()) {
    return it->second;
  }
  RankedTensorType sourceType = cast<RankedTensorType>(source.getType());
  RankedTensorType tileType =
      RankedTensorType::get({mu, mu}, sourceType.getElementType());
  SmallVector<OpFoldResult> offsets{b.getIndexAttr(tileRow * mu),
                                   b.getIndexAttr(tileCol * mu)};
  SmallVector<OpFoldResult> sizes{b.getIndexAttr(mu), b.getIndexAttr(mu)};
  SmallVector<OpFoldResult> strides(2, b.getIndexAttr(1));
  Value tile = tensor::ExtractSliceOp::create(b, tileType, source, offsets,
                                              sizes, strides);
  cache.insert({tileId, tile});
  return tile;
}

// Builds a [batchSize, mu, mu] LOGICAL batch tensor for one GEMM operand
// role (lhs, rhs, or init/dest), scattering each of `tasks`' tiles --
// located via `tileId`/`tileRow`/`tileCol`, reusing computeGemmTilePlan's own
// (i,q)/(q,j)/(i,j) task coordinates rather than recomputing tile indexing --
// into that task's own `insertIndex(task)` within a fresh, explicitly (and
// hence authoritative, per LayoutPropagation.cpp's destination-authority
// mechanism) zero-seeded batch of `batchSize` positions.
//
// Before P8.5, every caller passed `insertIndex = destTileId` and
// `batchSize = taskCount`: getMultiTileLayoutRelation's own construction
// (see lib/Utils/Layout/Utils.cpp) derives the physical (ct=group, slot)
// pair FROM the plain logical batch-tensor index via `ct = floor(tileId /
// tilesPerCiphertext)`, `p = tileId mod tilesPerCiphertext` internally --
// i.e. the mtpLayout attribute alone already encodes the group/position
// split for whatever plain index this function uses -- so a single
// [taskCount,mu,mu] batch indexed by destTileId could span multiple
// physical ciphertext groups directly.
//
// P8.5's per-group callers instead pass `insertIndex = position` and
// `batchSize = tilesPerCiphertext`, with `tasks` pre-filtered to one
// physical group: `position` (= destTileId mod tilesPerCiphertext) is
// unique WITHIN one group (though it collides ACROSS groups, e.g. destTileId
// 0 and destTileId 2 can share position 0 in different groups), so this
// produces a batch that is always exactly one physical ciphertext wide,
// regardless of how many groups the overall destination-tile grid spans.
// Passing fewer than `batchSize` tasks (a partially filled final group)
// leaves the remaining positions at their zero seed.
//
// Chained sibling inserts (one per task) are required here exactly as
// validated by the Phase 8 B2 spike fixture: the destination's fixed MTP
// addressing must survive unmodified through every insert, while each
// differently-positioned incoming tile is reconciled to it instead.
// `tasks` and `batchSize` are taken directly rather than a whole
// `GemmTilePlan` so that P8.4b and P8.5 can reuse this same, already-
// validated batch-assembly logic on an arbitrary subset of a plan's tasks
// (a per-q slice for P8.4b, a per-(q,group) slice for P8.5), not only on an
// entire plan's task list at once (the P8.2/P8.3 usage, where callers
// simply pass `plan->tasks, plan->taskCount`). For a single-task list
// (batchSize == 1, the P8.2 case) this reduces to exactly one iteration at
// index 0 and reproduces the prior assembleSingleTaskBatch's IR
// byte-for-byte.
Value assembleMultiTaskBatch(
    ImplicitLocOpBuilder& b, Value source, llvm::ArrayRef<GemmTask> tasks,
    int64_t batchSize, LayoutAttr mtpLayout, int64_t mu,
    llvm::DenseMap<int64_t, Value>& cache,
    llvm::function_ref<int64_t(const GemmTask&)> tileId,
    llvm::function_ref<int64_t(const GemmTask&)> tileRow,
    llvm::function_ref<int64_t(const GemmTask&)> tileCol,
    llvm::function_ref<int64_t(const GemmTask&)> insertIndex) {
  RankedTensorType sourceType = cast<RankedTensorType>(source.getType());
  Type elementType = sourceType.getElementType();
  RankedTensorType batchType =
      RankedTensorType::get({batchSize, mu, mu}, elementType);

  Value batch;
  for (const GemmTask& task : tasks) {
    // Extract (or reuse) this task's tile BEFORE lazily constructing the
    // zero-seeded authoritative destination on the first iteration --
    // matching assembleSingleTaskBatch's original op order (extract, then
    // zero/assign_layout, then insert) exactly, so the single-task (P8.2)
    // case reproduces byte-identical IR.
    Value tile = getOrExtractTile(b, source, cache, tileId(task),
                                  tileRow(task), tileCol(task), mu);
    if (!batch) {
      Value zero = arith::ConstantOp::create(
          b, batchType,
          DenseElementsAttr::get(batchType, b.getZeroAttr(elementType)));
      auto zeroLayoutOp = AssignLayoutOp::create(b, zero, mtpLayout);
      zeroLayoutOp->setAttr(TensorExtDialect::kLayoutAttrName, mtpLayout);
      batch = zeroLayoutOp.getResult();
    }
    SmallVector<OpFoldResult> insertOffsets{b.getIndexAttr(insertIndex(task)),
                                            b.getIndexAttr(0),
                                            b.getIndexAttr(0)};
    SmallVector<OpFoldResult> insertSizes{b.getIndexAttr(1), b.getIndexAttr(mu),
                                          b.getIndexAttr(mu)};
    SmallVector<OpFoldResult> insertStrides(3, b.getIndexAttr(1));
    batch = tensor::InsertSliceOp::create(b, tile, batch, insertOffsets,
                                          insertSizes, insertStrides);
  }
  if (!batch) {
    // tasks was empty (an entirely-absent group at this q -- not currently
    // producible by computeGemmTilePlan's own coverage guarantee, but
    // handled explicitly rather than left as a null Value): the batch is
    // simply its zero seed.
    RankedTensorType batchType2 =
        RankedTensorType::get({batchSize, mu, mu}, elementType);
    Value zero = arith::ConstantOp::create(
        b, batchType2,
        DenseElementsAttr::get(batchType2, b.getZeroAttr(elementType)));
    auto zeroLayoutOp = AssignLayoutOp::create(b, zero, mtpLayout);
    zeroLayoutOp->setAttr(TensorExtDialect::kLayoutAttrName, mtpLayout);
    batch = zeroLayoutOp.getResult();
  }
  return batch;
}

// Builds a fresh [1, mu, mu] batch tensor by extracting the mu x mu tile at
// logical tile-row `tileRow`, tile-column `tileCol` from `source`, and
// inserting it into a freshly zero-seeded, explicitly-laid-out (hence
// authoritative) [1,mu,mu] batch -- generalizing P8.2's original
// single-task gather to an arbitrary (tileRow,tileCol) instead of always
// (0,0). Used by the P8.4 (Q > 1, one destination) materialization below,
// once per q-step: unlike assembleMultiTaskBatch (which packs taskCount
// destination tiles into ONE shared batch), each q-step here gets its own
// independent [1,mu,mu] batch and its own linalg.batch_matmul, so that the
// Q partial products remain individually inspectable primitive
// multiplications rather than being folded into one opaque multi-task
// batch.
Value assembleOneTileBatch(ImplicitLocOpBuilder& b, Value source,
                           int64_t tileRow, int64_t tileCol, int64_t mu,
                           LayoutAttr mtpLayout) {
  RankedTensorType sourceType = cast<RankedTensorType>(source.getType());
  Type elementType = sourceType.getElementType();
  RankedTensorType tileType = RankedTensorType::get({mu, mu}, elementType);
  RankedTensorType batchType = RankedTensorType::get({1, mu, mu}, elementType);

  SmallVector<OpFoldResult> tileOffsets{b.getIndexAttr(tileRow * mu),
                                        b.getIndexAttr(tileCol * mu)};
  SmallVector<OpFoldResult> tileSizes{b.getIndexAttr(mu), b.getIndexAttr(mu)};
  SmallVector<OpFoldResult> tileStrides(2, b.getIndexAttr(1));
  Value tile = tensor::ExtractSliceOp::create(b, tileType, source, tileOffsets,
                                              tileSizes, tileStrides);

  Value zero = arith::ConstantOp::create(
      b, batchType, DenseElementsAttr::get(batchType, b.getZeroAttr(elementType)));
  auto zeroLayoutOp = AssignLayoutOp::create(b, zero, mtpLayout);
  zeroLayoutOp->setAttr(TensorExtDialect::kLayoutAttrName, mtpLayout);

  SmallVector<OpFoldResult> batchOffsets(3, b.getIndexAttr(0));
  SmallVector<OpFoldResult> batchSizes{b.getIndexAttr(1), b.getIndexAttr(mu),
                                       b.getIndexAttr(mu)};
  SmallVector<OpFoldResult> batchStrides(3, b.getIndexAttr(1));
  return tensor::InsertSliceOp::create(b, tile, zeroLayoutOp.getResult(),
                                       batchOffsets, batchSizes, batchStrides);
}

// Builds a fresh, explicitly-laid-out (authoritative) zero-seeded
// [taskCount,mu,mu] batch with no tile inserted -- the outs/accumulator for
// every q > 0 partial-product batch_matmul in the P8.4 (taskCount == 1) and
// P8.4b (taskCount > 1) reduction chains below, so that only q == 0's own
// batch_matmul (fed the real gathered Cinit tile(s) instead) ever
// contributes the real init.
Value zeroSeededBatch(ImplicitLocOpBuilder& b, int64_t taskCount, int64_t mu,
                      Type elementType, LayoutAttr mtpLayout) {
  RankedTensorType batchType =
      RankedTensorType::get({taskCount, mu, mu}, elementType);
  Value zero = arith::ConstantOp::create(
      b, batchType, DenseElementsAttr::get(batchType, b.getZeroAttr(elementType)));
  auto zeroLayoutOp = AssignLayoutOp::create(b, zero, mtpLayout);
  zeroLayoutOp->setAttr(TensorExtDialect::kLayoutAttrName, mtpLayout);
  return zeroLayoutOp.getResult();
}

// Builds a fresh, explicitly-laid-out (authoritative) zero-seeded [M,N]
// logical output accumulator whose physical layout is the SAME Convention-S
// MTP relation the [taskCount,mu,mu] batch itself uses --
// getMultiTileLayoutRelation on the plain [M,N] shape with rowAxis=0,
// columnAxis=1 (ordinary matrix row/column, exactly as the batch's own
// rowAxis=1, columnAxis=2 binds its axis 1/2), the same tileRows=tileColumns
// = mu, and -- critically -- the exact same `tilesPerCiphertext` the batch's
// own mtpLayout was already built with (never independently recomputed from
// the output's own shape).
//
// This is deliberately NOT getRowMajorLayoutRelation. Proof that this makes
// a completed destination tile's batch position and its corresponding
// position in this accumulator identical, not merely compatible after a
// conversion:
//
//   - computeGemmTilePlan defines destTileId = i*J + j (row-major over the
//     destination-tile grid, see lib/Utils/Layout/TilePlanningMtpJkls.cpp).
//   - getMultiTileLayoutRelation, given a [M,N] tensor with rowAxis=0,
//     columnAxis=1, computes qL = floor(row/mu) (= i), qD = floor(col/mu)
//     (= j), and flattens (qL,qD) row-major into its own internal `tileId`
//     as qL*J + qD -- algebraically the same formula as destTileId.
//   - Both then compute ct = floor(tileId/tilesPerCiphertext), p = tileId
//     mod tilesPerCiphertext, l = row mod mu, d = col mod mu, and
//     slot = d*(tilesPerCiphertext*mu) + p*mu + l from the identical
//     (tileId, l, d) triple -- so for the SAME (row,col) pair, the batch's
//     own per-task-tile physical location (at that task's destTileId, local
//     (l,d) = (row mod mu, col mod mu)) and this accumulator's own physical
//     location for logical (row,col) are the same (ct,slot), by
//     construction of the shared formula, not by a downstream reconciling
//     conversion. Scattering a finished destination tile into this
//     accumulator via an ordinary tensor.insert_slice therefore requires no
//     cross-ciphertext rotation, split, or tensor_ext.convert_layout: source
//     and destination already describe the same physical location.
FailureOr<std::pair<Value, LayoutAttr>> buildMtpOutputAccumulator(
    ImplicitLocOpBuilder& b, MLIRContext* ctx, RankedTensorType outputType,
    int64_t mu, int64_t tilesPerCiphertext, int64_t minSlotCount) {
  FailureOr<presburger::IntegerRelation> outputRelation =
      getMultiTileLayoutRelation(outputType, /*rowAxis=*/0, /*columnAxis=*/1,
                                 /*tileRows=*/mu, /*tileColumns=*/mu,
                                 tilesPerCiphertext, minSlotCount);
  if (failed(outputRelation)) {
    return failure();
  }
  LayoutAttr outputLayout =
      LayoutAttr::getFromIntegerRelation(ctx, *outputRelation);

  Type elementType = outputType.getElementType();
  Value outputZero = arith::ConstantOp::create(
      b, outputType, DenseElementsAttr::get(outputType, b.getZeroAttr(elementType)));
  auto outputZeroLayoutOp = AssignLayoutOp::create(b, outputZero, outputLayout);
  outputZeroLayoutOp->setAttr(TensorExtDialect::kLayoutAttrName, outputLayout);
  return std::make_pair(outputZeroLayoutOp.getResult(), outputLayout);
}

// P8.5: materializes a GEMM whose destination-tile grid spans more than one
// physical ciphertext (plan.numCiphertexts > 1) as one independent batch per
// PHYSICAL destination group, instead of one [taskCount,mu,mu] global batch
// spanning every group. This is what avoids ever needing a ciphertext-
// count-changing conversion when gathering lhs/rhs/init into a batch: each
// group's own [tilesPerCiphertext,mu,mu] batch always materializes to
// exactly one physical ciphertext, the same shape every pre-P8.5 single-
// group caller (P8.2/P8.3/P8.4) already gathers into successfully, so the
// gather machinery this reuses (assembleMultiTaskBatch, zeroSeededBatch) is
// never asked to place data across more physical ciphertexts than its own
// batch spans.
//
// A group's batch is indexed by GemmTask::position (unique within one
// group), NOT GemmTask::destTileId (unique globally, but not what
// distinguishes positions within a single-ciphertext-sized batch). Every
// group's batch uses `plan.tilesPerCiphertext` as its size (never the
// group's own, possibly smaller, actual task count), so a partially filled
// final group still gives every local position the identical slot
// placement the corresponding global output tile already has -- unfilled
// positions are simply left at their zero seed.
//
// The final per-destination scatter targets the SAME global [M,N] logical
// accumulator (`outputAccum`, built by the caller via
// buildMtpOutputAccumulator with the identical tilesPerCiphertext) the
// pre-P8.5 single-group paths already scatter into: that accumulator's own
// layout already places destTileId at its correct physical (group,
// position) location by construction (see buildMtpOutputAccumulator's own
// proof above), so a group-local physical position 0 becoming row `group`
// of the final physical container happens purely through this ordinary
// tensor.insert_slice -- ordinary ciphertext-semantic materialization of a
// ciphertext-spanning ordinary tensor insert, ALREADY validated by the
// existing secretScalarSecretTensor row-selection handling -- never through
// a tensor_ext.remap or a slot rotation.
//
// Returns failure() (without emitting malformed IR) only if the group
// batch's own layout relation cannot be constructed -- should not happen
// given the caller's own mtpRelation-equivalent construction already
// succeeded with the same mu/tilesPerCiphertext/minSlotCount.
FailureOr<Value> materializeMultiGroupGemm(ImplicitLocOpBuilder& b,
                                           MLIRContext* ctx, Value lhs,
                                           Value rhs, Value init,
                                           const GemmTilePlan& plan,
                                           int64_t mu, int64_t minSlotCount,
                                           Type elementType,
                                           Value outputAccum) {
  int64_t Q = plan.Q;
  int64_t taskCount = plan.taskCount;
  int64_t tilesPerCiphertext = plan.tilesPerCiphertext;
  int64_t numGroups = plan.numCiphertexts;
  llvm::ArrayRef<GemmTask> allTasks = plan.tasks;

  // Every group's batch shares this SAME [tilesPerCiphertext,mu,mu] type
  // and layout: `tileId` internal to getMultiTileLayoutRelation ranges over
  // [0,tilesPerCiphertext), so `ct = floor(tileId/tilesPerCiphertext)` is
  // always exactly 0 -- i.e. this relation, by construction, never admits
  // more than one physical ciphertext, regardless of group.
  //
  // NOTE (see this function's caller-facing report): giving each group's
  // batch its OWN layout, with the range-side ct variable shifted by `+g`
  // via shiftVar so it truthfully reports ct == g instead of ct == 0, was
  // also tried. It did not resolve the final-scatter blocker described
  // below, because layout-propagation's OWN (unmodified) tensor.extract_slice
  // layout computation for "extract position 0 of a [tilesPerCiphertext,mu,mu]
  // batch" independently reconstructs a canonical ct == 0 tile layout from
  // the slice's own shape, rather than substituting the source batch's
  // actual (possibly shifted) ct -- so the same "ct == 0 source, ct == group
  // destination" mismatch reappears at the same place regardless of what
  // ct this function's own batch layout claims. The shift also triggered a
  // segfault in a later pass once distinct per-group layouts coexisted, so
  // it is not used here; see the report for the exact evidence.
  RankedTensorType groupBatchType =
      RankedTensorType::get({tilesPerCiphertext, mu, mu}, elementType);
  FailureOr<presburger::IntegerRelation> groupRelation =
      getMultiTileLayoutRelation(groupBatchType, /*rowAxis=*/1,
                                 /*columnAxis=*/2, /*tileRows=*/mu,
                                 /*tileColumns=*/mu, tilesPerCiphertext,
                                 minSlotCount);
  if (failed(groupRelation)) {
    return failure();
  }
  LayoutAttr sharedGroupLayout =
      LayoutAttr::getFromIntegerRelation(ctx, *groupRelation);
  std::vector<LayoutAttr> groupLayout(static_cast<size_t>(numGroups),
                                      sharedGroupLayout);

  // q == 0's chunk has exactly one task per destination (GemmTask::group
  // and GemmTask::position depend only on destTileId = i*J+j, not q, so
  // this bucketing is valid for every q, not only q == 0), in increasing
  // destTileId order; grouping preserves increasing position within each
  // group since destTileId is contiguous per group by construction of
  // getBalancedMtpPacking (group = destTileId / tilesPerCiphertext,
  // position = destTileId % tilesPerCiphertext).
  llvm::ArrayRef<GemmTask> destTasks = allTasks.slice(0, taskCount);
  std::vector<std::vector<GemmTask>> tasksByGroup(numGroups);
  for (const GemmTask& t : destTasks) {
    tasksByGroup[t.group].push_back(t);
  }

  // One finished [tilesPerCiphertext,mu,mu] batch per group, reduced across
  // q independently -- exactly the existing single-group P8.4b reduction,
  // just repeated once per group instead of once for the whole plan.
  std::vector<Value> groupFinalBatch(static_cast<size_t>(numGroups));

  for (int64_t q = 0; q < Q; ++q) {
    llvm::ArrayRef<GemmTask> qTasks = allTasks.slice(
        static_cast<size_t>(q * taskCount), static_cast<size_t>(taskCount));

    // Per-q-scoped caches, shared ACROSS every group processed at this q
    // (not reset per group): a shared A or B source tile (identical
    // lhsTileId/rhsTileId, e.g. the same lhs row tile feeding multiple
    // destination groups) is extracted once here and its exact SSA value
    // is reused by every group that needs it, without aliasing distinct
    // groups' physical ciphertexts -- each group below still gets its own,
    // independently assembled and independently laid out, one-ciphertext
    // batch.
    llvm::DenseMap<int64_t, Value> lhsCache, rhsCache, initCache;

    std::vector<std::vector<GemmTask>> qTasksByGroup(numGroups);
    for (const GemmTask& t : qTasks) {
      qTasksByGroup[t.group].push_back(t);
    }

    for (int64_t g = 0; g < numGroups; ++g) {
      llvm::ArrayRef<GemmTask> groupTasks = qTasksByGroup[static_cast<size_t>(g)];
      LayoutAttr thisGroupLayout = groupLayout[static_cast<size_t>(g)];

      Value lhsBatch = assembleMultiTaskBatch(
          b, lhs, groupTasks, tilesPerCiphertext, thisGroupLayout, mu,
          lhsCache, [](const GemmTask& t) { return t.lhsTileId; },
          [](const GemmTask& t) { return t.i; },
          [](const GemmTask& t) { return t.q; },
          [](const GemmTask& t) { return t.position; });
      Value rhsBatch = assembleMultiTaskBatch(
          b, rhs, groupTasks, tilesPerCiphertext, thisGroupLayout, mu,
          rhsCache, [](const GemmTask& t) { return t.rhsTileId; },
          [](const GemmTask& t) { return t.q; },
          [](const GemmTask& t) { return t.j; },
          [](const GemmTask& t) { return t.position; });
      // The real Cinit destination tiles are included exactly once: only
      // q == 0's outs batch gathers them for THIS group; every later q
      // starts from a fresh zero batch of the same
      // [tilesPerCiphertext,mu,mu] shape.
      Value outsBatch =
          (q == 0)
              ? assembleMultiTaskBatch(
                    b, init, groupTasks, tilesPerCiphertext, thisGroupLayout,
                    mu, initCache,
                    [](const GemmTask& t) { return t.destTileId; },
                    [](const GemmTask& t) { return t.i; },
                    [](const GemmTask& t) { return t.j; },
                    [](const GemmTask& t) { return t.position; })
              : zeroSeededBatch(b, tilesPerCiphertext, mu, elementType,
                                thisGroupLayout);

      auto kernelAttr = secret::KernelAttr::get(
          ctx, KernelName::BatchMatmulMtpJkls, /*force=*/true);
      auto qGroupBatchMatmulOp = linalg::BatchMatmulOp::create(
          b, TypeRange{groupBatchType}, ValueRange{lhsBatch, rhsBatch},
          ValueRange{outsBatch});
      qGroupBatchMatmulOp->setAttr(secret::SecretDialect::kKernelAttrName,
                                   kernelAttr);
      qGroupBatchMatmulOp->setAttr(TensorExtDialect::kLayoutAttrName,
                                   thisGroupLayout);

      Value qGroupResult = qGroupBatchMatmulOp.getResult(0);
      groupFinalBatch[static_cast<size_t>(g)] =
          (q == 0) ? qGroupResult
                   : arith::AddFOp::create(
                         b, groupFinalBatch[static_cast<size_t>(g)],
                         qGroupResult)
                         .getResult();
    }
  }

  // Scatter each destination's finished tile -- extracted from its OWN
  // group's finished batch at its LOCAL position, never destTileId -- into
  // the shared global [M,N] accumulator. This insert is the same ordinary
  // per-tile scatter the pre-P8.5 single-group paths already use; no
  // group-aware logic is needed here because the accumulator's own layout
  // (built by the caller with the identical tilesPerCiphertext) already
  // places destTileId at the correct physical (group, position).
  for (int64_t g = 0; g < numGroups; ++g) {
    for (const GemmTask& task : tasksByGroup[static_cast<size_t>(g)]) {
      SmallVector<OpFoldResult> extractOffsets{b.getIndexAttr(task.position),
                                               b.getIndexAttr(0),
                                               b.getIndexAttr(0)};
      SmallVector<OpFoldResult> extractSizes{
          b.getIndexAttr(1), b.getIndexAttr(mu), b.getIndexAttr(mu)};
      SmallVector<OpFoldResult> extractStrides(3, b.getIndexAttr(1));
      Value tile = tensor::ExtractSliceOp::create(
          b, RankedTensorType::get({mu, mu}, elementType),
          groupFinalBatch[static_cast<size_t>(g)], extractOffsets,
          extractSizes, extractStrides);

      SmallVector<OpFoldResult> scatterOffsets{b.getIndexAttr(task.i * mu),
                                               b.getIndexAttr(task.j * mu)};
      SmallVector<OpFoldResult> scatterSizes{b.getIndexAttr(mu),
                                             b.getIndexAttr(mu)};
      SmallVector<OpFoldResult> scatterStrides(2, b.getIndexAttr(1));
      outputAccum = tensor::InsertSliceOp::create(
          b, tile, outputAccum, scatterOffsets, scatterSizes, scatterStrides);
    }
  }

  return outputAccum;
}

// If `bodyValue` is a block argument of the secret.generic body enclosing
// `opInsideGeneric`, AND that generic's own corresponding operand is
// itself a direct block argument of an enclosing func.func (i.e. a genuine
// function-boundary secret value, not a value computed by another op
// within the same generic), returns that func.func together with the
// argument index. Returns std::nullopt for every other case -- this
// function deliberately does not attempt to synthesize a function
// argument for a value that isn't one.
std::optional<std::pair<func::FuncOp, unsigned>> traceBlockArgToFuncArgument(
    Operation* opInsideGeneric, Value bodyValue) {
  auto genericOp = opInsideGeneric->getParentOfType<secret::GenericOp>();
  if (!genericOp) {
    return std::nullopt;
  }
  OpOperand* operand = genericOp.getOpOperandForBlockArgument(bodyValue);
  if (!operand) {
    return std::nullopt;
  }
  auto outerBlockArg = dyn_cast<BlockArgument>(operand->get());
  if (!outerBlockArg) {
    return std::nullopt;
  }
  auto funcOp =
      dyn_cast_or_null<func::FuncOp>(outerBlockArg.getOwner()->getParentOp());
  if (!funcOp) {
    return std::nullopt;
  }
  return std::make_pair(funcOp, outerBlockArg.getArgNumber());
}

// One multi-group GEMM's requirement that a specific function argument
// carry a specific Convention-S MTP layout, kept together with enough
// context (the requesting op and a human-readable role name) to produce
// clean diagnostics if this requirement conflicts with another one.
struct RequiredArgLayout {
  func::FuncOp funcOp;
  unsigned argIndex;
  LayoutAttr layout;
  linalg::MatmulOp op;
  StringRef role;
};

// P8.5 input-ABI integration: for every ELIGIBLE multi-group plan
// (plan->numCiphertexts > 1) among `eligibleOps`, derives the
// Convention-S MTP layout each of A (lhs), B (rhs), and Cinit (init) must
// carry for the existing, unmodified per-group implementation to gather
// every mu x mu tile from a single physical source ciphertext (proved
// sufficient by the P8.5 input-ABI checkpoint's fixture-only experiment),
// and -- only once every such requirement across the whole module has
// been collected and found free of conflicts -- annotates the
// corresponding OUTER func.func arguments with the standard
// tensor_ext.layout attribute. This is the exact mechanism
// LayoutPropagation already preserves (it treats an argument's own
// declared layout as authoritative) and client-pack generation already
// honors (its own boundary codegen reads argument layouts generically,
// with no P8.5-specific logic needed); no new attribute, operation, or
// encrypted preprocessing is introduced. Single-group plans
// (P8.2-P8.4b) are entirely unaffected: this function only ever looks at
// plans with numCiphertexts > 1.
//
// tensor_ext.assign_layout is deliberately not used here: per its own
// contract, it is for ingesting a plaintext value into a layout, not for
// declaring the layout of an already-encrypted secret value -- the
// function-argument tensor_ext.layout attribute is the correct mechanism
// for that, exactly as LayoutPropagation and client-pack generation
// already expect.
void annotateMultiGroupInputAbi(llvm::ArrayRef<linalg::MatmulOp> eligibleOps,
                                MLIRContext* ctx, int64_t mu,
                                int64_t minSlotCount) {
  std::vector<RequiredArgLayout> requirements;

  for (linalg::MatmulOp op : eligibleOps) {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Value init = op.getOutputs().front();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    auto rhsType = cast<RankedTensorType>(rhs.getType());
    auto initType = cast<RankedTensorType>(init.getType());

    FailureOr<GemmTilePlan> plan = computeGemmTilePlan(
        lhsType.getDimSize(0), lhsType.getDimSize(1), rhsType.getDimSize(1),
        mu, minSlotCount);
    if (failed(plan) || plan->numCiphertexts <= 1) {
      // Single-group (or otherwise ineligible): P8.2-P8.4b's existing IR
      // and boundary behavior must be left completely unaffected.
      continue;
    }

    for (auto& [value, type, role] :
         {std::make_tuple(lhs, lhsType, StringRef("lhs (A)")),
          std::make_tuple(rhs, rhsType, StringRef("rhs (B)")),
          std::make_tuple(init, initType, StringRef("init (Cinit)"))}) {
      FailureOr<presburger::IntegerRelation> relation =
          getMultiTileLayoutRelation(type, /*rowAxis=*/0, /*columnAxis=*/1,
                                     /*tileRows=*/mu, /*tileColumns=*/mu,
                                     plan->tilesPerCiphertext, minSlotCount);
      if (failed(relation)) {
        // Should not happen given computeGemmTilePlan already succeeded
        // with the same mu/tilesPerCiphertext/minSlotCount above; never
        // emit a spurious diagnostic or malformed IR on an unexpected
        // failure -- leave this operand's ABI selection alone instead.
        continue;
      }
      LayoutAttr required = LayoutAttr::getFromIntegerRelation(ctx, *relation);

      std::optional<std::pair<func::FuncOp, unsigned>> traced =
          traceBlockArgToFuncArgument(op, value);
      if (!traced) {
        op->emitRemark()
            << "P8.5 input-ABI selection: the " << role
            << " operand of this multi-group GEMM is not a direct "
               "function argument (it is produced by another operation), "
               "so its existing producer's layout is preserved unchanged; "
               "this operand may still require a downstream layout "
               "conversion depending on that producer's own layout";
        continue;
      }
      requirements.push_back(
          {traced->first, traced->second, required, op, role});
    }
  }

  if (requirements.empty()) {
    return;
  }

  // Collect every requirement before mutating anything, and detect
  // cross-GEMM conflicts (two selected multi-group GEMMs requiring
  // genuinely different layouts for the SAME function argument) in a
  // first, read-only pass over the complete collection. This ABI
  // selection is a best-effort optimization, not something the rest of
  // tile planning depends on to function correctly (an unannotated or
  // wrongly-annotated argument simply falls back to whatever layout
  // LayoutPropagation would otherwise have chosen, exactly as before this
  // feature existed) -- so a conflict is diagnosed cleanly and this
  // function makes NO mutation at all (not even to other, non-conflicting
  // arguments), per "before making any partial ABI changes", but does not
  // abort the rest of this pass.
  llvm::DenseMap<std::pair<Operation*, unsigned>, RequiredArgLayout*> byArg;
  for (RequiredArgLayout& req : requirements) {
    auto key = std::make_pair(req.funcOp.getOperation(), req.argIndex);
    auto it = byArg.find(key);
    if (it == byArg.end()) {
      byArg[key] = &req;
      continue;
    }
    if (it->second->layout != req.layout) {
      req.op->emitError()
          << "P8.5 input-ABI selection: two selected multi-group GEMMs "
             "require incompatible Convention-S MTP layouts for the "
             "same function argument #"
          << req.argIndex << " of @" << req.funcOp.getName() << " ("
          << it->second->role << " of one GEMM requires "
          << it->second->layout << "; " << req.role
          << " of another requires " << req.layout
          << "); refusing to make any partial ABI change";
      return;
    }
  }

  // No cross-GEMM conflicts: check each argument's own existing layout (if
  // any) against its single, agreed-upon requirement, and mutate only
  // where it is safe to do so. An incompatible pre-existing user layout on
  // one argument does not block annotating other, non-conflicting
  // arguments.
  for (auto& [key, reqPtr] : byArg) {
    RequiredArgLayout& req = *reqPtr;
    func::FuncOp funcOp = req.funcOp;
    unsigned argIndex = req.argIndex;
    Attribute existing =
        funcOp.getArgAttr(argIndex, TensorExtDialect::kLayoutAttrName);
    if (existing) {
      if (existing == req.layout) {
        continue;  // Already exactly the required layout: preserve as-is.
      }
      req.op->emitError()
          << "P8.5 input-ABI selection: function argument #" << argIndex
          << " of @" << funcOp.getName()
          << " already has an explicit tensor_ext.layout incompatible "
             "with the Convention-S MTP layout this multi-group GEMM's "
          << req.role
          << " operand requires; refusing to overwrite it (existing="
          << existing << ", required=" << req.layout << ")";
      continue;  // Do not overwrite.
    }
    funcOp.setArgAttr(argIndex, TensorExtDialect::kLayoutAttrName,
                      req.layout);
  }
}

}  // namespace

void TilePlanningMtpJkls::runOnOperation() {
  if (!enableMtpJklsGemm) return;

  if (mtpJklsGemmTileSize <= 0) {
    getOperation()->emitError()
        << "tile-planning-mtp-jkls-gemm is enabled but "
           "mtp-jkls-gemm-tile-size ("
        << mtpJklsGemmTileSize << ") is not positive";
    signalPassFailure();
    return;
  }
  int64_t mu = mtpJklsGemmTileSize;

  DataFlowSolver solver;
  dataflow::loadBaselineAnalyses(solver);
  solver.load<SecretnessAnalysis>();
  if (failed(solver.initializeAndRun(getOperation()))) {
    getOperation()->emitOpError() << "Failed to run secretness analysis.\n";
    signalPassFailure();
    return;
  }

  // Collect eligible ops first, then rewrite: mutating/erasing an op while
  // a walk over the same tree is in progress is unsafe in general, even
  // though linalg.matmul itself has no nested regions of interest here.
  std::vector<linalg::MatmulOp> eligibleOps;
  getOperation()->walk([&](linalg::MatmulOp op) {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    if (!isSecret(lhs, &solver) || !isSecret(rhs, &solver)) {
      return;  // pt-ct/ct-pt/cleartext matmul: leave untouched.
    }

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!lhsType || !rhsType || !resultType || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape() || !resultType.hasStaticShape() ||
        lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
        resultType.getRank() != 2) {
      return;
    }

    int64_t M = lhsType.getDimSize(0);
    int64_t K = lhsType.getDimSize(1);
    int64_t N = rhsType.getDimSize(1);
    if (rhsType.getDimSize(0) != K || resultType.getDimSize(0) != M ||
        resultType.getDimSize(1) != N) {
      return;  // Malformed matmul; leave for normal verification elsewhere.
    }

    FailureOr<GemmTilePlan> plan =
        computeGemmTilePlan(M, K, N, mu, minSlotCount);
    if (failed(plan)) {
      return;  // Ineligible mu/minSlotCount (e.g. mu*mu > minSlotCount):
               // fall through untouched.
    }

    // Every task must be a full, non-boundary mu x mu x mu tile: boundary
    // tiles (M, K, or N not evenly divisible by mu) require extracting only
    // the valid sub-rectangle and zero-padding the rest, which is a
    // deliberately separate, not-yet-implemented generalization (tracked for
    // a later phase). Rejecting here, rather than assuming full extents
    // always hold, is what keeps assembleMultiTaskBatch/assembleOneTileBatch
    // below safe: both always extract a full mu x mu region at each task's
    // tile offset. This check applies uniformly to every case below.
    bool hasBoundaryTile = llvm::any_of(plan->tasks, [&](const GemmTask& t) {
      return t.validRows != mu || t.validContraction != mu ||
             t.validColumns != mu;
    });
    if (hasBoundaryTile) {
      return;
    }

    // P8.5: plan->numCiphertexts > 1 (destination tiles spanning multiple
    // physical ciphertext groups) is no longer rejected here. The
    // [taskCount,mu,mu] batch this pass builds is always indexed by the
    // plain logical destTileId (see assembleMultiTaskBatch above), and
    // mtpLayout (built below from getMultiTileLayoutRelation with
    // plan->tilesPerCiphertext) already encodes that logical index's
    // physical (group, position) split -- the same multi-group layout
    // Phase 7's own automatic batch_matmul selection already materializes.
    // No additional per-group looping is needed in this pass's own IR
    // construction as a result.

    // Every (Q, taskCount) combination is eligible once the boundary guard
    // above passes (P8.5 also removed the single-ciphertext-group
    // restriction, so numCiphertexts is unconstrained here): Q == 1
    // (P8.2/P8.3), Q > 1 && taskCount == 1 (P8.4), or Q > 1 &&
    // taskCount > 1 (P8.4b/P8.5). The materialization loop dispatches on
    // the same (plan->Q, plan->taskCount) distinction.

    eligibleOps.push_back(op);
  });

  MLIRContext* ctx = &getContext();

  // P8.5 input-ABI integration: before materializing anything, give every
  // eligible multi-group plan's raw A/B/Cinit function arguments the
  // Convention-S MTP layout the per-group implementation needs to gather
  // every mu x mu tile from a single physical source ciphertext, so
  // unannotated logical inputs receive the same layouts the input-ABI
  // checkpoint's fixture-only experiment proved sufficient. Single-group
  // plans are untouched by this call (see annotateMultiGroupInputAbi). This
  // is a best-effort ABI optimization: a diagnosed conflict does not abort
  // the rest of this pass, which proceeds to materialize eligible GEMMs
  // exactly as it would have before this feature existed.
  annotateMultiGroupInputAbi(eligibleOps, ctx, mu, minSlotCount);

  for (linalg::MatmulOp op : eligibleOps) {
    Value lhs = op.getOperand(0);
    Value rhs = op.getOperand(1);
    Value init = op.getOutputs().front();
    auto lhsType = cast<RankedTensorType>(lhs.getType());
    Type elementType = lhsType.getElementType();

    mlir::IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(op);
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    // Recompute the plan here (cheap, pure arithmetic) rather than threading
    // it through eligibleOps: every field this loop needs (taskCount,
    // tilesPerCiphertext, and the tasks themselves) is deterministically
    // reproduced from the same M/K/N/mu/minSlotCount already validated above.
    FailureOr<GemmTilePlan> plan = computeGemmTilePlan(
        lhsType.getDimSize(0), lhsType.getDimSize(1),
        cast<RankedTensorType>(rhs.getType()).getDimSize(1), mu, minSlotCount);
    RankedTensorType batchType =
        RankedTensorType::get({plan->taskCount, mu, mu}, elementType);
    // Convention S: batch axis 1 is ordinary matrix row and maps to `l`;
    // batch axis 2 is ordinary matrix column and maps to `d`.
    FailureOr<presburger::IntegerRelation> mtpRelation =
        getMultiTileLayoutRelation(batchType, /*rowAxis=*/1, /*columnAxis=*/2,
                                   /*tileRows=*/mu, /*tileColumns=*/mu,
                                   plan->tilesPerCiphertext, minSlotCount);
    if (failed(mtpRelation)) {
      // Should not happen given computeGemmTilePlan already succeeded with
      // the same parameters above, but never emit malformed IR on an
      // unexpected failure -- leave this op untouched instead.
      continue;
    }
    LayoutAttr mtpLayout = LayoutAttr::getFromIntegerRelation(ctx, *mtpRelation);

    if (plan->Q > 1 && plan->taskCount == 1) {
      // P8.4: exactly one destination tile; derive its (i,j) from the
      // plan's own first task rather than assuming (0,0), and use
      // reductionChainForDest -- not a hand-rolled q-loop over plan->tasks
      // -- to get its Q contributing tasks in the planner's own
      // documented, strictly increasing-q accumulation order.
      const GemmTask& firstTask = plan->tasks.front();
      std::vector<GemmTask> chain =
          plan->reductionChainForDest(firstTask.i, firstTask.j);

      Value finalResult;
      for (size_t k = 0; k < chain.size(); ++k) {
        const GemmTask& task = chain[k];
        Value lhsTile =
            assembleOneTileBatch(b, lhs, task.i, task.q, mu, mtpLayout);
        Value rhsTile =
            assembleOneTileBatch(b, rhs, task.q, task.j, mu, mtpLayout);
        // The real Cinit is included exactly once: only q == 0's own
        // batch_matmul (the first entry of the reduction chain, per
        // reductionChainForDest's documented contract) is fed the real
        // gathered init tile as its outs; every later q starts from a
        // fresh zero, so the primitive's own outs-accumulation semantics
        // never re-add the init.
        Value outsTile = (k == 0)
            ? assembleOneTileBatch(b, init, task.i, task.j, mu, mtpLayout)
            : zeroSeededBatch(b, /*taskCount=*/1, mu, elementType, mtpLayout);

        auto kernelAttr = secret::KernelAttr::get(
            ctx, KernelName::BatchMatmulMtpJkls, /*force=*/true);
        auto qBatchMatmulOp = linalg::BatchMatmulOp::create(
            b, TypeRange{batchType}, ValueRange{lhsTile, rhsTile},
            ValueRange{outsTile});
        qBatchMatmulOp->setAttr(secret::SecretDialect::kKernelAttrName,
                                kernelAttr);
        qBatchMatmulOp->setAttr(TensorExtDialect::kLayoutAttrName, mtpLayout);

        SmallVector<OpFoldResult> resultOffsets(3, b.getIndexAttr(0));
        SmallVector<OpFoldResult> resultSizes{
            b.getIndexAttr(1), b.getIndexAttr(mu), b.getIndexAttr(mu)};
        SmallVector<OpFoldResult> resultStrides(3, b.getIndexAttr(1));
        Value resultTile = tensor::ExtractSliceOp::create(
            b, RankedTensorType::get({mu, mu}, elementType),
            qBatchMatmulOp.getResult(0), resultOffsets, resultSizes,
            resultStrides);

        // The reduction across q is a plain, standard arith.addf chain in
        // strictly increasing q order (matching reductionChainForDest's own
        // documented linear, not tree-shaped, accumulation order) --
        // visible in ordinary tensor/linalg/arith IR before any
        // primitive/kernel expansion, and structurally distinct from the
        // forced-kernel batch_matmul each partial product comes from.
        finalResult = (k == 0) ? resultTile
                               : arith::AddFOp::create(b, finalResult,
                                                       resultTile)
                                     .getResult();
      }

      rewriter.replaceOp(op, finalResult);
      continue;
    }

    if (plan->Q > 1 && plan->taskCount > 1) {
      if (plan->numCiphertexts > 1) {
        // P8.5: taskCount destinations span multiple physical ciphertext
        // groups. Dispatch to the dedicated per-group materialization
        // BEFORE building any [taskCount,mu,mu] global batch, so the
        // multi-ciphertext-spanning global batch the P8.4b path below
        // builds is never constructed in this case.
        RankedTensorType outputType =
            cast<RankedTensorType>(op->getResult(0).getType());
        FailureOr<std::pair<Value, LayoutAttr>> outputAccumulatorResult =
            buildMtpOutputAccumulator(b, ctx, outputType, mu,
                                      plan->tilesPerCiphertext, minSlotCount);
        if (failed(outputAccumulatorResult)) {
          continue;
        }
        FailureOr<Value> multiGroupResult = materializeMultiGroupGemm(
            b, ctx, lhs, rhs, init, *plan, mu, minSlotCount, elementType,
            outputAccumulatorResult->first);
        if (failed(multiGroupResult)) {
          continue;
        }
        rewriter.replaceOp(op, *multiGroupResult);
        continue;
      }

      // P8.4b: taskCount > 1 destinations, each independently accumulating
      // Q > 1 contraction-tile partial products, all sharing a single
      // physical ciphertext group (numCiphertexts == 1, established above).
      // Follows the preferred representation: for each q, assemble one
      // shared [taskCount,mu,mu] lhs/rhs/outs batch (in destTileId order)
      // and run ONE forced batch_matmul over the whole q-batch, then reduce
      // the Q per-q result batches (in increasing-q order) and scatter each
      // destination exactly once.
      //
      // plan->tasks is already laid out, per GemmTilePlan::tasks' own
      // documented canonical order (outer q, inner destTileId), as Q
      // contiguous chunks of taskCount tasks each -- so each q's chunk is
      // exactly `plan->tasks.slice(q * taskCount, taskCount)`, reusing the
      // planner's own task list and ArrayRef::slice rather than
      // recomputing any per-q task selection.
      int64_t taskCount = plan->taskCount;
      int64_t Q = plan->Q;
      llvm::ArrayRef<GemmTask> allTasks = plan->tasks;

      std::vector<Value> qResults;
      qResults.reserve(static_cast<size_t>(Q));
      for (int64_t q = 0; q < Q; ++q) {
        llvm::ArrayRef<GemmTask> qTasks =
            allTasks.slice(static_cast<size_t>(q * taskCount),
                          static_cast<size_t>(taskCount));

        // Fresh per-q caches: lhs/rhs SSA reuse is scoped to within this
        // q's own batch (required -- a destination-position task's
        // lhsTileId/rhsTileId always encodes q, so a cache shared across q
        // would never actually collide, but resetting here makes
        // per-q-scoped reuse an intentional property of the construction,
        // not an accident of the cache's lifetime).
        llvm::DenseMap<int64_t, Value> lhsCache, rhsCache, initCache;
        Value lhsBatch = assembleMultiTaskBatch(
            b, lhs, qTasks, taskCount, mtpLayout, mu, lhsCache,
            [](const GemmTask& t) { return t.lhsTileId; },
            [](const GemmTask& t) { return t.i; },
            [](const GemmTask& t) { return t.q; },
            [](const GemmTask& t) { return t.destTileId; });
        Value rhsBatch = assembleMultiTaskBatch(
            b, rhs, qTasks, taskCount, mtpLayout, mu, rhsCache,
            [](const GemmTask& t) { return t.rhsTileId; },
            [](const GemmTask& t) { return t.q; },
            [](const GemmTask& t) { return t.j; },
            [](const GemmTask& t) { return t.destTileId; });
        // The real Cinit destination tiles are included exactly once: only
        // q == 0's outs batch gathers them (one real tile per destination,
        // via the same assembleMultiTaskBatch machinery); every later q
        // starts from a fresh zero batch of the same [taskCount,mu,mu]
        // shape.
        Value outsBatch =
            (q == 0)
                ? assembleMultiTaskBatch(
                      b, init, qTasks, taskCount, mtpLayout, mu, initCache,
                      [](const GemmTask& t) { return t.destTileId; },
                      [](const GemmTask& t) { return t.i; },
                      [](const GemmTask& t) { return t.j; },
                      [](const GemmTask& t) { return t.destTileId; })
                : zeroSeededBatch(b, taskCount, mu, elementType, mtpLayout);

        auto kernelAttr = secret::KernelAttr::get(
            ctx, KernelName::BatchMatmulMtpJkls, /*force=*/true);
        auto qBatchMatmulOp = linalg::BatchMatmulOp::create(
            b, TypeRange{batchType}, ValueRange{lhsBatch, rhsBatch},
            ValueRange{outsBatch});
        qBatchMatmulOp->setAttr(secret::SecretDialect::kKernelAttrName,
                                kernelAttr);
        qBatchMatmulOp->setAttr(TensorExtDialect::kLayoutAttrName, mtpLayout);
        qResults.push_back(qBatchMatmulOp.getResult(0));
      }

      // Reduce across q (strictly increasing order, matching
      // reductionChainForDest's own documented per-destination contract)
      // and scatter each destination's final tile into its own output
      // region. Every reduction step at a given destination identity
      // (destTileId, NOT position -- position collides across ciphertext
      // groups once taskCount > tilesPerCiphertext, P8.5) uses ONLY that
      // same destTileId across every q's batch -- destTileId is stable
      // across q by construction (it depends only on (i,j), not q) -- so
      // destinations are never cross-reduced even when they land in
      // different physical ciphertext groups.
      RankedTensorType outputType =
          cast<RankedTensorType>(op->getResult(0).getType());
      FailureOr<std::pair<Value, LayoutAttr>> outputAccumulatorResult =
          buildMtpOutputAccumulator(b, ctx, outputType, mu,
                                    plan->tilesPerCiphertext, minSlotCount);
      if (failed(outputAccumulatorResult)) {
        // Should not happen: the batch's own mtpRelation above already
        // succeeded with the identical mu/tilesPerCiphertext/minSlotCount,
        // and the output's capacity check depends only on those, not on
        // shape/rank (already validated static/rank-2 by the eligibility
        // walk). Never emit malformed IR on an unexpected failure.
        continue;
      }
      Value outputAccum = outputAccumulatorResult->first;

      // q=0's chunk has exactly one task per destination, in destTileId
      // order -- reused directly instead of recomputing the (i,j) <->
      // destTileId correspondence.
      llvm::ArrayRef<GemmTask> destTasks =
          allTasks.slice(0, static_cast<size_t>(taskCount));
      for (const GemmTask& destTask : destTasks) {
        Value finalTile;
        for (int64_t q = 0; q < Q; ++q) {
          SmallVector<OpFoldResult> extractOffsets{
              b.getIndexAttr(destTask.destTileId), b.getIndexAttr(0),
              b.getIndexAttr(0)};
          SmallVector<OpFoldResult> extractSizes{
              b.getIndexAttr(1), b.getIndexAttr(mu), b.getIndexAttr(mu)};
          SmallVector<OpFoldResult> extractStrides(3, b.getIndexAttr(1));
          Value tile = tensor::ExtractSliceOp::create(
              b, RankedTensorType::get({mu, mu}, elementType),
              qResults[static_cast<size_t>(q)], extractOffsets, extractSizes,
              extractStrides);
          finalTile = (q == 0)
              ? tile
              : arith::AddFOp::create(b, finalTile, tile).getResult();
        }

        SmallVector<OpFoldResult> scatterOffsets{
            b.getIndexAttr(destTask.i * mu), b.getIndexAttr(destTask.j * mu)};
        SmallVector<OpFoldResult> scatterSizes{b.getIndexAttr(mu),
                                               b.getIndexAttr(mu)};
        SmallVector<OpFoldResult> scatterStrides(2, b.getIndexAttr(1));
        outputAccum = tensor::InsertSliceOp::create(
            b, finalTile, outputAccum, scatterOffsets, scatterSizes,
            scatterStrides);
      }

      rewriter.replaceOp(op, outputAccum);
      continue;
    }

    if (plan->numCiphertexts > 1) {
      // P8.5: Q == 1's own multi-group case (e.g. the required Q=1
      // capacity-boundary fixture): taskCount destinations span multiple
      // physical ciphertext groups even though there is only one
      // contraction step. Dispatch to the same dedicated per-group
      // materialization used by the Q > 1 case above (with Q == 1 as its
      // single, degenerate q-iteration) BEFORE building any
      // [taskCount,mu,mu] global batch, so the multi-ciphertext-spanning
      // global batch the P8.2/P8.3 path below builds is never constructed
      // in this case.
      RankedTensorType outputType =
          cast<RankedTensorType>(op->getResult(0).getType());
      FailureOr<std::pair<Value, LayoutAttr>> outputAccumulatorResult =
          buildMtpOutputAccumulator(b, ctx, outputType, mu,
                                    plan->tilesPerCiphertext, minSlotCount);
      if (failed(outputAccumulatorResult)) {
        continue;
      }
      FailureOr<Value> multiGroupResult = materializeMultiGroupGemm(
          b, ctx, lhs, rhs, init, *plan, mu, minSlotCount, elementType,
          outputAccumulatorResult->first);
      if (failed(multiGroupResult)) {
        continue;
      }
      rewriter.replaceOp(op, *multiGroupResult);
      continue;
    }

    // P8.2/P8.3: Q == 1, single physical ciphertext group (numCiphertexts
    // == 1, established above). lhsTileId = i*Q+q identifies each task's
    // source lhs tile at (row=i, col=q); with Q == 1 established above,
    // every task sharing a given i shares the same lhsTileId (P8.3's "same
    // lhs feeds multiple destination-column tasks" case) and
    // getOrExtractTile reuses that extraction. rhsTileId = q*J+j and
    // destTileId = i*J+j are each unique per (q,j) and per (i,j)
    // respectively, so rhs and init/dest tiles are
    // never spuriously shared across distinct tasks.
    llvm::DenseMap<int64_t, Value> lhsCache, rhsCache, initCache;
    Value lhsBatch = assembleMultiTaskBatch(
        b, lhs, plan->tasks, plan->taskCount, mtpLayout, mu, lhsCache,
        [](const GemmTask& t) { return t.lhsTileId; },
        [](const GemmTask& t) { return t.i; },
        [](const GemmTask& t) { return t.q; },
        [](const GemmTask& t) { return t.destTileId; });
    Value rhsBatch = assembleMultiTaskBatch(
        b, rhs, plan->tasks, plan->taskCount, mtpLayout, mu, rhsCache,
        [](const GemmTask& t) { return t.rhsTileId; },
        [](const GemmTask& t) { return t.q; },
        [](const GemmTask& t) { return t.j; },
        [](const GemmTask& t) { return t.destTileId; });
    // The real init/outs operand is gathered exactly like lhs/rhs, one real
    // Cinit tile per destination. This is what preserves linalg.matmul's own
    // `result = init + lhs @ rhs` semantics through the rewrite: the
    // forced-kernel batch_matmul below accumulates onto whatever is passed
    // as its own `outs`, and the JKLS primitive it reuses already correctly
    // does so, per batch element, for a real, possibly nonzero init
    // (validated by the Phase 8 spike's B3 fixture).
    Value initBatch = assembleMultiTaskBatch(
        b, init, plan->tasks, plan->taskCount, mtpLayout, mu, initCache,
        [](const GemmTask& t) { return t.destTileId; },
        [](const GemmTask& t) { return t.i; },
        [](const GemmTask& t) { return t.j; },
        [](const GemmTask& t) { return t.destTileId; });

    auto kernelAttr = secret::KernelAttr::get(ctx, KernelName::BatchMatmulMtpJkls,
                                              /*force=*/true);
    auto batchMatmulOp = linalg::BatchMatmulOp::create(
        b, TypeRange{batchType}, ValueRange{lhsBatch, rhsBatch},
        ValueRange{initBatch});
    batchMatmulOp->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);
    batchMatmulOp->setAttr(TensorExtDialect::kLayoutAttrName, mtpLayout);

    Value finalResult;
    if (plan->taskCount == 1) {
      // The single destination tile IS the whole output already (the
      // established P8.2 behavior): extract it directly, with no separate
      // scatter into a shared destination.
      SmallVector<OpFoldResult> resultOffsets(3, b.getIndexAttr(0));
      SmallVector<OpFoldResult> resultSizes{
          b.getIndexAttr(1), b.getIndexAttr(mu), b.getIndexAttr(mu)};
      SmallVector<OpFoldResult> resultStrides(3, b.getIndexAttr(1));
      finalResult = tensor::ExtractSliceOp::create(
          b, RankedTensorType::get({mu, mu}, elementType),
          batchMatmulOp.getResult(0), resultOffsets, resultSizes,
          resultStrides);
    } else {
      // P8.3: multiple destination tiles. Scatter each task's result tile
      // into its own (task.i*mu, task.j*mu) region of a fresh, explicitly
      // (and hence authoritative) row-major MxN output accumulator -- the
      // same destination-authority scatter pattern already validated for
      // the standalone Phase 8 B2 LayoutPropagation fixture, now driven end
      // to end from an ordinary linalg.matmul. Every (i,j) in [0,I)x[0,J) has
      // exactly one task (Q == 1 and no boundary tiles were both established
      // above), so every element of the accumulator is written exactly once
      // and the initial zero never survives to the final result.
      RankedTensorType outputType =
          cast<RankedTensorType>(op->getResult(0).getType());
      FailureOr<std::pair<Value, LayoutAttr>> outputAccumulatorResult =
          buildMtpOutputAccumulator(b, ctx, outputType, mu,
                                    plan->tilesPerCiphertext, minSlotCount);
      if (failed(outputAccumulatorResult)) {
        // Should not happen: the batch's own mtpRelation above already
        // succeeded with the identical mu/tilesPerCiphertext/minSlotCount,
        // and the output's capacity check depends only on those, not on
        // shape/rank (already validated static/rank-2 by the eligibility
        // walk). Never emit malformed IR on an unexpected failure.
        continue;
      }
      Value outputAccum = outputAccumulatorResult->first;
      for (const GemmTask& task : plan->tasks) {
        SmallVector<OpFoldResult> extractOffsets{
            b.getIndexAttr(task.destTileId), b.getIndexAttr(0),
            b.getIndexAttr(0)};
        SmallVector<OpFoldResult> extractSizes{
            b.getIndexAttr(1), b.getIndexAttr(mu), b.getIndexAttr(mu)};
        SmallVector<OpFoldResult> extractStrides(3, b.getIndexAttr(1));
        Value resultTile = tensor::ExtractSliceOp::create(
            b, RankedTensorType::get({mu, mu}, elementType),
            batchMatmulOp.getResult(0), extractOffsets, extractSizes,
            extractStrides);

        SmallVector<OpFoldResult> scatterOffsets{
            b.getIndexAttr(task.i * mu), b.getIndexAttr(task.j * mu)};
        SmallVector<OpFoldResult> scatterSizes{b.getIndexAttr(mu),
                                               b.getIndexAttr(mu)};
        SmallVector<OpFoldResult> scatterStrides(2, b.getIndexAttr(1));
        outputAccum = tensor::InsertSliceOp::create(
            b, resultTile, outputAccum, scatterOffsets, scatterSizes,
            scatterStrides);
      }
      finalResult = outputAccum;
    }

    rewriter.replaceOp(op, finalResult);
  }
}

}  // namespace heir
}  // namespace mlir
