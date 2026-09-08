#include "lib/Transforms/TilePlanningMtpJkls/TilePlanningMtpJkls.h"

#include <cstdint>
#include <vector>

#include "lib/Analysis/SecretnessAnalysis/SecretnessAnalysis.h"
#include "lib/Dialect/Secret/IR/SecretAttributes.h"
#include "lib/Dialect/Secret/IR/SecretDialect.h"
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

// Builds a [taskCount, mu, mu] physical batch tensor for one GEMM operand
// role (lhs, rhs, or init/dest), scattering each of `tasks`' tiles --
// located via `tileId`/`tileRow`/`tileCol`, reusing computeGemmTilePlan's own
// (i,q)/(q,j)/(i,j) task coordinates rather than recomputing tile indexing --
// into that task's own `position` within a fresh, explicitly (and hence
// authoritative, per LayoutPropagation.cpp's destination-authority
// mechanism) zero-seeded batch. Chained sibling inserts (one per task) are
// required here exactly as validated by the Phase 8 B2 spike fixture: the
// destination's fixed MTP addressing must survive unmodified through every
// insert, while each differently-positioned incoming tile is reconciled to
// it instead. `tasks` and `taskCount` are taken directly rather than a
// whole `GemmTilePlan` so that P8.4b can reuse this same, already-validated
// batch-assembly logic on a per-q SLICE of a plan's tasks (see below), not
// only on an entire plan's task list at once (the P8.2/P8.3 usage, where
// callers simply pass `plan->tasks, plan->taskCount`). For a single-task
// list (taskCount == 1, the P8.2 case) this reduces to exactly one
// iteration at position 0 and reproduces the prior assembleSingleTaskBatch's
// IR byte-for-byte.
Value assembleMultiTaskBatch(
    ImplicitLocOpBuilder& b, Value source, llvm::ArrayRef<GemmTask> tasks,
    int64_t taskCount, LayoutAttr mtpLayout, int64_t mu,
    llvm::DenseMap<int64_t, Value>& cache,
    llvm::function_ref<int64_t(const GemmTask&)> tileId,
    llvm::function_ref<int64_t(const GemmTask&)> tileRow,
    llvm::function_ref<int64_t(const GemmTask&)> tileCol) {
  RankedTensorType sourceType = cast<RankedTensorType>(source.getType());
  Type elementType = sourceType.getElementType();
  RankedTensorType batchType =
      RankedTensorType::get({taskCount, mu, mu}, elementType);

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
    SmallVector<OpFoldResult> insertOffsets{b.getIndexAttr(task.position),
                                            b.getIndexAttr(0),
                                            b.getIndexAttr(0)};
    SmallVector<OpFoldResult> insertSizes{b.getIndexAttr(1), b.getIndexAttr(mu),
                                          b.getIndexAttr(mu)};
    SmallVector<OpFoldResult> insertStrides(3, b.getIndexAttr(1));
    batch = tensor::InsertSliceOp::create(b, tile, batch, insertOffsets,
                                          insertSizes, insertStrides);
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

    if (plan->numCiphertexts != 1) {
      // More than one physical ciphertext group means the taskCount
      // destination tiles cannot all be assembled into a single [taskCount,
      // mu, mu] batch tensor addressed by getMultiTileLayoutRelation alone --
      // scattering across multiple ciphertext groups is out-of-scope
      // multi-group scheduling work, not this checkpoint's tile-planning
      // generalization. Leave such shapes untouched rather than emit a
      // partially-correct single-group batch. (For plan->taskCount == 1 --
      // the P8.4 branch below -- this is always satisfied, since one
      // destination tile trivially fits in one ciphertext whenever
      // mu*mu <= minSlotCount, which computeGemmTilePlan already required to
      // succeed at all; the check still runs uniformly here rather than
      // being assumed.)
      return;
    }

    // Every (Q, taskCount) combination is eligible once the boundary and
    // single-ciphertext-group guards above pass: Q == 1 (P8.2/P8.3),
    // Q > 1 && taskCount == 1 (P8.4), or Q > 1 && taskCount > 1 (P8.4b,
    // the integration below). The materialization loop dispatches on the
    // same (plan->Q, plan->taskCount) distinction.

    eligibleOps.push_back(op);
  });

  MLIRContext* ctx = &getContext();
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
    FailureOr<presburger::IntegerRelation> mtpRelation =
        getMultiTileLayoutRelation(batchType, /*rowAxis=*/2, /*columnAxis=*/1,
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
      // P8.4b: taskCount > 1 destinations, each independently accumulating
      // Q > 1 contraction-tile partial products, within one ciphertext
      // group. Follows the preferred representation: for each q, assemble
      // one shared [taskCount,mu,mu] lhs/rhs/outs batch (in destination-
      // position order) and run ONE forced batch_matmul over the whole
      // q-batch, then reduce the Q per-q result batches (in increasing-q
      // order) and scatter each destination position exactly once.
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
            [](const GemmTask& t) { return t.q; });
        Value rhsBatch = assembleMultiTaskBatch(
            b, rhs, qTasks, taskCount, mtpLayout, mu, rhsCache,
            [](const GemmTask& t) { return t.rhsTileId; },
            [](const GemmTask& t) { return t.q; },
            [](const GemmTask& t) { return t.j; });
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
                      [](const GemmTask& t) { return t.j; })
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
      // region. Every reduction step at a given destination position p
      // uses ONLY that same position p across every q's batch -- positions
      // are stable across q by construction (they depend only on
      // destTileId, not q) -- so destinations are never cross-reduced.
      RankedTensorType outputType =
          cast<RankedTensorType>(op->getResult(0).getType());
      presburger::IntegerRelation outputRelation =
          getRowMajorLayoutRelation(outputType, minSlotCount);
      LayoutAttr outputLayout =
          LayoutAttr::getFromIntegerRelation(ctx, outputRelation);

      Value outputZero = arith::ConstantOp::create(
          b, outputType,
          DenseElementsAttr::get(outputType, b.getZeroAttr(elementType)));
      auto outputZeroLayoutOp =
          AssignLayoutOp::create(b, outputZero, outputLayout);
      outputZeroLayoutOp->setAttr(TensorExtDialect::kLayoutAttrName,
                                  outputLayout);
      Value outputAccum = outputZeroLayoutOp.getResult();

      // q=0's chunk has exactly one task per destination, in destTileId
      // (== position, since taskCount == tilesPerCiphertext here) order --
      // reused directly instead of recomputing the (i,j) <-> position
      // correspondence.
      llvm::ArrayRef<GemmTask> destTasks =
          allTasks.slice(0, static_cast<size_t>(taskCount));
      for (const GemmTask& destTask : destTasks) {
        Value finalTile;
        for (int64_t q = 0; q < Q; ++q) {
          SmallVector<OpFoldResult> extractOffsets{
              b.getIndexAttr(destTask.position), b.getIndexAttr(0),
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

    // lhsTileId = i*Q+q identifies each task's source lhs tile at (row=i,
    // col=q); with Q == 1 established above, every task sharing a given i
    // shares the same lhsTileId (P8.3's "same lhs feeds multiple
    // destination-column tasks" case) and getOrExtractTile reuses that
    // extraction. rhsTileId = q*J+j and destTileId = i*J+j are each unique
    // per (q,j) and per (i,j) respectively, so rhs and init/dest tiles are
    // never spuriously shared across distinct tasks.
    llvm::DenseMap<int64_t, Value> lhsCache, rhsCache, initCache;
    Value lhsBatch = assembleMultiTaskBatch(
        b, lhs, plan->tasks, plan->taskCount, mtpLayout, mu, lhsCache,
        [](const GemmTask& t) { return t.lhsTileId; },
        [](const GemmTask& t) { return t.i; },
        [](const GemmTask& t) { return t.q; });
    Value rhsBatch = assembleMultiTaskBatch(
        b, rhs, plan->tasks, plan->taskCount, mtpLayout, mu, rhsCache,
        [](const GemmTask& t) { return t.rhsTileId; },
        [](const GemmTask& t) { return t.q; },
        [](const GemmTask& t) { return t.j; });
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
        [](const GemmTask& t) { return t.j; });

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
      presburger::IntegerRelation outputRelation =
          getRowMajorLayoutRelation(outputType, minSlotCount);
      LayoutAttr outputLayout =
          LayoutAttr::getFromIntegerRelation(ctx, outputRelation);

      Value outputZero = arith::ConstantOp::create(
          b, outputType,
          DenseElementsAttr::get(outputType, b.getZeroAttr(elementType)));
      auto outputZeroLayoutOp =
          AssignLayoutOp::create(b, outputZero, outputLayout);
      outputZeroLayoutOp->setAttr(TensorExtDialect::kLayoutAttrName,
                                  outputLayout);

      Value outputAccum = outputZeroLayoutOp.getResult();
      for (const GemmTask& task : plan->tasks) {
        SmallVector<OpFoldResult> extractOffsets{b.getIndexAttr(task.position),
                                                 b.getIndexAttr(0),
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
