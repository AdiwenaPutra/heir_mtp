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
// role (lhs, rhs, or init/dest), scattering each of `plan`'s tasks' tiles --
// located via `tileId`/`tileRow`/`tileCol`, reusing computeGemmTilePlan's own
// (i,q)/(q,j)/(i,j) task coordinates rather than recomputing tile indexing --
// into that task's own `position` within a fresh, explicitly (and hence
// authoritative, per LayoutPropagation.cpp's destination-authority
// mechanism) zero-seeded batch. Chained sibling inserts (one per task) are
// required here exactly as validated by the Phase 8 B2 spike fixture: the
// destination's fixed MTP addressing must survive unmodified through every
// insert, while each differently-positioned incoming tile is reconciled to
// it instead. For a single-task plan (taskCount == 1, the P8.2 case) this
// reduces to exactly one iteration at position 0 and reproduces the prior
// assembleSingleTaskBatch's IR byte-for-byte.
Value assembleMultiTaskBatch(
    ImplicitLocOpBuilder& b, Value source, const GemmTilePlan& plan,
    LayoutAttr mtpLayout, int64_t mu, llvm::DenseMap<int64_t, Value>& cache,
    llvm::function_ref<int64_t(const GemmTask&)> tileId,
    llvm::function_ref<int64_t(const GemmTask&)> tileRow,
    llvm::function_ref<int64_t(const GemmTask&)> tileCol) {
  RankedTensorType sourceType = cast<RankedTensorType>(source.getType());
  Type elementType = sourceType.getElementType();
  RankedTensorType batchType =
      RankedTensorType::get({plan.taskCount, mu, mu}, elementType);

  Value batch;
  for (const GemmTask& task : plan.tasks) {
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

    if (plan->Q != 1) {
      // A contraction depth spanning more than one tile requires chaining
      // the batch_matmul's `outs` across q-steps to accumulate partial
      // products -- that reduction is P8.4, not this checkpoint. Leave any
      // such shape (a B3-style case) completely untouched.
      return;
    }

    // Every task must be a full, non-boundary mu x mu x mu tile: boundary
    // tiles (M, K, or N not evenly divisible by mu) require extracting only
    // the valid sub-rectangle and zero-padding the rest, which is a
    // deliberately separate, not-yet-implemented generalization (tracked for
    // a later phase). Rejecting here, rather than assuming full extents
    // always hold, is what keeps assembleMultiTaskBatch below safe: it always
    // extracts a full mu x mu region at each task's tile offset.
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
      // partially-correct single-group batch.
      return;
    }

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

    // lhsTileId = i*Q+q identifies each task's source lhs tile at (row=i,
    // col=q); with Q == 1 established above, every task sharing a given i
    // shares the same lhsTileId (P8.3's "same lhs feeds multiple
    // destination-column tasks" case) and getOrExtractTile reuses that
    // extraction. rhsTileId = q*J+j and destTileId = i*J+j are each unique
    // per (q,j) and per (i,j) respectively, so rhs and init/dest tiles are
    // never spuriously shared across distinct tasks.
    llvm::DenseMap<int64_t, Value> lhsCache, rhsCache, initCache;
    Value lhsBatch = assembleMultiTaskBatch(
        b, lhs, *plan, mtpLayout, mu, lhsCache,
        [](const GemmTask& t) { return t.lhsTileId; },
        [](const GemmTask& t) { return t.i; },
        [](const GemmTask& t) { return t.q; });
    Value rhsBatch = assembleMultiTaskBatch(
        b, rhs, *plan, mtpLayout, mu, rhsCache,
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
        b, init, *plan, mtpLayout, mu, initCache,
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
