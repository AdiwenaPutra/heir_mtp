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

// Builds a [1, mu, mu] tensor by extracting the mu x mu sub-rectangle of
// `source` at offset (0,0) and inserting it into a fresh zero-seeded,
// explicitly laid-out batch tensor -- the same zero-seed-plus-valid-extent
// scatter primitive already validated by hand-written fixtures for this
// pass's eventual generalization to multiple tasks. This always extracts
// the FULL mu x mu source with no boundary handling: it is only ever
// called on an operand already confirmed (by the caller's exact
// M == K == N == mu eligibility check, not by computeGemmTilePlan alone --
// see the caller) to be exactly mu x mu, never smaller.
Value assembleSingleTaskBatch(ImplicitLocOpBuilder& b, Value source,
                              int64_t mu, LayoutAttr mtpLayout) {
  RankedTensorType sourceType = cast<RankedTensorType>(source.getType());
  Type elementType = sourceType.getElementType();
  RankedTensorType tileType = RankedTensorType::get({mu, mu}, elementType);
  RankedTensorType batchType = RankedTensorType::get({1, mu, mu}, elementType);

  SmallVector<OpFoldResult> tileOffsets(2, b.getIndexAttr(0));
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

    if (M != mu || K != mu || N != mu) {
      // Anything other than an exact full mu x mu x mu tile is out of scope
      // for this P8.2 checkpoint -- including a smaller-than-mu single-task
      // shape, which computeGemmTilePlan still legitimately reports as
      // taskCount=1 (I=Q=J=1) with a boundary (validRows/validContraction/
      // validColumns < mu): that is a real, well-defined plan, but
      // assembleSingleTaskBatch below always extracts a full mu x mu region
      // at offset (0,0), which would run out-of-bounds -- and hence produce
      // invalid IR, not just an imprecise result -- for anything smaller.
      // Handling that boundary correctly (extract the valid sub-rectangle,
      // zero-pad) is exactly the P8.3+ generalization, not this checkpoint.
      return;
    }

    FailureOr<GemmTilePlan> plan =
        computeGemmTilePlan(M, K, N, mu, minSlotCount);
    if (failed(plan)) {
      return;  // Ineligible mu/minSlotCount (e.g. mu*mu > minSlotCount):
               // fall through untouched.
    }
    if (plan->taskCount != 1 || plan->Q != 1) {
      // Unreachable given M=K=N=mu above (which always yields I=Q=J=1), but
      // never rewrite on an assumption alone -- if this weren't
      // taskCount=1, it would be a general B2/B3/B6 case, out of scope for
      // this checkpoint.
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

    RankedTensorType batchType = RankedTensorType::get({1, mu, mu}, elementType);
    // taskCount == 1 was already established above, so
    // getBalancedMtpPacking(1, mu, mu, minSlotCount) -- called again inside
    // computeGemmTilePlan -- always yields tilesPerCiphertext == 1; recompute
    // the plan here (cheap, pure arithmetic) rather than threading it through
    // eligibleOps, since only tilesPerCiphertext is needed at this point.
    FailureOr<GemmTilePlan> plan = computeGemmTilePlan(
        lhsType.getDimSize(0), lhsType.getDimSize(1),
        cast<RankedTensorType>(rhs.getType()).getDimSize(1), mu, minSlotCount);
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

    Value lhsBatch = assembleSingleTaskBatch(b, lhs, mu, mtpLayout);
    Value rhsBatch = assembleSingleTaskBatch(b, rhs, mu, mtpLayout);
    // The real init/outs operand is gathered exactly like lhs/rhs. This is
    // what preserves linalg.matmul's own `result = init + lhs @ rhs`
    // semantics through the rewrite: the forced-kernel batch_matmul below
    // accumulates onto whatever is passed as its own `outs`, and the JKLS
    // primitive it reuses already correctly does so for a real, possibly
    // nonzero init (validated by the Phase 8 spike's B3 fixture).
    Value initBatch = assembleSingleTaskBatch(b, init, mu, mtpLayout);

    auto kernelAttr = secret::KernelAttr::get(ctx, KernelName::BatchMatmulMtpJkls,
                                              /*force=*/true);
    auto batchMatmulOp = linalg::BatchMatmulOp::create(
        b, TypeRange{batchType}, ValueRange{lhsBatch, rhsBatch},
        ValueRange{initBatch});
    batchMatmulOp->setAttr(secret::SecretDialect::kKernelAttrName, kernelAttr);
    batchMatmulOp->setAttr(TensorExtDialect::kLayoutAttrName, mtpLayout);

    SmallVector<OpFoldResult> resultOffsets(3, b.getIndexAttr(0));
    SmallVector<OpFoldResult> resultSizes{b.getIndexAttr(1), b.getIndexAttr(mu),
                                          b.getIndexAttr(mu)};
    SmallVector<OpFoldResult> resultStrides(3, b.getIndexAttr(1));
    Value resultTile = tensor::ExtractSliceOp::create(
        b, RankedTensorType::get({mu, mu}, elementType),
        batchMatmulOp.getResult(0), resultOffsets, resultSizes, resultStrides);

    // M = N = mu exactly for this single-task (I = J = 1) case, so
    // resultTile's type already exactly matches the original op's result
    // type: it IS the whole output, and no separate scatter into a shared
    // destination is needed here (that step is what P8.3's I*J > 1 case
    // requires).
    rewriter.replaceOp(op, resultTile);
  }
}

}  // namespace heir
}  // namespace mlir
