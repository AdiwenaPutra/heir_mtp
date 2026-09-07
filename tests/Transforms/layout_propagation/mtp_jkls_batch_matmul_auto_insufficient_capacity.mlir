// RUN: not heir-opt --layout-propagation='min-slot-count=2 enable-mtp-jkls=true' %s

// Phase 7 graceful rejection: enable-mtp-jkls=true, and the shape would
// otherwise be MTP-eligible, but a single mu=2 tile (mu*mu=4 slots) does not
// fit in min-slot-count=2. Eligibility must reject this without overflow or a
// crash, and — since this square shape is not tricyclic-eligible either —
// layout propagation must fail cleanly (non-crashing exit code) exactly as
// before Phase 7, via the same silent notifyMatchFailure path (no diagnostic
// text to FileCheck against — see mtp_jkls_batch_matmul_auto_disabled.mlir).

module {
  func.func @mtp_auto_insufficient_capacity(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) -> !secret.secret<tensor<2x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) {
    ^body(%lhs: tensor<2x2x2xf32>, %rhs: tensor<2x2x2xf32>):
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x2x2xf32>, tensor<2x2x2xf32>) outs(%cst : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
      secret.yield %1 : tensor<2x2x2xf32>
    } -> !secret.secret<tensor<2x2x2xf32>>
    return %0 : !secret.secret<tensor<2x2x2xf32>>
  }
}
