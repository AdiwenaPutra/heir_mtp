// RUN: not heir-opt --layout-propagation=min-slot-count=8 %s

// Phase 7 default-off control: the same shape that mtp_jkls_batch_matmul_auto.mlir
// selects MTP-JKLS for, but with enable-mtp-jkls omitted (defaults to false).
// Existing behavior must be unchanged: this square mu=2 shape is not tricyclic-
// eligible either (gcd(mu, mu) != 1), so layout propagation must fail exactly
// as it did before Phase 7 existed (a clean pass failure via
// notifyMatchFailure, which is silent on a bare IRRewriter — there is no
// diagnostic text to FileCheck here, only the non-crashing failing exit code),
// not silently select MTP.

module {
  func.func @mtp_auto_disabled_by_default(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) -> !secret.secret<tensor<2x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) {
    ^body(%lhs: tensor<2x2x2xf32>, %rhs: tensor<2x2x2xf32>):
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x2x2xf32>, tensor<2x2x2xf32>) outs(%cst : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
      secret.yield %1 : tensor<2x2x2xf32>
    } -> !secret.secret<tensor<2x2x2xf32>>
    return %0 : !secret.secret<tensor<2x2x2xf32>>
  }
}
