// RUN: heir-opt --layout-propagation='min-slot-count=20 enable-mtp-jkls=true' --mlir-print-local-scope %s | FileCheck %s

// The counterexample that actually distinguishes balanced packing from an
// earlier, incorrect maximum-fill tile-capacity formula: batch=7, mu=2,
// min-slot-count=20 -> capacity=floor(20/4)=5, numCiphertexts=ceil(7/5)=2.
//
//   old (incorrect) maximum-fill: tilesPerCiphertext = min(7,5) = 5
//   current balanced packing:     tilesPerCiphertext = ceil(7/2) = 4
//
// ConvertLinalgBatchMatmul::mtpJklsKernel independently recovers
// tilesPerCiphertext as ceil(batch/numCiphertexts) = ceil(7/2) = 4 from the
// physical shape alone, so the old formula's 5 would not match, and
// materialization would fail outright (see the companion test of the same
// name under tests/Transforms/convert_to_ciphertext_semantics). This file
// only checks that selection succeeds; unlike batch=3 against capacity=2 (in
// mtp_jkls_batch_matmul_auto.mlir), batch=9 against capacity=5, or other
// pairs where the two formulas coincidentally agree despite a nonzero
// remainder, this (batch, capacity) pair is a real disagreement.

module {
  // CHECK: func.func @mtp_auto_disagreement
  func.func @mtp_auto_disagreement(%arg0: !secret.secret<tensor<7x2x2xf32>>, %arg1: !secret.secret<tensor<7x2x2xf32>>) -> !secret.secret<tensor<7x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<7x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<7x2x2xf32>>, %arg1: !secret.secret<tensor<7x2x2xf32>>) {
    ^body(%lhs: tensor<7x2x2xf32>, %rhs: tensor<7x2x2xf32>):
      // CHECK: linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = false>
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<7x2x2xf32>, tensor<7x2x2xf32>) outs(%cst : tensor<7x2x2xf32>) -> tensor<7x2x2xf32>
      secret.yield %1 : tensor<7x2x2xf32>
    } -> !secret.secret<tensor<7x2x2xf32>>
    return %0 : !secret.secret<tensor<7x2x2xf32>>
  }
}
