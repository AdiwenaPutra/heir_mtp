// RUN: heir-opt --layout-propagation='min-slot-count=8 enable-mtp-jkls=true' --mlir-print-local-scope %s | FileCheck %s

// Automatic MTP-JKLS selection for ordinary, unannotated secret-secret
// linalg.batch_matmul, with no handwritten secret.kernel or
// tensor_ext.layout anywhere in the source. Covers one ciphertext, an evenly
// filled multi-ciphertext case, and an unevenly filled multi-ciphertext case
// (balanced packing leaves one MTP tile position unused). See
// mtp_jkls_batch_matmul_auto_disagreement.mlir in this directory for a case
// that distinguishes balanced packing from an earlier, incorrect
// maximum-fill formula — this file's uneven case does not (both formulas
// happen to agree for batch=3, capacity=2).

module {
  // CHECK: func.func @mtp_auto_one_ct
  func.func @mtp_auto_one_ct(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) -> !secret.secret<tensor<2x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) {
    ^body(%lhs: tensor<2x2x2xf32>, %rhs: tensor<2x2x2xf32>):
      // CHECK: tensor_ext.convert_layout
      // CHECK: tensor_ext.convert_layout
      // CHECK: linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = false>
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x2x2xf32>, tensor<2x2x2xf32>) outs(%cst : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
      secret.yield %1 : tensor<2x2x2xf32>
    } -> !secret.secret<tensor<2x2x2xf32>>
    return %0 : !secret.secret<tensor<2x2x2xf32>>
  }

  // CHECK: func.func @mtp_auto_multi_ct_even
  func.func @mtp_auto_multi_ct_even(%arg0: !secret.secret<tensor<4x2x2xf32>>, %arg1: !secret.secret<tensor<4x2x2xf32>>) -> !secret.secret<tensor<4x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<4x2x2xf32>>, %arg1: !secret.secret<tensor<4x2x2xf32>>) {
    ^body(%lhs: tensor<4x2x2xf32>, %rhs: tensor<4x2x2xf32>):
      // CHECK: linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = false>
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<4x2x2xf32>, tensor<4x2x2xf32>) outs(%cst : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
      secret.yield %1 : tensor<4x2x2xf32>
    } -> !secret.secret<tensor<4x2x2xf32>>
    return %0 : !secret.secret<tensor<4x2x2xf32>>
  }

  // Uneven case: batch=3, mu=2, min-slot-count=8 -> capacity=2, numCiphertexts
  // = ceil(3/2) = 2, tilesPerCiphertext = ceil(3/2) = 2 (one unused tile
  // position in the second ciphertext). Proves selection and layout
  // construction handle partial occupancy; the balanced-packing formula and
  // an earlier, incorrect maximum-fill formula happen to agree for this
  // particular (batch, capacity) pair, so this case alone does not
  // distinguish them — see mtp_jkls_batch_matmul_auto_disagreement.mlir for
  // one that does.
  // CHECK: func.func @mtp_auto_multi_ct_uneven
  func.func @mtp_auto_multi_ct_uneven(%arg0: !secret.secret<tensor<3x2x2xf32>>, %arg1: !secret.secret<tensor<3x2x2xf32>>) -> !secret.secret<tensor<3x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x2x2xf32>>, %arg1: !secret.secret<tensor<3x2x2xf32>>) {
    ^body(%lhs: tensor<3x2x2xf32>, %rhs: tensor<3x2x2xf32>):
      // CHECK: linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = false>
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<3x2x2xf32>, tensor<3x2x2xf32>) outs(%cst : tensor<3x2x2xf32>) -> tensor<3x2x2xf32>
      secret.yield %1 : tensor<3x2x2xf32>
    } -> !secret.secret<tensor<3x2x2xf32>>
    return %0 : !secret.secret<tensor<3x2x2xf32>>
  }
}
