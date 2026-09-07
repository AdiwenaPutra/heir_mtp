// Phase 7: automatic-selection counterpart of mtp_jkls_batch_matmul_multi_ct.mlir
// (batch=4, evenly packed 2 tiles/ciphertext). No secret.kernel or
// tensor_ext.layout anywhere in this source.

module {
  func.func @mtp_auto_multi_ct_even(%arg0: !secret.secret<tensor<4x2x2xf32>>, %arg1: !secret.secret<tensor<4x2x2xf32>>) -> !secret.secret<tensor<4x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<4x2x2xf32>>, %arg1: !secret.secret<tensor<4x2x2xf32>>) {
    ^body(%lhs: tensor<4x2x2xf32>, %rhs: tensor<4x2x2xf32>):
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<4x2x2xf32>, tensor<4x2x2xf32>) outs(%cst : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
      secret.yield %1 : tensor<4x2x2xf32>
    } -> !secret.secret<tensor<4x2x2xf32>>
    return %0 : !secret.secret<tensor<4x2x2xf32>>
  }
}
