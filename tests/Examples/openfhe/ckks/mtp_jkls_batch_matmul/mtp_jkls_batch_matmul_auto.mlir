// Phase 7: automatic-selection counterpart of mtp_jkls_batch_matmul.mlir. No
// secret.kernel or tensor_ext.layout anywhere in this source — MTP-JKLS
// selection and layout construction happen automatically via
// enable-mtp-jkls=true (see BUILD).

module {
  func.func @mtp_auto_one_ct(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) -> !secret.secret<tensor<2x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) {
    ^body(%lhs: tensor<2x2x2xf32>, %rhs: tensor<2x2x2xf32>):
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x2x2xf32>, tensor<2x2x2xf32>) outs(%cst : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
      secret.yield %1 : tensor<2x2x2xf32>
    } -> !secret.secret<tensor<2x2x2xf32>>
    return %0 : !secret.secret<tensor<2x2x2xf32>>
  }
}
