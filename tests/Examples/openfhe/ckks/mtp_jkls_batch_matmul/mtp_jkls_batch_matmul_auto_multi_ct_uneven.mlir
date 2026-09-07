// The uneven-batch encrypted regression (batch=3, mu=2, min-slot-count=8 ->
// capacity=2, numCiphertexts=2, tilesPerCiphertext=2 balanced -> physical
// [2,8] with one unused tile position in the second ciphertext). The
// corresponding test harness's encrypted oracle comparison over all 12 valid
// logical outputs proves the unused tile position does not contaminate the
// valid tile sharing its ciphertext, through real client packing,
// encryption, and homomorphic evaluation — not just at the abstract-DAG
// level. It does not by itself observe the unused position's raw value (the
// generated decrypt helper only unpacks the 12 real logical positions); that
// the position is zero-initialized is a property of the generated client
// packing code, inspectable separately. No secret.kernel or
// tensor_ext.layout anywhere in this source.

module {
  func.func @mtp_auto_multi_ct_uneven(%arg0: !secret.secret<tensor<3x2x2xf32>>, %arg1: !secret.secret<tensor<3x2x2xf32>>) -> !secret.secret<tensor<3x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<3x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<3x2x2xf32>>, %arg1: !secret.secret<tensor<3x2x2xf32>>) {
    ^body(%lhs: tensor<3x2x2xf32>, %rhs: tensor<3x2x2xf32>):
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<3x2x2xf32>, tensor<3x2x2xf32>) outs(%cst : tensor<3x2x2xf32>) -> tensor<3x2x2xf32>
      secret.yield %1 : tensor<3x2x2xf32>
    } -> !secret.secret<tensor<3x2x2xf32>>
    return %0 : !secret.secret<tensor<3x2x2xf32>>
  }
}
