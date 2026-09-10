// Convention S: l (tensor axis 1) is the local matrix row, d (tensor axis 2)
// is the local matrix column, slot = d*(tilesPerCiphertext*mu) + p*mu + l.
#mtp = #tensor_ext.layout<"{ [p, l, d] -> [ct, slot] : 0 <= p <= 3 and 0 <= l <= 1 and 0 <= d <= 1 and 0 <= ct <= 1 and 0 <= slot <= 7 and exists q : p - 2 * ct - q = 0 and slot - 4 * d - 2 * q - l = 0 and 0 <= q <= 1 }">
#kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>

module {
  func.func @mtp_jkls_batch_matmul_multi_ct(
      %arg0: !secret.secret<tensor<4x2x2xf32>> {tensor_ext.layout = #mtp},
      %arg1: !secret.secret<tensor<4x2x2xf32>> {tensor_ext.layout = #mtp})
      -> (!secret.secret<tensor<4x2x2xf32>> {tensor_ext.layout = #mtp}) {
    %zero = arith.constant dense<0.0> : tensor<4x2x2xf32>
    %result = secret.generic(
        %arg0: !secret.secret<tensor<4x2x2xf32>> {tensor_ext.layout = #mtp},
        %arg1: !secret.secret<tensor<4x2x2xf32>> {tensor_ext.layout = #mtp}) {
    ^body(%lhs: tensor<4x2x2xf32>, %rhs: tensor<4x2x2xf32>):
      %init = tensor_ext.assign_layout %zero
          {layout = #mtp, tensor_ext.layout = #mtp}
          : tensor<4x2x2xf32>
      %product = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #mtp
        } ins(%lhs, %rhs : tensor<4x2x2xf32>, tensor<4x2x2xf32>)
          outs(%init : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
      secret.yield %product : tensor<4x2x2xf32>
    } -> (!secret.secret<tensor<4x2x2xf32>> {tensor_ext.layout = #mtp})
    return %result : !secret.secret<tensor<4x2x2xf32>>
  }
}
