// RUN: not heir-opt --convert-to-ciphertext-semantics=min-slot-count=8 %s 2>&1 | FileCheck %s

// This batch-major layout has the same logical and physical shapes as the MTP
// layout, but it does not use the required d -> p -> l physical order.
#not_mtp = #tensor_ext.layout<"{ [p, d, l] -> [ct, slot] : ct = 0 and slot - 4 * p - 2 * d - l = 0 and 0 <= p <= 1 and 0 <= d <= 1 and 0 <= l <= 1 and 0 <= slot <= 7 }">
#kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>

module {
  // CHECK: MTP-JKLS requires matching row-major multi-tile layouts
  func.func @reject_non_mtp_layout(
      %arg0: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #not_mtp},
      %arg1: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #not_mtp})
      -> (!secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #not_mtp}) {
    %zero = arith.constant dense<0.0> : tensor<2x2x2xf32>
    %result = secret.generic(
        %arg0: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #not_mtp},
        %arg1: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #not_mtp}) {
    ^body(%lhs: tensor<2x2x2xf32>, %rhs: tensor<2x2x2xf32>):
      %init = tensor_ext.assign_layout %zero
          {layout = #not_mtp, tensor_ext.layout = #not_mtp}
          : tensor<2x2x2xf32>
      %product = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #not_mtp
        } ins(%lhs, %rhs : tensor<2x2x2xf32>, tensor<2x2x2xf32>)
          outs(%init : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
      secret.yield %product : tensor<2x2x2xf32>
    } -> (!secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #not_mtp})
    return %result : !secret.secret<tensor<2x2x2xf32>>
  }
}
