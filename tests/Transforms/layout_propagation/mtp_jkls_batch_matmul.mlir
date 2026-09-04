// RUN: heir-opt --layout-propagation=min-slot-count=8 %s | FileCheck %s

#mtp = #tensor_ext.layout<"{ [p, d, l] -> [ct, slot] : ct = 0 and slot - 4 * d - 2 * p - l = 0 and 0 <= p <= 1 and 0 <= d <= 1 and 0 <= l <= 1 and 0 <= slot <= 7 }">
#kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>

module {
  // CHECK: func.func @mtp_jkls_batch_matmul
  // CHECK-SAME: tensor_ext.layout = #{{.*}}
  func.func @mtp_jkls_batch_matmul(
      %arg0: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp},
      %arg1: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp})
      -> (!secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp}) {
    %zero = arith.constant dense<0.0> : tensor<2x2x2xf32>
    %result = secret.generic(
        %arg0: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp},
        %arg1: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp}) {
    ^body(%lhs: tensor<2x2x2xf32>, %rhs: tensor<2x2x2xf32>):
      // CHECK: tensor_ext.assign_layout
      // CHECK-SAME: layout = #{{.*}}
      %init = tensor_ext.assign_layout %zero
          {layout = #mtp, tensor_ext.layout = #mtp}
          : tensor<2x2x2xf32>
      // CHECK: linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #kernel
      // CHECK-SAME: tensor_ext.layout = #{{.*}}
      %product = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #mtp
        } ins(%lhs, %rhs : tensor<2x2x2xf32>, tensor<2x2x2xf32>)
          outs(%init : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
      secret.yield %product : tensor<2x2x2xf32>
    } -> (!secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp})
    return %result : !secret.secret<tensor<2x2x2xf32>>
  }
}
