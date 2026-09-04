// RUN: heir-opt --annotate-module="backend=openfhe scheme=ckks" --mlir-to-ckks='min-slot-count=8' --scheme-to-openfhe='entry-function=mtp_jkls_batch_matmul' %s | FileCheck %s

#mtp = #tensor_ext.layout<"{ [p, d, l] -> [ct, slot] : ct = 0 and slot - 4 * d - 2 * p - l = 0 and 0 <= p <= 1 and 0 <= d <= 1 and 0 <= l <= 1 and 0 <= slot <= 7 }">
#kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>

module {
  // CHECK: ![[CT:.*]] = !openfhe.ciphertext
  // CHECK: ![[PT:.*]] = !openfhe.plaintext
  // CHECK: func.func @mtp_jkls_batch_matmul__preprocessing
  // CHECK-SAME: -> memref<6x![[PT]]>
  // CHECK-COUNT-6: openfhe.make_ckks_packed_plaintext

  // CHECK: func.func @mtp_jkls_batch_matmul__preprocessed
  // CHECK: openfhe.fast_rotation_precompute
  // CHECK: openfhe.mul_plain
  // CHECK: openfhe.fast_rotation
  // CHECK: openfhe.mul_no_relin
  // CHECK: openfhe.relin_inplace
  // CHECK: openfhe.mod_reduce

  // CHECK: func.func @mtp_jkls_batch_matmul(
  // CHECK-SAME: tensor<1x![[CT]]>

  // CHECK: func.func @mtp_jkls_batch_matmul__encrypt__arg0
  // CHECK-SAME: tensor<2x2x2xf32>
  // CHECK: openfhe.encrypt
  // CHECK: func.func @mtp_jkls_batch_matmul__decrypt__result0
  // CHECK-SAME: -> tensor<2x2x2xf32>
  // CHECK: openfhe.decrypt

  // CHECK: func.func @mtp_jkls_batch_matmul__configure_crypto_context
  // CHECK: openfhe.gen_mulkey
  // CHECK: openfhe.gen_rotkey
  // CHECK-SAME: indices = array<i64: 7, 4, 1>
  func.func @mtp_jkls_batch_matmul(
      %arg0: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp},
      %arg1: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp})
      -> (!secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp}) {
    %zero = arith.constant dense<0.0> : tensor<2x2x2xf32>
    %result = secret.generic(
        %arg0: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp},
        %arg1: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #mtp}) {
    ^body(%lhs: tensor<2x2x2xf32>, %rhs: tensor<2x2x2xf32>):
      %init = tensor_ext.assign_layout %zero
          {layout = #mtp, tensor_ext.layout = #mtp}
          : tensor<2x2x2xf32>
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
