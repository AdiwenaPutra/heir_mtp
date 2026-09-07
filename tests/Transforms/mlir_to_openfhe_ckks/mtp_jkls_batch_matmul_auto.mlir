// RUN: heir-opt --annotate-module="backend=openfhe scheme=ckks" --mlir-to-ckks='min-slot-count=8 enable-mtp-jkls=true' --scheme-to-openfhe='entry-function=mtp_auto_one_ct' %s | FileCheck %s

// Phase 7: automatic-selection input (no secret.kernel, no tensor_ext.layout)
// reached through the main --mlir-to-ckks pipeline with enable-mtp-jkls=true,
// all the way to the OpenFHE dialect. Same established Phase 6 evidence as
// mtp_jkls_batch_matmul.mlir (rotation keys, mul/relin/mod-reduce management),
// now produced without any handwritten MTP contract in the source.

module {
  // CHECK: ![[CT:.*]] = !openfhe.ciphertext
  // CHECK: ![[PT:.*]] = !openfhe.plaintext
  // CHECK: func.func @mtp_auto_one_ct__preprocessing
  // CHECK-SAME: -> memref<6x![[PT]]>
  // CHECK-COUNT-6: openfhe.make_ckks_packed_plaintext

  // CHECK: func.func @mtp_auto_one_ct__preprocessed
  // CHECK: openfhe.fast_rotation_precompute
  // CHECK: openfhe.mul_plain
  // CHECK: openfhe.fast_rotation
  // CHECK: openfhe.mul_no_relin
  // CHECK: openfhe.relin_inplace
  // CHECK: openfhe.mod_reduce

  // CHECK: func.func @mtp_auto_one_ct(
  // CHECK-SAME: tensor<1x![[CT]]>

  // CHECK: func.func @mtp_auto_one_ct__encrypt__arg0
  // CHECK-SAME: tensor<2x2x2xf32>
  // CHECK: openfhe.encrypt
  // CHECK: func.func @mtp_auto_one_ct__decrypt__result0
  // CHECK-SAME: -> tensor<2x2x2xf32>
  // CHECK: openfhe.decrypt

  // CHECK: func.func @mtp_auto_one_ct__configure_crypto_context
  // CHECK: openfhe.gen_mulkey
  // CHECK: openfhe.gen_rotkey
  // CHECK-SAME: indices = array<i64: 7, 4, 1>
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
