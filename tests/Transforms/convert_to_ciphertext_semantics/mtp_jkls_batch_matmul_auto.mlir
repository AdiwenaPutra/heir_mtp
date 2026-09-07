// RUN: heir-opt --layout-propagation='min-slot-count=8 enable-mtp-jkls=true' --convert-to-ciphertext-semantics=min-slot-count=8 %s | FileCheck %s

// Automatic-selection input (no secret.kernel, no tensor_ext.layout
// anywhere in this source) run through layout propagation AND materialization,
// covering one ciphertext, an evenly filled multi-ciphertext case, and an
// unevenly filled multi-ciphertext case (balanced packing leaves one unused
// tile position). linalg.batch_matmul must disappear, physical types must be
// [ciphertexts, min-slot-count], and MTP rotations, repeated dense masks, and
// ciphertext-ciphertext multiply/add must appear — the same materialization
// already proven for the forced path in mtp_jkls_batch_matmul.mlir, now
// reached without any handwritten contract. This file's uneven case
// (batch=3, capacity=2) does not distinguish balanced packing from an
// earlier, incorrect maximum-fill formula, since both happen to agree for
// that pair — see mtp_jkls_batch_matmul_auto_disagreement.mlir for a case
// that does.

module {
  // CHECK: func.func @mtp_auto_one_ct
  // CHECK-SAME: tensor<1x8xf32>
  // CHECK-NOT: linalg.batch_matmul
  // CHECK-DAG: tensor_ext.rotate
  // CHECK-DAG: arith.constant dense<
  // CHECK-DAG: arith.mulf
  // CHECK-DAG: arith.addf
  // CHECK: return
  func.func @mtp_auto_one_ct(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) -> !secret.secret<tensor<2x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<2x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x2x2xf32>>, %arg1: !secret.secret<tensor<2x2x2xf32>>) {
    ^body(%lhs: tensor<2x2x2xf32>, %rhs: tensor<2x2x2xf32>):
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<2x2x2xf32>, tensor<2x2x2xf32>) outs(%cst : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>
      secret.yield %1 : tensor<2x2x2xf32>
    } -> !secret.secret<tensor<2x2x2xf32>>
    return %0 : !secret.secret<tensor<2x2x2xf32>>
  }

  // CHECK: func.func @mtp_auto_multi_ct_even
  // CHECK-SAME: tensor<2x8xf32>
  // CHECK-NOT: linalg.batch_matmul
  // CHECK-DAG: tensor_ext.rotate
  // CHECK-DAG: arith.constant dense<
  // CHECK-DAG: arith.mulf
  // CHECK-DAG: arith.addf
  // CHECK: return
  func.func @mtp_auto_multi_ct_even(%arg0: !secret.secret<tensor<4x2x2xf32>>, %arg1: !secret.secret<tensor<4x2x2xf32>>) -> !secret.secret<tensor<4x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<4x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<4x2x2xf32>>, %arg1: !secret.secret<tensor<4x2x2xf32>>) {
    ^body(%lhs: tensor<4x2x2xf32>, %rhs: tensor<4x2x2xf32>):
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<4x2x2xf32>, tensor<4x2x2xf32>) outs(%cst : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>
      secret.yield %1 : tensor<4x2x2xf32>
    } -> !secret.secret<tensor<4x2x2xf32>>
    return %0 : !secret.secret<tensor<4x2x2xf32>>
  }

  // Uneven: batch=3, mu=2, min-slot-count=8 -> capacity=2, numCiphertexts=2,
  // tilesPerCiphertext=2 (balanced), physical [2,8], one unused tile position.
  // CHECK: func.func @mtp_auto_multi_ct_uneven
  // CHECK-SAME: tensor<2x8xf32>
  // CHECK-NOT: linalg.batch_matmul
  // CHECK-DAG: tensor_ext.rotate
  // CHECK-DAG: arith.constant dense<
  // CHECK-DAG: arith.mulf
  // CHECK-DAG: arith.addf
  // CHECK: return
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
