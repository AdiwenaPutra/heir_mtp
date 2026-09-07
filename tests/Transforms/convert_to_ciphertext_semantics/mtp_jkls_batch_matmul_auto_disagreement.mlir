// RUN: heir-opt --layout-propagation='min-slot-count=20 enable-mtp-jkls=true' --convert-to-ciphertext-semantics=min-slot-count=20 %s | FileCheck %s

// The essential regression for the tile-capacity formula fix: batch=7, mu=2,
// min-slot-count=20 -> capacity=5, numCiphertexts=ceil(7/5)=2. An earlier,
// incorrect maximum-fill formula would pick tilesPerCiphertext=min(7,5)=5,
// but ConvertLinalgBatchMatmul::mtpJklsKernel recovers
// ceil(batch/numCiphertexts)=ceil(7/2)=4 from the physical shape alone —
// those disagree, so layout propagation using the old formula would have
// built a relation materialization then rejects
// ("MTP-JKLS requires matching row-major multi-tile layouts..."), not a
// silent wrong answer. The current balanced-packing formula
// (tilesPerCiphertext=ceil(7/2)=4, matching the recovery) makes this chain
// succeed: linalg.batch_matmul disappears and the physical type is [2,20].
// batch=3 against capacity=2 (in mtp_jkls_batch_matmul_auto.mlir) does not
// exercise this — both formulas happen to agree there.

module {
  // CHECK: func.func @mtp_auto_disagreement
  // CHECK-SAME: tensor<2x20xf32>
  // CHECK-NOT: linalg.batch_matmul
  // CHECK-DAG: tensor_ext.rotate
  // CHECK-DAG: arith.constant dense<
  // CHECK-DAG: arith.mulf
  // CHECK-DAG: arith.addf
  // CHECK: return
  func.func @mtp_auto_disagreement(%arg0: !secret.secret<tensor<7x2x2xf32>>, %arg1: !secret.secret<tensor<7x2x2xf32>>) -> !secret.secret<tensor<7x2x2xf32>> {
    %cst = arith.constant dense<0.000000e+00> : tensor<7x2x2xf32>
    %0 = secret.generic(%arg0: !secret.secret<tensor<7x2x2xf32>>, %arg1: !secret.secret<tensor<7x2x2xf32>>) {
    ^body(%lhs: tensor<7x2x2xf32>, %rhs: tensor<7x2x2xf32>):
      %1 = linalg.batch_matmul ins(%lhs, %rhs : tensor<7x2x2xf32>, tensor<7x2x2xf32>) outs(%cst : tensor<7x2x2xf32>) -> tensor<7x2x2xf32>
      secret.yield %1 : tensor<7x2x2xf32>
    } -> !secret.secret<tensor<7x2x2xf32>>
    return %0 : !secret.secret<tensor<7x2x2xf32>>
  }
}
