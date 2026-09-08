// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// Phase 8 P8.2 (B1 compatibility checkpoint): an ordinary, otherwise
// unannotated secret-secret linalg.matmul with M=K=N=mu=2 (computeGemmTilePlan
// reports exactly one task) is rewritten into standard tensor/linalg/arith
// operations that gather lhs, rhs, and the REAL (here nonzero-shaped) init
// operand into [1,mu,mu] batch tensors, feed a forced-kernel
// linalg.batch_matmul, and extract the single result tile directly as the
// final answer (no separate scatter is needed when there is exactly one
// destination tile).
//
// Each gather's CHECK below pins down not just the shape of the extract/
// insert/assign_layout sequence but WHICH block argument feeds it (%[[A]],
// %[[B]], or %[[C]] specifically) -- a wildcard source here would not
// distinguish "gathered the real init" from "silently substituted a
// different operand and still produced structurally similar IR", which is
// exactly the class of bug this checkpoint's init-preservation requirement
// must catch.

// CHECK: #[[KERNEL:[a-zA-Z0-9_]+]] = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>
// CHECK: func.func @b1_matmul
module {
  func.func @b1_matmul(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x2xf32>>) {
    // CHECK: secret.generic
    %result = secret.generic(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) {
    // CHECK-NEXT: ^body(%[[A:.*]]: tensor<2x2xf32>, %[[B:.*]]: tensor<2x2xf32>, %[[C:.*]]: tensor<2x2xf32>)
    ^body(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>):
      // Gather lhs (%[[A]] specifically) into a [1,2,2] batch.
      // CHECK: %[[ATILE:.*]] = tensor.extract_slice %[[A]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[AZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT:[a-zA-Z0-9_]+]]
      // CHECK: %[[LHSBATCH:.*]] = tensor.insert_slice %[[ATILE]] into %[[AZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]

      // Gather rhs (%[[B]] specifically) into its own [1,2,2] batch, same
      // layout.
      // CHECK: %[[BTILE:.*]] = tensor.extract_slice %[[B]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[BZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[RHSBATCH:.*]] = tensor.insert_slice %[[BTILE]] into %[[BZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]

      // Gather the REAL init (%[[C]] specifically, not %[[A]]/%[[B]] again
      // and not a bare zero constant) into its own [1,2,2] batch, same
      // layout: this is what preserves linalg.matmul's own
      // `result = init + lhs @ rhs` semantics through the rewrite.
      // CHECK: %[[CTILE:.*]] = tensor.extract_slice %[[C]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[CZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[INITBATCH:.*]] = tensor.insert_slice %[[CTILE]] into %[[CZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]

      // The forced-kernel batch_matmul reuses the existing MTP-JKLS
      // primitive (via the unmodified layout-propagation/
      // convert-to-ciphertext-semantics pipeline this op feeds into), fed
      // by the three real gathered operands above in their correct roles --
      // not by any freshly reinitialized zero tensor or a swapped operand.
      // CHECK: linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #[[KERNEL]]
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK-SAME: ins(%[[LHSBATCH]], %[[RHSBATCH]]
      // CHECK-SAME: outs(%[[INITBATCH]]
      %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>

      // The single destination tile IS the whole output: extract it
      // directly, with no additional scatter into a shared buffer.
      // CHECK: %[[RESULTTILE:.*]] = tensor.extract_slice %{{.*}}[0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: secret.yield %[[RESULTTILE]]
      secret.yield %r : tensor<2x2xf32>
    } -> (!secret.secret<tensor<2x2xf32>>)
    return %result : !secret.secret<tensor<2x2xf32>>
  }
}
