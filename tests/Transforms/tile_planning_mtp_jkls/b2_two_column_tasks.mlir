// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// Phase 8 P8.3 (first I*J > 1 generalization, Q == 1): mu=2, A: 2x2, B: 2x4,
// Cinit: 2x4 -- computeGemmTilePlan reports I=1, Q=1, J=2, i.e. two
// destination-column tasks (i=0,j=0,q=0) and (i=0,j=1,q=0) that share the
// SAME lhs tile (lhsTileId = i*Q+q = 0 for both) but consume DISTINCT rhs
// tiles (rhsTileId = q*J+j = 0 and 1) and preserve DISTINCT real Cinit
// tiles, scattering into DISTINCT output regions.
//
// Every CHECK below pins down operand PROVENANCE, not just shape: the same
// extracted lhs tile SSA value must feed both sibling inserts (proving
// reuse, not two structurally-similar-but-separate extractions), while rhs
// and Cinit must come from two textually distinct extract_slice offsets --
// exactly the class of bug (a swapped, dropped, or aliased-when-it-
// shouldn't-be operand) that a wildcard-source CHECK would miss, per the
// P8.2 checkpoint's own self-caught wildcard-source lesson.

// CHECK: #[[KERNEL:[a-zA-Z0-9_]+]] = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>
// CHECK: func.func @b2_two_col
module {
  func.func @b2_two_col(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x4xf32>>, %Cinit: !secret.secret<tensor<2x4xf32>>) -> (!secret.secret<tensor<2x4xf32>>) {
    // CHECK: secret.generic
    %result = secret.generic(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x4xf32>>, %Cinit: !secret.secret<tensor<2x4xf32>>) {
    // CHECK-NEXT: ^body(%[[A:.*]]: tensor<2x2xf32>, %[[B:.*]]: tensor<2x4xf32>, %[[C:.*]]: tensor<2x4xf32>)
    ^body(%a: tensor<2x2xf32>, %b: tensor<2x4xf32>, %c: tensor<2x4xf32>):
      // The lhs tile is extracted from %[[A]] exactly ONCE ...
      // CHECK: %[[ATILE:.*]] = tensor.extract_slice %[[A]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[AZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT:[a-zA-Z0-9_]+]]
      // CHECK: %[[LHSBATCH0:.*]] = tensor.insert_slice %[[ATILE]] into %[[AZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // ... and the SAME %[[ATILE]] SSA value is reused for the second
      // destination-column task's sibling insert -- proving the lhs tile is
      // genuinely shared, not merely structurally similar.
      // CHECK: %[[LHSBATCH:.*]] = tensor.insert_slice %[[ATILE]] into %[[LHSBATCH0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // rhs, by contrast, is extracted from two DISTINCT offsets of %[[B]]
      // (columns [0,2) and [2,4)) into two DISTINCT tile values.
      // CHECK: %[[BTILE0:.*]] = tensor.extract_slice %[[B]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[BZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[RHSBATCH0:.*]] = tensor.insert_slice %[[BTILE0]] into %[[BZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[BTILE1:.*]] = tensor.extract_slice %[[B]][0, 2] [2, 2] [1, 1]
      // CHECK: %[[RHSBATCH:.*]] = tensor.insert_slice %[[BTILE1]] into %[[RHSBATCH0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // The REAL Cinit is likewise gathered as two DISTINCT tiles from
      // %[[C]] (not a zero constant, not %[[A]]/%[[B]] again): this is what
      // preserves linalg.matmul's own `result = init + lhs @ rhs` semantics
      // per destination tile through the rewrite.
      // CHECK: %[[CTILE0:.*]] = tensor.extract_slice %[[C]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[CZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[INITBATCH0:.*]] = tensor.insert_slice %[[CTILE0]] into %[[CZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[CTILE1:.*]] = tensor.extract_slice %[[C]][0, 2] [2, 2] [1, 1]
      // CHECK: %[[INITBATCH:.*]] = tensor.insert_slice %[[CTILE1]] into %[[INITBATCH0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // One forced-kernel batch_matmul over the whole [2,2,2] batch handles
      // both destination tasks at once.
      // CHECK: %[[PRODUCT:.*]] = linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #[[KERNEL]]
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK-SAME: ins(%[[LHSBATCH]], %[[RHSBATCH]]
      // CHECK-SAME: outs(%[[INITBATCH]]
      %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x4xf32>) outs(%c : tensor<2x4xf32>) -> tensor<2x4xf32>

      // Finally, the two result tiles (physical batch positions 0 and 1)
      // scatter into two DISTINCT (row, col) regions of a fresh, explicitly
      // laid-out (authoritative) row-major 2x4 output accumulator.
      // CHECK: %[[OUTZERO:.*]] = tensor_ext.assign_layout
      // CHECK: %[[RTILE0:.*]] = tensor.extract_slice %[[PRODUCT]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[OUT0:.*]] = tensor.insert_slice %[[RTILE0]] into %[[OUTZERO]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[RTILE1:.*]] = tensor.extract_slice %[[PRODUCT]][1, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[OUT1:.*]] = tensor.insert_slice %[[RTILE1]] into %[[OUT0]][0, 2] [2, 2] [1, 1]
      // CHECK: secret.yield %[[OUT1]]
      secret.yield %r : tensor<2x4xf32>
    } -> (!secret.secret<tensor<2x4xf32>>)
    return %result : !secret.secret<tensor<2x4xf32>>
  }
}
