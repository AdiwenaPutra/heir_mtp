// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// Phase 8 P8.4 (B3 checkpoint, Q > 1): mu=2, A: 2x4, B: 4x2, Cinit: 2x2 --
// computeGemmTilePlan reports I=1, Q=2, J=1: one destination tile
// accumulating two contraction-tile tasks, (i=0,j=0,q=0) and
// (i=0,j=0,q=1), which share the SAME logical destination but use DISTINCT
// lhs tiles (A[:,0:2] vs A[:,2:4]) and DISTINCT rhs tiles (B[0:2,:] vs
// B[2:4,:]).
//
// Chosen init/reduction representation (see PHASE_8_HANDOFF.md for the
// full justification): q=0's own batch_matmul is fed the REAL Cinit as its
// outs (reusing the primitive's already-validated init-accumulation
// semantics), producing Cinit+P0 directly; q=1's batch_matmul is fed a
// fresh ZERO outs, producing raw P1 alone. The two extracted result tiles
// are then combined by one standard arith.addf -- Cinit therefore appears
// in the IR exactly once (as q=0's outs), and the q=0/q=1 reduction is a
// plain, inspectable arith op, not hidden inside a second forced-kernel
// accumulation.

// CHECK: #[[KERNEL:[a-zA-Z0-9_]+]] = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>
// CHECK: func.func @b3_two_q
module {
  func.func @b3_two_q(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x2xf32>>) {
    // CHECK: secret.generic
    %result = secret.generic(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) {
    // CHECK-NEXT: ^body(%[[A:.*]]: tensor<2x4xf32>, %[[B:.*]]: tensor<4x2xf32>, %[[C:.*]]: tensor<2x2xf32>)
    ^body(%a: tensor<2x4xf32>, %b: tensor<4x2xf32>, %c: tensor<2x2xf32>):
      // q=0: gather A[:,0:2] and B[0:2,:] -- the FIRST distinct lhs/rhs
      // slice pair.
      // CHECK: %[[ATILE0:.*]] = tensor.extract_slice %[[A]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[AZERO0:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT:[a-zA-Z0-9_]+]]
      // CHECK: %[[LHS0:.*]] = tensor.insert_slice %[[ATILE0]] into %[[AZERO0]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[BTILE0:.*]] = tensor.extract_slice %[[B]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[BZERO0:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[RHS0:.*]] = tensor.insert_slice %[[BTILE0]] into %[[BZERO0]][0, 0, 0] [1, 2, 2] [1, 1, 1]

      // q=0's outs is the REAL Cinit (gathered from %[[C]], not a zero
      // constant) -- this is what includes the real init exactly once.
      // CHECK: %[[CTILE:.*]] = tensor.extract_slice %[[C]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[CZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[INITBATCH:.*]] = tensor.insert_slice %[[CTILE]] into %[[CZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]

      // CHECK: %[[PRODUCT0:.*]] = linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #[[KERNEL]]
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK-SAME: ins(%[[LHS0]], %[[RHS0]]
      // CHECK-SAME: outs(%[[INITBATCH]]
      // CHECK: %[[TILE0:.*]] = tensor.extract_slice %[[PRODUCT0]][0, 0, 0] [1, 2, 2] [1, 1, 1]

      // q=1: gather A[:,2:4] and B[2:4,:] -- DISTINCT lhs/rhs slices from
      // q=0's.
      // CHECK: %[[ATILE1:.*]] = tensor.extract_slice %[[A]][0, 2] [2, 2] [1, 1]
      // CHECK: %[[AZERO1:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[LHS1:.*]] = tensor.insert_slice %[[ATILE1]] into %[[AZERO1]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[BTILE1:.*]] = tensor.extract_slice %[[B]][2, 0] [2, 2] [1, 1]
      // CHECK: %[[BZERO1:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[RHS1:.*]] = tensor.insert_slice %[[BTILE1]] into %[[BZERO1]][0, 0, 0] [1, 2, 2] [1, 1, 1]

      // q=1's outs is a FRESH zero -- NOT another extraction from %[[C]] --
      // proving the real init is not re-added.
      // CHECK: %[[ZOUTS:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[PRODUCT1:.*]] = linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #[[KERNEL]]
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK-SAME: ins(%[[LHS1]], %[[RHS1]]
      // CHECK-SAME: outs(%[[ZOUTS]]
      // CHECK: %[[TILE1:.*]] = tensor.extract_slice %[[PRODUCT1]][0, 0, 0] [1, 2, 2] [1, 1, 1]

      // Both q's extracted tiles feed the SAME destination via one visible,
      // standard arith.addf -- the reduction remains ordinary
      // tensor/linalg/arith IR, not hidden inside a second forced kernel.
      // CHECK: %[[SUM:.*]] = arith.addf %[[TILE0]], %[[TILE1]]
      // CHECK: secret.yield %[[SUM]]
      %r = linalg.matmul ins(%a, %b : tensor<2x4xf32>, tensor<4x2xf32>) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
      secret.yield %r : tensor<2x2xf32>
    } -> (!secret.secret<tensor<2x2xf32>>)
    return %result : !secret.secret<tensor<2x2xf32>>
  }
}
