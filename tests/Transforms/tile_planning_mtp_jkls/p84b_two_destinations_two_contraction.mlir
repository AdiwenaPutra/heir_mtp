// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// Phase 8 P8.4b (integration checkpoint, NOT P8.5 -- P8.5 remains
// multiple-ciphertext-group support): combines P8.3's multiple
// destinations (I*J > 1) with P8.4's multiple contraction tiles (Q > 1)
// within one ciphertext group, full non-boundary tiles only.
//
// mu=2, A: 2x4, B: 4x4, Cinit: 2x4 -- computeGemmTilePlan reports I=1,
// Q=2, J=2: two destinations (j=0, j=1), each independently accumulating
// two contraction-tile partial products (q=0, q=1). Exactly 4 product
// tasks (I*Q*J = 4) are represented as two [taskCount=2,mu,mu] batches (one
// per q), each holding both destinations' tasks for that q, per the
// preferred representation.
//
// C[0,0] = Cinit[0,0] + A[0,0]xB[0,0] + A[0,1]xB[1,0]
// C[0,1] = Cinit[0,1] + A[0,0]xB[0,1] + A[0,1]xB[1,1]
//
// Every CHECK below pins down operand PROVENANCE: at q=0, the SAME lhs
// tile A[0,0] must feed both j destinations by SSA reuse (matching P8.3's
// lhs-sharing pattern within one q's batch); at q=1, the SAME lhs tile
// A[0,1] must similarly be reused (a DIFFERENT SSA value from q=0's, since
// it is a different logical tile); rhs tiles must be distinct per (q,j);
// the two real Cinit tiles must appear only in q=0's outs batch; and the
// two destinations' final tiles must never be cross-reduced or scattered
// to the same offset.

// CHECK: #[[KERNEL:[a-zA-Z0-9_]+]] = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>
// CHECK: func.func @p84b_primary
module {
  func.func @p84b_primary(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x4xf32>>, %Cinit: !secret.secret<tensor<2x4xf32>>) -> (!secret.secret<tensor<2x4xf32>>) {
    // CHECK: secret.generic
    %result = secret.generic(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x4xf32>>, %Cinit: !secret.secret<tensor<2x4xf32>>) {
    // CHECK-NEXT: ^body(%[[A:.*]]: tensor<2x4xf32>, %[[B:.*]]: tensor<4x4xf32>, %[[C:.*]]: tensor<2x4xf32>)
    ^body(%a: tensor<2x4xf32>, %b: tensor<4x4xf32>, %c: tensor<2x4xf32>):
      // --- q=0 batch: lhs A[0,0] extracted ONCE and reused for BOTH
      // destination positions (SSA reuse independently within this q's
      // batch). ---
      // CHECK: %[[ATILE0:.*]] = tensor.extract_slice %[[A]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[AZERO0:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT:[a-zA-Z0-9_]+]]
      // CHECK: %[[LHS0POS0:.*]] = tensor.insert_slice %[[ATILE0]] into %[[AZERO0]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[LHS0:.*]] = tensor.insert_slice %[[ATILE0]] into %[[LHS0POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // rhs B[0,0] and B[0,2] -- DISTINCT tiles for the two (q=0,j) tasks.
      // CHECK: %[[BTILE00:.*]] = tensor.extract_slice %[[B]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[BZERO0:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[RHS0POS0:.*]] = tensor.insert_slice %[[BTILE00]] into %[[BZERO0]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[BTILE01:.*]] = tensor.extract_slice %[[B]][0, 2] [2, 2] [1, 1]
      // CHECK: %[[RHS0:.*]] = tensor.insert_slice %[[BTILE01]] into %[[RHS0POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // The two REAL Cinit destination tiles -- appear ONLY here, in q=0's
      // outs batch.
      // CHECK: %[[CTILE0:.*]] = tensor.extract_slice %[[C]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[CZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[INIT0POS0:.*]] = tensor.insert_slice %[[CTILE0]] into %[[CZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[CTILE1:.*]] = tensor.extract_slice %[[C]][0, 2] [2, 2] [1, 1]
      // CHECK: %[[INIT0:.*]] = tensor.insert_slice %[[CTILE1]] into %[[INIT0POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // CHECK: %[[Q0RESULT:.*]] = linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #[[KERNEL]]
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK-SAME: ins(%[[LHS0]], %[[RHS0]]
      // CHECK-SAME: outs(%[[INIT0]]

      // --- q=1 batch: lhs A[0,2] -- a DIFFERENT tile from q=0's, extracted
      // ONCE and reused for BOTH destination positions again. ---
      // CHECK: %[[ATILE1:.*]] = tensor.extract_slice %[[A]][0, 2] [2, 2] [1, 1]
      // CHECK: %[[AZERO1:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[LHS1POS0:.*]] = tensor.insert_slice %[[ATILE1]] into %[[AZERO1]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[LHS1:.*]] = tensor.insert_slice %[[ATILE1]] into %[[LHS1POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // rhs B[2,0] and B[2,2] -- DISTINCT from each other AND from q=0's.
      // CHECK: %[[BTILE10:.*]] = tensor.extract_slice %[[B]][2, 0] [2, 2] [1, 1]
      // CHECK: %[[BZERO1:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[RHS1POS0:.*]] = tensor.insert_slice %[[BTILE10]] into %[[BZERO1]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[BTILE11:.*]] = tensor.extract_slice %[[B]][2, 2] [2, 2] [1, 1]
      // CHECK: %[[RHS1:.*]] = tensor.insert_slice %[[BTILE11]] into %[[RHS1POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // q=1's outs is a FRESH zero batch -- no extraction from %[[C]] here.
      // CHECK: %[[ZOUTS1:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[Q1RESULT:.*]] = linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #[[KERNEL]]
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK-SAME: ins(%[[LHS1]], %[[RHS1]]
      // CHECK-SAME: outs(%[[ZOUTS1]]
      %r = linalg.matmul ins(%a, %b : tensor<2x4xf32>, tensor<4x4xf32>) outs(%c : tensor<2x4xf32>) -> tensor<2x4xf32>

      // Destination 0 (position 0): reduces q=0's and q=1's position-0
      // tiles ONLY, in increasing-q order, then scatters to output offset
      // [0,0].
      // CHECK: %[[OUTZERO:.*]] = tensor_ext.assign_layout
      // CHECK: %[[D0Q0:.*]] = tensor.extract_slice %[[Q0RESULT]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[D0Q1:.*]] = tensor.extract_slice %[[Q1RESULT]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[D0SUM:.*]] = arith.addf %[[D0Q0]], %[[D0Q1]]
      // CHECK: %[[OUT0:.*]] = tensor.insert_slice %[[D0SUM]] into %[[OUTZERO]][0, 0] [2, 2] [1, 1]

      // Destination 1 (position 1): reduces q=0's and q=1's position-1
      // tiles ONLY -- never destination 0's -- then scatters to a DISTINCT
      // output offset [0,2].
      // CHECK: %[[D1Q0:.*]] = tensor.extract_slice %[[Q0RESULT]][1, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[D1Q1:.*]] = tensor.extract_slice %[[Q1RESULT]][1, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[D1SUM:.*]] = arith.addf %[[D1Q0]], %[[D1Q1]]
      // CHECK: %[[OUT1:.*]] = tensor.insert_slice %[[D1SUM]] into %[[OUT0]][0, 2] [2, 2] [1, 1]
      // CHECK: secret.yield %[[OUT1]]
      secret.yield %r : tensor<2x4xf32>
    } -> (!secret.secret<tensor<2x4xf32>>)
    return %result : !secret.secret<tensor<2x4xf32>>
  }
}
