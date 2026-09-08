// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// Complementary P8.4b fixture: mu=2, A: 4x4, B: 4x2, Cinit: 4x2 ->
// I=2, Q=2, J=1. The two destinations here differ along the ROW dimension
// (i=0 vs i=1), not the column dimension as in the primary fixture --
// catching accidental assumptions that multiple destinations always vary
// along J. Consequently the SHARED (SSA-reused) operand at each q is now
// rhs (B[q,0], the same for both i), while lhs (A[i,q]) is the one that
// differs per destination -- the mirror image of the primary fixture's
// sharing pattern.

// CHECK: #[[KERNEL:[a-zA-Z0-9_]+]] = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>
// CHECK: func.func @p84b_row
module {
  func.func @p84b_row(%A: !secret.secret<tensor<4x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<4x2xf32>>) -> (!secret.secret<tensor<4x2xf32>>) {
    // CHECK: secret.generic
    %result = secret.generic(%A: !secret.secret<tensor<4x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<4x2xf32>>) {
    // CHECK-NEXT: ^body(%[[A:.*]]: tensor<4x4xf32>, %[[B:.*]]: tensor<4x2xf32>, %[[C:.*]]: tensor<4x2xf32>)
    ^body(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>, %c: tensor<4x2xf32>):
      // q=0: lhs DIFFERS per destination (A[0,0] vs A[2,0], distinct
      // extractions)...
      // CHECK: %[[ATILE00:.*]] = tensor.extract_slice %[[A]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[AZERO0:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT:[a-zA-Z0-9_]+]]
      // CHECK: %[[LHS0POS0:.*]] = tensor.insert_slice %[[ATILE00]] into %[[AZERO0]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[ATILE01:.*]] = tensor.extract_slice %[[A]][2, 0] [2, 2] [1, 1]
      // CHECK: %[[LHS0:.*]] = tensor.insert_slice %[[ATILE01]] into %[[LHS0POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // ...while rhs B[0,0] is extracted ONCE and reused for BOTH
      // destinations -- the shared operand this time.
      // CHECK: %[[BTILE0:.*]] = tensor.extract_slice %[[B]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[BZERO0:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[RHS0POS0:.*]] = tensor.insert_slice %[[BTILE0]] into %[[BZERO0]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[RHS0:.*]] = tensor.insert_slice %[[BTILE0]] into %[[RHS0POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // The two REAL Cinit destination tiles, q=0 only.
      // CHECK: %[[CTILE0:.*]] = tensor.extract_slice %[[C]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[CZERO:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[INIT0POS0:.*]] = tensor.insert_slice %[[CTILE0]] into %[[CZERO]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[CTILE1:.*]] = tensor.extract_slice %[[C]][2, 0] [2, 2] [1, 1]
      // CHECK: %[[INIT0:.*]] = tensor.insert_slice %[[CTILE1]] into %[[INIT0POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // CHECK: %[[Q0RESULT:.*]] = linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #[[KERNEL]]
      // CHECK-SAME: ins(%[[LHS0]], %[[RHS0]]
      // CHECK-SAME: outs(%[[INIT0]]

      // q=1: lhs differs per destination again (A[0,2] vs A[2,2])...
      // CHECK: %[[ATILE10:.*]] = tensor.extract_slice %[[A]][0, 2] [2, 2] [1, 1]
      // CHECK: %[[AZERO1:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[LHS1POS0:.*]] = tensor.insert_slice %[[ATILE10]] into %[[AZERO1]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[ATILE11:.*]] = tensor.extract_slice %[[A]][2, 2] [2, 2] [1, 1]
      // CHECK: %[[LHS1:.*]] = tensor.insert_slice %[[ATILE11]] into %[[LHS1POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // ...while rhs B[2,0] is again shared between both destinations.
      // CHECK: %[[BTILE1:.*]] = tensor.extract_slice %[[B]][2, 0] [2, 2] [1, 1]
      // CHECK: %[[BZERO1:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[RHS1POS0:.*]] = tensor.insert_slice %[[BTILE1]] into %[[BZERO1]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[RHS1:.*]] = tensor.insert_slice %[[BTILE1]] into %[[RHS1POS0]][1, 0, 0] [1, 2, 2] [1, 1, 1]

      // q=1's outs is a fresh zero batch.
      // CHECK: %[[ZOUTS1:.*]] = tensor_ext.assign_layout
      // CHECK-SAME: tensor_ext.layout = #[[MTPLAYOUT]]
      // CHECK: %[[Q1RESULT:.*]] = linalg.batch_matmul
      // CHECK-SAME: secret.kernel = #[[KERNEL]]
      // CHECK-SAME: ins(%[[LHS1]], %[[RHS1]]
      // CHECK-SAME: outs(%[[ZOUTS1]]
      %r = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x2xf32>) outs(%c : tensor<4x2xf32>) -> tensor<4x2xf32>

      // Destination 0 (position 0) scatters to row offset [0,0]; destination
      // 1 (position 1) scatters to a DISTINCT row offset [2,0] -- never
      // cross-reduced.
      // CHECK: %[[OUTZERO:.*]] = tensor_ext.assign_layout
      // CHECK: %[[D0Q0:.*]] = tensor.extract_slice %[[Q0RESULT]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[D0Q1:.*]] = tensor.extract_slice %[[Q1RESULT]][0, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[D0SUM:.*]] = arith.addf %[[D0Q0]], %[[D0Q1]]
      // CHECK: %[[OUT0:.*]] = tensor.insert_slice %[[D0SUM]] into %[[OUTZERO]][0, 0] [2, 2] [1, 1]
      // CHECK: %[[D1Q0:.*]] = tensor.extract_slice %[[Q0RESULT]][1, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[D1Q1:.*]] = tensor.extract_slice %[[Q1RESULT]][1, 0, 0] [1, 2, 2] [1, 1, 1]
      // CHECK: %[[D1SUM:.*]] = arith.addf %[[D1Q0]], %[[D1Q1]]
      // CHECK: %[[OUT1:.*]] = tensor.insert_slice %[[D1SUM]] into %[[OUT0]][2, 0] [2, 2] [1, 1]
      // CHECK: secret.yield %[[OUT1]]
      secret.yield %r : tensor<4x2xf32>
    } -> (!secret.secret<tensor<4x2xf32>>)
    return %result : !secret.secret<tensor<4x2xf32>>
  }
}
