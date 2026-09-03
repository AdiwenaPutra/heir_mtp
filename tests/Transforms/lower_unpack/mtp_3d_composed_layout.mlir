// RUN: heir-opt --lower-unpack --canonicalize %s | FileCheck %s
#mtp_3d = #tensor_ext.layout<
  "{ [h, i, j] -> [ct, slot] : 0 <= h <= 1 and 0 <= i <= 3 and 0 <= j <= 7 and 0 <= ct <= 7 and 0 <= slot <= 7 and ct - 4 * h - 2 * floor((i) / 2) - floor((j) / 4) = 0 and slot - i - 4 * j + 2 * floor((i) / 2) + 6 * floor((j) / 2) + 4 * floor((j) / 4) = 0 }"
>
#original_3d = #tensor_ext.original_type<
  originalType = tensor<2x4x8xf32>,
  layout = #mtp_3d
>

// CHECK-LABEL: func.func @unpack_mtp_3d_composed

// Constants used by the generated address calculations.
// CHECK-DAG: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[THREE:.*]] = arith.constant 3 : i32
// CHECK-DAG: %[[FOUR:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[SEVEN:.*]] = arith.constant 7 : i32

// Three reverse-order loops produce logical coordinates h, i, and j.
// CHECK: scf.for
// CHECK: %[[H:.*]] = arith.subi %[[ONE]], %{{.*}} : i32
// CHECK: scf.for
// CHECK: %[[I:.*]] = arith.subi %[[THREE]], %{{.*}} : i32
// CHECK: scf.for
// CHECK: %[[J:.*]] = arith.subi %[[SEVEN]], %{{.*}} : i32

// Compute qL = floor(i/2) and l = i - 2*qL.
// CHECK: %[[QL_CT:.*]] = arith.floordivsi %[[I]], %[[TWO]] : i32
// CHECK: %[[TWO_QL_CT:.*]] = arith.muli %[[QL_CT]], %[[TWO]] : i32
// CHECK: %[[L_CT:.*]] = arith.subi %[[I]], %[[TWO_QL_CT]] : i32

// Compute 4*h + 2*qL as 4*h - l + i.
// CHECK: %[[NEG_L:.*]] = arith.subi %[[ZERO]], %[[L_CT]] : i32
// CHECK: %[[FOUR_H:.*]] = arith.muli %[[H]], %[[FOUR]] : i32
// CHECK: %[[FOUR_H_MINUS_L:.*]] = arith.addi %[[NEG_L]], %[[FOUR_H]] : i32
// CHECK: %[[CT_BASE:.*]] = arith.addi %[[FOUR_H_MINUS_L]], %[[I]] : i32

// Compute g = floor(j/4), then ct = 4*h + 2*qL + g.
// CHECK: %[[G_CT:.*]] = arith.floordivsi %[[J]], %[[FOUR]] : i32
// CHECK: %[[CT:.*]] = arith.addi %[[CT_BASE]], %[[G_CT]] : i32

// Recompute l = i mod 2 for the slot expression.
// CHECK: %[[QL_SLOT:.*]] = arith.floordivsi %[[I]], %[[TWO]] : i32
// CHECK: %[[TWO_QL_SLOT:.*]] = arith.muli %[[QL_SLOT]], %[[TWO]] : i32
// CHECK: %[[L_SLOT:.*]] = arith.subi %[[I]], %[[TWO_QL_SLOT]] : i32

// Compute d = j mod 2.
// CHECK: %[[QD:.*]] = arith.floordivsi %[[J]], %[[TWO]] : i32
// CHECK: %[[TWO_QD:.*]] = arith.muli %[[QD]], %[[TWO]] : i32
// CHECK: %[[D:.*]] = arith.subi %[[J]], %[[TWO_QD]] : i32

// Begin slot = l + 4*d + 2*p.
// CHECK: %[[THREE_D:.*]] = arith.muli %[[D]], %[[THREE]] : i32
// CHECK: %[[L_PLUS_THREE_D:.*]] = arith.addi %[[L_SLOT]], %[[THREE_D]] : i32

// j - 4*g equals 2*p + d.
// CHECK: %[[G_SLOT:.*]] = arith.floordivsi %[[J]], %[[FOUR]] : i32
// CHECK: %[[FOUR_G:.*]] = arith.muli %[[G_SLOT]], %[[FOUR]] : i32
// CHECK: %[[TWO_P_PLUS_D:.*]] = arith.subi %[[J]], %[[FOUR_G]] : i32

// (l + 3*d) + (2*p + d) = l + 4*d + 2*p.
// CHECK: %[[SLOT:.*]] = arith.addi %[[L_PLUS_THREE_D]], %[[TWO_P_PLUS_D]] : i32

// Verify that the calculated ct and slot address the packed tensor.
// CHECK: %[[CT_INDEX:.*]] = arith.index_cast %[[CT]] : i32 to index
// CHECK: %[[SLOT_INDEX:.*]] = arith.index_cast %[[SLOT]] : i32 to index
// CHECK: tensor.extract %{{.*}}[%[[CT_INDEX]], %[[SLOT_INDEX]]]

// Verify reconstruction at the original logical coordinate [h,i,j].
// CHECK: %[[H_INDEX:.*]] = arith.index_cast %[[H]] : i32 to index
// CHECK: %[[I_INDEX:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK: %[[J_INDEX:.*]] = arith.index_cast %[[J]] : i32 to index
// CHECK: tensor.insert %{{.*}} into %{{.*}}[%[[H_INDEX]], %[[I_INDEX]], %[[J_INDEX]]]
// CHECK: return

func.func @unpack_mtp_3d_composed(
    %packed: tensor<8x8xf32>
      {tensor_ext.original_type = #original_3d}
) -> tensor<2x4x8xf32> {
  %result = tensor_ext.unpack %packed {layout = #mtp_3d}
      : (tensor<8x8xf32>) -> tensor<2x4x8xf32>
  return %result : tensor<2x4x8xf32>
}