// RUN: heir-opt --lower-unpack --canonicalize %s | FileCheck %s

#mtp = #tensor_ext.layout<
  "{ [i, j] -> [ct, slot] : 0 <= i <= 3 and 0 <= j <= 3 and 0 <= ct <= 1 and 0 <= slot <= 7 and exists qL, qD, l, d, tile, p : i - 2 * qL - l = 0 and j - 2 * qD - d = 0 and tile - 2 * qL - qD = 0 and tile - 2 * ct - p = 0 and slot - 4 * d - 2 * p - l = 0 and 0 <= qL <= 1 and 0 <= qD <= 1 and 0 <= l <= 1 and 0 <= d <= 1 and 0 <= tile <= 3 and 0 <= p <= 1 }"
>

#original = #tensor_ext.original_type<
  originalType = tensor<4x4xf32>,
  layout = #mtp
>

// CHECK-LABEL: func.func @unpack_mtp_composed

// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : i32
// CHECK-DAG: %[[THREE:.*]] = arith.constant 3 : i32

// HEIR traverses the logical coordinates in reverse order.
// CHECK: scf.for
// CHECK: %[[I:.*]] = arith.subi %[[THREE]], %{{.*}} : i32
// CHECK: scf.for
// CHECK: %[[J:.*]] = arith.subi %[[THREE]], %{{.*}} : i32

// ct = floor(i / 2)
// CHECK: %[[CT:.*]] = arith.floordivsi %[[I]], %[[TWO]] : i32

// l = i - 2 * floor(i / 2) = i mod 2
// CHECK: %[[I_QUOTIENT:.*]] = arith.floordivsi %[[I]], %[[TWO]] : i32
// CHECK: %[[TWO_I_QUOTIENT:.*]] = arith.muli %[[I_QUOTIENT]], %[[TWO]] : i32
// CHECK: %[[L:.*]] = arith.subi %[[I]], %[[TWO_I_QUOTIENT]] : i32

// d = j - 2 * floor(j / 2) = j mod 2
// CHECK: %[[J_QUOTIENT:.*]] = arith.floordivsi %[[J]], %[[TWO]] : i32
// CHECK: %[[TWO_J_QUOTIENT:.*]] = arith.muli %[[J_QUOTIENT]], %[[TWO]] : i32
// CHECK: %[[D:.*]] = arith.subi %[[J]], %[[TWO_J_QUOTIENT]] : i32

// slot = l + 3*d + j
// This is equivalent to slot = l + 4*d + 2*p because j = 2*p + d.
// CHECK: %[[THREE_D:.*]] = arith.muli %[[D]], %[[THREE]] : i32
// CHECK: %[[L_PLUS_THREE_D:.*]] = arith.addi %[[L]], %[[THREE_D]] : i32
// CHECK: %[[SLOT:.*]] = arith.addi %[[L_PLUS_THREE_D]], %[[J]] : i32

// Verify that ct and slot are the two physical tensor indices.
// CHECK: %[[CT_INDEX:.*]] = arith.index_cast %[[CT]] : i32 to index
// CHECK: %[[SLOT_INDEX:.*]] = arith.index_cast %[[SLOT]] : i32 to index
// CHECK: tensor.extract %{{.*}}[%[[CT_INDEX]], %[[SLOT_INDEX]]]

// Verify reconstruction of logical result[i,j].
// CHECK: %[[I_INDEX:.*]] = arith.index_cast %[[I]] : i32 to index
// CHECK: %[[J_INDEX:.*]] = arith.index_cast %[[J]] : i32 to index
// CHECK: tensor.insert %{{.*}} into %{{.*}}[%[[I_INDEX]], %[[J_INDEX]]]
// CHECK: return

func.func @unpack_mtp_composed(
    %packed: tensor<2x8xf32>
      {tensor_ext.original_type = #original}
) -> tensor<4x4xf32> {
  %result = tensor_ext.unpack %packed {layout = #mtp}
      : (tensor<2x8xf32>) -> tensor<4x4xf32>
  return %result : tensor<4x4xf32>
}