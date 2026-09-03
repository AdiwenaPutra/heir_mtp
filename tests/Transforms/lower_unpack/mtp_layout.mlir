// RUN: heir-opt --lower-unpack --canonicalize %s | FileCheck %s

#mtp = #tensor_ext.layout<
  "{ [p, l, d] -> [ct, slot] : ct = 0 and slot - 4 * d - 2 * p - l = 0 and 0 <= p <= 1 and 0 <= l <= 1 and 0 <= d <= 1 and 0 <= slot <= 7 }"
>

#original = #tensor_ext.original_type<
  originalType = tensor<2x2x2xf32>,
  layout = #mtp
>

// CHECK-LABEL: func.func @unpack_mtp

// Constants used by slot = 4*d + 2*p + l.
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : i32

// Three loops corresponding to p, l, and d.
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for

// Construct 2*p + l + 4*d.
// CHECK: %[[TWO_P:.*]] = arith.muli %{{.*}}, %[[C2]] : i32
// CHECK: %[[TWO_P_PLUS_L:.*]] = arith.addi %[[TWO_P]], %{{.*}} : i32
// CHECK: %[[FOUR_D:.*]] = arith.muli %{{.*}}, %[[C4]] : i32
// CHECK: %[[SLOT:.*]] = arith.addi %[[TWO_P_PLUS_L]], %[[FOUR_D]] : i32
// CHECK: %[[SLOT_INDEX:.*]] = arith.index_cast %[[SLOT]] : i32 to index

// Ciphertext index is zero; slot index is the formula above.
// CHECK: tensor.extract %{{.*}}[%{{.*}}, %[[SLOT_INDEX]]]

// CHECK: tensor.insert
// CHECK: return

func.func @unpack_mtp(
    %packed: tensor<1x8xf32>
      {tensor_ext.original_type = #original}
) -> tensor<2x2x2xf32> {
  %result = tensor_ext.unpack %packed {layout = #mtp}
      : (tensor<1x8xf32>) -> tensor<2x2x2xf32>
  return %result : tensor<2x2x2xf32>
}