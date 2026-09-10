// RUN: heir-opt --layout-propagation=min-slot-count=4 --mlir-print-local-scope %s | FileCheck %s

// A genuinely multi-row (two-ciphertext) standalone source, locally indexed
// from zero (ct in {0, 1}), inserted into a non-leading, contiguous
// two-position interval [2, 4) of a four-position authoritative
// destination. After normalization (shifting the composed requirement's
// range ciphertext variable down by its own tight lower bound, 2), the
// requirement becomes ct in {0, 1} -- exactly the source's own layout, so
// no conversion should be inserted. This proves the fix's normalization
// generalizes beyond a single-row source to a genuinely multi-row one
// landing at a non-leading destination interval.
#dest = #tensor_ext.layout<"{ [p, l, d] -> [ct, slot] : ct = p and slot - 2 * l - d = 0 and 0 <= p <= 3 and 0 <= l <= 1 and 0 <= d <= 1 and 0 <= slot <= 3 }">
#source = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : ct = i0 and slot = 2i1 + i2 and 0 <= i0 <= 1 and 0 <= i1 <= 1 and 0 <= i2 <= 1 }">

module {
  // CHECK-LABEL: @non_leading_multi_row
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[2, 0, 0]
  func.func @non_leading_multi_row(%pair: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #source}) -> (!secret.secret<tensor<4x2x2xf32>>) {
    %zero = arith.constant dense<0.0> : tensor<4x2x2xf32>
    %result = secret.generic(%pair: !secret.secret<tensor<2x2x2xf32>> {tensor_ext.layout = #source}) {
    ^body(%p: tensor<2x2x2xf32>):
      %authDest = tensor_ext.assign_layout %zero {layout = #dest, tensor_ext.layout = #dest} : tensor<4x2x2xf32>
      %inserted = tensor.insert_slice %p into %authDest[2, 0, 0] [2, 2, 2] [1, 1, 1] : tensor<2x2x2xf32> into tensor<4x2x2xf32>
      secret.yield %inserted : tensor<4x2x2xf32>
    } -> (!secret.secret<tensor<4x2x2xf32>>)
    return %result : !secret.secret<tensor<4x2x2xf32>>
  }
}
