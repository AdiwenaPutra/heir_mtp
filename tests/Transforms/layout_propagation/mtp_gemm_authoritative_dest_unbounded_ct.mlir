// RUN: heir-opt %s --layout-propagation=min-slot-count=4 --verify-diagnostics

// getRequiredSliceLayoutForAuthoritativeDest's normalization requires a
// CONSTANT tight lower bound on the composed requirement's range-side
// ciphertext variable before it can zero-shift it. Here the destination's
// own declared layout leaves ct unbounded below (only `ct <= 3` is stated),
// so no constant lower bound exists after composing through the insertion
// geometry. This must be rejected with the existing clean diagnostic --
// never silently treated as already-zero-indexed, and never a crash.
#dest = #tensor_ext.layout<"{ [p, l, d] -> [ct, slot] : ct <= 3 and slot - 2 * l - d = 0 and 0 <= p <= 1 and 0 <= l <= 1 and 0 <= d <= 1 and 0 <= slot <= 3 }">

module {
  func.func @unbounded_ct(%tile: !secret.secret<tensor<1x2x2xf32>>) -> (!secret.secret<tensor<2x2x2xf32>>) {
    %zero = arith.constant dense<0.0> : tensor<2x2x2xf32>
    %result = secret.generic(%tile: !secret.secret<tensor<1x2x2xf32>>) {
    ^body(%t: tensor<1x2x2xf32>):
      %authDest = tensor_ext.assign_layout %zero {layout = #dest, tensor_ext.layout = #dest} : tensor<2x2x2xf32>
      // expected-error@+1 {{cannot reconcile tensor.insert_slice into an authoritative destination layout: unsupported dynamic offsets/sizes/strides, non-unit strides, or a dropped-dimension mask getSliceInsertionRelation cannot express}}
      %inserted = tensor.insert_slice %t into %authDest[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<1x2x2xf32> into tensor<2x2x2xf32>
      secret.yield %inserted : tensor<2x2x2xf32>
    } -> (!secret.secret<tensor<2x2x2xf32>>)
    return %result : !secret.secret<tensor<2x2x2xf32>>
  }
}
