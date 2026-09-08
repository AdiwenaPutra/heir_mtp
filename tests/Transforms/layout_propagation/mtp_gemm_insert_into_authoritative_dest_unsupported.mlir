// RUN: heir-opt %s --layout-propagation=min-slot-count=16 --verify-diagnostics

// Phase 8 graceful rejection: a tensor.insert_slice into an authoritative
// destination (LayoutPropagation.cpp's authoritativeDestinations) with a
// non-unit stride is unsupported by
// getRequiredSliceLayoutForAuthoritativeDest, and must be cleanly rejected
// with a diagnostic -- not crash or silently relax the destination's
// addressing the way naively generalizing the older, offset-blind
// pushSliceLayoutThroughInsertSlice inference would.

#dest = #tensor_ext.layout<"{ [p, d, l] -> [ct, slot] : ct = 0 and slot - 8 * p - 2 * d - l = 0 and 0 <= p <= 1 and 0 <= d <= 3 and 0 <= l <= 1 and 0 <= slot <= 15 }">

module {
  func.func @non_unit_stride_into_authoritative_dest(%a: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x4x2xf32>>) {
    %zero = arith.constant dense<0.0> : tensor<2x4x2xf32>
    %result = secret.generic(%a: !secret.secret<tensor<2x2xf32>>) {
    ^body(%a_inner: tensor<2x2xf32>):
      %authDest = tensor_ext.assign_layout %zero {layout = #dest, tensor_ext.layout = #dest} : tensor<2x4x2xf32>
      // A non-unit stride (2) on the "d" dimension, which has room (extent
      // 4) for a genuinely non-trivial strided placement.
      // expected-error@+1 {{cannot reconcile tensor.insert_slice into an authoritative destination layout: unsupported dynamic offsets/sizes/strides, non-unit strides, or a dropped-dimension mask getSliceInsertionRelation cannot express}}
      %inserted = tensor.insert_slice %a_inner into %authDest[0, 0, 0] [1, 2, 2] [1, 2, 1] : tensor<2x2xf32> into tensor<2x4x2xf32>
      secret.yield %inserted : tensor<2x4x2xf32>
    } -> (!secret.secret<tensor<2x4x2xf32>>)
    return %result : !secret.secret<tensor<2x4x2xf32>>
  }
}
