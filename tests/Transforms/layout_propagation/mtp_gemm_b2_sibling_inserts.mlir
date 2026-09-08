// RUN: heir-opt --layout-propagation=min-slot-count=8 --mlir-print-local-scope %s | FileCheck %s

// Acceptance test for Phase 8's B2 fixture: two sibling tensor.insert_slice
// ops scatter distinct lhs/rhs tiles into a shared, explicitly assigned
// (authoritative) destination layout -- lhs reuses the SAME %aTile for both
// sibling positions, but each needs a genuinely different physical
// conversion, since the two positions land at different slot offsets.
//
// Before the destination-authority mechanism
// (LayoutPropagation.cpp's authoritativeDestinations set), the second
// sibling insert either crashed (Hoisting.cpp's
// pushSliceLayoutThroughInsertSlice assertion on a partial, non-unit,
// non-full-dimension insert) or, if that assertion were merely weakened
// instead of replaced, silently relaxed the destination's addressing,
// producing a layout where slot no longer depends on the row index at all
// (confirmed with a reverted diagnostic experiment during development).
// This test proves the destination keeps its EXACT original row/column-to-slot addressing
// through both sibling inserts (not merely that materialization succeeds),
// and that each incoming tile instead gets its own distinct, correctly
// derived conversion.

#mtp = #tensor_ext.layout<"{ [p, d, l] -> [ct, slot] : ct = 0 and slot - 4 * d - 2 * p - l = 0 and 0 <= p <= 1 and 0 <= d <= 1 and 0 <= l <= 1 and 0 <= slot <= 7 }">
#rowmajor2x4 = #tensor_ext.layout<"{ [i, j] -> [ct, slot] : ct = 0 and slot - 4 * i - j = 0 and 0 <= i <= 1 and 0 <= j <= 3 and 0 <= slot <= 7 }">
#kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>

module {
  // CHECK: func.func @mtp_gemm_b2
  func.func @mtp_gemm_b2(
      %A: !secret.secret<tensor<2x2xf32>>,
      %B: !secret.secret<tensor<2x4xf32>>)
      -> (!secret.secret<tensor<2x4xf32>>) {
    %zero2 = arith.constant dense<0.0> : tensor<2x2x2xf32>
    %outZero = arith.constant dense<0.0> : tensor<2x4xf32>
    %result = secret.generic(
        %A: !secret.secret<tensor<2x2xf32>>,
        %B: !secret.secret<tensor<2x4xf32>>) {
    ^body(%a: tensor<2x2xf32>, %b: tensor<2x4xf32>):
      %mtpZero = tensor_ext.assign_layout %zero2 {layout = #mtp, tensor_ext.layout = #mtp} : tensor<2x2x2xf32>
      %outInit = tensor_ext.assign_layout %outZero {layout = #rowmajor2x4, tensor_ext.layout = #rowmajor2x4} : tensor<2x4xf32>

      %aTile = tensor.extract_slice %a[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> to tensor<2x2xf32>
      // The destination's own fixed row/column addressing must survive
      // unmodified at the FIRST sibling insert:
      // CHECK: tensor_ext.convert_layout
      // CHECK-SAME: to_layout = #tensor_ext.layout<"{{.*}}slot = 4i0 + i1{{.*}}">
      // CHECK: tensor.insert_slice
      // CHECK-SAME: tensor_ext.layout = #tensor_ext.layout<"{{.*}}slot - 4 * d - 2 * p - l = 0{{.*}}">
      %lhs0 = tensor.insert_slice %aTile into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<2x2x2xf32>
      // ...and unmodified again, byte-for-byte the same relation, at the
      // SECOND sibling insert -- this is the destination-authority
      // invariant. The reused %aTile needs a genuinely different
      // conversion here (slot = 2 + 4i0 + i1, not 4i0 + i1), proving this
      // is a per-task, offset-aware derivation and not a single shared
      // inference copied across siblings.
      // CHECK: tensor_ext.convert_layout
      // CHECK-SAME: to_layout = #tensor_ext.layout<"{{.*}}slot = 2 + 4i0 + i1{{.*}}">
      // CHECK: tensor.insert_slice
      // CHECK-SAME: tensor_ext.layout = #tensor_ext.layout<"{{.*}}slot - 4 * d - 2 * p - l = 0{{.*}}">
      %lhs1 = tensor.insert_slice %aTile into %lhs0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<2x2x2xf32>

      %bTile0 = tensor.extract_slice %b[0, 0] [2, 2] [1, 1] : tensor<2x4xf32> to tensor<2x2xf32>
      %rhs0 = tensor.insert_slice %bTile0 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<2x2x2xf32>
      %bTile1 = tensor.extract_slice %b[0, 2] [2, 2] [1, 1] : tensor<2x4xf32> to tensor<2x2xf32>
      %rhs1 = tensor.insert_slice %bTile1 into %rhs0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<2x2x2xf32>

      %product = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #mtp
        } ins(%lhs1, %rhs1 : tensor<2x2x2xf32>, tensor<2x2x2xf32>)
          outs(%mtpZero : tensor<2x2x2xf32>) -> tensor<2x2x2xf32>

      %rTile0 = tensor.extract_slice %product[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2x2xf32> to tensor<2x2xf32>
      %out0 = tensor.insert_slice %rTile0 into %outInit[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x4xf32>
      %rTile1 = tensor.extract_slice %product[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2x2xf32> to tensor<2x2xf32>
      // The final output scatter's two sibling inserts must likewise both
      // keep the SAME output destination addressing (row-major, slot = 4*i
      // + j), unmodified, with no conversion needed since the two
      // extracted result tiles already match it exactly. Anchor past the
      // first scatter insert (%out0, above) so this checks the second one.
      // CHECK: tensor.extract_slice %{{.*}}[1, 0, 0]
      // CHECK: tensor.insert_slice
      // CHECK-SAME: tensor_ext.layout = #tensor_ext.layout<"{{.*}}slot - 4 * i - j = 0{{.*}}">
      %C = tensor.insert_slice %rTile1 into %out0[0, 2] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<2x4xf32>
      secret.yield %C : tensor<2x4xf32>
    } -> (!secret.secret<tensor<2x4xf32>>)
    return %result : !secret.secret<tensor<2x4xf32>>
  }
}
