// RUN: heir-opt --layout-propagation=min-slot-count=4 --mlir-print-local-scope %s | FileCheck %s

// Isolated regression for getRequiredSliceLayoutForAuthoritativeDest's
// zero-normalization fix (LayoutPropagation.cpp): a standalone one-
// ciphertext source, locally indexed from zero (ct = 0), is inserted into
// an authoritative destination slice whose absolute physical location is
// ciphertext 1. This is structural tensor assembly (insertion geometry
// alone determines which absolute destination row receives the source),
// not an FHE rotation -- so no conversion should ever be inserted between
// the source and the insert.
//
// Before the fix, getRequiredSliceLayoutForAuthoritativeDest composed the
// insertion geometry through the destination's OWN absolute addressing
// (ct = 1 here) and compared that directly against the source's ct = 0
// layout -- a spurious mismatch that forced a tensor_ext.convert_layout
// whose to_layout (ct = 1) tensor_ext.remap's AllTypesMatch trait cannot
// satisfy from a genuinely one-ciphertext carrier (this is the P8.5
// per-group scatter bug). The fix renormalizes the computed REQUIREMENT
// (never the destination's own relation, which keeps its absolute
// ct = 0, 1, ... addressing) to start at zero before comparing it against
// the standalone source, exactly as tensor.extract_slice's own result-
// layout computation and ConvertTensorInsertSlice::secretScalarSecretTensor
// already do.
//
// The source's layout string below is written in the same canonical form
// LayoutAttr::getFromIntegerRelation itself produces (`slot = 2i0 + i1`,
// not an algebraically-equivalent but differently-formatted
// `slot - 2*i0 - i1 = 0`), since compatibility is checked by strict
// attribute-string equality, not relation equivalence.
#dest = #tensor_ext.layout<"{ [p, l, d] -> [ct, slot] : ct = p and slot - 2 * l - d = 0 and 0 <= p <= 1 and 0 <= l <= 1 and 0 <= d <= 1 and 0 <= slot <= 3 }">
#source = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 2i0 + i1 and 0 <= i0 <= 1 and 0 <= i1 <= 1 }">

module {
  // CHECK-LABEL: @group1_insert
  // A mutant that omits normalization (or shifts in the wrong direction, or
  // uses the domain-side bound instead of the range ciphertext bound) would
  // compute a required source layout of ct = 1 (or some other value != 0)
  // here, disagreeing with %src's own ct = 0 layout and forcing a
  // convert_layout that the correct, normalized comparison never needs.
  // CHECK-NOT: tensor_ext.convert_layout
  // CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[1, 0, 0]
  func.func @group1_insert(%src: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #source}) -> (!secret.secret<tensor<2x2x2xf32>>) {
    %zero = arith.constant dense<0.0> : tensor<2x2x2xf32>
    %result = secret.generic(%src: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #source}) {
    ^body(%s: tensor<2x2xf32>):
      %authDest = tensor_ext.assign_layout %zero {layout = #dest, tensor_ext.layout = #dest} : tensor<2x2x2xf32>
      %inserted = tensor.insert_slice %s into %authDest[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<2x2x2xf32>
      secret.yield %inserted : tensor<2x2x2xf32>
    } -> (!secret.secret<tensor<2x2x2xf32>>)
    return %result : !secret.secret<tensor<2x2x2xf32>>
  }
}
