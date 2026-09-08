// RUN: heir-opt --convert-to-ciphertext-semantics=min-slot-count=8 --split-input-file %s | FileCheck %s

// Regression test for a bug where ConvertTensorExtractSlice tagged an
// extracted slice's ongoing tensor_ext.layout context with the physical
// (ciphertext, slot) -> (ciphertext, slot) remap relation it builds
// internally to materialize the extraction, instead of the slice's own
// already-known logical-index -> (ciphertext, slot) result layout (the
// value bound to `resultLayout` in secretSourceSecretResult). A later op
// reading that mistagged context (here, the second tensor.insert_slice,
// consuming a tile extracted at a nonzero offset) composed against the
// wrong relation and could silently produce an always-false ("1 = 0")
// tensor_ext.remap permutation, which forces every slot to zero, discarding
// the extracted data. The fix matches the existing ConvertTensorPad
// pattern's convention just above in this file:
// the composed physical relation is used only to build the tensor_ext.remap
// op, while the produced value's tracked layout stays the logical result
// layout that was already known before this pattern ran.
//
// This input is the verbatim --layout-propagation output of a hand-written
// two-tile scatter (extract two adjacent 2x2 tiles of a 2x4 tensor, insert
// each into its own zero-seeded 1x2x2 batch slot, then add), pinned here as
// a static fixture so this test exercises only --convert-to-ciphertext-semantics.

module {
  // The bug produced an always-false "1 = 0" tensor_ext.remap permutation
  // relation. MLIR may print it either inline or hoisted into a named
  // #layoutN alias ahead of the function (when not using
  // --mlir-print-local-scope, as here), so the two CHECK-NOT lines below
  // deliberately have no other FileCheck directive between them and their
  // respective bounding match, to cover both the alias preamble and the
  // function body without gaps.
  // CHECK-NOT: 1 = 0
  // CHECK: func.func @extract_then_insert
  // CHECK-NOT: 1 = 0
  func.func @extract_then_insert(%arg0: !secret.secret<tensor<2x4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (4i0 - i1 + slot) mod 8 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= slot <= 7 }">}) -> (!secret.secret<tensor<1x2x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">}) {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x2x2xf32>
    // CHECK: return
    %0 = secret.generic(%arg0: !secret.secret<tensor<2x4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (4i0 - i1 + slot) mod 8 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 3 and 0 <= slot <= 7 }">}) {
    ^body(%input0: tensor<2x4xf32>):
      %1 = tensor_ext.assign_layout %cst {layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (2i1 - i2 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">, tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (2i1 - i2 + slot) mod 4 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">} : tensor<1x2x2xf32>
      %2 = tensor_ext.assign_layout %1 {layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (-2 + 4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">, tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (-2 + 4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">} : tensor<1x2x2xf32>
      %3 = tensor_ext.assign_layout %1 {layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">, tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">} : tensor<1x2x2xf32>
      %extracted_slice = tensor.extract_slice %input0[0, 0] [2, 2] [1, 1] {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (4i0 - i1 + slot) mod 8 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 1 and 0 <= slot <= 7 }">} : tensor<2x4xf32> to tensor<2x2xf32>
      %inserted_slice = tensor.insert_slice %extracted_slice into %3[0, 0, 0] [1, 2, 2] [1, 1, 1] {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">} : tensor<2x2xf32> into tensor<1x2x2xf32>
      %extracted_slice_0 = tensor.extract_slice %input0[0, 2] [2, 2] [1, 1] {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-2 + 4i0 - i1 + slot) mod 8 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 1 and 0 <= slot <= 7 }">} : tensor<2x4xf32> to tensor<2x2xf32>
      %inserted_slice_1 = tensor.insert_slice %extracted_slice_0 into %2[0, 0, 0] [1, 2, 2] [1, 1, 1] {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (-2 + 4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">} : tensor<2x2xf32> into tensor<1x2x2xf32>
      %4 = tensor_ext.convert_layout %inserted_slice_1 {from_layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (-2 + 4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">, tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">, to_layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">} : tensor<1x2x2xf32>
      %5 = arith.addf %inserted_slice, %4 {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">} : tensor<1x2x2xf32>
      secret.yield %5 : tensor<1x2x2xf32>
    } -> (!secret.secret<tensor<1x2x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : i0 = 0 and ct = 0 and (4i1 - i2 + slot) mod 8 = 0 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= slot <= 7 }">})
    return %0 : !secret.secret<tensor<1x2x2xf32>>
  }
}
