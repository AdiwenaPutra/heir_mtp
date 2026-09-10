// RUN: heir-opt --convert-to-ciphertext-semantics=min-slot-count=4 %s | FileCheck %s

// Stage-A integration regression for remapAndExtractResult
// (lib/Transforms/ConvertToCiphertextSemantics/ConvertToCiphertextSemantics.cpp):
// the extraction offset must come from the destination (range) side's tight
// ciphertext lower bound, not be hardcoded to 0.
//
// The first secret.generic assembles a genuine two-ciphertext carrier out of
// two DISTINCT real tiles: row 0 holds %tileA, row 1 holds %tileB (mirroring
// how a P8.5 per-group batch is assembled out of two real, differently-
// positioned tiles). The second secret.generic then extracts row 1's tile
// back out, with its own layout deliberately reporting itself at ciphertext
// index 1 (rather than a fresh ciphertext 0) -- the same relabeling used
// when scattering a finished tile into a multi-ciphertext destination that
// already holds other real data at ciphertext 0. Splitting the assembly and
// extraction into two secret.generic regions (rather than one) is what
// prevents ordinary insert_slice/extract_slice canonicalization from folding
// the whole chain away before this bug could ever be exercised.
//
// %tileA (row 0 of the carrier) is a real, distinct, nonzero value: if the
// extraction offset were wrongly hardcoded to 0, the numeric companion
// (remap_extract_nonzero_offset_test) would observe %tileA's own values,
// not zero and not garbage -- a genuine contamination signature, not merely
// "wrong by an off-by-one."
#tile = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = i1 and i0 = 0 and 0 <= i1 <= 3 }">
#carrier = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = i0 and slot = i1 and 0 <= i0 <= 1 and 0 <= i1 <= 3 }">
#result = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 1 and slot = i1 and i0 = 0 and 0 <= i1 <= 3 }">

module {
  // CHECK-LABEL: @nonzero_offset
  // CHECK: secret.generic
  // CHECK: secret.generic
  // CHECK: %[[REMAP:.*]] = tensor_ext.remap
  // A mutant restoring the old hardcoded offset would emit
  // `tensor.extract_slice %[[REMAP]][0, 0]` here instead, silently reading
  // row 0's real, distinct %tileA data (genuine contamination) rather than
  // row 1's %tileB data -- see the numeric companion test.
  // CHECK-NEXT: tensor.extract_slice %[[REMAP]][1, 0] [1, 4] [1, 1]
  func.func @nonzero_offset(%tileA: !secret.secret<tensor<1x4xf32>> {tensor_ext.layout = #tile},
                            %tileB: !secret.secret<tensor<1x4xf32>> {tensor_ext.layout = #tile})
      -> (!secret.secret<tensor<1x4xf32>> {tensor_ext.layout = #result}) {
    %carrier = secret.generic(%tileA: !secret.secret<tensor<1x4xf32>> {tensor_ext.layout = #tile},
                              %tileB: !secret.secret<tensor<1x4xf32>> {tensor_ext.layout = #tile}) {
    ^body(%a: tensor<1x4xf32>, %b: tensor<1x4xf32>):
      %zero = arith.constant dense<0.0> : tensor<2x4xf32>
      %z = tensor_ext.assign_layout %zero {layout = #carrier, tensor_ext.layout = #carrier} : tensor<2x4xf32>
      %ins0 = tensor.insert_slice %a into %z[0, 0] [1, 4] [1, 1] {tensor_ext.layout = #carrier} : tensor<1x4xf32> into tensor<2x4xf32>
      %ins1 = tensor.insert_slice %b into %ins0[1, 0] [1, 4] [1, 1] {tensor_ext.layout = #carrier} : tensor<1x4xf32> into tensor<2x4xf32>
      secret.yield %ins1 : tensor<2x4xf32>
    } -> (!secret.secret<tensor<2x4xf32>> {tensor_ext.layout = #carrier})
    %0 = secret.generic(%carrier: !secret.secret<tensor<2x4xf32>> {tensor_ext.layout = #carrier}) {
    ^body(%c: tensor<2x4xf32>):
      %res = tensor.extract_slice %c[1, 0] [1, 4] [1, 1] {tensor_ext.layout = #result} : tensor<2x4xf32> to tensor<1x4xf32>
      secret.yield %res : tensor<1x4xf32>
    } -> (!secret.secret<tensor<1x4xf32>> {tensor_ext.layout = #result})
    return %0 : !secret.secret<tensor<1x4xf32>>
  }
}
