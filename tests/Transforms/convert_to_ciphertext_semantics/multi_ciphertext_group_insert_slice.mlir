// RUN: heir-opt --layout-propagation=min-slot-count=8 --convert-to-ciphertext-semantics=min-slot-count=8 %s | FileCheck %s

// Stage 1 regression for the P8.5 multi-ciphertext secret
// tensor.insert_slice bug: a fix in ConvertToCiphertextSemantics's
// materializeLayout (lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.cpp),
// independent of the GEMM tile planner -- this fixture uses only
// hand-written, explicitly-laid-out tensor.insert_slice chains.
//
// #mtp lays out a [3,2,2] logical tensor (three 2x2 tiles, mu=2) across
// min-slot-count=8 with tilesPerCiphertext=2 (capacity floor(8/4)=2),
// giving numCiphertexts=ceil(3/2)=2: destTileId 0 and 1 share ciphertext
// group 0 at local positions 0 and 1; destTileId 2 alone occupies
// ciphertext group 1 at local position 0, leaving group 1's second tile
// position unused. destTileId 0 and destTileId 2 therefore share the SAME
// local position (0) in DIFFERENT groups -- exactly the case that, before
// the fix, made the reconciled source operand for the destTileId=2 insert
// materialize with a genuine-but-inflated multi-ciphertext type
// (tensor<2x8xf32> instead of tensor<1x8xf32>), crashing an assertion
// several steps later (`lhs.getType() == rhs.getType()` in
// makeAppropriatelyTypedMulOp, lib/Utils/Utils.cpp) when that oversized
// value was used in a per-ciphertext masked multiply expecting a genuine
// single-ciphertext [1, numSlots] operand.
//
// Three distinct secret 2x2 tiles are chain-inserted at destTileId 0, 1,
// and 2 in turn. Numerical equivalence (every tile landing in its own
// intended ciphertext, destination values outside each insert's own slice
// preserved across later insertions, and the unused final-group tile
// position confirmed zero) is verified separately with a CPU-side
// ISL-based interpreter extended to genuinely model [numCiphertexts,
// numSlots] physical tensors (stage1_repro_check.py, evidence directory);
// this test is the permanent structural regression that must keep passing
// on its own.

#mtp = #tensor_ext.layout<"{ [i0, i1, i2] -> [ct, slot] : (-2i0 + 2i1 - i2 - 2ct + slot) mod 6 = 0 and 0 <= i0 <= 2 and 0 <= i1 <= 1 and 0 <= i2 <= 1 and 0 <= ct <= 1 and slot >= -6 + 8i0 + 4i1 + i2 - 16ct and slot >= 2i0 + i1 + i2 - 4ct and 0 <= slot <= 7 and slot <= 3 + 2i0 + i1 + i2 - 4ct and slot <= 8i0 + 4i1 + i2 - 16ct }">

// The synthesized always-false relation this bug class has historically
// been adjacent to (an unrelated but easy-to-reintroduce failure mode) must
// not appear either; these two CHECK-NOT lines bound both the possible
// hoisted-alias preamble and the function body, matching this directory's
// established convention.
// CHECK-NOT: 1 = 0
// CHECK: func.func @multi_group_insert
// CHECK-NOT: 1 = 0
// The final physical result spans exactly two ciphertexts.
// CHECK: tensor<2x8xf32>
module {
  func.func @multi_group_insert(%a0: !secret.secret<tensor<2x2xf32>>, %a1: !secret.secret<tensor<2x2xf32>>, %a2: !secret.secret<tensor<2x2xf32>>) -> !secret.secret<tensor<3x2x2xf32>> {
    %result = secret.generic(%a0: !secret.secret<tensor<2x2xf32>>, %a1: !secret.secret<tensor<2x2xf32>>, %a2: !secret.secret<tensor<2x2xf32>>) {
    ^body(%t0: tensor<2x2xf32>, %t1: tensor<2x2xf32>, %t2: tensor<2x2xf32>):
      %zero = arith.constant dense<0.000000e+00> : tensor<3x2x2xf32>
      %dest0 = tensor_ext.assign_layout %zero {layout = #mtp, tensor_ext.layout = #mtp} : tensor<3x2x2xf32>
      // destTileId 0 -> group 0, local position 0.
      %dest1 = tensor.insert_slice %t0 into %dest0[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<3x2x2xf32>
      // destTileId 1 -> group 0, local position 1 (shares group 0 with
      // destTileId 0, at a distinct local position).
      %dest2 = tensor.insert_slice %t1 into %dest1[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<3x2x2xf32>
      // destTileId 2 -> group 1, local position 0 (shares LOCAL POSITION 0
      // with destTileId 0, but in a DIFFERENT ciphertext group -- the
      // aliasing-by-position-alone scenario this fix must not collide on).
      %dest3 = tensor.insert_slice %t2 into %dest2[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<3x2x2xf32>
      secret.yield %dest3 : tensor<3x2x2xf32>
    } -> !secret.secret<tensor<3x2x2xf32>>
    return %result : !secret.secret<tensor<3x2x2xf32>>
  }
}
