// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --mlir-print-local-scope %s | FileCheck %s

// UPDATED for P8.5 (previously updated for P8.4b, see the file's own
// earlier history: this M=4,K=4,N=2 shape was once the negative case for
// "any combined Q>1 && taskCount>1 stays untouched" under P8.4, then
// repurposed as P8.4b's own positive single-group test elsewhere, then
// repurposed AGAIN here as the negative "multi-group Q>1 && taskCount>1
// stays untouched" case for P8.4b -- and now that claim is ALSO obsolete).
//
// M=6, K=4, N=2 with mu=2: computeGemmTilePlan reports I=3, Q=2, J=1 --
// taskCount=3 destinations, each needing a two-contraction-tile reduction,
// spanning numCiphertexts=2 physical ciphertext groups at
// tilesPerCiphertext=2. Multi-group support is now P8.5's own per-group
// implementation, and the P8.5 input-ABI integration automatically
// annotates A, B, and Cinit with the Convention-S MTP layout the
// per-group implementation needs.

// CHECK: func.func @multi_group_combined
// CHECK-SAME: %arg0: !secret.secret<tensor<6x4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - 2i1 + 2ct - slot + 2*floor((i0)/2)) mod 6 = 0 and 0 <= i0 <= 5 and 0 <= i1 <= 3 and 0 <= ct <= 2 and 0 <= slot <= 7 and -3 - i0 - i1 + 4ct + slot <= 2*floor((i0)/2) <= -i0 - i1 + 4ct + slot and -i0 - 4i1 + 16ct + slot <= 14*floor((i0)/2) <= 6 - i0 - 4i1 + 16ct + slot }">
// CHECK-SAME: %arg1: !secret.secret<tensor<4x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-i0 + 2i1 + slot) mod 6 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 1 and slot >= i0 + i1 and 0 <= slot <= 7 and slot <= 3 + i0 + i1 and -i0 - 4i1 + slot <= 6*floor((i0)/2) <= 6 - i0 - 4i1 + slot }">
// CHECK-SAME: %arg2: !secret.secret<tensor<6x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-i0 + 2i1 - 2ct + slot) mod 6 = 0 and 0 <= i0 <= 5 and 0 <= i1 <= 1 and 0 <= ct <= 1 and slot >= i0 + i1 - 4ct and 0 <= slot <= 7 and slot <= 3 + i0 + i1 - 4ct and -i0 - 4i1 + 16ct + slot <= 6*floor((i0)/2) <= 6 - i0 - 4i1 + 16ct + slot }">
// CHECK-NOT: linalg.matmul
// CHECK: linalg.batch_matmul
module {
  func.func @multi_group_combined(%A: !secret.secret<tensor<6x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<6x2xf32>>) -> (!secret.secret<tensor<6x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<6x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<6x2xf32>>) {
    ^body(%a: tensor<6x4xf32>, %b: tensor<4x2xf32>, %c: tensor<6x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<6x4xf32>, tensor<4x2xf32>) outs(%c : tensor<6x2xf32>) -> tensor<6x2xf32>
      secret.yield %r : tensor<6x2xf32>
    } -> (!secret.secret<tensor<6x2xf32>>)
    return %result : !secret.secret<tensor<6x2xf32>>
  }
}
