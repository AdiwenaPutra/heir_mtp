// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --mlir-print-local-scope %s | FileCheck %s

// UPDATED for P8.5: M=K=N=4, mu=2: computeGemmTilePlan reports I=Q=J=2,
// taskCount=4, which at tilesPerCiphertext=2 spans numCiphertexts=2
// physical ciphertext groups -- the general B2/B3/B6-shaped multi-group
// case, once explicitly out of scope for this P8.2 checkpoint (see the
// original, now-obsolete claim this test carried), is now P8.5's own
// per-group implementation, and the P8.5 input-ABI integration
// automatically annotates A, B, and Cinit (which here all share the
// identical [4,4] shape and hence the identical Convention-S MTP layout)
// with the layout the per-group implementation needs.

// CHECK: func.func @four_by_four
// CHECK-SAME: %arg0: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - 2i1 + 2ct - slot + 2*floor((i0)/2)) mod 6 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= ct <= 1 and 0 <= slot <= 7 and -3 - i0 - i1 + 4ct + slot <= 2*floor((i0)/2) <= -i0 - i1 + 4ct + slot and -i0 - 4i1 + 16ct + slot <= 14*floor((i0)/2) <= 6 - i0 - 4i1 + 16ct + slot }">
// CHECK-SAME: %arg1: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - 2i1 + 2ct - slot + 2*floor((i0)/2)) mod 6 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= ct <= 1 and 0 <= slot <= 7 and -3 - i0 - i1 + 4ct + slot <= 2*floor((i0)/2) <= -i0 - i1 + 4ct + slot and -i0 - 4i1 + 16ct + slot <= 14*floor((i0)/2) <= 6 - i0 - 4i1 + 16ct + slot }">
// CHECK-SAME: %arg2: !secret.secret<tensor<4x4xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (i0 - 2i1 + 2ct - slot + 2*floor((i0)/2)) mod 6 = 0 and 0 <= i0 <= 3 and 0 <= i1 <= 3 and 0 <= ct <= 1 and 0 <= slot <= 7 and -3 - i0 - i1 + 4ct + slot <= 2*floor((i0)/2) <= -i0 - i1 + 4ct + slot and -i0 - 4i1 + 16ct + slot <= 14*floor((i0)/2) <= 6 - i0 - 4i1 + 16ct + slot }">
// CHECK-NOT: linalg.matmul
// CHECK: linalg.batch_matmul
module {
  func.func @four_by_four(%A: !secret.secret<tensor<4x4xf32>>, %B: !secret.secret<tensor<4x4xf32>>, %Cinit: !secret.secret<tensor<4x4xf32>>) -> (!secret.secret<tensor<4x4xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<4x4xf32>>, %B: !secret.secret<tensor<4x4xf32>>, %Cinit: !secret.secret<tensor<4x4xf32>>) {
    ^body(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>, %c: tensor<4x4xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x4xf32>) outs(%c : tensor<4x4xf32>) -> tensor<4x4xf32>
      secret.yield %r : tensor<4x4xf32>
    } -> (!secret.secret<tensor<4x4xf32>>)
    return %result : !secret.secret<tensor<4x4xf32>>
  }
}
