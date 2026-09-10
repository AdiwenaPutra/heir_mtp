// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --mlir-print-local-scope %s | FileCheck %s

// UPDATED for P8.5: M=6, K=2, N=2 with mu=2: computeGemmTilePlan reports
// I=3, Q=1, J=1 (no contraction reduction, no boundary tiles), and
// taskCount=3 spans numCiphertexts=2 physical ciphertext groups at
// tilesPerCiphertext=2. Scattering across more than one physical
// ciphertext group -- once explicitly out of scope for this P8.3
// checkpoint (see the original, now-obsolete claim this test carried) --
// is now P8.5's own per-group implementation, and the P8.5 input-ABI
// integration automatically annotates A, B, and Cinit with the
// Convention-S MTP layout the per-group implementation needs.

// CHECK: func.func @multi_group
// CHECK-SAME: %arg0: !secret.secret<tensor<6x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-i0 + 2i1 - 2ct + slot) mod 6 = 0 and 0 <= i0 <= 5 and 0 <= i1 <= 1 and 0 <= ct <= 1 and slot >= i0 + i1 - 4ct and 0 <= slot <= 7 and slot <= 3 + i0 + i1 - 4ct and -i0 - 4i1 + 16ct + slot <= 6*floor((i0)/2) <= 6 - i0 - 4i1 + 16ct + slot }">
// CHECK-SAME: %arg1: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-i0 + 2i1 + slot) mod 6 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 1 and slot >= -6 + i0 + 4i1 and slot >= i0 + i1 and 0 <= slot <= 7 and slot <= 3 + i0 + i1 and slot <= i0 + 4i1 }">
// CHECK-SAME: %arg2: !secret.secret<tensor<6x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-i0 + 2i1 - 2ct + slot) mod 6 = 0 and 0 <= i0 <= 5 and 0 <= i1 <= 1 and 0 <= ct <= 1 and slot >= i0 + i1 - 4ct and 0 <= slot <= 7 and slot <= 3 + i0 + i1 - 4ct and -i0 - 4i1 + 16ct + slot <= 6*floor((i0)/2) <= 6 - i0 - 4i1 + 16ct + slot }">
// CHECK-NOT: linalg.matmul
// CHECK: linalg.batch_matmul
module {
  func.func @multi_group(%A: !secret.secret<tensor<6x2xf32>>, %B: !secret.secret<tensor<2x2xf32>>, %Cinit: !secret.secret<tensor<6x2xf32>>) -> (!secret.secret<tensor<6x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<6x2xf32>>, %B: !secret.secret<tensor<2x2xf32>>, %Cinit: !secret.secret<tensor<6x2xf32>>) {
    ^body(%a: tensor<6x2xf32>, %b: tensor<2x2xf32>, %c: tensor<6x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<6x2xf32>, tensor<2x2xf32>) outs(%c : tensor<6x2xf32>) -> tensor<6x2xf32>
      secret.yield %r : tensor<6x2xf32>
    } -> (!secret.secret<tensor<6x2xf32>>)
    return %result : !secret.secret<tensor<6x2xf32>>
  }
}
