// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --mlir-print-local-scope %s | FileCheck %s

// P8.5 input-ABI integration test 1: A, B, and Cinit start with no explicit
// tensor_ext.layout at all. Since this GEMM's plan has numCiphertexts > 1
// (Q1 capacity-boundary shape: M=2,K=2,N=6,mu=2,minSlotCount=8 gives
// taskCount=3, tilesPerCiphertext=2, numCiphertexts=2), tile planning must
// derive and attach the Convention-S MTP layout (via the existing
// getMultiTileLayoutRelation, rowAxis=0, columnAxis=1) to each of the three
// OUTER func.func arguments directly -- these are the exact relation
// strings the P8.5 input-ABI checkpoint's fixture-only experiment proved
// sufficient for the unmodified per-group implementation to complete with
// zero convert_layout ops.
// CHECK: func.func @q1_boundary
// CHECK-SAME: %arg0: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-i0 + 2i1 + slot) mod 6 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 1 and slot >= -6 + i0 + 4i1 and slot >= i0 + i1 and 0 <= slot <= 7 and slot <= 3 + i0 + i1 and slot <= i0 + 4i1 }">
// CHECK-SAME: %arg1: !secret.secret<tensor<2x6xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-i0 + 2i1 - 2ct + slot) mod 6 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 5 and 0 <= ct <= 1 and slot >= -6 + i0 + 4i1 - 16ct and slot >= i0 + i1 - 4ct and 0 <= slot <= 7 and slot <= 3 + i0 + i1 - 4ct and slot <= i0 + 4i1 - 16ct }">
// CHECK-SAME: %arg2: !secret.secret<tensor<2x6xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-i0 + 2i1 - 2ct + slot) mod 6 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 5 and 0 <= ct <= 1 and slot >= -6 + i0 + 4i1 - 16ct and slot >= i0 + i1 - 4ct and 0 <= slot <= 7 and slot <= 3 + i0 + i1 - 4ct and slot <= i0 + 4i1 - 16ct }">
module {
  func.func @q1_boundary(%a: !secret.secret<tensor<2x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) -> (!secret.secret<tensor<2x6xf32>>) {
    %result = secret.generic(%a: !secret.secret<tensor<2x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) {
    ^body(%input0: tensor<2x2xf32>, %input1: tensor<2x6xf32>, %input2: tensor<2x6xf32>):
      %r = linalg.matmul ins(%input0, %input1 : tensor<2x2xf32>, tensor<2x6xf32>) outs(%input2 : tensor<2x6xf32>) -> tensor<2x6xf32>
      secret.yield %r : tensor<2x6xf32>
    } -> (!secret.secret<tensor<2x6xf32>>)
    return %result : !secret.secret<tensor<2x6xf32>>
  }
}
