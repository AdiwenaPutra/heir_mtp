// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --mlir-print-local-scope %s | FileCheck %s

// P8.5 input-ABI integration test 2: A already carries the EXACT
// Convention-S MTP layout tile planning would itself derive. This must be
// preserved byte-for-byte (not merely "close enough") -- no diagnostic, no
// change to the attribute string.
#A = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-i0 + 2i1 + slot) mod 6 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 1 and slot >= -6 + i0 + 4i1 and slot >= i0 + i1 and 0 <= slot <= 7 and slot <= 3 + i0 + i1 and slot <= i0 + 4i1 }">

// CHECK: func.func @q1_boundary
// CHECK-SAME: %arg0: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and (-i0 + 2i1 + slot) mod 6 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 1 and slot >= -6 + i0 + 4i1 and slot >= i0 + i1 and 0 <= slot <= 7 and slot <= 3 + i0 + i1 and slot <= i0 + 4i1 }">
module {
  func.func @q1_boundary(%a: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #A}, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) -> (!secret.secret<tensor<2x6xf32>>) {
    %result = secret.generic(%a: !secret.secret<tensor<2x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) {
    ^body(%input0: tensor<2x2xf32>, %input1: tensor<2x6xf32>, %input2: tensor<2x6xf32>):
      %r = linalg.matmul ins(%input0, %input1 : tensor<2x2xf32>, tensor<2x6xf32>) outs(%input2 : tensor<2x6xf32>) -> tensor<2x6xf32>
      secret.yield %r : tensor<2x6xf32>
    } -> (!secret.secret<tensor<2x6xf32>>)
    return %result : !secret.secret<tensor<2x6xf32>>
  }
}
