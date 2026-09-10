// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --mlir-print-local-scope --verify-diagnostics %s | FileCheck %s

// P8.5 input-ABI integration test 3: A carries an explicit user layout
// (a plain row-major relation) that is INCOMPATIBLE with the Convention-S
// MTP layout this multi-group GEMM requires. Tile planning must not
// overwrite it -- the original, wrong-for-MTP layout must survive
// unchanged in the output -- and must instead emit a clean diagnostic
// naming both the existing and required layouts.
#WrongA = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 2 * i0 + i1 and 0 <= i0 <= 1 and 0 <= i1 <= 1 }">

module {
  // CHECK: func.func @q1_boundary
  // CHECK-SAME: %arg0: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 2 * i0 + i1 and 0 <= i0 <= 1 and 0 <= i1 <= 1 }">
  func.func @q1_boundary(%a: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #WrongA}, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) -> (!secret.secret<tensor<2x6xf32>>) {
    %result = secret.generic(%a: !secret.secret<tensor<2x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) {
    ^body(%input0: tensor<2x2xf32>, %input1: tensor<2x6xf32>, %input2: tensor<2x6xf32>):
      // expected-error@+1 {{P8.5 input-ABI selection: function argument #0 of @q1_boundary already has an explicit tensor_ext.layout incompatible with the Convention-S MTP layout this multi-group GEMM's lhs (A) operand requires; refusing to overwrite it}}
      %r = linalg.matmul ins(%input0, %input1 : tensor<2x2xf32>, tensor<2x6xf32>) outs(%input2 : tensor<2x6xf32>) -> tensor<2x6xf32>
      secret.yield %r : tensor<2x6xf32>
    } -> (!secret.secret<tensor<2x6xf32>>)
    return %result : !secret.secret<tensor<2x6xf32>>
  }
}
