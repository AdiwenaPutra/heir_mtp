// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --layout-propagation=min-slot-count=8 --convert-to-ciphertext-semantics=min-slot-count=8 --mlir-print-local-scope %s | FileCheck %s

// P8.5 input-ABI integration test 8: the final materialized function's own
// RESULT metadata (tensor_ext.original_type, attached once the logical
// tensor<2x6xf32> result is converted to its physical tensor<2x8xf32>
// ciphertext-semantic form) records the same Convention-S MTP relation
// that governs C's own physical (ct,slot) placement -- proving the MTP-ness
// of the ABI is exposed at the function boundary in both directions, not
// only for inputs.
// CHECK: func.func @q1_boundary
// CHECK-SAME: -> (!secret.secret<tensor<2x8xf32>>
// CHECK-SAME: tensor_ext.original_type
// CHECK-SAME: originalType = tensor<2x6xf32>
// CHECK-SAME: layout = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : (-i0 + 2i1 - 2ct + slot) mod 6 = 0 and 0 <= i0 <= 1 and 0 <= i1 <= 5 and 0 <= ct <= 1
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
