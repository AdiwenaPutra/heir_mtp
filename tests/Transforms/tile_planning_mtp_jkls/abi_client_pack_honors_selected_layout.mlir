// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --annotate-module="backend=openfhe scheme=ckks" --mlir-to-ckks="min-slot-count=8" --scheme-to-openfhe="entry-function=q1_boundary" %s | FileCheck %s

// P8.5 input-ABI integration test 7: the existing, generic client-pack
// codegen (which derives its own packing loop directly from whatever
// tensor_ext.layout ends up on a function argument, with no P8.5-specific
// logic) honors the AUTOMATICALLY selected Convention-S MTP layout for
// %arg1 (B) exactly as it already honors an explicit, hand-written one --
// its own generated __encrypt__arg1 function builds a genuinely
// two-ciphertext (tensor<2x!lwe.lwe_ciphertext<...>>) plaintext packing
// via an scf-based loop, matching B's [2,8]-physical, tile-aligned MTP
// layout rather than a naive row-major one.
// CHECK: func.func @q1_boundary__encrypt__arg1
// CHECK-SAME: -> tensor<2x!{{.*}}>
// CHECK: scf.for
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
