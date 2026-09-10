// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=12" --verify-diagnostics %s | FileCheck %s

// P8.5 input-ABI integration test 4: two separate multi-group GEMMs share
// the SAME function argument %b (a [2,2] secret tensor) as their rhs, but
// each has a different M (8 vs 10), giving different taskCount (4 vs 5)
// and, after getBalancedMtpPacking's own balancing at capacity
// floor(12/4)=3, genuinely different tilesPerCiphertext (2 vs 3) --
// taskCount=4 -> numCiphertexts=ceil(4/3)=2, tilesPerCiphertext=ceil(4/2)=2;
// taskCount=5 -> numCiphertexts=ceil(5/3)=2, tilesPerCiphertext=ceil(5/3)=3.
// Both are genuinely multi-group (numCiphertexts=2), and both require
// %b's Convention-S MTP layout to be built with a different
// tilesPerCiphertext -- an irreconcilable requirement on one argument.
// This must be diagnosed cleanly, and refuse to make ANY partial ABI
// change (not even to the non-conflicting arguments), per the conflict
// policy's "before making any partial ABI changes" requirement. The ABI
// selection is a best-effort optimization, so the diagnostic does not
// abort the rest of the pass: both GEMMs still materialize into
// linalg.batch_matmul normally, just without the ABI annotation.
// No function argument may carry a tensor_ext.layout: the conflict must be
// discovered before any partial ABI mutation, including to %a1/%c1, which
// have no conflict of their own and would otherwise be safe to annotate.
// CHECK: func.func @conflict
// CHECK-NOT: tensor_ext.layout
// CHECK: secret.generic
// CHECK: linalg.batch_matmul
// CHECK: linalg.batch_matmul
module {
  func.func @conflict(%a1: !secret.secret<tensor<8x2xf32>>, %b: !secret.secret<tensor<2x2xf32>>, %c1: !secret.secret<tensor<8x2xf32>>, %a2: !secret.secret<tensor<10x2xf32>>, %c2: !secret.secret<tensor<10x2xf32>>) -> (!secret.secret<tensor<8x2xf32>>, !secret.secret<tensor<10x2xf32>>) {
    %r1 = secret.generic(%a1: !secret.secret<tensor<8x2xf32>>, %b: !secret.secret<tensor<2x2xf32>>, %c1: !secret.secret<tensor<8x2xf32>>) {
    ^body(%x: tensor<8x2xf32>, %y: tensor<2x2xf32>, %z: tensor<8x2xf32>):
      %r = linalg.matmul ins(%x, %y : tensor<8x2xf32>, tensor<2x2xf32>) outs(%z : tensor<8x2xf32>) -> tensor<8x2xf32>
      secret.yield %r : tensor<8x2xf32>
    } -> (!secret.secret<tensor<8x2xf32>>)
    %r2 = secret.generic(%a2: !secret.secret<tensor<10x2xf32>>, %b: !secret.secret<tensor<2x2xf32>>, %c2: !secret.secret<tensor<10x2xf32>>) {
    ^body(%x: tensor<10x2xf32>, %y: tensor<2x2xf32>, %z: tensor<10x2xf32>):
      // expected-error@+1 {{P8.5 input-ABI selection: two selected multi-group GEMMs require incompatible Convention-S MTP layouts for the same function argument #1 of @conflict}}
      %r = linalg.matmul ins(%x, %y : tensor<10x2xf32>, tensor<2x2xf32>) outs(%z : tensor<10x2xf32>) -> tensor<10x2xf32>
      secret.yield %r : tensor<10x2xf32>
    } -> (!secret.secret<tensor<10x2xf32>>)
    return %r1, %r2 : !secret.secret<tensor<8x2xf32>>, !secret.secret<tensor<10x2xf32>>
  }
}
