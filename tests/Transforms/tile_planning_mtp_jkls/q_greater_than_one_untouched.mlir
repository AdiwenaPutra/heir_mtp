// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// M=2, K=4, N=2 with mu=2: computeGemmTilePlan reports I=1, Q=2, J=1 -- a
// B3-style shape whose single destination tile requires chaining the
// batch_matmul's `outs` across two q-steps to accumulate partial products.
// That contraction-dimension reduction is explicitly P8.4, not this P8.3
// checkpoint (which only generalizes I*J > 1 for Q == 1). Must be left
// completely untouched, not partially or incorrectly rewritten.

// CHECK: func.func @q_two
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @q_two(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) {
    ^body(%a: tensor<2x4xf32>, %b: tensor<4x2xf32>, %c: tensor<2x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<2x4xf32>, tensor<4x2xf32>) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
      secret.yield %r : tensor<2x2xf32>
    } -> (!secret.secret<tensor<2x2xf32>>)
    return %result : !secret.secret<tensor<2x2xf32>>
  }
}
