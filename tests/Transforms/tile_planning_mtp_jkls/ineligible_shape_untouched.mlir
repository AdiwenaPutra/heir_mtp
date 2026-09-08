// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// M=2, K=3: not equal, so this shape is never a single-task (M=K=N=mu) B1
// case for any mu -- computeGemmTilePlan may still succeed (Q > 1), but the
// resulting plan's taskCount/Q will not both be 1, so this checkpoint must
// leave the op completely untouched rather than attempt a partial rewrite.

// CHECK: func.func @non_square
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @non_square(%A: !secret.secret<tensor<2x3xf32>>, %B: !secret.secret<tensor<3x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<2x3xf32>>, %B: !secret.secret<tensor<3x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) {
    ^body(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>, %c: tensor<2x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
      secret.yield %r : tensor<2x2xf32>
    } -> (!secret.secret<tensor<2x2xf32>>)
    return %result : !secret.secret<tensor<2x2xf32>>
  }
}
