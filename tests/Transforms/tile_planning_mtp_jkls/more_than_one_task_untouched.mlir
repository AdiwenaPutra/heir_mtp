// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// M=K=N=4, mu=2: computeGemmTilePlan reports I=Q=J=2, taskCount=4 (more
// than one task) -- the general B2/B3/B6 case, explicitly out of scope for
// this P8.2 checkpoint. Must be left completely untouched, not partially
// or incorrectly rewritten.

// CHECK: func.func @four_by_four
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
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
