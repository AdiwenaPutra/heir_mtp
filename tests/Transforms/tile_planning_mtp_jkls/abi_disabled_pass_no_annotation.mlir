// RUN: heir-opt --tile-planning-mtp-jkls-gemm="mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// P8.5 input-ABI integration test 5: enable-mtp-jkls-gemm defaults to
// false, so this pass must be a complete no-op -- no ABI annotation, no
// rewrite -- even though the shape below (M=2,K=2,N=6) would otherwise be
// an eligible multi-group plan.
// CHECK: func.func @q1_boundary
// CHECK-SAME: %arg0: !secret.secret<tensor<2x2xf32>>
// CHECK-NOT: tensor_ext.layout
// CHECK: linalg.matmul
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
