// RUN: heir-opt --tile-planning-mtp-jkls-gemm %s | FileCheck %s

// With enable-mtp-jkls-gemm left at its default (false), the pass is a
// complete no-op regardless of the shape: the ordinary linalg.matmul must
// survive unchanged, with no batch_matmul, no gather/scatter, and no
// tensor_ext content introduced.

// CHECK: func.func @b1_matmul
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @b1_matmul(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) {
    ^body(%a: tensor<2x2xf32>, %b: tensor<2x2xf32>, %c: tensor<2x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x2xf32>) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
      secret.yield %r : tensor<2x2xf32>
    } -> (!secret.secret<tensor<2x2xf32>>)
    return %result : !secret.secret<tensor<2x2xf32>>
  }
}
