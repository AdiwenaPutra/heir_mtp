// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// M=K=N=1 with mu=2: square, but still smaller than mu in every dimension
// -- complementary to smaller_than_mu_untouched.mlir's asymmetric 1x2*2x1
// case, confirming a fully symmetric sub-mu shape is equally untouched
// (not just one where distinct dimensions might mask an off-by-one in the
// eligibility check).

// CHECK: func.func @onebyone
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @onebyone(%A: !secret.secret<tensor<1x1xf32>>, %B: !secret.secret<tensor<1x1xf32>>, %Cinit: !secret.secret<tensor<1x1xf32>>) -> (!secret.secret<tensor<1x1xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<1x1xf32>>, %B: !secret.secret<tensor<1x1xf32>>, %Cinit: !secret.secret<tensor<1x1xf32>>) {
    ^body(%a: tensor<1x1xf32>, %b: tensor<1x1xf32>, %c: tensor<1x1xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<1x1xf32>, tensor<1x1xf32>) outs(%c : tensor<1x1xf32>) -> tensor<1x1xf32>
      secret.yield %r : tensor<1x1xf32>
    } -> (!secret.secret<tensor<1x1xf32>>)
    return %result : !secret.secret<tensor<1x1xf32>>
  }
}
