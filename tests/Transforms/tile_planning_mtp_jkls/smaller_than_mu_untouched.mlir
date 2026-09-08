// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// M=1, K=2, N=1 with mu=2: computeGemmTilePlan legitimately reports exactly
// one task (I=Q=J=1, since M and N are each smaller than mu and round up to
// one tile), but M != mu and N != mu -- this is a smaller-than-mu single
// task, not an exact full mu x mu x mu B1 tile. The eligibility check must
// require M == K == N == mu exactly, not merely taskCount == 1: before this
// check existed, this exact shape made the pass emit invalid IR (an
// out-of-bounds tensor.extract_slice, since the gather helper always
// extracts a full mu x mu region), not just an imprecise result. Chosen
// asymmetric (1x2 * 2x1, not 1x1 * 1x1) so a transposed-dimension mixup in
// any future eligibility check would also be caught here.

// CHECK: func.func @small
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @small(%A: !secret.secret<tensor<1x2xf32>>, %B: !secret.secret<tensor<2x1xf32>>, %Cinit: !secret.secret<tensor<1x1xf32>>) -> (!secret.secret<tensor<1x1xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<1x2xf32>>, %B: !secret.secret<tensor<2x1xf32>>, %Cinit: !secret.secret<tensor<1x1xf32>>) {
    ^body(%a: tensor<1x2xf32>, %b: tensor<2x1xf32>, %c: tensor<1x1xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<1x2xf32>, tensor<2x1xf32>) outs(%c : tensor<1x1xf32>) -> tensor<1x1xf32>
      secret.yield %r : tensor<1x1xf32>
    } -> (!secret.secret<tensor<1x1xf32>>)
    return %result : !secret.secret<tensor<1x1xf32>>
  }
}
