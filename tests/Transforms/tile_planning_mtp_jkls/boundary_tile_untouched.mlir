// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// M=2, K=2, N=3 with mu=2: computeGemmTilePlan reports I=1, Q=1, J=2,
// taskCount=2 (fits in one ciphertext at capacity floor(8/4)=2 -- the
// multi-ciphertext-group guard does NOT apply here, isolating this test to
// the boundary check alone), but N is not evenly divisible by mu, so the
// column-tile at j=1 is a boundary tile with validColumns=1 < mu=2.
// Boundary-safe extraction (extract only the valid sub-rectangle, zero-pad
// the rest) is deliberately not implemented yet -- assembleMultiTaskBatch's
// getOrExtractTile always extracts a full mu x mu region at each task's
// tile offset, which would run out-of-bounds for this shape's second
// column tile. Must be left completely untouched, not partially or
// incorrectly rewritten.

// CHECK: func.func @column_boundary
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @column_boundary(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x3xf32>>, %Cinit: !secret.secret<tensor<2x3xf32>>) -> (!secret.secret<tensor<2x3xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x3xf32>>, %Cinit: !secret.secret<tensor<2x3xf32>>) {
    ^body(%a: tensor<2x2xf32>, %b: tensor<2x3xf32>, %c: tensor<2x3xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x3xf32>) outs(%c : tensor<2x3xf32>) -> tensor<2x3xf32>
      secret.yield %r : tensor<2x3xf32>
    } -> (!secret.secret<tensor<2x3xf32>>)
    return %result : !secret.secret<tensor<2x3xf32>>
  }
}
