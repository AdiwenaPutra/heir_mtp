// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// M=2, K=3, N=2 with mu=2: computeGemmTilePlan reports I=1, Q=2, J=1 -- a
// single destination with two contraction tiles, in scope for the P8.4
// Q > 1 checkpoint -- but K is not evenly divisible by mu, so the q=1
// contraction tile is a boundary tile with validContraction=1 < mu=2.
// Boundary-safe extraction is deliberately not implemented for this
// checkpoint either (assembleOneTileBatch always extracts a full mu x mu
// region). Must be left completely untouched, isolating that the P8.4
// (Q > 1) code path shares the same boundary guard as P8.2/P8.3, not just
// that some other guard happens to also reject this shape.

// CHECK: func.func @q_boundary
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @q_boundary(%A: !secret.secret<tensor<2x3xf32>>, %B: !secret.secret<tensor<3x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<2x3xf32>>, %B: !secret.secret<tensor<3x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) {
    ^body(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>, %c: tensor<2x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
      secret.yield %r : tensor<2x2xf32>
    } -> (!secret.secret<tensor<2x2xf32>>)
    return %result : !secret.secret<tensor<2x2xf32>>
  }
}
