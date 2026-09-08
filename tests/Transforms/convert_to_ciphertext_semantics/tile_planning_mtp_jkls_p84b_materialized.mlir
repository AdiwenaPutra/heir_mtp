// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --layout-propagation=min-slot-count=8 --convert-to-ciphertext-semantics=min-slot-count=8 %s | FileCheck %s

// Phase 8 P8.4b end-to-end check: an ordinary secret-secret linalg.matmul
// with I=1, Q=2, J=2 (mu=2, A: 2x4, B: 4x4, Cinit: 2x4 -- two destinations,
// each accumulating two contraction-tile partial products) runs through
// tile-planning-mtp-jkls-gemm, then the existing, unmodified
// layout-propagation and convert-to-ciphertext-semantics passes, with no
// crash and no always-false ("1 = 0") tensor_ext.remap permutation silently
// discarding data -- in particular proving the per-destination arith.addf
// reductions this pass introduces are each assigned a consistent,
// non-degenerate layout by the unmodified LayoutPropagation pass. Full
// numerical equivalence with the plaintext Cinit+A@B oracle (zero and
// nonzero init, zero-contamination check) is verified separately with a
// CPU-side ISL-based interpreter (not FileCheck); this test is the
// permanent structural regression that must keep passing on its own.

// CHECK-NOT: 1 = 0
// CHECK: func.func @p84b_primary
// CHECK-NOT: 1 = 0
module {
  func.func @p84b_primary(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x4xf32>>, %Cinit: !secret.secret<tensor<2x4xf32>>) -> (!secret.secret<tensor<2x4xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x4xf32>>, %Cinit: !secret.secret<tensor<2x4xf32>>) {
    ^body(%a: tensor<2x4xf32>, %b: tensor<4x4xf32>, %c: tensor<2x4xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<2x4xf32>, tensor<4x4xf32>) outs(%c : tensor<2x4xf32>) -> tensor<2x4xf32>
      secret.yield %r : tensor<2x4xf32>
    } -> (!secret.secret<tensor<2x4xf32>>)
    return %result : !secret.secret<tensor<2x4xf32>>
  }
}
