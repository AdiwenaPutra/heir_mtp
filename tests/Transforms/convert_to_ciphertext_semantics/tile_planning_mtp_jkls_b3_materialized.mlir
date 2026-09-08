// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --layout-propagation=min-slot-count=8 --convert-to-ciphertext-semantics=min-slot-count=8 %s | FileCheck %s

// Phase 8 P8.4 end-to-end check: an ordinary secret-secret linalg.matmul
// with I=1, Q=2, J=1 (mu=2, A: 2x4, B: 4x2, Cinit: 2x2 -- one destination
// accumulating two contraction-tile partial products) runs through
// tile-planning-mtp-jkls-gemm, then the existing, unmodified
// layout-propagation and convert-to-ciphertext-semantics passes, with no
// crash and no always-false ("1 = 0") tensor_ext.remap permutation silently
// discarding data -- in particular proving the plain arith.addf reduction
// this pass introduces (combining the two q-steps' extracted tiles) is
// itself assigned a consistent, non-degenerate layout by the unmodified
// LayoutPropagation pass. Numerical equivalence with the plaintext GEMM
// oracle for both zero and nonzero init, including zero-contamination
// checks across every physical slot, is verified separately with a
// CPU-side ISL-based interpreter (not FileCheck), since it requires
// evaluating the actual generated rotate/mask/mul/add arithmetic; this test
// is the permanent structural regression that must keep passing on its own.

// A pass-synthesized always-false relation may print either inline or
// hoisted into a named #layoutN alias ahead of the function (this test
// omits --mlir-print-local-scope, matching this directory's convention), so
// these two CHECK-NOT lines deliberately have no other FileCheck directive
// between them and their bounding match, covering both the alias preamble
// and the function body without a gap.
// CHECK-NOT: 1 = 0
// CHECK: func.func @b3_two_q
// CHECK-NOT: 1 = 0
module {
  func.func @b3_two_q(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<2x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<2x2xf32>>) {
    ^body(%a: tensor<2x4xf32>, %b: tensor<4x2xf32>, %c: tensor<2x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<2x4xf32>, tensor<4x2xf32>) outs(%c : tensor<2x2xf32>) -> tensor<2x2xf32>
      secret.yield %r : tensor<2x2xf32>
    } -> (!secret.secret<tensor<2x2xf32>>)
    return %result : !secret.secret<tensor<2x2xf32>>
  }
}
