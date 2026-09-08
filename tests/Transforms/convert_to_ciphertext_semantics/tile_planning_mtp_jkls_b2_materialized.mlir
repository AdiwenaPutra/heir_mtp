// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --layout-propagation=min-slot-count=8 --convert-to-ciphertext-semantics=min-slot-count=8 %s | FileCheck %s

// Phase 8 P8.3 end-to-end check: an ordinary secret-secret linalg.matmul
// with I=1, Q=1, J=2 (mu=2, A: 2x2, B: 2x4, Cinit: 2x4 -- two
// destination-column tasks sharing one lhs tile) runs through
// tile-planning-mtp-jkls-gemm, then the existing, unmodified
// layout-propagation and convert-to-ciphertext-semantics passes, with no
// crash and no always-false ("1 = 0") tensor_ext.remap permutation silently
// discarding data. Numerical equivalence with the plaintext GEMM oracle,
// including the real nonzero init operand and zero-contamination checks
// across every physical slot, is verified separately with a CPU-side
// ISL-based interpreter (not FileCheck), since it requires evaluating the
// actual generated rotate/mask/mul/add arithmetic; this test is the
// permanent structural regression that must keep passing on its own.

// A pass-synthesized always-false relation may print either inline or
// hoisted into a named #layoutN alias ahead of the function (this test
// omits --mlir-print-local-scope, matching this directory's convention), so
// these two CHECK-NOT lines deliberately have no other FileCheck directive
// between them and their bounding match, covering both the alias preamble
// and the function body without a gap.
// CHECK-NOT: 1 = 0
// CHECK: func.func @b2_two_col
// CHECK-NOT: 1 = 0
module {
  func.func @b2_two_col(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x4xf32>>, %Cinit: !secret.secret<tensor<2x4xf32>>) -> (!secret.secret<tensor<2x4xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<2x2xf32>>, %B: !secret.secret<tensor<2x4xf32>>, %Cinit: !secret.secret<tensor<2x4xf32>>) {
    ^body(%a: tensor<2x2xf32>, %b: tensor<2x4xf32>, %c: tensor<2x4xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<2x2xf32>, tensor<2x4xf32>) outs(%c : tensor<2x4xf32>) -> tensor<2x4xf32>
      secret.yield %r : tensor<2x4xf32>
    } -> (!secret.secret<tensor<2x4xf32>>)
    return %result : !secret.secret<tensor<2x4xf32>>
  }
}
