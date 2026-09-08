// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" --layout-propagation=min-slot-count=8 --convert-to-ciphertext-semantics=min-slot-count=8 %s | FileCheck %s

// Phase 8 P8.4b end-to-end check for the complementary row-oriented
// fixture: mu=2, A: 4x4, B: 4x2, Cinit: 4x2 -> I=2, Q=2, J=1 (destinations
// differing along the row dimension). No crash, no always-false ("1 = 0")
// remap. Numerical equivalence verified separately (p84b_row_check.py).

// CHECK-NOT: 1 = 0
// CHECK: func.func @p84b_row
// CHECK-NOT: 1 = 0
module {
  func.func @p84b_row(%A: !secret.secret<tensor<4x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<4x2xf32>>) -> (!secret.secret<tensor<4x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<4x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<4x2xf32>>) {
    ^body(%a: tensor<4x4xf32>, %b: tensor<4x2xf32>, %c: tensor<4x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<4x4xf32>, tensor<4x2xf32>) outs(%c : tensor<4x2xf32>) -> tensor<4x2xf32>
      secret.yield %r : tensor<4x2xf32>
    } -> (!secret.secret<tensor<4x2xf32>>)
    return %result : !secret.secret<tensor<4x2xf32>>
  }
}
