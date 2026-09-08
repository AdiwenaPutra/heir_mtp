// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true" %s --verify-diagnostics

// Enabling the pass while leaving mtp-jkls-gemm-tile-size at its default
// (<= 0) is a pass-time diagnostic, not a silent no-op: a misconfiguration
// must never be mistaken for "no eligible ops found".

// expected-error@+1 {{tile-planning-mtp-jkls-gemm is enabled but mtp-jkls-gemm-tile-size}}
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
