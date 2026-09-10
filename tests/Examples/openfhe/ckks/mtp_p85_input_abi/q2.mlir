// P8.5 input-ABI integration, Q=2 multi-group fixture (A:2x4, B:4x6,
// Cinit/C:2x6, mu=2, minSlotCount=8): a plain, unannotated secret-secret
// linalg.matmul requiring four kernel calls (two contraction steps times
// two destination groups). The P8.5 input-ABI integration must
// automatically select and annotate A, B, and Cinit -- B's own required
// layout here spans three physical ciphertexts (2x3=6 tiles at
// tilesPerCiphertext=2).
module {
  func.func @q2_multi_group(%a: !secret.secret<tensor<2x4xf32>>, %b: !secret.secret<tensor<4x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) -> (!secret.secret<tensor<2x6xf32>>) {
    %result = secret.generic(%a: !secret.secret<tensor<2x4xf32>>, %b: !secret.secret<tensor<4x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) {
    ^body(%input0: tensor<2x4xf32>, %input1: tensor<4x6xf32>, %input2: tensor<2x6xf32>):
      %r = linalg.matmul ins(%input0, %input1 : tensor<2x4xf32>, tensor<4x6xf32>) outs(%input2 : tensor<2x6xf32>) -> tensor<2x6xf32>
      secret.yield %r : tensor<2x6xf32>
    } -> (!secret.secret<tensor<2x6xf32>>)
    return %result : !secret.secret<tensor<2x6xf32>>
  }
}
