// P8.5 input-ABI integration, Q=1 capacity-boundary fixture: a plain,
// unannotated secret-secret linalg.matmul. The P8.5 input-ABI integration
// (TilePlanningMtpJkls.cpp) must automatically select and annotate A, B,
// and Cinit with the Convention-S MTP layout the per-group implementation
// needs -- no explicit tensor_ext.layout or secret.kernel appears in this
// source, unlike the sibling mtp_jkls_batch_matmul example.
module {
  func.func @q1_boundary(%a: !secret.secret<tensor<2x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) -> (!secret.secret<tensor<2x6xf32>>) {
    %result = secret.generic(%a: !secret.secret<tensor<2x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<2x6xf32>>) {
    ^body(%input0: tensor<2x2xf32>, %input1: tensor<2x6xf32>, %input2: tensor<2x6xf32>):
      %r = linalg.matmul ins(%input0, %input1 : tensor<2x2xf32>, tensor<2x6xf32>) outs(%input2 : tensor<2x6xf32>) -> tensor<2x6xf32>
      secret.yield %r : tensor<2x6xf32>
    } -> (!secret.secret<tensor<2x6xf32>>)
    return %result : !secret.secret<tensor<2x6xf32>>
  }
}
