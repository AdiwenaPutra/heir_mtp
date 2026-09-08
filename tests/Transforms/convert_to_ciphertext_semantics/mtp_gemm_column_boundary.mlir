// RUN: heir-opt --layout-propagation=min-slot-count=16 --convert-to-ciphertext-semantics=min-slot-count=16 %s | FileCheck %s

// Acceptance test complementing mtp_gemm_b6_boundary.mlir: a partial
// output-COLUMN boundary (mu=2, A:4x4, B:4x3, C:4x3 -> I=2, Q=2, J=2,
// taskCount=4; M and K divide mu evenly, so there is no row or contraction
// boundary here, isolating the column case). Column tile j=1 has valid
// extent 1 of mu=2, so its final scatter into C is a narrower-than-mu-wide
// insert into the shared row-major output destination -- a boundary
// direction the row/contraction-focused B6 fixture does not exercise. The
// destination-authority mechanism must reconcile this scatter without
// crashing, without relaxing the destination's addressing, and without an
// always-false ("1 = 0") tensor_ext.remap permutation discarding data.
// Full numerical verification against a plaintext GEMM oracle requires an
// out-of-tree ISL-based interpreter, not FileCheck, and is not part of this
// test; this test is the permanent structural regression that must keep
// passing on its own.

// A pass-synthesized always-false relation may print either inline or
// hoisted into a named #layoutN alias ahead of the function (this test
// omits --mlir-print-local-scope, matching this directory's convention), so
// these two CHECK-NOT lines deliberately have no other FileCheck directive
// between them and their bounding match, covering both the alias preamble
// and the function body without a gap.
// CHECK-NOT: 1 = 0
// CHECK: func.func @mtp_gemm_column_boundary
// CHECK-NOT: 1 = 0

#mtp4 = #tensor_ext.layout<"{ [p, d, l] -> [ct, slot] : ct = 0 and slot - 8*d - 2*p - l = 0 and 0 <= p <= 3 and 0 <= d <= 1 and 0 <= l <= 1 and 0 <= slot <= 15 }">
#rowmajorC = #tensor_ext.layout<"{ [i, j] -> [ct, slot] : ct = 0 and slot - 3*i - j = 0 and 0 <= i <= 3 and 0 <= j <= 2 and 0 <= slot <= 15 }">
#tileRowMajor = #tensor_ext.layout<"{ [d, l] -> [ct, slot] : ct = 0 and slot - 2*d - l = 0 and 0 <= d <= 1 and 0 <= l <= 1 and 0 <= slot <= 15 }">
#kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>

module {
  func.func @mtp_gemm_column_boundary(
      %A: !secret.secret<tensor<4x4xf32>>,
      %B: !secret.secret<tensor<4x3xf32>>)
      -> (!secret.secret<tensor<4x3xf32>>) {
    %zero4 = arith.constant dense<0.0> : tensor<4x2x2xf32>
    %outZero = arith.constant dense<0.0> : tensor<4x3xf32>
    %result = secret.generic(
        %A: !secret.secret<tensor<4x4xf32>>,
        %B: !secret.secret<tensor<4x3xf32>>) {
    ^body(%a: tensor<4x4xf32>, %b: tensor<4x3xf32>):
      %mtpZero = tensor_ext.assign_layout %zero4 {layout = #mtp4, tensor_ext.layout = #mtp4} : tensor<4x2x2xf32>
      %outInit = tensor_ext.assign_layout %outZero {layout = #rowmajorC, tensor_ext.layout = #rowmajorC} : tensor<4x3xf32>

      // ---- q=0 ----
      // lhs task p=0 (i=0,q=0): A[0:2, 0:2]
      %g0 = tensor.extract_slice %a[0, 0] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      // lhs task p=1 (i=0,q=0): A[0:2, 0:2]
      %g1 = tensor.extract_slice %a[0, 0] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      // lhs task p=2 (i=1,q=0): A[2:4, 0:2]
      %g2 = tensor.extract_slice %a[2, 0] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      // lhs task p=3 (i=1,q=0): A[2:4, 0:2]
      %g3 = tensor.extract_slice %a[2, 0] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      %lhsQ0_0 = tensor.insert_slice %g0 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ0_1 = tensor.insert_slice %g1 into %lhsQ0_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ0_2 = tensor.insert_slice %g2 into %lhsQ0_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ0_3 = tensor.insert_slice %g3 into %lhsQ0_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      // rhs task p=0 (q=0,j=0): B[0:2, 0:2]
      %g4 = tensor.extract_slice %b[0, 0] [2, 2] [1, 1] : tensor<4x3xf32> to tensor<2x2xf32>
      // rhs task p=1 (q=0,j=1): B[0:2, 2:3]
      %g5 = tensor.extract_slice %b[0, 2] [2, 1] [1, 1] : tensor<4x3xf32> to tensor<2x1xf32>
      %zc6 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z7 = tensor_ext.assign_layout %zc6 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p8 = tensor.insert_slice %g5 into %z7[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<2x2xf32>
      // rhs task p=2 (q=0,j=0): B[0:2, 0:2]
      %g9 = tensor.extract_slice %b[0, 0] [2, 2] [1, 1] : tensor<4x3xf32> to tensor<2x2xf32>
      // rhs task p=3 (q=0,j=1): B[0:2, 2:3]
      %g10 = tensor.extract_slice %b[0, 2] [2, 1] [1, 1] : tensor<4x3xf32> to tensor<2x1xf32>
      %zc11 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z12 = tensor_ext.assign_layout %zc11 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p13 = tensor.insert_slice %g10 into %z12[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<2x2xf32>
      %rhsQ0_0 = tensor.insert_slice %g4 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ0_1 = tensor.insert_slice %p8 into %rhsQ0_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ0_2 = tensor.insert_slice %g9 into %rhsQ0_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ0_3 = tensor.insert_slice %p13 into %rhsQ0_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %product0 = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #mtp4
        } ins(%lhsQ0_3, %rhsQ0_3 : tensor<4x2x2xf32>, tensor<4x2x2xf32>)
          outs(%mtpZero : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>

      // ---- q=1 ----
      // lhs task p=0 (i=0,q=1): A[0:2, 2:4]
      %g14 = tensor.extract_slice %a[0, 2] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      // lhs task p=1 (i=0,q=1): A[0:2, 2:4]
      %g15 = tensor.extract_slice %a[0, 2] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      // lhs task p=2 (i=1,q=1): A[2:4, 2:4]
      %g16 = tensor.extract_slice %a[2, 2] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      // lhs task p=3 (i=1,q=1): A[2:4, 2:4]
      %g17 = tensor.extract_slice %a[2, 2] [2, 2] [1, 1] : tensor<4x4xf32> to tensor<2x2xf32>
      %lhsQ1_0 = tensor.insert_slice %g14 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ1_1 = tensor.insert_slice %g15 into %lhsQ1_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ1_2 = tensor.insert_slice %g16 into %lhsQ1_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ1_3 = tensor.insert_slice %g17 into %lhsQ1_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      // rhs task p=0 (q=1,j=0): B[2:4, 0:2]
      %g18 = tensor.extract_slice %b[2, 0] [2, 2] [1, 1] : tensor<4x3xf32> to tensor<2x2xf32>
      // rhs task p=1 (q=1,j=1): B[2:4, 2:3]
      %g19 = tensor.extract_slice %b[2, 2] [2, 1] [1, 1] : tensor<4x3xf32> to tensor<2x1xf32>
      %zc20 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z21 = tensor_ext.assign_layout %zc20 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p22 = tensor.insert_slice %g19 into %z21[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<2x2xf32>
      // rhs task p=2 (q=1,j=0): B[2:4, 0:2]
      %g23 = tensor.extract_slice %b[2, 0] [2, 2] [1, 1] : tensor<4x3xf32> to tensor<2x2xf32>
      // rhs task p=3 (q=1,j=1): B[2:4, 2:3]
      %g24 = tensor.extract_slice %b[2, 2] [2, 1] [1, 1] : tensor<4x3xf32> to tensor<2x1xf32>
      %zc25 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z26 = tensor_ext.assign_layout %zc25 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p27 = tensor.insert_slice %g24 into %z26[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<2x2xf32>
      %rhsQ1_0 = tensor.insert_slice %g18 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ1_1 = tensor.insert_slice %p22 into %rhsQ1_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ1_2 = tensor.insert_slice %g23 into %rhsQ1_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ1_3 = tensor.insert_slice %p27 into %rhsQ1_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %product1 = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #mtp4
        } ins(%lhsQ1_3, %rhsQ1_3 : tensor<4x2x2xf32>, tensor<4x2x2xf32>)
          outs(%product0 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>

      // ---- scatter ----
      %rTile0 = tensor.extract_slice %product1[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf32> to tensor<2x2xf32>
      %out0 = tensor.insert_slice %rTile0 into %outInit[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x3xf32>
      %rTile1 = tensor.extract_slice %product1[1, 0, 0] [1, 2, 1] [1, 1, 1] : tensor<4x2x2xf32> to tensor<2x1xf32>
      %out1 = tensor.insert_slice %rTile1 into %out0[0, 2] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<4x3xf32>
      %rTile2 = tensor.extract_slice %product1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf32> to tensor<2x2xf32>
      %out2 = tensor.insert_slice %rTile2 into %out1[2, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<4x3xf32>
      %rTile3 = tensor.extract_slice %product1[3, 0, 0] [1, 2, 1] [1, 1, 1] : tensor<4x2x2xf32> to tensor<2x1xf32>
      %out3 = tensor.insert_slice %rTile3 into %out2[2, 2] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<4x3xf32>
      secret.yield %out3 : tensor<4x3xf32>
    } -> (!secret.secret<tensor<4x3xf32>>)
    return %result : !secret.secret<tensor<4x3xf32>>
  }
}
