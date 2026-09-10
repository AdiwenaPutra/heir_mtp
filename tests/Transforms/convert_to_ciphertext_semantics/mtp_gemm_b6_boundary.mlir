// RUN: heir-opt --layout-propagation=min-slot-count=32 --convert-to-ciphertext-semantics=min-slot-count=32 %s | FileCheck %s

// Acceptance test for Phase 8's B6 fixture: the mandatory rectangular,
// non-divisible-boundary GEMM (mu=2, A:3x5, B:5x4, C:3x4 -> I=2, Q=3, J=2,
// taskCount=4), exercising both a partial row tile (i=1, valid 1 of mu=2
// rows) and a partial contraction tile (q=2, valid 1 of mu=2). Every
// boundary gather is zero-padded via its own explicitly assigned
// (authoritative) tile layout before being scattered into the shared
// taskCount-wide batch, and the destination-authority mechanism
// (LayoutPropagation.cpp's authoritativeDestinations) must reconcile every
// one of the resulting partial/full inserts without crashing, without
// silently relaxing addressing, and without an always-false ("1 = 0")
// tensor_ext.remap permutation discarding a boundary contribution -- a
// class of bug that previously escaped detection because it manifests as a
// silent numerical error, not a compile failure. Full numerical
// verification against a plaintext GEMM oracle (M=3,K=5,N=4, non-divisible
// in both M and K) requires an out-of-tree ISL-based interpreter, not
// FileCheck, and is not part of this test; this test is the permanent
// structural regression that must keep passing on its own.

// A pass-synthesized always-false relation may print either inline or
// hoisted into a named #layoutN alias ahead of the function (this test
// omits --mlir-print-local-scope, matching this directory's convention), so
// these two CHECK-NOT lines deliberately have no other FileCheck directive
// between them and their bounding match, covering both the alias preamble
// and the function body without a gap. The plain-string form "1 = 0" (used
// elsewhere in this directory) false-positives here: this fixture's
// Convention-S boundary relations legitimately contain both "i1 = 0"
// (a domain variable pinned to a single value at a size-1 boundary) and
// "mod 31 = 0" (an ordinary modulus), both of which contain "1 = 0" as a
// substring. The regex below requires a non-alphanumeric character
// immediately before "1 = 0", which still matches a genuine standalone
// always-false constraint (printed as "... : 1 = 0 ...", preceded by a
// space or punctuation) while excluding both false positives.
// CHECK-NOT: {{[^0-9A-Za-z]1 = 0}}
// CHECK: func.func @mtp_gemm_b6
// CHECK-NOT: {{[^0-9A-Za-z]1 = 0}}

// Convention S: l (tensor axis 1) is the local matrix row, d (tensor axis 2)
// is the local matrix column, slot = d*(tilesPerCiphertext*mu) + p*mu + l.
#mtp4 = #tensor_ext.layout<"{ [p, l, d] -> [ct, slot] : ct = 0 and slot - 8*d - 2*p - l = 0 and 0 <= p <= 3 and 0 <= l <= 1 and 0 <= d <= 1 and 0 <= slot <= 31 }">
#rowmajorC = #tensor_ext.layout<"{ [i, j] -> [ct, slot] : ct = 0 and slot - 4*i - j = 0 and 0 <= i <= 2 and 0 <= j <= 3 and 0 <= slot <= 31 }">
#tileRowMajor = #tensor_ext.layout<"{ [d, l] -> [ct, slot] : ct = 0 and slot - 2*d - l = 0 and 0 <= d <= 1 and 0 <= l <= 1 and 0 <= slot <= 31 }">
#kernel = #secret.kernel<name = "BatchMatmulMtpJkls", force = true>

module {
  func.func @mtp_gemm_b6(
      %A: !secret.secret<tensor<3x5xf32>>,
      %B: !secret.secret<tensor<5x4xf32>>)
      -> (!secret.secret<tensor<3x4xf32>>) {
    %zero4 = arith.constant dense<0.0> : tensor<4x2x2xf32>
    %outZero = arith.constant dense<0.0> : tensor<3x4xf32>
    %result = secret.generic(
        %A: !secret.secret<tensor<3x5xf32>>,
        %B: !secret.secret<tensor<5x4xf32>>) {
    ^body(%a: tensor<3x5xf32>, %b: tensor<5x4xf32>):
      %mtpZero = tensor_ext.assign_layout %zero4 {layout = #mtp4, tensor_ext.layout = #mtp4} : tensor<4x2x2xf32>
      %outInit = tensor_ext.assign_layout %outZero {layout = #rowmajorC, tensor_ext.layout = #rowmajorC} : tensor<3x4xf32>

      // ---- q=0 ----
      // lhs task p=0 (i=0,q=0): A[0:2, 0:2]
      %g0 = tensor.extract_slice %a[0, 0] [2, 2] [1, 1] : tensor<3x5xf32> to tensor<2x2xf32>
      // lhs task p=1 (i=0,q=0): A[0:2, 0:2]
      %g1 = tensor.extract_slice %a[0, 0] [2, 2] [1, 1] : tensor<3x5xf32> to tensor<2x2xf32>
      // lhs task p=2 (i=1,q=0): A[2:3, 0:2]
      %g2 = tensor.extract_slice %a[2, 0] [1, 2] [1, 1] : tensor<3x5xf32> to tensor<1x2xf32>
      %zc3 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z4 = tensor_ext.assign_layout %zc3 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p5 = tensor.insert_slice %g2 into %z4[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      // lhs task p=3 (i=1,q=0): A[2:3, 0:2]
      %g6 = tensor.extract_slice %a[2, 0] [1, 2] [1, 1] : tensor<3x5xf32> to tensor<1x2xf32>
      %zc7 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z8 = tensor_ext.assign_layout %zc7 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p9 = tensor.insert_slice %g6 into %z8[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      %lhsQ0_0 = tensor.insert_slice %g0 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ0_1 = tensor.insert_slice %g1 into %lhsQ0_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ0_2 = tensor.insert_slice %p5 into %lhsQ0_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ0_3 = tensor.insert_slice %p9 into %lhsQ0_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      // rhs task p=0 (q=0,j=0): B[0:2, 0:2]
      %g10 = tensor.extract_slice %b[0, 0] [2, 2] [1, 1] : tensor<5x4xf32> to tensor<2x2xf32>
      // rhs task p=1 (q=0,j=1): B[0:2, 2:4]
      %g11 = tensor.extract_slice %b[0, 2] [2, 2] [1, 1] : tensor<5x4xf32> to tensor<2x2xf32>
      // rhs task p=2 (q=0,j=0): B[0:2, 0:2]
      %g12 = tensor.extract_slice %b[0, 0] [2, 2] [1, 1] : tensor<5x4xf32> to tensor<2x2xf32>
      // rhs task p=3 (q=0,j=1): B[0:2, 2:4]
      %g13 = tensor.extract_slice %b[0, 2] [2, 2] [1, 1] : tensor<5x4xf32> to tensor<2x2xf32>
      %rhsQ0_0 = tensor.insert_slice %g10 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ0_1 = tensor.insert_slice %g11 into %rhsQ0_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ0_2 = tensor.insert_slice %g12 into %rhsQ0_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ0_3 = tensor.insert_slice %g13 into %rhsQ0_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %product0 = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #mtp4
        } ins(%lhsQ0_3, %rhsQ0_3 : tensor<4x2x2xf32>, tensor<4x2x2xf32>)
          outs(%mtpZero : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>

      // ---- q=1 ----
      // lhs task p=0 (i=0,q=1): A[0:2, 2:4]
      %g14 = tensor.extract_slice %a[0, 2] [2, 2] [1, 1] : tensor<3x5xf32> to tensor<2x2xf32>
      // lhs task p=1 (i=0,q=1): A[0:2, 2:4]
      %g15 = tensor.extract_slice %a[0, 2] [2, 2] [1, 1] : tensor<3x5xf32> to tensor<2x2xf32>
      // lhs task p=2 (i=1,q=1): A[2:3, 2:4]
      %g16 = tensor.extract_slice %a[2, 2] [1, 2] [1, 1] : tensor<3x5xf32> to tensor<1x2xf32>
      %zc17 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z18 = tensor_ext.assign_layout %zc17 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p19 = tensor.insert_slice %g16 into %z18[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      // lhs task p=3 (i=1,q=1): A[2:3, 2:4]
      %g20 = tensor.extract_slice %a[2, 2] [1, 2] [1, 1] : tensor<3x5xf32> to tensor<1x2xf32>
      %zc21 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z22 = tensor_ext.assign_layout %zc21 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p23 = tensor.insert_slice %g20 into %z22[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      %lhsQ1_0 = tensor.insert_slice %g14 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ1_1 = tensor.insert_slice %g15 into %lhsQ1_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ1_2 = tensor.insert_slice %p19 into %lhsQ1_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ1_3 = tensor.insert_slice %p23 into %lhsQ1_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      // rhs task p=0 (q=1,j=0): B[2:4, 0:2]
      %g24 = tensor.extract_slice %b[2, 0] [2, 2] [1, 1] : tensor<5x4xf32> to tensor<2x2xf32>
      // rhs task p=1 (q=1,j=1): B[2:4, 2:4]
      %g25 = tensor.extract_slice %b[2, 2] [2, 2] [1, 1] : tensor<5x4xf32> to tensor<2x2xf32>
      // rhs task p=2 (q=1,j=0): B[2:4, 0:2]
      %g26 = tensor.extract_slice %b[2, 0] [2, 2] [1, 1] : tensor<5x4xf32> to tensor<2x2xf32>
      // rhs task p=3 (q=1,j=1): B[2:4, 2:4]
      %g27 = tensor.extract_slice %b[2, 2] [2, 2] [1, 1] : tensor<5x4xf32> to tensor<2x2xf32>
      %rhsQ1_0 = tensor.insert_slice %g24 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ1_1 = tensor.insert_slice %g25 into %rhsQ1_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ1_2 = tensor.insert_slice %g26 into %rhsQ1_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ1_3 = tensor.insert_slice %g27 into %rhsQ1_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %product1 = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #mtp4
        } ins(%lhsQ1_3, %rhsQ1_3 : tensor<4x2x2xf32>, tensor<4x2x2xf32>)
          outs(%product0 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>

      // ---- q=2 ----
      // lhs task p=0 (i=0,q=2): A[0:2, 4:5]
      %g28 = tensor.extract_slice %a[0, 4] [2, 1] [1, 1] : tensor<3x5xf32> to tensor<2x1xf32>
      %zc29 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z30 = tensor_ext.assign_layout %zc29 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p31 = tensor.insert_slice %g28 into %z30[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<2x2xf32>
      // lhs task p=1 (i=0,q=2): A[0:2, 4:5]
      %g32 = tensor.extract_slice %a[0, 4] [2, 1] [1, 1] : tensor<3x5xf32> to tensor<2x1xf32>
      %zc33 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z34 = tensor_ext.assign_layout %zc33 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p35 = tensor.insert_slice %g32 into %z34[0, 0] [2, 1] [1, 1] : tensor<2x1xf32> into tensor<2x2xf32>
      // lhs task p=2 (i=1,q=2): A[2:3, 4:5]
      %g36 = tensor.extract_slice %a[2, 4] [1, 1] [1, 1] : tensor<3x5xf32> to tensor<1x1xf32>
      %zc37 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z38 = tensor_ext.assign_layout %zc37 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p39 = tensor.insert_slice %g36 into %z38[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<2x2xf32>
      // lhs task p=3 (i=1,q=2): A[2:3, 4:5]
      %g40 = tensor.extract_slice %a[2, 4] [1, 1] [1, 1] : tensor<3x5xf32> to tensor<1x1xf32>
      %zc41 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z42 = tensor_ext.assign_layout %zc41 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p43 = tensor.insert_slice %g40 into %z42[0, 0] [1, 1] [1, 1] : tensor<1x1xf32> into tensor<2x2xf32>
      %lhsQ2_0 = tensor.insert_slice %p31 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ2_1 = tensor.insert_slice %p35 into %lhsQ2_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ2_2 = tensor.insert_slice %p39 into %lhsQ2_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %lhsQ2_3 = tensor.insert_slice %p43 into %lhsQ2_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      // rhs task p=0 (q=2,j=0): B[4:5, 0:2]
      %g44 = tensor.extract_slice %b[4, 0] [1, 2] [1, 1] : tensor<5x4xf32> to tensor<1x2xf32>
      %zc45 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z46 = tensor_ext.assign_layout %zc45 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p47 = tensor.insert_slice %g44 into %z46[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      // rhs task p=1 (q=2,j=1): B[4:5, 2:4]
      %g48 = tensor.extract_slice %b[4, 2] [1, 2] [1, 1] : tensor<5x4xf32> to tensor<1x2xf32>
      %zc49 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z50 = tensor_ext.assign_layout %zc49 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p51 = tensor.insert_slice %g48 into %z50[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      // rhs task p=2 (q=2,j=0): B[4:5, 0:2]
      %g52 = tensor.extract_slice %b[4, 0] [1, 2] [1, 1] : tensor<5x4xf32> to tensor<1x2xf32>
      %zc53 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z54 = tensor_ext.assign_layout %zc53 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p55 = tensor.insert_slice %g52 into %z54[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      // rhs task p=3 (q=2,j=1): B[4:5, 2:4]
      %g56 = tensor.extract_slice %b[4, 2] [1, 2] [1, 1] : tensor<5x4xf32> to tensor<1x2xf32>
      %zc57 = arith.constant dense<0.0> : tensor<2x2xf32>
      %z58 = tensor_ext.assign_layout %zc57 {layout = #tileRowMajor, tensor_ext.layout = #tileRowMajor} : tensor<2x2xf32>
      %p59 = tensor.insert_slice %g56 into %z58[0, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<2x2xf32>
      %rhsQ2_0 = tensor.insert_slice %p47 into %mtpZero[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ2_1 = tensor.insert_slice %p51 into %rhsQ2_0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ2_2 = tensor.insert_slice %p55 into %rhsQ2_1[2, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %rhsQ2_3 = tensor.insert_slice %p59 into %rhsQ2_2[3, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<4x2x2xf32>
      %product2 = linalg.batch_matmul {
          secret.kernel = #kernel,
          tensor_ext.layout = #mtp4
        } ins(%lhsQ2_3, %rhsQ2_3 : tensor<4x2x2xf32>, tensor<4x2x2xf32>)
          outs(%product1 : tensor<4x2x2xf32>) -> tensor<4x2x2xf32>

      // ---- scatter ----
      %rTile0 = tensor.extract_slice %product2[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf32> to tensor<2x2xf32>
      %out0 = tensor.insert_slice %rTile0 into %outInit[0, 0] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<3x4xf32>
      %rTile1 = tensor.extract_slice %product2[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<4x2x2xf32> to tensor<2x2xf32>
      %out1 = tensor.insert_slice %rTile1 into %out0[0, 2] [2, 2] [1, 1] : tensor<2x2xf32> into tensor<3x4xf32>
      %rTile2 = tensor.extract_slice %product2[2, 0, 0] [1, 1, 2] [1, 1, 1] : tensor<4x2x2xf32> to tensor<1x2xf32>
      %out2 = tensor.insert_slice %rTile2 into %out1[2, 0] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<3x4xf32>
      %rTile3 = tensor.extract_slice %product2[3, 0, 0] [1, 1, 2] [1, 1, 1] : tensor<4x2x2xf32> to tensor<1x2xf32>
      %out3 = tensor.insert_slice %rTile3 into %out2[2, 2] [1, 2] [1, 1] : tensor<1x2xf32> into tensor<3x4xf32>
      secret.yield %out3 : tensor<3x4xf32>
    } -> (!secret.secret<tensor<3x4xf32>>)
    return %result : !secret.secret<tensor<3x4xf32>>
  }
}
