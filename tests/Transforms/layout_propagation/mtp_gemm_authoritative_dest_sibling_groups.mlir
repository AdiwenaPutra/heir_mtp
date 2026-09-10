// RUN: heir-opt --layout-propagation=min-slot-count=4 --mlir-print-local-scope %s | FileCheck %s

// Sibling counterpart to mtp_gemm_authoritative_dest_group1_normalization:
// two independent standalone one-ciphertext sources (each locally indexed
// ct = 0) are inserted into the SAME authoritative destination's group 0
// and group 1 respectively. Both must be recognized as already compatible
// after normalization -- neither needs a conversion, and a mutant that
// only fixes normalization for one destination group (e.g. hardcoding a
// shift of exactly -1, or always treating a sibling as if it targets
// group 0) would be caught by the per-insert offset checks below.
#dest = #tensor_ext.layout<"{ [p, l, d] -> [ct, slot] : ct = p and slot - 2 * l - d = 0 and 0 <= p <= 1 and 0 <= l <= 1 and 0 <= d <= 1 and 0 <= slot <= 3 }">
#source = #tensor_ext.layout<"{ [i0, i1] -> [ct, slot] : ct = 0 and slot = 2i0 + i1 and 0 <= i0 <= 1 and 0 <= i1 <= 1 }">

module {
  // CHECK-LABEL: @sibling_group0_group1
  func.func @sibling_group0_group1(%a: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #source}, %b: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #source}) -> (!secret.secret<tensor<2x2x2xf32>>) {
    %zero = arith.constant dense<0.0> : tensor<2x2x2xf32>
    %result = secret.generic(%a: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #source}, %b: !secret.secret<tensor<2x2xf32>> {tensor_ext.layout = #source}) {
    ^body(%sa: tensor<2x2xf32>, %sb: tensor<2x2xf32>):
      %authDest = tensor_ext.assign_layout %zero {layout = #dest, tensor_ext.layout = #dest} : tensor<2x2x2xf32>
      // group 0's insert: no conversion needed even before the fix, since
      // group 0's own required ct is already 0 with no shift involved.
      // CHECK-NOT: tensor_ext.convert_layout
      // CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[0, 0, 0]
      %ins0 = tensor.insert_slice %sa into %authDest[0, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<2x2x2xf32>
      // group 1's insert: THIS is the one a mutant that fixes only group 0
      // (or omits normalization for any non-zero-offset group) would break.
      // CHECK-NOT: tensor_ext.convert_layout
      // CHECK: tensor.insert_slice %{{.*}} into %{{.*}}[1, 0, 0]
      %ins1 = tensor.insert_slice %sb into %ins0[1, 0, 0] [1, 2, 2] [1, 1, 1] : tensor<2x2xf32> into tensor<2x2x2xf32>
      secret.yield %ins1 : tensor<2x2x2xf32>
    } -> (!secret.secret<tensor<2x2x2xf32>>)
    return %result : !secret.secret<tensor<2x2x2xf32>>
  }
}
