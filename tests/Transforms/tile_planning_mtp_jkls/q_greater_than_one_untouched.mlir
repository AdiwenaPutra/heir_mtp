// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// UPDATED for P8.4b (previously this file's M=4,K=4,N=2 shape was the
// negative case for "any combined Q>1 && taskCount>1 stays untouched"
// under P8.4 -- P8.4b now implements exactly that shape (see
// p84b_row_oriented_two_destinations.mlir, the positive structural test
// for this identical fixture), so that old claim became false and this
// file was repurposed again rather than left stale.
//
// M=6, K=4, N=2 with mu=2: computeGemmTilePlan reports I=3, Q=2, J=1 --
// taskCount=3 destinations, each needing a two-contraction-tile reduction
// (in scope for P8.4b's Q>1 && taskCount>1 integration), no boundary tile,
// but taskCount=3 does not fit in a single ciphertext at capacity
// floor(8/(2*2))=2: getBalancedMtpPacking packs this across
// numCiphertexts=2 physical ciphertext groups. Scattering destination
// tiles (or per-q task batches) across more than one physical ciphertext
// group remains out of scope -- P8.4b is restricted to "exactly one
// ciphertext group" exactly like every earlier checkpoint; multi-group
// support is P8.5, still not started. Must be left completely untouched.

// CHECK: func.func @multi_group_combined
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @multi_group_combined(%A: !secret.secret<tensor<6x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<6x2xf32>>) -> (!secret.secret<tensor<6x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<6x4xf32>>, %B: !secret.secret<tensor<4x2xf32>>, %Cinit: !secret.secret<tensor<6x2xf32>>) {
    ^body(%a: tensor<6x4xf32>, %b: tensor<4x2xf32>, %c: tensor<6x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<6x4xf32>, tensor<4x2xf32>) outs(%c : tensor<6x2xf32>) -> tensor<6x2xf32>
      secret.yield %r : tensor<6x2xf32>
    } -> (!secret.secret<tensor<6x2xf32>>)
    return %result : !secret.secret<tensor<6x2xf32>>
  }
}
