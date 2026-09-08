// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// M=6, K=2, N=2 with mu=2: computeGemmTilePlan reports I=3, Q=1, J=1 (no
// contraction reduction, no boundary tiles -- M, K, N are all exact
// multiples of mu), but taskCount=3 does not fit in a single ciphertext at
// tilesPerCiphertext capacity floor(8/(2*2))=2: getBalancedMtpPacking packs
// this across numCiphertexts=2 physical ciphertext groups. Scattering
// destination tiles across more than one physical ciphertext group is
// deliberately out of scope for this P8.3 checkpoint (a single
// [taskCount, mu, mu] batch tensor addressed by one getMultiTileLayoutRelation
// call, as assembleMultiTaskBatch builds here, only ever spans one group) --
// tracked as later multi-group scheduling work. Must be left completely
// untouched, not partially or incorrectly rewritten.

// CHECK: func.func @multi_group
// CHECK-NOT: linalg.batch_matmul
// CHECK-NOT: tensor_ext
// CHECK: linalg.matmul
module {
  func.func @multi_group(%A: !secret.secret<tensor<6x2xf32>>, %B: !secret.secret<tensor<2x2xf32>>, %Cinit: !secret.secret<tensor<6x2xf32>>) -> (!secret.secret<tensor<6x2xf32>>) {
    %result = secret.generic(%A: !secret.secret<tensor<6x2xf32>>, %B: !secret.secret<tensor<2x2xf32>>, %Cinit: !secret.secret<tensor<6x2xf32>>) {
    ^body(%a: tensor<6x2xf32>, %b: tensor<2x2xf32>, %c: tensor<6x2xf32>):
      %r = linalg.matmul ins(%a, %b : tensor<6x2xf32>, tensor<2x2xf32>) outs(%c : tensor<6x2xf32>) -> tensor<6x2xf32>
      secret.yield %r : tensor<6x2xf32>
    } -> (!secret.secret<tensor<6x2xf32>>)
    return %result : !secret.secret<tensor<6x2xf32>>
  }
}
