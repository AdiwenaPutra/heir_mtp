// RUN: heir-opt --tile-planning-mtp-jkls-gemm="enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8" %s | FileCheck %s

// P8.5 input-ABI integration test 6: a boundary-tile GEMM (whose M is not
// an exact multiple of mu, so the existing eligibility walk rejects it
// before this GEMM ever reaches ABI selection), a dynamic-shape (hence
// ineligible) GEMM, and a genuinely SINGLE-GROUP eligible GEMM (which DOES
// reach annotateMultiGroupInputAbi, but whose plan->numCiphertexts == 1
// must exclude it there) may none of them receive an automatic MTP
// input-ABI annotation -- even though the boundary-tile shape below
// (M=3,K=2,N=6) would otherwise plan as a genuine multi-group GEMM
// (I=2,J=3,taskCount=6,tilesPerCiphertext=2,numCiphertexts=3).

// CHECK: func.func @boundary_tile
// CHECK-SAME: %arg0: !secret.secret<tensor<3x2xf32>>
// CHECK-NOT: tensor_ext.layout
// CHECK: linalg.matmul
func.func @boundary_tile(%a: !secret.secret<tensor<3x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<3x6xf32>>) -> (!secret.secret<tensor<3x6xf32>>) {
  %result = secret.generic(%a: !secret.secret<tensor<3x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<3x6xf32>>) {
  ^body(%input0: tensor<3x2xf32>, %input1: tensor<2x6xf32>, %input2: tensor<3x6xf32>):
    %r = linalg.matmul ins(%input0, %input1 : tensor<3x2xf32>, tensor<2x6xf32>) outs(%input2 : tensor<3x6xf32>) -> tensor<3x6xf32>
    secret.yield %r : tensor<3x6xf32>
  } -> (!secret.secret<tensor<3x6xf32>>)
  return %result : !secret.secret<tensor<3x6xf32>>
}

// CHECK: func.func @dynamic_shape_ineligible
// CHECK-SAME: %arg0: !secret.secret<tensor<?x2xf32>>
// CHECK-NOT: tensor_ext.layout
// CHECK: linalg.matmul
func.func @dynamic_shape_ineligible(%a: !secret.secret<tensor<?x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<?x6xf32>>) -> (!secret.secret<tensor<?x6xf32>>) {
  %result = secret.generic(%a: !secret.secret<tensor<?x2xf32>>, %b: !secret.secret<tensor<2x6xf32>>, %c: !secret.secret<tensor<?x6xf32>>) {
  ^body(%input0: tensor<?x2xf32>, %input1: tensor<2x6xf32>, %input2: tensor<?x6xf32>):
    %r = linalg.matmul ins(%input0, %input1 : tensor<?x2xf32>, tensor<2x6xf32>) outs(%input2 : tensor<?x6xf32>) -> tensor<?x6xf32>
    secret.yield %r : tensor<?x6xf32>
  } -> (!secret.secret<tensor<?x6xf32>>)
  return %result : !secret.secret<tensor<?x6xf32>>
}

// A genuinely single-group GEMM (M=K=N=2,mu=2: taskCount=1,
// numCiphertexts=1) DOES reach annotateMultiGroupInputAbi via
// eligibleOps, unlike the two cases above -- only its own
// plan->numCiphertexts <= 1 check excludes it there.
// CHECK: func.func @single_group_eligible
// CHECK-SAME: %arg0: !secret.secret<tensor<2x2xf32>>
// CHECK-NOT: tensor_ext.layout
// CHECK: secret.generic
// CHECK: linalg.batch_matmul
func.func @single_group_eligible(%a: !secret.secret<tensor<2x2xf32>>, %b: !secret.secret<tensor<2x2xf32>>, %c: !secret.secret<tensor<2x2xf32>>) -> (!secret.secret<tensor<2x2xf32>>) {
  %result = secret.generic(%a: !secret.secret<tensor<2x2xf32>>, %b: !secret.secret<tensor<2x2xf32>>, %c: !secret.secret<tensor<2x2xf32>>) {
  ^body(%input0: tensor<2x2xf32>, %input1: tensor<2x2xf32>, %input2: tensor<2x2xf32>):
    %r = linalg.matmul ins(%input0, %input1 : tensor<2x2xf32>, tensor<2x2xf32>) outs(%input2 : tensor<2x2xf32>) -> tensor<2x2xf32>
    secret.yield %r : tensor<2x2xf32>
  } -> (!secret.secret<tensor<2x2xf32>>)
  return %result : !secret.secret<tensor<2x2xf32>>
}
