#ifndef LIB_TRANSFORMS_TILEPLANNINGMTPJKLS_TILEPLANNINGMTPJKLS_TD_
#define LIB_TRANSFORMS_TILEPLANNINGMTPJKLS_TILEPLANNINGMTPJKLS_TD_

include "mlir/Pass/PassBase.td"

def TilePlanningMtpJkls : Pass<"tile-planning-mtp-jkls-gemm"> {
  let summary = "Rewrite an eligible ordinary secret-secret linalg.matmul "
                "into an explicit MTP-JKLS tile schedule";
  let description = [{
  Phase 8 checkpoint P8.2 (B1 compatibility only): for an ordinary, otherwise
  unannotated secret-secret `linalg.matmul` (rank 2, `[M,K] x [K,N] -> [M,N]`,
  static shapes), computes the general outer-tiled GEMM plan
  (`computeGemmTilePlan`, `lib/Utils/Layout/TilePlanningMtpJkls.h`) for the
  given primitive tile size `mu`, and, only when that plan has exactly one
  task (the `M = K = N = mu`, `I = Q = J = 1` case), rewrites the op into
  standard `tensor`/`linalg`/`arith` operations that assemble `[1, mu, mu]`
  batch tensors for the lhs, rhs, and the matmul's own (possibly nonzero)
  `outs`/init operand, feed them through a `linalg.batch_matmul` carrying a
  forced `BatchMatmulMtpJkls` kernel and its required physical layout
  (`getMultiTileLayoutRelation`, the same constructor Phase 7's own
  automatic MTP-JKLS selection already uses), and replace the original op's
  result directly with the single resulting tile (no separate scatter step
  is needed when there is exactly one destination tile occupying the whole
  output).

  Any `linalg.matmul` whose plan has more than one task (the general B2/B3/B6
  case) is left completely untouched by this checkpoint -- no attribute is
  added, no IR is emitted -- and remains available for the existing,
  unmodified layout-propagation matmul paths. Extending this pass to those
  cases is future work (Phase 8 P8.3 onward), not part of P8.2.

  This pass runs standalone; it is not yet wired into the shared
  `--mlir-to-ckks` pipeline (`ArithmeticPipelineRegistration.cpp`), matching
  P8.2's scope as an isolated compatibility checkpoint.
  }];
  let options = [
    Option<
      "enableMtpJklsGemm",
      "enable-mtp-jkls-gemm",
      "bool",
      /*default=*/"false",
      "If true, rewrite an eligible unannotated secret-secret linalg.matmul "
      "into an explicit MTP-JKLS tile schedule. Independent of "
      "layout-propagation's own enable-mtp-jkls flag, which governs "
      "already-batch-shaped ops this pass does not touch."
    >,
    Option<
      "mtpJklsGemmTileSize",
      "mtp-jkls-gemm-tile-size",
      "int64_t",
      /*default=*/"0",
      "The primitive square tile size mu. A rank-2 [M,K]x[K,N] matmul has "
      "no dimension that uniquely determines mu, so it must be given "
      "explicitly. A value <= 0 makes this pass inert. Enabling the pass "
      "while leaving this <= 0 is a pass-time diagnostic, not a silent "
      "no-op, since a misconfiguration must not be mistaken for \"no "
      "eligible ops found\"."
    >,
    Option<
      "minSlotCount",
      "min-slot-count",
      "int64_t",
      /*default=*/"1024",
      "Minimum number of slots per ciphertext, used identically to "
      "layout-propagation's own option of the same name to compute the "
      "physical MTP layout for the batch tensors this pass constructs."
    >
  ];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::secret::SecretDialect",
    "mlir::heir::tensor_ext::TensorExtDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

#endif  // LIB_TRANSFORMS_TILEPLANNINGMTPJKLS_TILEPLANNINGMTPJKLS_TD_
