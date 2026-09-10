load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

# Real encrypted execution of the P8.5 input-ABI integration: unlike
# mtp_jkls_batch_matmul's sources (which hand-annotate tensor_ext.layout
# and secret.kernel directly), q1.mlir and q2.mlir are plain, unannotated
# secret-secret linalg.matmul ops. --tile-planning-mtp-jkls-gemm must
# itself select the MTP-JKLS kernel/layout AND automatically annotate the
# raw A/B/Cinit function arguments with the Convention-S MTP layout the
# per-group implementation needs, entirely from tile planning's own
# eligibility/plan logic -- exactly the P8.5 input-ABI checkpoint's
# fixture-only experiment, now running for real, encrypted, end to end.

openfhe_end_to_end_test(
    name = "q1_test",
    generated_lib_header = "q1_lib.h",
    heir_opt_flags = [
        "--tile-planning-mtp-jkls-gemm=enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8",
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=min-slot-count=8",
        "--scheme-to-openfhe=entry-function=q1_boundary",
    ],
    mlir_src = "q1.mlir",
    size = "medium",
    test_src = "q1_test.cpp",
)

openfhe_end_to_end_test(
    name = "q2_test",
    generated_lib_header = "q2_lib.h",
    heir_opt_flags = [
        "--tile-planning-mtp-jkls-gemm=enable-mtp-jkls-gemm=true mtp-jkls-gemm-tile-size=2 min-slot-count=8",
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=min-slot-count=8",
        "--scheme-to-openfhe=entry-function=q2_multi_group",
    ],
    mlir_src = "q2.mlir",
    size = "medium",
    test_src = "q2_test.cpp",
)
