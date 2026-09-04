load("@heir//tests/Examples/openfhe:test.bzl", "openfhe_end_to_end_test")

package(default_applicable_licenses = ["@heir//:license"])

openfhe_end_to_end_test(
    name = "mtp_jkls_batch_matmul_test",
    generated_lib_header = "mtp_jkls_batch_matmul_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=min-slot-count=8",
        "--scheme-to-openfhe=entry-function=mtp_jkls_batch_matmul",
    ],
    mlir_src = "mtp_jkls_batch_matmul.mlir",
    size = "medium",
    test_src = "mtp_jkls_batch_matmul_test.cpp",
)

openfhe_end_to_end_test(
    name = "mtp_jkls_batch_matmul_multi_ct_test",
    generated_lib_header = "mtp_jkls_batch_matmul_multi_ct_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=min-slot-count=8",
        "--scheme-to-openfhe=entry-function=mtp_jkls_batch_matmul_multi_ct",
    ],
    mlir_src = "mtp_jkls_batch_matmul_multi_ct.mlir",
    size = "medium",
    test_src = "mtp_jkls_batch_matmul_multi_ct_test.cpp",
)
