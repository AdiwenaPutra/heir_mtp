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

# Automatic-selection counterparts of the two forced-path tests above, plus
# the required uneven-batch case. None of these three sources contain a
# secret.kernel or tensor_ext.layout — enable-mtp-jkls=true makes
# LayoutPropagation select and construct the MTP-JKLS kernel/layout on its
# own.

openfhe_end_to_end_test(
    name = "mtp_jkls_batch_matmul_auto_test",
    generated_lib_header = "mtp_jkls_batch_matmul_auto_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=min-slot-count=8 enable-mtp-jkls=true",
        "--scheme-to-openfhe=entry-function=mtp_auto_one_ct",
    ],
    mlir_src = "mtp_jkls_batch_matmul_auto.mlir",
    size = "medium",
    test_src = "mtp_jkls_batch_matmul_auto_test.cpp",
)

openfhe_end_to_end_test(
    name = "mtp_jkls_batch_matmul_auto_multi_ct_even_test",
    generated_lib_header = "mtp_jkls_batch_matmul_auto_multi_ct_even_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=min-slot-count=8 enable-mtp-jkls=true",
        "--scheme-to-openfhe=entry-function=mtp_auto_multi_ct_even",
    ],
    mlir_src = "mtp_jkls_batch_matmul_auto_multi_ct_even.mlir",
    size = "medium",
    test_src = "mtp_jkls_batch_matmul_auto_multi_ct_even_test.cpp",
)

openfhe_end_to_end_test(
    name = "mtp_jkls_batch_matmul_auto_multi_ct_uneven_test",
    generated_lib_header = "mtp_jkls_batch_matmul_auto_multi_ct_uneven_lib.h",
    heir_opt_flags = [
        "--annotate-module=backend=openfhe scheme=ckks",
        "--mlir-to-ckks=min-slot-count=8 enable-mtp-jkls=true",
        "--scheme-to-openfhe=entry-function=mtp_auto_multi_ct_uneven",
    ],
    mlir_src = "mtp_jkls_batch_matmul_auto_multi_ct_uneven.mlir",
    size = "medium",
    test_src = "mtp_jkls_batch_matmul_auto_multi_ct_uneven_test.cpp",
)
