#include <vector>

#include "gtest/gtest.h"  // from @googletest

#include "tests/Examples/openfhe/ckks/mtp_p85_input_abi/q1_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

// Real encrypted execution of the P8.5 input-ABI Q1 capacity-boundary
// fixture: A automatically packed into one ciphertext, B and Cinit into
// two, entirely from tile planning's own automatic ABI selection (this
// source has no explicit tensor_ext.layout or secret.kernel at all).
// Asymmetric, signed values so row/column swaps, tile swaps, and
// cross-group contamination would all be visible as wrong numbers.
TEST(MtpP85InputAbiQ1Test, ExactCinitPlusAB) {
  auto cryptoContext = q1_boundary__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      q1_boundary__configure_crypto_context(cryptoContext, secretKey);

  // A: 2x2, row-major.
  std::vector<float> a = {3.0, -7.0, 11.0, -2.0};
  // B: 2x6, row-major.
  std::vector<float> b = {5.0, -6.0, 13.0, -1.0, 8.0, -9.0,
                          -4.0, 17.0, 2.0, -15.0, 6.0, -3.0};
  // Cinit: 2x6, row-major.
  std::vector<float> cinit = {100.0, -200.0, 30.0,  -40.0, 500.0, -600.0,
                              -10.0, 20.0,   -300.0, 400.0, -5.0,  6.0};

  auto aEnc = q1_boundary__encrypt__arg0(cryptoContext, a, publicKey);
  auto bEnc = q1_boundary__encrypt__arg1(cryptoContext, b, publicKey);
  auto cinitEnc = q1_boundary__encrypt__arg2(cryptoContext, cinit, publicKey);

  auto resultEnc = q1_boundary(cryptoContext, aEnc, bEnc, cinitEnc);
  auto result =
      q1_boundary__decrypt__result0(cryptoContext, resultEnc, secretKey);

  // Expected: Cinit + A @ B, 2x6 row-major.
  const std::vector<float> expected = {143.0, -337.0, 55.0,  62.0, 482.0, -606.0,
                                       53.0,  -80.0,  -161.0, 419.0, 71.0,  -87.0};

  ASSERT_EQ(result.size(), expected.size());
  constexpr float kErrorThreshold = 1e-2;
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], kErrorThreshold)
        << "Mismatch at flattened output index " << i;
  }
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
