#include <vector>

#include "gtest/gtest.h"  // from @googletest

#include "tests/Examples/openfhe/ckks/mtp_p85_input_abi/q2_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

// Real encrypted execution of the P8.5 input-ABI Q2 multi-group fixture:
// A automatically packed into one ciphertext, B into three, Cinit into
// two, entirely from tile planning's own automatic ABI selection. Four
// kernel calls (two contraction steps times two destination groups) run
// under the hood. Asymmetric, signed values throughout.
TEST(MtpP85InputAbiQ2Test, ExactCinitPlusAB) {
  auto cryptoContext = q2_multi_group__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      q2_multi_group__configure_crypto_context(cryptoContext, secretKey);

  // A: 2x4, row-major.
  std::vector<float> a = {3.0, -7.0, 2.0, -5.0, -9.0, 4.0, 6.0, -1.0};
  // B: 4x6, row-major.
  std::vector<float> b = {5.0, -6.0, 13.0, -1.0, 8.0, -9.0,
                          -4.0, 17.0, 2.0, -15.0, 6.0, -3.0,
                          9.0, -1.0, -8.0, 3.0, -12.0, 14.0,
                          -6.0, 5.0, 1.0, -2.0, 7.0, -10.0};
  // Cinit: 2x6, row-major.
  std::vector<float> cinit = {100.0, -200.0, 30.0,  -40.0, 500.0, -600.0,
                              -10.0, 20.0,   -300.0, 400.0, -5.0,  6.0};

  auto aEnc = q2_multi_group__encrypt__arg0(cryptoContext, a, publicKey);
  auto bEnc = q2_multi_group__encrypt__arg1(cryptoContext, b, publicKey);
  auto cinitEnc =
      q2_multi_group__encrypt__arg2(cryptoContext, cinit, publicKey);

  auto resultEnc = q2_multi_group(cryptoContext, aEnc, bEnc, cinitEnc);
  auto result =
      q2_multi_group__decrypt__result0(cryptoContext, resultEnc, secretKey);

  // Expected: Cinit + A @ B, 2x6 row-major (computed offline in plain
  // Python and cross-checked independently of this test).
  const std::vector<float> expected = {191.0, -364.0, 34.0,  78.0,  423.0, -528.0,
                                       -11.0,  131.0, -458.0, 369.0, -132.0, 169.0};

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
