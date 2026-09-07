#include <chrono>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/mtp_jkls_batch_matmul/mtp_jkls_batch_matmul_auto_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

// Phase 7: automatic-selection counterpart of
// MtpJklsBatchMatmulOpenFHETest.TwoPackedTiles in mtp_jkls_batch_matmul_test.cpp
// — same numeric example, but the source MLIR has no secret.kernel or
// tensor_ext.layout; MTP-JKLS is selected automatically.
TEST(MtpJklsBatchMatmulAutoOpenFHETest, TwoPackedTilesAutoSelected) {
  using Clock = std::chrono::steady_clock;
  auto milliseconds = [](Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  };

  auto setupStart = Clock::now();
  auto cryptoContext = mtp_auto_one_ct__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext =
      mtp_auto_one_ct__configure_crypto_context(cryptoContext, secretKey);
  auto setupEnd = Clock::now();

  // Two logical 2x2 matrices, flattened in [batch, row, column] order.
  std::vector<float> lhs = {
      1.0, 2.0, 3.0, 4.0,
      2.0, 0.0, 1.0, 2.0,
  };
  std::vector<float> rhs = {
      5.0, 6.0, 7.0, 8.0,
      3.0, 1.0, 4.0, 2.0,
  };

  auto encryptionStart = Clock::now();
  auto lhsEncrypted =
      mtp_auto_one_ct__encrypt__arg0(cryptoContext, lhs, publicKey);
  auto rhsEncrypted =
      mtp_auto_one_ct__encrypt__arg1(cryptoContext, rhs, publicKey);
  auto encryptionEnd = Clock::now();

  ASSERT_FALSE(lhsEncrypted.empty());
  ASSERT_FALSE(rhsEncrypted.empty());
  std::cout << "LHS input ciphertext level: "
            << lhsEncrypted.front()->GetLevel() << '\n';
  std::cout << "RHS input ciphertext level: "
            << rhsEncrypted.front()->GetLevel() << '\n';

  auto evaluationStart = Clock::now();
  auto resultEncrypted =
      mtp_auto_one_ct(cryptoContext, lhsEncrypted, rhsEncrypted);
  auto evaluationEnd = Clock::now();

  ASSERT_FALSE(resultEncrypted.empty());
  std::cout << "Output ciphertext level: "
            << resultEncrypted.front()->GetLevel() << '\n';

  auto decryptionStart = Clock::now();
  auto result = mtp_auto_one_ct__decrypt__result0(cryptoContext,
                                                   resultEncrypted, secretKey);
  auto decryptionEnd = Clock::now();

  std::cout << "Context and key setup: "
            << milliseconds(setupEnd - setupStart) << " ms\n";
  std::cout << "Encryption: "
            << milliseconds(encryptionEnd - encryptionStart) << " ms\n";
  std::cout << "Homomorphic evaluation: "
            << milliseconds(evaluationEnd - evaluationStart) << " ms\n";
  std::cout << "Decryption and unpacking: "
            << milliseconds(decryptionEnd - decryptionStart) << " ms\n";
  std::cout << "Measured total: "
            << milliseconds(decryptionEnd - setupStart) << " ms\n";

  // Expected products:
  //
  //   [1 2] [5 6] = [19 22]      [2 0] [3 1] = [ 6 2]
  //   [3 4] [7 8]   [43 50]      [1 2] [4 2]   [11 5]
  const std::vector<float> expected = {
      19.0, 22.0, 43.0, 50.0,
      6.0,  2.0,  11.0, 5.0,
  };

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
