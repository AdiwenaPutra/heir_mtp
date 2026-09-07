#include <chrono>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"  // from @googletest

// Generated headers (block clang-format from messing up order)
#include "tests/Examples/openfhe/ckks/mtp_jkls_batch_matmul/mtp_jkls_batch_matmul_auto_multi_ct_uneven_lib.h"

namespace mlir {
namespace heir {
namespace openfhe {

// Phase 7's required uneven-batch encrypted regression: batch=3, mu=2,
// min-slot-count=8 -> capacity=2, numCiphertexts=2, tilesPerCiphertext=2
// (balanced) -> physical [2,8], with the second ciphertext holding only one
// real tile plus one unused MTP tile position. Reuses tiles 0-2 of the
// established four-tile FourTilesAcrossTwoCiphertexts example (dropping
// tile 3) so the numeric data is directly traceable to that regression.
//
// Comparing every one of the 12 decrypted logical outputs against the
// plaintext oracle — including tile 2, which shares ciphertext 1 with the
// unused fourth tile position — is itself the contamination check: if the
// unused slot's content leaked into a real rotation/multiply within that
// ciphertext, tile 2's decrypted values would diverge from the oracle here.
TEST(MtpJklsBatchMatmulAutoOpenFHETest, ThreeTilesAcrossTwoCiphertextsUnevenAutoSelected) {
  using Clock = std::chrono::steady_clock;
  auto milliseconds = [](Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
  };

  auto setupStart = Clock::now();
  auto cryptoContext =
      mtp_auto_multi_ct_uneven__generate_crypto_context();
  auto keyPair = cryptoContext->KeyGen();
  auto publicKey = keyPair.publicKey;
  auto secretKey = keyPair.secretKey;
  cryptoContext = mtp_auto_multi_ct_uneven__configure_crypto_context(
      cryptoContext, secretKey);
  auto setupEnd = Clock::now();

  // Three logical 2x2 matrices, flattened in [batch, row, column] order.
  std::vector<float> lhs = {
      1.0, 2.0, 3.0, 4.0,  // tile 0
      2.0, 0.0, 1.0, 2.0,  // tile 1
      0.0, 1.0, 2.0, 3.0,  // tile 2 (alone in ciphertext 1)
  };
  std::vector<float> rhs = {
      5.0, 6.0, 7.0, 8.0,  // tile 0
      3.0, 1.0, 4.0, 2.0,  // tile 1
      1.0, 2.0, 3.0, 4.0,  // tile 2 (alone in ciphertext 1)
  };

  std::vector<float> expected(12, 0.0);
  for (size_t batch = 0; batch < 3; ++batch) {
    for (size_t row = 0; row < 2; ++row) {
      for (size_t column = 0; column < 2; ++column) {
        for (size_t reduction = 0; reduction < 2; ++reduction) {
          expected[batch * 4 + row * 2 + column] +=
              lhs[batch * 4 + row * 2 + reduction] *
              rhs[batch * 4 + reduction * 2 + column];
        }
      }
    }
  }

  auto encryptionStart = Clock::now();
  auto lhsEncrypted = mtp_auto_multi_ct_uneven__encrypt__arg0(
      cryptoContext, lhs, publicKey);
  auto rhsEncrypted = mtp_auto_multi_ct_uneven__encrypt__arg1(
      cryptoContext, rhs, publicKey);
  auto encryptionEnd = Clock::now();

  ASSERT_EQ(lhsEncrypted.size(), 2);
  ASSERT_EQ(rhsEncrypted.size(), 2);
  for (size_t i = 0; i < lhsEncrypted.size(); ++i) {
    std::cout << "LHS ciphertext " << i
              << " input level: " << lhsEncrypted[i]->GetLevel() << '\n';
    std::cout << "RHS ciphertext " << i
              << " input level: " << rhsEncrypted[i]->GetLevel() << '\n';
  }

  auto evaluationStart = Clock::now();
  auto resultEncrypted = mtp_auto_multi_ct_uneven(
      cryptoContext, lhsEncrypted, rhsEncrypted);
  auto evaluationEnd = Clock::now();

  ASSERT_EQ(resultEncrypted.size(), 2);
  for (size_t i = 0; i < resultEncrypted.size(); ++i) {
    std::cout << "Output ciphertext " << i
              << " level: " << resultEncrypted[i]->GetLevel() << '\n';
  }

  auto decryptionStart = Clock::now();
  auto result = mtp_auto_multi_ct_uneven__decrypt__result0(
      cryptoContext, resultEncrypted, secretKey);
  auto decryptionEnd = Clock::now();

  // The generated decrypt function unpacks according to the original
  // [3,2,2] logical shape, not the [2,8] physical capacity, so the unused
  // fourth tile position must not appear in the output at all.
  ASSERT_EQ(result.size(), expected.size());
  constexpr float kErrorThreshold = 1e-2;
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_NEAR(result[i], expected[i], kErrorThreshold)
        << "Mismatch at flattened output index " << i;
  }

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
}

}  // namespace openfhe
}  // namespace heir
}  // namespace mlir
