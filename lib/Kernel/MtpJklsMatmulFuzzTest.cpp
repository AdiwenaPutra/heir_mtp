#include <cstdint>
#include <tuple>
#include <vector>

#include "gtest/gtest.h"  // from @googletest
#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/EvalVisitor.h"
#include "lib/Kernel/KernelImplementation.h"

// copybara hack: avoid reordering include
#include "fuzztest/fuzztest.h"  // from @fuzztest

namespace mlir {
namespace heir {
namespace kernel {
namespace {

using FuzzArgs =
    std::tuple<int64_t, int64_t, int64_t, std::vector<int>, std::vector<int>>;

// Converts tile-major matrices [p][d][l] into the MTP physical order
// d -> p -> l. Slots beyond the occupied prefix are filled with sentinels so
// the test also verifies that the kernel clears excess ciphertext capacity.
std::vector<int> packMtp(const std::vector<int>& matrices, int64_t tileSize,
                         int64_t tilesPerCiphertext, int64_t excessSlots) {
  int64_t occupiedSlots = tileSize * tileSize * tilesPerCiphertext;
  std::vector<int> packed(occupiedSlots + excessSlots, 31337);

  for (int64_t p = 0; p < tilesPerCiphertext; ++p) {
    for (int64_t d = 0; d < tileSize; ++d) {
      for (int64_t l = 0; l < tileSize; ++l) {
        int64_t matrixIndex = (p * tileSize + d) * tileSize + l;
        int64_t slot = d * (tilesPerCiphertext * tileSize) + p * tileSize + l;
        packed[slot] = matrices[matrixIndex];
      }
    }
  }

  return packed;
}

std::vector<int> naiveTilewiseMatmul(const std::vector<int>& matricesA,
                                     const std::vector<int>& matricesB,
                                     int64_t tileSize,
                                     int64_t tilesPerCiphertext) {
  std::vector<int> result(tileSize * tileSize * tilesPerCiphertext, 0);

  for (int64_t p = 0; p < tilesPerCiphertext; ++p) {
    for (int64_t row = 0; row < tileSize; ++row) {
      for (int64_t column = 0; column < tileSize; ++column) {
        int value = 0;
        for (int64_t k = 0; k < tileSize; ++k) {
          int64_t lhsIndex = (p * tileSize + row) * tileSize + k;
          int64_t rhsIndex = (p * tileSize + k) * tileSize + column;
          value += matricesA[lhsIndex] * matricesB[rhsIndex];
        }
        result[(p * tileSize + row) * tileSize + column] = value;
      }
    }
  }

  return result;
}

void mtpJklsMatmulMatchesNaive(const FuzzArgs& args) {
  const auto& [tileSize, tilesPerCiphertext, excessSlots, matricesA,
               matricesB] = args;

  std::vector<int> packedA =
      packMtp(matricesA, tileSize, tilesPerCiphertext, excessSlots);
  std::vector<int> packedB =
      packMtp(matricesB, tileSize, tilesPerCiphertext, excessSlots);
  int64_t numSlots = static_cast<int64_t>(packedA.size());

  LiteralValue packedAValue(packedA);
  LiteralValue packedBValue(packedB);
  auto dag = implementMtpJklsMatmul(
      packedAValue, packedBValue, tileSize, tilesPerCiphertext,
      DagType::intTensor(32, {numSlots}));

  LiteralValue evaluated = evalKernel(dag)[0];
  const auto& actualPacked = std::get<std::vector<int>>(evaluated.get());

  std::vector<int> expectedMatrices = naiveTilewiseMatmul(
      matricesA, matricesB, tileSize, tilesPerCiphertext);
  std::vector<int> expectedPacked =
      packMtp(expectedMatrices, tileSize, tilesPerCiphertext, excessSlots);
  for (int64_t slot = tileSize * tileSize * tilesPerCiphertext;
       slot < numSlots; ++slot) {
    expectedPacked[slot] = 0;
  }

  EXPECT_EQ(actualPacked, expectedPacked);
}

auto mtpShapeAndMatrices() {
  return fuzztest::FlatMap(
      [](int64_t tileSize, int64_t tilesPerCiphertext,
         int64_t excessSlots) {
        int64_t matrixElements =
            tileSize * tileSize * tilesPerCiphertext;
        return fuzztest::TupleOf(
            fuzztest::Just(tileSize), fuzztest::Just(tilesPerCiphertext),
            fuzztest::Just(excessSlots),
            fuzztest::VectorOf(fuzztest::InRange(-10, 10))
                .WithSize(matrixElements),
            fuzztest::VectorOf(fuzztest::InRange(-10, 10))
                .WithSize(matrixElements));
      },
      /*tileSize=*/fuzztest::InRange<int64_t>(1, 4),
      /*tilesPerCiphertext=*/fuzztest::InRange<int64_t>(1, 4),
      /*excessSlots=*/fuzztest::InRange<int64_t>(0, 8));
}

TEST(MtpJklsMatmulFuzzTest, SignedMultipleTilesWithExcessCapacityRegression) {
  mtpJklsMatmulMatchesNaive(
      {/*tileSize=*/2,
       /*tilesPerCiphertext=*/2,
       /*excessSlots=*/3,
       /*matricesA=*/{1, -2, 3, 4, -1, 5, 2, -3},
       /*matricesB=*/{5, 6, -7, 8, 4, -2, 1, 3}});
}

FUZZ_TEST(MtpJklsMatmulFuzzTest, mtpJklsMatmulMatchesNaive)
    .WithDomains(mtpShapeAndMatrices());

}  // namespace
}  // namespace kernel
}  // namespace heir
}  // namespace mlir
