#ifndef LIB_KERNEL_KERNELIMPLEMENTATION_H_
#define LIB_KERNEL_KERNELIMPLEMENTATION_H_

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

#include "lib/Kernel/AbstractValue.h"
#include "lib/Kernel/ArithmeticDag.h"
#include "lib/Kernel/KernelName.h"
#include "lib/Utils/APIntUtils.h"
#include "lib/Utils/MathUtils.h"
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project

namespace mlir {
namespace heir {
namespace kernel {

// A function that generalizes the reduction operation in all kernels in this
// file. E.g., whether to use `add` or `mul`
template <typename T>
using DagReducer = std::function<std::shared_ptr<ArithmeticDagNode<T>>(
    std::shared_ptr<ArithmeticDagNode<T>>,
    std::shared_ptr<ArithmeticDagNode<T>>)>;

template <typename T>
using DagExtractor = std::function<std::shared_ptr<ArithmeticDagNode<T>>(
    std::shared_ptr<ArithmeticDagNode<T>>, int64_t)>;

// Dynamic extraction: takes tensor and DAG node representing runtime index
template <typename T>
using DagExtractorDynamic = std::function<std::shared_ptr<ArithmeticDagNode<T>>(
    std::shared_ptr<ArithmeticDagNode<T>>,
    std::shared_ptr<ArithmeticDagNode<T>>)>;

struct MtpPackedShape {
  int64_t numCiphertexts;
  int64_t numSlots;
};

// Standalone kernel tests use [slots], while ciphertext-semantics conversion
// represents a ciphertext tensor as [ciphertexts, slots]. In both cases,
// ciphertext rotations act independently on the final slot dimension.
inline MtpPackedShape getMtpPackedShape(const DagType& baseType) {
  const auto& shape = baseType.getShape();
  assert((shape.size() == 1 || shape.size() == 2) &&
         "MTP expects [slots] or [ciphertexts, slots]");
  int64_t numCiphertexts = shape.size() == 1 ? 1 : shape[0];
  int64_t numSlots = shape.back();
  assert(numCiphertexts > 0 &&
         "MTP requires a positive ciphertext count");
  assert(numSlots > 0 && "MTP requires a positive slot count");
  return {numCiphertexts, numSlots};
}

inline std::vector<double> repeatMtpSlotMask(
    const std::vector<double>& slotMask, int64_t numCiphertexts) {
  std::vector<double> repeated;
  repeated.reserve(numCiphertexts * slotMask.size());
  for (int64_t ct = 0; ct < numCiphertexts; ++ct) {
    repeated.insert(repeated.end(), slotMask.begin(), slotMask.end());
  }
  return repeated;
}

// Applies the JKLS Phi_k permutation independently to every MTP tile:
//
//   Phi_k(X)[l,d,p] = X[(l+k) mod tileRows,d,p]
//
// The MTP physical slot organization is:
//
//   slot = d * (tilesPerCiphertext * tileRows)
//        + p * tileRows
//        + l
//
// Phi_k changes only the local row coordinate l. Therefore, its rotation
// offsets and masks depend on tileRows, while tileColumns determines how many
// occupied tile segments are present.
//
// Two rotations and repeated plaintext masks prevent values from crossing
// primitive-tile boundaries.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMtpPhi(const T& packed, int64_t tileRows, int64_t tileColumns,
                int64_t tilesPerCiphertext, int64_t shift,
                const DagType& baseType) {
  using NodeTy = ArithmeticDagNode<T>;

  assert(tileRows > 0 && "MTP tile row count must be positive");
  assert(tileColumns > 0 && "MTP tile column count must be positive");
  assert(tilesPerCiphertext > 0 &&
         "MTP tiles per ciphertext must be positive");
  assert(packed.getShape() == baseType.getShape() &&
         "packed value and DAG type must have the same shape");

  auto [numCiphertexts, numSlots] = getMtpPackedShape(baseType);

  // Validate:
  //
  //   occupiedSlots =
  //       tileRows * tileColumns * tilesPerCiphertext
  //
  // without overflowing signed int64_t.
  assert(tileRows <= numSlots / tileColumns &&
         "one MTP tile does not fit in the available slots");
  int64_t slotsPerTile = tileRows * tileColumns;

  assert(tilesPerCiphertext <= numSlots / slotsPerTile &&
         "MTP tiles do not fit in the available slots");
  int64_t occupiedSlots = tilesPerCiphertext * slotsPerTile;

  // Phi shifts the local row coordinate, so normalize modulo tileRows.
  int64_t k = ((shift % tileRows) + tileRows) % tileRows;

  auto input = NodeTy::leaf(packed);

  // Phi_0 is the identity and requires no masks or rotations.
  if (k == 0) return input;

  // These are public plaintext masks. Slots outside the occupied MTP prefix
  // remain zero in both masks.
  std::vector<double> nonWrappingMask(numSlots, 0.0);
  std::vector<double> wrappingMask(numSlots, 0.0);

  for (int64_t slot = 0; slot < occupiedSlots; ++slot) {
    // Under the D -> p -> L physical order, l is innermost.
    int64_t l = slot % tileRows;

    if (l < tileRows - k) {
      nonWrappingMask[slot] = 1.0;
    } else {
      wrappingMask[slot] = 1.0;
    }
  }

  auto nonWrappingMaskDag = NodeTy::constantTensor(
      repeatMtpSlotMask(nonWrappingMask, numCiphertexts), baseType);
  auto wrappingMaskDag = NodeTy::constantTensor(
      repeatMtpSlotMask(wrappingMask, numCiphertexts), baseType);

  // Non-wrapping output positions:
  //
  //   output[l,d,p] = input[l+k,d,p]
  auto nonWrappingRotation = NodeTy::leftRotate(input, k);
  auto nonWrappingPart =
      NodeTy::mul(nonWrappingRotation, nonWrappingMaskDag);

  // Wrapping output positions:
  //
  //   output[l,d,p] = input[l+k-tileRows,d,p]
  auto wrappingRotation =
      NodeTy::leftRotate(input, k - tileRows);
  auto wrappingPart =
      NodeTy::mul(wrappingRotation, wrappingMaskDag);

  return NodeTy::add(nonWrappingPart, wrappingPart);
}

// Applies the JKLS Psi_k permutation independently to every MTP tile:
//
//   Psi_k(X)[l,d,p] = X[l,(d+k) mod tileColumns,p]
//
// Under the MTP physical layout, adjacent values of d are separated by
// tilesPerCiphertext * tileRows slots. Two rotations and masks implement the
// cyclic shift without wrapping values between the first and last d regions.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMtpPsi(const T& packed, int64_t tileRows, int64_t tileColumns,
                int64_t tilesPerCiphertext, int64_t shift,
                const DagType& baseType) {
  using NodeTy = ArithmeticDagNode<T>;

  assert(tileRows > 0 && "MTP tile row count must be positive");
  assert(tileColumns > 0 && "MTP tile column count must be positive");
  assert(tilesPerCiphertext > 0 &&
         "MTP tiles per ciphertext must be positive");
  assert(packed.getShape() == baseType.getShape() &&
         "packed value and DAG type must have the same shape");

  auto [numCiphertexts, numSlots] = getMtpPackedShape(baseType);

  // Validate tileRows * tileColumns * tilesPerCiphertext <= numSlots
  // without overflowing signed int64_t.
  assert(tileRows <= numSlots / tileColumns &&
         "one MTP tile does not fit in the available slots");
  int64_t slotsPerTile = tileRows * tileColumns;

  assert(tilesPerCiphertext <= numSlots / slotsPerTile &&
         "MTP tiles do not fit in the available slots");
  int64_t occupiedSlots = tilesPerCiphertext * slotsPerTile;

  // Psi shifts the local column coordinate, so normalize modulo tileColumns.
  int64_t k = ((shift % tileColumns) + tileColumns) % tileColumns;

  auto input = NodeTy::leaf(packed);
  if (k == 0) return input;

  int64_t columnStride = tilesPerCiphertext * tileRows;
  std::vector<double> nonWrappingMask(numSlots, 0.0);
  std::vector<double> wrappingMask(numSlots, 0.0);

  for (int64_t slot = 0; slot < occupiedSlots; ++slot) {
    // Each contiguous region of columnStride slots has one d coordinate.
    int64_t d = slot / columnStride;

    if (d < tileColumns - k) {
      nonWrappingMask[slot] = 1.0;
    } else {
      wrappingMask[slot] = 1.0;
    }
  }

  auto nonWrappingMaskDag = NodeTy::constantTensor(
      repeatMtpSlotMask(nonWrappingMask, numCiphertexts), baseType);
  auto wrappingMaskDag = NodeTy::constantTensor(
      repeatMtpSlotMask(wrappingMask, numCiphertexts), baseType);

  // Non-wrapping output positions read from d+k.
  auto nonWrappingRotation =
      NodeTy::leftRotate(input, k * columnStride);
  auto nonWrappingPart =
      NodeTy::mul(nonWrappingRotation, nonWrappingMaskDag);

  // Wrapping output positions read from d+k-tileColumns.
  auto wrappingRotation =
      NodeTy::leftRotate(input, (k - tileColumns) * columnStride);
  auto wrappingPart = NodeTy::mul(wrappingRotation, wrappingMaskDag);

  return NodeTy::add(nonWrappingPart, wrappingPart);
}

// Builds an MTP slot permutation from a map that gives the source slot for
// every occupied output slot. Output slots requiring the same cyclic rotation
// are grouped behind one repeated public mask. Slots outside occupiedSlots are
// zeroed by construction.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMtpSlotPermutation(
    std::shared_ptr<ArithmeticDagNode<T>> input, int64_t occupiedSlots,
    const DagType& baseType,
    const std::function<int64_t(int64_t)>& sourceSlotForOutput) {
  using NodeTy = ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;

  assert(input && "MTP permutation requires a non-null input DAG");

  auto [numCiphertexts, numSlots] = getMtpPackedShape(baseType);
  assert(occupiedSlots > 0 && occupiedSlots <= numSlots &&
         "occupied MTP slots must fit in the packed tensor");

  // A left rotation by r makes output slot s read input slot (s+r) mod N.
  // Group output positions by that required rotation so every distinct shift
  // is performed only once.
  std::map<int64_t, std::vector<double>> masksByShift;
  for (int64_t outputSlot = 0; outputSlot < occupiedSlots; ++outputSlot) {
    int64_t sourceSlot = sourceSlotForOutput(outputSlot);
    assert(sourceSlot >= 0 && sourceSlot < occupiedSlots &&
           "MTP permutation source must be an occupied slot");

    int64_t shift = sourceSlot - outputSlot;
    shift = ((shift % numSlots) + numSlots) % numSlots;
    auto it = masksByShift
                  .try_emplace(shift,
                               std::vector<double>(numSlots, 0.0))
                  .first;
    it->second[outputSlot] = 1.0;
  }

  NodePtr result;
  for (auto& [shift, mask] : masksByShift) {
    auto maskDag = NodeTy::constantTensor(
        repeatMtpSlotMask(mask, numCiphertexts), baseType);
    auto shifted = shift == 0 ? input : NodeTy::leftRotate(input, shift);
    auto selected = NodeTy::mul(shifted, maskDag);
    result = result ? NodeTy::add(result, selected) : selected;
  }

  assert(result && "an MTP permutation must have at least one output slot");
  return result;
}

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMtpSlotPermutation(
    const T& packed, int64_t occupiedSlots, const DagType& baseType,
    const std::function<int64_t(int64_t)>& sourceSlotForOutput) {
  assert(packed.getShape() == baseType.getShape() &&
         "packed value and DAG type must have the same shape");
  return implementMtpSlotPermutation<T>(
      ArithmeticDagNode<T>::leaf(packed), occupiedSlots, baseType,
      sourceSlotForOutput);
}

// Applies Phi_k to an existing arithmetic DAG. This node-level form is used
// when composing Phi_k after sigma inside the complete JKLS kernel.
template <typename T>
std::shared_ptr<ArithmeticDagNode<T>> implementMtpPhiOnDag(
    std::shared_ptr<ArithmeticDagNode<T>> input, int64_t tileRows,
    int64_t tileColumns, int64_t tilesPerCiphertext, int64_t shift,
    const DagType& baseType) {
  assert(tileRows > 0 && "MTP tile row count must be positive");
  assert(tileColumns > 0 && "MTP tile column count must be positive");
  assert(tilesPerCiphertext > 0 &&
         "MTP tiles per ciphertext must be positive");
  assert(input && "MTP Phi requires a non-null input DAG");

  int64_t numSlots = getMtpPackedShape(baseType).numSlots;
  assert(tileRows <= numSlots / tileColumns &&
         "one MTP tile does not fit in the available slots");
  int64_t slotsPerTile = tileRows * tileColumns;
  assert(tilesPerCiphertext <= numSlots / slotsPerTile &&
         "MTP tiles do not fit in the available slots");
  int64_t occupiedSlots = tilesPerCiphertext * slotsPerTile;
  int64_t columnStride = tilesPerCiphertext * tileRows;
  int64_t k = ((shift % tileRows) + tileRows) % tileRows;
  if (k == 0) return input;

  return implementMtpSlotPermutation<T>(
      input, occupiedSlots, baseType, [&](int64_t outputSlot) {
        int64_t d = outputSlot / columnStride;
        int64_t withinColumn = outputSlot % columnStride;
        int64_t p = withinColumn / tileRows;
        int64_t l = withinColumn % tileRows;
        int64_t sourceL = (l + k) % tileRows;
        return d * columnStride + p * tileRows + sourceL;
      });
}

// Applies Psi_k to an existing arithmetic DAG. This node-level form is used
// when composing Psi_k after tau inside the complete JKLS kernel.
template <typename T>
std::shared_ptr<ArithmeticDagNode<T>> implementMtpPsiOnDag(
    std::shared_ptr<ArithmeticDagNode<T>> input, int64_t tileRows,
    int64_t tileColumns, int64_t tilesPerCiphertext, int64_t shift,
    const DagType& baseType) {
  assert(tileRows > 0 && "MTP tile row count must be positive");
  assert(tileColumns > 0 && "MTP tile column count must be positive");
  assert(tilesPerCiphertext > 0 &&
         "MTP tiles per ciphertext must be positive");
  assert(input && "MTP Psi requires a non-null input DAG");

  int64_t numSlots = getMtpPackedShape(baseType).numSlots;
  assert(tileRows <= numSlots / tileColumns &&
         "one MTP tile does not fit in the available slots");
  int64_t slotsPerTile = tileRows * tileColumns;
  assert(tilesPerCiphertext <= numSlots / slotsPerTile &&
         "MTP tiles do not fit in the available slots");
  int64_t occupiedSlots = tilesPerCiphertext * slotsPerTile;
  int64_t columnStride = tilesPerCiphertext * tileRows;
  int64_t k = ((shift % tileColumns) + tileColumns) % tileColumns;
  if (k == 0) return input;

  return implementMtpSlotPermutation<T>(
      input, occupiedSlots, baseType, [&](int64_t outputSlot) {
        int64_t d = outputSlot / columnStride;
        int64_t withinColumn = outputSlot % columnStride;
        int64_t p = withinColumn / tileRows;
        int64_t l = withinColumn % tileRows;
        int64_t sourceD = (d + k) % tileColumns;
        return sourceD * columnStride + p * tileRows + l;
      });
}

// Applies the JKLS sigma permutation independently to every MTP tile:
//
//   sigma(X)[l,d,p] = X[(l+d) mod tileRows,d,p]
//
// The source displacement is independent of p, so the number of rotations
// does not grow with tilesPerCiphertext.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMtpSigma(const T& packed, int64_t tileRows, int64_t tileColumns,
                  int64_t tilesPerCiphertext, const DagType& baseType) {
  assert(tileRows > 0 && "MTP tile row count must be positive");
  assert(tileColumns > 0 && "MTP tile column count must be positive");
  assert(tilesPerCiphertext > 0 &&
         "MTP tiles per ciphertext must be positive");
  int64_t numSlots = getMtpPackedShape(baseType).numSlots;
  assert(tileRows <= numSlots / tileColumns &&
         "one MTP tile does not fit in the available slots");
  int64_t slotsPerTile = tileRows * tileColumns;
  assert(tilesPerCiphertext <= numSlots / slotsPerTile &&
         "MTP tiles do not fit in the available slots");
  int64_t occupiedSlots = tilesPerCiphertext * slotsPerTile;
  int64_t columnStride = tilesPerCiphertext * tileRows;

  return implementMtpSlotPermutation<T>(
      packed, occupiedSlots, baseType, [&](int64_t outputSlot) {
        int64_t d = outputSlot / columnStride;
        int64_t withinColumn = outputSlot % columnStride;
        int64_t p = withinColumn / tileRows;
        int64_t l = withinColumn % tileRows;
        int64_t sourceL = (l + d) % tileRows;
        return d * columnStride + p * tileRows + sourceL;
      });
}

// Applies the JKLS tau permutation independently to every MTP tile:
//
//   tau(X)[l,d,p] = X[l,(d+l) mod tileColumns,p]
//
// The source displacement is independent of p, so the number of rotations
// does not grow with tilesPerCiphertext. When the occupied MTP region fills
// the ciphertext exactly, cyclic ciphertext rotation also implements the
// wrap between the last and first d regions; square tiles then require only
// tileRows - 1 nonzero rotations.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMtpTau(const T& packed, int64_t tileRows, int64_t tileColumns,
                int64_t tilesPerCiphertext, const DagType& baseType) {
  assert(tileRows > 0 && "MTP tile row count must be positive");
  assert(tileColumns > 0 && "MTP tile column count must be positive");
  assert(tilesPerCiphertext > 0 &&
         "MTP tiles per ciphertext must be positive");
  int64_t numSlots = getMtpPackedShape(baseType).numSlots;
  assert(tileRows <= numSlots / tileColumns &&
         "one MTP tile does not fit in the available slots");
  int64_t slotsPerTile = tileRows * tileColumns;
  assert(tilesPerCiphertext <= numSlots / slotsPerTile &&
         "MTP tiles do not fit in the available slots");
  int64_t occupiedSlots = tilesPerCiphertext * slotsPerTile;
  int64_t columnStride = tilesPerCiphertext * tileRows;

  return implementMtpSlotPermutation<T>(
      packed, occupiedSlots, baseType, [&](int64_t outputSlot) {
        int64_t d = outputSlot / columnStride;
        int64_t withinColumn = outputSlot % columnStride;
        int64_t p = withinColumn / tileRows;
        int64_t l = withinColumn % tileRows;
        int64_t sourceD = (d + l) % tileColumns;
        return sourceD * columnStride + p * tileRows + l;
      });
}

// Transposes every MTP tile. The input tile has shape tileRows x tileColumns;
// the output is interpreted using the MTP layout for tileColumns x tileRows:
//
//   transpose(X)[l,d,p] = X[d,l,p]
//
// For square tiles the tile-position term cancels from every displacement, so
// the rotation count is independent of tilesPerCiphertext. Rectangular tiles
// are also supported, but changing the physical segment width makes some
// displacements depend on p.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMtpTranspose(const T& packed, int64_t tileRows,
                      int64_t tileColumns, int64_t tilesPerCiphertext,
                      const DagType& baseType) {
  assert(tileRows > 0 && "MTP tile row count must be positive");
  assert(tileColumns > 0 && "MTP tile column count must be positive");
  assert(tilesPerCiphertext > 0 &&
         "MTP tiles per ciphertext must be positive");
  int64_t numSlots = getMtpPackedShape(baseType).numSlots;
  assert(tileRows <= numSlots / tileColumns &&
         "one MTP tile does not fit in the available slots");
  int64_t slotsPerTile = tileRows * tileColumns;
  assert(tilesPerCiphertext <= numSlots / slotsPerTile &&
         "MTP tiles do not fit in the available slots");
  int64_t occupiedSlots = tilesPerCiphertext * slotsPerTile;

  int64_t outputTileRows = tileColumns;
  int64_t outputColumnStride = tilesPerCiphertext * outputTileRows;
  int64_t inputColumnStride = tilesPerCiphertext * tileRows;

  return implementMtpSlotPermutation<T>(
      packed, occupiedSlots, baseType, [&](int64_t outputSlot) {
        int64_t outputD = outputSlot / outputColumnStride;
        int64_t withinColumn = outputSlot % outputColumnStride;
        int64_t p = withinColumn / outputTileRows;
        int64_t outputL = withinColumn % outputTileRows;

        // Output coordinate (outputL, outputD) reads input coordinate
        // (outputD, outputL).
        int64_t sourceL = outputD;
        int64_t sourceD = outputL;
        return sourceD * inputColumnStride + p * tileRows + sourceL;
      });
}

// Implements the JKLS ciphertext-ciphertext matrix multiplication on every
// aligned square MTP tile pair:
//
//   C = sum_k Phi_k(sigma(A)) * Psi_k(tau(B)).
//
// Under the JKLS/MTP coordinate correspondence, d is the matrix row and l is
// the matrix column. The returned slots therefore encode C = A * B in the same
// d -> p -> l physical order. Relinearization and rescaling are intentionally
// left to HEIR's scheme-management passes.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMtpJklsMatmul(const T& packedA, const T& packedB,
                       int64_t tileSize, int64_t tilesPerCiphertext,
                       const DagType& baseType) {
  using NodeTy = ArithmeticDagNode<T>;

  assert(tileSize > 0 && "JKLS tile size must be positive");
  assert(tilesPerCiphertext > 0 &&
         "MTP tiles per ciphertext must be positive");
  assert(packedA.getShape() == baseType.getShape() &&
         "packed A and DAG type must have the same shape");
  assert(packedB.getShape() == baseType.getShape() &&
         "packed B and DAG type must have the same shape");

  int64_t numSlots = getMtpPackedShape(baseType).numSlots;
  assert(tileSize <= numSlots / tileSize &&
         "one square JKLS tile does not fit in the available slots");
  int64_t slotsPerTile = tileSize * tileSize;
  assert(tilesPerCiphertext <= numSlots / slotsPerTile &&
         "MTP tiles do not fit in the available slots");

  auto sigmaA = implementMtpSigma(packedA, tileSize, tileSize,
                                  tilesPerCiphertext, baseType);
  auto tauB = implementMtpTau(packedB, tileSize, tileSize,
                              tilesPerCiphertext, baseType);

  auto result = NodeTy::mul(sigmaA, tauB);
  for (int64_t k = 1; k < tileSize; ++k) {
    auto shiftedA = implementMtpPhiOnDag(
        sigmaA, tileSize, tileSize, tilesPerCiphertext, k, baseType);
    auto shiftedB = implementMtpPsiOnDag(
        tauB, tileSize, tileSize, tilesPerCiphertext, k, baseType);
    result = NodeTy::add(result, NodeTy::mul(shiftedA, shiftedB));
  }

  return result;
}

// Returns an arithmetic DAG that implements a matvec kernel. Ensure this is
// only generated for T a subclass of AbstractValue.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementMatvec(KernelName kernelName, const T& matrix, const T& vector) {
  using NodeTy = ArithmeticDagNode<T>;
  assert(kernelName == KernelName::MatvecDiagonal);
  auto matrixDag = NodeTy::leaf(matrix);
  auto vectorDag = NodeTy::leaf(vector);

  int numRows = matrix.getShape()[0];
  assert(numRows > 0);

  auto firstTerm = NodeTy::mul(NodeTy::leftRotate(vectorDag, 0),
                               NodeTy::extract(matrixDag, 0));

  auto accumulatedSum = firstTerm;
  for (int i = 1; i < numRows; ++i) {
    auto term = NodeTy::mul(NodeTy::leftRotate(vectorDag, i),
                            NodeTy::extract(matrixDag, i));
    accumulatedSum = NodeTy::add(accumulatedSum, term);
  }
  return accumulatedSum;
}

// Returns an arithmetic DAG that implements a logarithmic rotate-and-reduce
// accumulation of an input ciphertext.
//
// This is a special case of `tensor_ext.rotate_and_reduce`
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementRotateAndReduceAccumulation(
    std::shared_ptr<ArithmeticDagNode<T>> vectorDag, int64_t period,
    int64_t steps, DagReducer<T> reduceFunc) {
  using NodeTy = ArithmeticDagNode<T>;
  for (int64_t shiftSize = steps / 2; shiftSize > 0; shiftSize /= 2) {
    auto rotated = NodeTy::leftRotate(vectorDag, shiftSize * period);
    auto reduced = reduceFunc(vectorDag, rotated);
    vectorDag = reduced;
  }
  return vectorDag;
}

template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementRotateAndReduceAccumulation(const T& vector, int64_t period,
                                     int64_t steps, DagReducer<T> reduceFunc) {
  using NodeTy = ArithmeticDagNode<T>;
  return implementRotateAndReduceAccumulation<T>(NodeTy::leaf(vector), period,
                                                 steps, reduceFunc);
}

// Rolled version of implementRotateAndReduceAccumulation.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementRotateAndReduceAccumulationRolled(
    std::shared_ptr<ArithmeticDagNode<T>> vectorDag, int64_t period,
    int64_t steps, DagReducer<T> reduceFunc, const DagType& baseType) {
  using NodeTy = ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;

  int64_t numIterations = static_cast<int64_t>(std::log2(steps));
  if (numIterations <= 0) return vectorDag;

  auto initialShift = NodeTy::constantScalar(steps / 2, DagType::index());
  auto loopNode = NodeTy::loop(
      {vectorDag, initialShift}, {baseType, DagType::index()}, /*lower=*/0,
      /*upper=*/numIterations,
      /*step=*/1, [&](NodePtr i, const std::vector<NodePtr>& iterArgs) {
        auto currentVector = iterArgs[0];
        auto currentShift = iterArgs[1];

        auto rotated = NodeTy::leftRotate(
            currentVector,
            NodeTy::mul(currentShift,
                        NodeTy::constantScalar(period, DagType::index())));
        auto reduced = reduceFunc(currentVector, rotated);

        auto nextShift = NodeTy::floorDiv(currentShift, 2);
        return NodeTy::yield({reduced, nextShift});
      });

  return NodeTy::resultAt(loopNode, 0);
}

// Rolled version of implementRotateAndReduceAccumulation.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementRotateAndReduceAccumulationRolled(const T& vector, int64_t period,
                                           int64_t steps,
                                           DagReducer<T> reduceFunc,
                                           const DagType& baseType) {
  using NodeTy = ArithmeticDagNode<T>;
  return implementRotateAndReduceAccumulationRolled<T>(
      NodeTy::leaf(vector), period, steps, reduceFunc, baseType);
}

// A function that generalizes the choice of rotation for the "baby stepped
// operand" of a baby-step giant-step algorithm. This is required because
// the rotation used in Halevi-Shoup matvec differs from that of bicyclic
// matmul.
using DerivedRotationIndexFn = std::function<int64_t(
    // giant step size
    int64_t,
    // current giant step index
    int64_t,
    // current baby step index
    int64_t,
    // period
    int64_t)>;

inline int64_t defaultDerivedRotationIndexFn(int64_t giantStepSize,
                                             int64_t giantStepIndex,
                                             int64_t babyStepIndex,
                                             int64_t period) {
  return -giantStepSize * giantStepIndex * period;
}

// Dynamic version: builds DAG expression for rotation amount from DAG node
// indices
template <typename T>
using DagDerivedRotationIndexFn =
    std::function<std::shared_ptr<ArithmeticDagNode<T>>(
        // giant step size (constant)
        int64_t,
        // current giant step index (DAG node)
        std::shared_ptr<ArithmeticDagNode<T>>,
        // current baby step index (DAG node)
        std::shared_ptr<ArithmeticDagNode<T>>,
        // period (constant)
        int64_t)>;

template <typename T>
std::shared_ptr<ArithmeticDagNode<T>> defaultDagDerivedRotationIndexFn(
    int64_t giantStepSize, std::shared_ptr<ArithmeticDagNode<T>> giantStepIndex,
    std::shared_ptr<ArithmeticDagNode<T>> babyStepIndex, int64_t period) {
  using NodeTy = ArithmeticDagNode<T>;
  // Build: -(giantStepSize * giantStepIndex * period)
  auto gsSize = NodeTy::constantScalar(giantStepSize, DagType::index());
  auto periodNode = NodeTy::constantScalar(period, DagType::index());
  auto negOne = NodeTy::constantScalar(-1, DagType::index());

  auto temp = NodeTy::mul(giantStepIndex, gsSize);
  temp = NodeTy::mul(temp, periodNode);
  return NodeTy::mul(temp, negOne);
}

// Returns an arithmetic DAG that implements a baby-step-giant-step
// rotate-and-reduce accumulation between an input ciphertext
// (giantSteppedOperand) and an abstraction over the other argument
// (babySteppedOperand). In particular, the babySteppedOperand may be a list of
// plaintexts like in Halevi-Shoup matvec, or a single ciphertext like in
// bicyclic matmul, and this abstracts over both by taking in an extraction
// callback.
//
// This is a special case of `tensor_ext.rotate_and_reduce`, but with the added
// abstractions it also supports situations not currently expressible by
// `tensor_ext.rotate_and_reduce`.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementBabyStepGiantStep(
    const T& giantSteppedOperand, const T& babySteppedOperand, int64_t period,
    int64_t steps, DagType dagType, DagExtractor<T> extractFunc,
    const std::map<int, bool>& zeroDiagonals = {},
    const DerivedRotationIndexFn& derivedRotationIndexFn =
        defaultDerivedRotationIndexFn) {
  using NodeTy = ArithmeticDagNode<T>;
  auto giantSteppedDag = NodeTy::leaf(giantSteppedOperand);
  auto babySteppedDag = NodeTy::leaf(babySteppedOperand);

  // Use a value of sqrt(n) as the baby step / giant step size.
  int64_t numBabySteps = static_cast<int64_t>(std::ceil(std::sqrt(steps)));
  int64_t giantStepSize = numBabySteps;
  // numGiantSteps = ceil(steps / numBabySteps)
  int64_t numGiantSteps = (steps + numBabySteps - 1) / numBabySteps;

  // Compute sqrt(n) ciphertext rotations of the input as baby-steps.
  SmallVector<std::shared_ptr<NodeTy>> babyStepVals;
  babyStepVals.push_back(giantSteppedDag);  // rot by zero
  for (int64_t i = 1; i < numBabySteps; ++i) {
    babyStepVals.push_back(NodeTy::leftRotate(giantSteppedDag, period * i));
  }

  // Compute the inner baby step sums.
  std::shared_ptr<NodeTy> result = nullptr;
  for (int64_t j = 0; j < numGiantSteps; ++j) {
    std::shared_ptr<NodeTy> innerSum = nullptr;
    for (int64_t i = 0; i < numBabySteps; ++i) {
      if (j * giantStepSize + i >= steps) {
        break;
      }
      int64_t innerRotAmount =
          derivedRotationIndexFn(giantStepSize, j, i, period);
      size_t extractionIndex = i + j * giantStepSize;

      // Skip the multiplication if the extraction index is zero.
      if (zeroDiagonals.contains(extractionIndex)) {
        continue;
      }

      auto plaintext = extractFunc(babySteppedDag, extractionIndex);
      auto rotatedPlaintext = NodeTy::leftRotate(plaintext, innerRotAmount);
      auto multiplied = NodeTy::mul(rotatedPlaintext, babyStepVals[i]);
      innerSum =
          innerSum == nullptr ? multiplied : NodeTy::add(innerSum, multiplied);
    }

    // The innerSum may be nullptr if all the multiplications were skipped.
    auto rotatedSum =
        innerSum == nullptr
            ? nullptr
            : NodeTy::leftRotate(innerSum, period * j * giantStepSize);
    if (result == nullptr) {
      result = rotatedSum;
    } else {
      result = rotatedSum == nullptr ? result : NodeTy::add(result, rotatedSum);
    }
  }

  return result == nullptr ? NodeTy::splat(0, dagType) : result;
}

// Default dynamic extractor: simple extraction at runtime index
template <typename T>
std::shared_ptr<ArithmeticDagNode<T>> defaultDagExtractor(
    std::shared_ptr<ArithmeticDagNode<T>> tensor,
    std::shared_ptr<ArithmeticDagNode<T>> index) {
  return ArithmeticDagNode<T>::extract(tensor, index);
}

// Rolled version of Baby-Step-Giant-Step algorithm.
//
// TODO(#2704): support skipping over zero diagonals
//
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementBabyStepGiantStepRolled(
    const T& giantSteppedOperand, const T& babySteppedOperand, int64_t period,
    int64_t steps, const DagType baseType,
    DagExtractorDynamic<T> extractFunc = defaultDagExtractor<T>,
    const std::map<int, bool>& zeroDiagonals = {},
    const DagDerivedRotationIndexFn<T>& dagRotationFn =
        defaultDagDerivedRotationIndexFn<T>) {
  using NodeTy = ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;

  auto giantSteppedDag = NodeTy::leaf(giantSteppedOperand);
  auto babySteppedDag = NodeTy::leaf(babySteppedOperand);

  int64_t numBabySteps = static_cast<int64_t>(std::ceil(std::sqrt(steps)));
  int64_t giantStepSize = numBabySteps;
  int64_t numGiantSteps = (steps + numBabySteps - 1) / numBabySteps;

  // Create an empty tensor of shape {numBabySteps, numSlots} and emit a loop to
  // compute baby-step rotations.
  std::vector<int64_t> packedShape = {numBabySteps};
  if (!baseType.getShape().empty()) {
    packedShape.push_back(baseType.getShape().back());
  }
  auto emptyTensor = NodeTy::empty(packedShape, baseType);
  auto precomputeLoop = NodeTy::loop(
      {emptyTensor}, {baseType}, /*lower=*/0, /*upper=*/numBabySteps,
      /*step=*/1, [&](NodePtr i, const std::vector<NodePtr>& iterArgs) {
        auto currentTensor = iterArgs[0];
        auto rotAmount =
            NodeTy::mul(i, NodeTy::constantScalar(period, DagType::index()));
        auto rotated = NodeTy::leftRotate(giantSteppedDag, rotAmount);
        auto updatedTensor = NodeTy::insert(rotated, currentTensor, i);
        return NodeTy::yield({updatedTensor});
      });
  NodePtr packedBabySteps = NodeTy::resultAt(precomputeLoop, 0);

  // Initialize outer sum to zero and pass packedBabySteps as loop invariant
  std::vector<NodePtr> outerInits = {NodeTy::splat(0, baseType),
                                     packedBabySteps};
  std::vector<DagType> outerTypes = {baseType, baseType};

  NodePtr isNotZeroDiagTensor = nullptr;
  if (!zeroDiagonals.empty()) {
    std::vector<double> isNotZeroDiagVec(steps, 1.0);
    for (const auto& [idx, isZero] : zeroDiagonals) {
      if (isZero && idx >= 0 && idx < steps) {
        isNotZeroDiagVec[idx] = 0.0;
      }
    }
    isNotZeroDiagTensor = NodeTy::constantTensor(
        isNotZeroDiagVec, DagType::floatTensor(64, {steps}));
  }

  // Outer loop over giant steps (j = 0 to numGiantSteps)
  auto outerLoop = NodeTy::loop(
      outerInits, outerTypes, /*lower=*/0, /*upper=*/numGiantSteps, /*step=*/1,
      [&](NodePtr j, const std::vector<NodePtr>& outerIterArgs) {
        auto outerSum = outerIterArgs[0];
        // Inner loop over baby steps (i = 0 to numBabySteps)
        auto innerLoop = NodeTy::loop(
            {NodeTy::splat(0, baseType)}, {baseType}, /*lower=*/0,
            /*upper=*/numBabySteps, /*step=*/1,
            [&](NodePtr i, const std::vector<NodePtr>& innerIterArgs) {
              auto innerSum = innerIterArgs[0];
              auto gsSize =
                  NodeTy::constantScalar(giantStepSize, DagType::index());
              auto jOffset = NodeTy::mul(j, gsSize);
              auto extractIdx = NodeTy::add(i, jOffset);

              auto stepsNode = NodeTy::constantScalar(steps, DagType::index());
              auto isBound = NodeTy::comparison(extractIdx, stepsNode,
                                                ComparisonPredicate::LT);

              auto buildMultiplyAndAdd = [&]() {
                auto plaintext = extractFunc(babySteppedDag, extractIdx);
                auto innerRotAmount =
                    dagRotationFn(giantStepSize, j, i, period);

                auto rotatedPlaintext =
                    NodeTy::leftRotate(plaintext, innerRotAmount);

                auto babyStepVal = NodeTy::extract(outerIterArgs[1], i);
                auto multiplied = NodeTy::mul(rotatedPlaintext, babyStepVal);
                return NodeTy::yield({NodeTy::add(innerSum, multiplied)});
              };

              auto buildInnerSumYield = [&]() {
                return NodeTy::yield({innerSum});
              };

              NodePtr term;
              if (isNotZeroDiagTensor != nullptr) {
                auto isNotZeroDiag =
                    NodeTy::extract(isNotZeroDiagTensor, extractIdx);
                auto zero = NodeTy::constantScalar(0.0, DagType::floatTy(64));
                auto isNotZero = NodeTy::comparison(isNotZeroDiag, zero,
                                                    ComparisonPredicate::NE);
                term = NodeTy::ifElse(
                    isBound,
                    [&]() {
                      auto innerIf = NodeTy::ifElse(
                          isNotZero, buildMultiplyAndAdd, buildInnerSumYield);
                      return NodeTy::yield({NodeTy::resultAt(innerIf, 0)});
                    },
                    buildInnerSumYield);
              } else {
                term = NodeTy::ifElse(isBound, buildMultiplyAndAdd,
                                      buildInnerSumYield);
              }

              return NodeTy::yield({NodeTy::resultAt(term, 0)});
            });

        auto innerSum = NodeTy::resultAt(innerLoop, 0);

        // Rotate by j * giantStepSize * period
        auto gsSize = NodeTy::constantScalar(giantStepSize, DagType::index());
        auto periodNode = NodeTy::constantScalar(period, DagType::index());
        auto outerRotAmount = NodeTy::mul(j, gsSize);
        outerRotAmount = NodeTy::mul(outerRotAmount, periodNode);

        auto rotatedSum = NodeTy::leftRotate(innerSum, outerRotAmount);

        // Accumulate into outer sum
        auto newOuterSum = NodeTy::add(outerSum, rotatedSum);

        return NodeTy::yield({newOuterSum, outerIterArgs[1]});
      });

  return NodeTy::resultAt(outerLoop, 0);
}

// Returns an arithmetic DAG that implements a tensor_ext.rotate_and_reduce op.
//
// See TensorExtOps.td docs for RotateAndReduceOp for more details.
//
// The `vector` argument is a ciphertext value that will be rotated O(sqrt(n))
// times when the `plaintexts` argument is set (Baby Step Giant Step), or
// O(log(n)) times when the `plaintexts` argument is not set (log-style
// rotate-and-reduce accumulation).
//
// The `plaintexts` argument, when present, represents a vector of pre-packed
// plaintexts that will be rotated and multiplied with the rotated `vector`
// argument in BSGS style.
//
// Note that using this kernel results in places in the pipeline where a
// plaintext type is rotated, but most FHE implementations don't have a
// plaintext rotation operation (it would be wasteful) and instead expect the
// "plaintext rotation" to apply to the cleartext. HEIR has places in the
// pipeline that support this by converting a rotate(encode(cleartext)) to
// encode(rotate(cleartext)).
//
// Note: If this implementation is updated, it's likely that
// rotateAndReduceRotationIndices in RotationUtils.h needs updating, too
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementRotateAndReduce(const T& vector, std::optional<T> plaintexts,
                         int64_t period, int64_t steps, const DagType& dagType,
                         const std::map<int, bool>& zeroDiagonals = {},
                         const std::string& reduceOp = "arith.addi",
                         bool unroll = true) {
  using NodeTy = ArithmeticDagNode<T>;
  auto performReduction = [&](std::shared_ptr<NodeTy> left,
                              std::shared_ptr<NodeTy> right) {
    if (reduceOp == "arith.addi" || reduceOp == "arith.addf") {
      return NodeTy::add(left, right);
    }

    if (reduceOp == "arith.muli" || reduceOp == "arith.mulf") {
      return NodeTy::mul(left, right);
    }

    // Default to add for unknown operations
    return NodeTy::add(left, right);
  };

  if (!plaintexts.has_value()) {
    if (unroll) {
      return implementRotateAndReduceAccumulation<T>(vector, period, steps,
                                                     performReduction);
    }
    return implementRotateAndReduceAccumulationRolled<T>(
        vector, period, steps, performReduction, dagType);
  }

  assert(reduceOp == "arith.addi" ||
         reduceOp == "arith.addf" &&
             "Baby-step-giant-step rotate-and-reduce only supports addition "
             "as the reduction operation");

  if (unroll) {
    // Unrolled version: uses static extraction
    auto extractFunc = [](std::shared_ptr<NodeTy> babySteppedDag,
                          int64_t extractionIndex) {
      return NodeTy::extract(babySteppedDag, extractionIndex);
    };

    return implementBabyStepGiantStep<T>(
        vector, plaintexts.value(), period, steps, dagType, extractFunc,
        zeroDiagonals, defaultDerivedRotationIndexFn);
  }

  // Rolled version: uses dynamic extraction and DAG rotation function
  auto dynamicExtractFunc = [](std::shared_ptr<NodeTy> babySteppedDag,
                               std::shared_ptr<NodeTy> extractionIndex) {
    return NodeTy::extract(babySteppedDag, extractionIndex);
  };

  return implementBabyStepGiantStepRolled<T>(
      vector, plaintexts.value(), period, steps, dagType, dynamicExtractFunc,
      zeroDiagonals, defaultDagDerivedRotationIndexFn<T>);
}

// Returns an arithmetic DAG that implements a dot product kernel.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementDot(const T& lhs, const T& rhs, int64_t steps,
             const DagType& dagType) {
  using NodeTy = ArithmeticDagNode<T>;
  auto mulDag = NodeTy::mul(NodeTy::leaf(lhs), NodeTy::leaf(rhs));
  return implementRotateAndReduceAccumulation<T>(mulDag, /*period=*/1, steps,
                                                 NodeTy::add);
}

// Returns an arithmetic DAG that implements a broadcasted reduce kernel.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementBroadcastedReduce(
    std::shared_ptr<ArithmeticDagNode<T>> vectorDag,
    std::optional<std::shared_ptr<ArithmeticDagNode<T>>> cleanupMaskDag,
    int64_t period, int64_t steps, int64_t numSlots, const DagType& dagType,
    const std::string& reduceOp = "arith.addi", bool unroll = true) {
  using NodeTy = ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;

  DagReducer<T> reduceFunc = [&](NodePtr lhs, NodePtr rhs) {
    if (reduceOp == "arith.addi" || reduceOp == "arith.addf") {
      return NodeTy::add(lhs, rhs);
    }
    if (reduceOp == "arith.muli" || reduceOp == "arith.mulf") {
      return NodeTy::mul(lhs, rhs);
    }
    return NodeTy::add(lhs, rhs);
  };

  NodePtr reduced;
  if (unroll) {
    reduced = implementRotateAndReduceAccumulation<T>(vectorDag, period, steps,
                                                      reduceFunc);
  } else {
    reduced = implementRotateAndReduceAccumulationRolled<T>(
        vectorDag, period, steps, reduceFunc, dagType);
  }

  // Check Natural Replication
  if (steps * period == numSlots) {
    return reduced;
  }

  // Shift to last slots
  int64_t shiftToLast = numSlots - (steps - 1) * period;
  auto shifted = NodeTy::leftRotate(reduced, shiftToLast);

  NodePtr current = shifted;
  // Cleanup Mask
  if (cleanupMaskDag.has_value()) {
    current = NodeTy::mul(current, cleanupMaskDag.value());
  }

  // Replication Tree (Left rotations only)
  for (int64_t rep_shift = 1; rep_shift < steps; rep_shift *= 2) {
    int64_t rotateAmount = rep_shift * period;
    auto rotated = NodeTy::leftRotate(current, rotateAmount);
    current = reduceFunc(current, rotated);
  }

  return current;
}

// Returns an arithmetic DAG that implements a baby-step-giant-step between
// ciphertexts.
//
// This implements equation 21 in 6.2.2 of LKAA25: "Tricycle: Private
// Transformer Inference with Tricyclic Encodings"
// https://eprint.iacr.org/2025/1200
//
// This differs from the above implementRotateAndReduce in that, instead of a
// set of pre-computed plaintexts, both arguments are individual ciphertexts.
// Normally with one ciphertext, the naive approach uses n - 1 rotations that
// BSGS reduces to c sqrt(n) + O(1) rotations, if both inputs are ciphertexts
// then it converts 2n - 2 total rotations to n + c sqrt(n) + O(1) rotations.
// Essentially, the "n to sqrt(n)" redution applies to the `vector` argument
// only, while the `plaintexts` argument still gets n-1 rotations.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementCiphertextCiphertextBabyStepGiantStep(
    const T& giantSteppedOperand, const T& babySteppedOperand, int64_t period,
    int64_t steps, const DagType& dagType,
    DerivedRotationIndexFn derivedRotationIndexFn) {
  using NodeTy = ArithmeticDagNode<T>;

  // Avoid replicating and re-extracting by simulating the extraction step by
  // just returning the single ciphertext.
  auto extractFunc = [](std::shared_ptr<NodeTy> babySteppedDag,
                        int64_t extractionIndex) { return babySteppedDag; };

  return implementBabyStepGiantStep<T>(giantSteppedOperand, babySteppedOperand,
                                       period, steps, dagType, extractFunc, {},
                                       derivedRotationIndexFn);
}

// Returns an arithmetic DAG that implements the Halevi-Shoup matrix
// multiplication algorithm. This implementation uses a rotate-and-reduce
// operation, followed by a summation of partial sums if the matrix is not
// square.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementHaleviShoup(const T& vector, const T& matrix,
                     std::vector<int64_t> originalMatrixShape,
                     const DagType& dagType,
                     std::map<int, bool> zeroDiagonals = {},
                     bool unroll = true) {
  using NodeTy = ArithmeticDagNode<T>;
  using NodePtr = std::shared_ptr<NodeTy>;
  int64_t numRotations = matrix.getShape()[0];

  auto rotateAndReduceResult = implementRotateAndReduce<T>(
      vector, std::optional<T>(matrix), /*period=*/1,
      /*steps=*/numRotations, dagType, zeroDiagonals,
      /*reduceOp=*/"arith.addi",
      /*unroll=*/unroll);

  auto summedShifts = rotateAndReduceResult;

  int64_t matrixNumRows = nextPowerOfTwo(originalMatrixShape[0]);
  int64_t matrixNumCols = nextPowerOfTwo(originalMatrixShape[1]);

  if (matrixNumRows == matrixNumCols) {
    return summedShifts;
  }

  // Post-processing partial-rotate-and-reduce step required for
  // squat-diagonal packing.
  int64_t numShifts = (int64_t)(log2(matrixNumCols) - log2(matrixNumRows));
  if (unroll) {
    int64_t shift = matrixNumCols / 2;
    for (int64_t i = 0; i < numShifts; ++i) {
      auto rotated = NodeTy::leftRotate(summedShifts, shift);
      summedShifts = NodeTy::add(summedShifts, rotated);
      shift /= 2;
    }

    return summedShifts;
  }

  auto shift = NodeTy::constantScalar(matrixNumCols / 2, DagType::index());
  auto loopNode = NodeTy::loop(
      {summedShifts, shift}, {dagType, DagType::index()},
      /*lower=*/0, /*upper=*/numShifts, /*step=*/1,
      [&](NodePtr iv, const std::vector<NodePtr>& iterArgs) {
        auto currentSum = iterArgs[0];
        auto currentShift = iterArgs[1];
        auto rotated = NodeTy::leftRotate(currentSum, currentShift);
        auto newSum = NodeTy::add(currentSum, rotated);
        auto newShift = NodeTy::floorDiv(currentShift, 2);
        return NodeTy::yield({newSum, newShift});
      });
  return NodeTy::resultAt(loopNode, 0);
}

// Returns an arithmetic DAG that implements the bicyclic matrix multiplication
// algorithm.
//
// The input matrices packedA and packedB are assumed to be properly packed to
// meet the conditions for bicyclic multiplication. That is: both matrices are
// zero-padded so that their dimensions are coprime, they are cyclically
// repeated to fill all the slots of the ciphertext, and they are packed
// according to the bicyclic ordering.
//
// This function produces a kernel using roughly n + 2sqrt(n) - 3 rotations
// (for matrix dimensions all order n), by applying the baby-step-giant-step
// method to reduce the number of rotations of packedA.
//
// This implements the BMM-I algorithm from https://eprint.iacr.org/2024/1762
// with modifications from LKAA25 (https://eprint.iacr.org/2025/1200):
//
//  - A simplification of the rotation formula in Sec 5.2.1 (equation 9).
//  - A baby-step-giant-step optimization of the summation below, from Sec
//    6.2.2 (equation 21).
//
// It computes
//
// C = sum_{c=0}^{n-1} rot(A, r1(c)) * rot(B, r2(c))
//
// where
//
//  r1(c) = cm
//  r2(c) = p(cm(p^{-1}) mod n) mod np
//
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementBicyclicMatmul(const T& packedA, const T& packedB, int64_t m,
                        int64_t n, int64_t p, const DagType& dagType) {
  APInt mAPInt = APInt(64, m);
  APInt nAPInt = APInt(64, n);
  APInt pAPInt = APInt(64, p);

  APInt mInvModN = multiplicativeInverse(mAPInt.urem(nAPInt), nAPInt);
  APInt pInvModN = multiplicativeInverse(pAPInt.urem(nAPInt), nAPInt);

  auto derivedRotationIndexFn = [&](int64_t giantStepSize,
                                    int64_t giantStepIndex,
                                    int64_t babyStepIndex, int64_t period) {
    APInt c(64, giantStepIndex * giantStepSize + babyStepIndex);
    APInt mAPInt(64, m);

    // RotY(c) = (p * (c * m * p^{-1} mod n)) mod (n * p)
    APInt rotyInner = (c * mAPInt * pInvModN.getSExtValue()).urem(nAPInt);
    APInt roty = (rotyInner * pAPInt).urem(nAPInt * pAPInt);

    APInt result = roty - APInt(64, period) * APInt(64, giantStepSize) *
                              APInt(64, giantStepIndex);
    // Ensure rotation index is a positive modulo representative [0, np-1].
    result = result.srem(nAPInt * pAPInt);
    if (result.isNegative()) {
      result += nAPInt * pAPInt;
    }
    return result.getSExtValue();
  };

  return implementCiphertextCiphertextBabyStepGiantStep<T>(
      packedA, packedB, /*period=*/m, /*steps=*/n, dagType,
      derivedRotationIndexFn);
}

// Returns an arithmetic DAG that implements the tricyclic batch matrix
// multiplication algorithm (ciphertext-ciphertext). Uses the tricyclic
// rotation formulas from LKAA25 (Tricycle paper) and applies BSGS to
// reduce rotations on the φ(A) side.
//
// The inputs packedA and packedB are expected to be tricyclic encodings
// φ(A) and φ(B) for tensors A ∈ R^{h×m×n} and B ∈ R^{h×n×p}. The function
// applies equation (22) and the ct-ct BSGS decomposition in Section 6.2.2.
//
// Parameters:
//  - packedA: tricyclic-encoded ciphertext for A (φ(A))
//  - packedB: tricyclic-encoded ciphertext for B (φ(B))
//  - h, m, n, p: tricyclic tensor dimensions (h: batch count / heads)
//
// Returns φ(Z) where Z = batch_matmul(A, B) as a DAG node.
template <typename T>
std::enable_if_t<std::is_base_of<AbstractValue, T>::value,
                 std::shared_ptr<ArithmeticDagNode<T>>>
implementTricyclicBatchMatmul(const T& packedA, const T& packedB, int64_t h,
                              int64_t m, int64_t n, int64_t p,
                              const DagType& dagType) {
  APInt hAPInt = APInt(64, h);
  APInt mAPInt = APInt(64, m);
  APInt nAPInt = APInt(64, n);
  APInt pAPInt = APInt(64, p);

  APInt pInvModN = multiplicativeInverse(pAPInt.urem(nAPInt), nAPInt);

  APInt modulus = (hAPInt * nAPInt * pAPInt);
  // This follows Eq. (22) and the ct-ct BSGS decomposition.
  // RotY(c) = (h * p * ( (c * m * p^{-1}) mod n )) mod (h * n * p)
  auto derivedRotationIndexFn = [&](int64_t giantStepSize,
                                    int64_t giantStepIndex,
                                    int64_t babyStepIndex, int64_t period) {
    APInt c(64, giantStepIndex * giantStepSize + babyStepIndex);

    // rotyInner = (c * m * p^{-1}) mod n
    APInt rotyInner = (c * mAPInt * pInvModN.getSExtValue()).urem(nAPInt);

    // rotY calculation from LKAA25 Eq. (22):
    // RotY(c) = (h * p * (c * m * p^{-1} mod n)) mod (h * n * p)
    APInt roty = (rotyInner * hAPInt * pAPInt).urem(modulus);

    APInt result = roty - APInt(64, period) * APInt(64, giantStepSize) *
                              APInt(64, giantStepIndex);
    result = result.srem(modulus);
    if (result.isNegative()) {
      result += modulus;
    }
    return result.getSExtValue();
  };

  int64_t period = h * m;
  return implementCiphertextCiphertextBabyStepGiantStep<T>(
      packedA, packedB, /*period=*/period, /*steps=*/n, dagType,
      derivedRotationIndexFn);
}

}  // namespace kernel
}  // namespace heir
}  // namespace mlir

#endif  // LIB_KERNEL_KERNELIMPLEMENTATION_H_
