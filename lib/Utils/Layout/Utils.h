#ifndef LIB_UTILS_LAYOUT_UTILS_H_
#define LIB_UTILS_LAYOUT_UTILS_H_

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/SmallBitVector.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/Utils/Utils.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"   // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"            // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"               // from @llvm-project

// ISL
#include "include/isl/ctx.h"  // from @isl
#include "include/isl/map.h"  // from @isl

namespace mlir {
namespace heir {

// Helper that adds constraints built from the array of positions and coeffs.
// Inequalities are given by (>= 0).
void addConstraint(presburger::IntegerRelation& result,
                   ArrayRef<std::pair<int64_t, int64_t>> posAndCoeff,
                   bool equality);
// Helper that adds inclusive lower and upper bounds for a given position and
// value.
void addBounds(presburger::IntegerRelation& result, int64_t pos, int64_t lower,
               std::optional<int64_t> upper = std::nullopt);

// Lifts a relation R : [a...] -> [b...] to [p, a...] -> [p, b...] by prepending
// a "passthrough" dimension as the new leading variable of both the domain and
// range, constrained so the new domain variable equals the new range variable
// (i.e. p is carried through unchanged). When `lb`/`ub` are provided, the
// passthrough dimension is additionally bounded to [lb, ub].
void prependPassthroughDim(presburger::IntegerRelation& relation,
                           std::optional<int64_t> lb = std::nullopt,
                           std::optional<int64_t> ub = std::nullopt);

// Adds a new local variable q to the relation that represents expr % modulus.
// Returns the index of the new local variable in the relation.
unsigned int addModConstraint(presburger::IntegerRelation& result,
                              ArrayRef<int64_t> exprs, int64_t modulus);

/// Cheap one-sided inequality check on two layout relations. Returns success()
/// when they are provably unequal, and failure() when the check cannot tell --
/// a failure() is not evidence of equality.
LogicalResult tryProveUnequalByVolume(
    const presburger::IntegerRelation& layout1,
    const presburger::IntegerRelation& layout2);

// Returns an IntegerRelation that enforces a row-major layout for the given
// tensor type and number of slots. This is used for IntegerRelations that
// represent data layouts in ciphertexts. It expects that the number of domain
// variables match the rank of the tensor, and that there are two range
// variables representing the ciphertext index and slot index in that order.
presburger::IntegerRelation getRowMajorLayoutRelation(
    RankedTensorType tensorType, int64_t numSlots);

// Returns an IntegerRelation that represents a diagonalized layout for a matrix
// such that the ith diagonal of the matrix is in the ith row of the
// result. The number of rows of the input and output must match.
presburger::IntegerRelation getDiagonalLayoutRelation(
    RankedTensorType matrixType, int64_t minSlotCount);

// Applies a diagonal layout onto a given 2-D matrix layout.
FailureOr<presburger::IntegerRelation> diagonalize2dMatrix(
    presburger::IntegerRelation relation, RankedTensorType originalType,
    int64_t minSlotCount);

// Returns an IntegerRelation that represents a bicyclic layout for a matrix.
// See https://eprint.iacr.org/2024/1762 for details.
presburger::IntegerRelation getBicyclicLayoutRelation(
    RankedTensorType matrixType, int64_t numSlots);

// Returns an IntegerRelation that represents a tricyclic layout for a 3-D
// tensor. The domain ordering is (h, m, n) and the range is (ct, slot).
presburger::IntegerRelation getTricyclicLayoutRelation(
    RankedTensorType tensorType, int64_t numSlots);

// Returns an IntegerRelation representing a Multi-Tile Packing (MTP) layout.
//
// The logical tensor is decomposed into primitive 2-D tiles along rowAxis and
// columnAxis. All remaining tensor dimensions remain outer dimensions of the
// tile tensor. Tile-tensor entries are enumerated in row-major order and
// grouped consecutively, with tilesPerCiphertext primitive tiles assigned to
// each ciphertext.
//
// For tile position p and intra-tile coordinates (l, d), the physical slot
// embedding is:
//
//   slot = d * (tilesPerCiphertext * tileRows) + p * tileRows + l
//
// The returned relation maps the original logical tensor coordinates to
// [ct, slot]. Conceptually multidimensional ciphertext coordinates are
// flattened into ct in row-major order.
//
// Returns failure if the tensor rank or axis selection is invalid, tile
// parameters are non-positive, or one ciphertext does not have enough slots
// for tilesPerCiphertext tiles.
FailureOr<presburger::IntegerRelation> getMultiTileLayoutRelation(
    RankedTensorType tensorType, int64_t rowAxis, int64_t columnAxis,
    int64_t tileRows, int64_t tileColumns, int64_t tilesPerCiphertext,
    int64_t numSlots);

// Returns the generalized diagonal packing relation for the cleartext
// operand of a bicyclic matrix multiplication.
//
// This layout is specific to the cleartext matrix. It decomposes the matrix
// into n diagonal vectors (one for each step along the contracting dimension).
// Each diagonal vector pre-arranges matrix elements to align with a specific
// rotation of the encrypted operand and match the target output slots. This
// avoids single-ciphertext capacity limits and removes the coprimality
// requirement on the cleartext matrix dimensions. It also ensures that the
// multiplied result directly matches the output bicyclic layout without needing
// layout conversions. However, it incurs the overhead of eagerly materializing
// n separate plaintext vectors of size numSlots.
presburger::IntegerRelation getBicyclicDiagonalRelation(
    RankedTensorType matrixType, int64_t contractionDim, int64_t stride,
    int64_t numSlots);

// Returns an IntegerRelation with domain and range space both (ct, slot) that
// maps each slot s in [0, period) of a ciphertext to every slot s' in [0,
// numSlots) with s' equiv s (mod period). Excepts numCiphertexts == 1.
presburger::IntegerRelation getPeriodicReplicationRelation(
    int64_t numCiphertexts, int64_t numSlots, int64_t period);

// Returns an IntegerRelation that represents a per-row layout for a matrix
// such that each row of the matrix is in a separate ciphertext.
presburger::IntegerRelation getPerRowLayoutRelation(RankedTensorType matrixType,
                                                    int64_t minSlotCount);

// Returns true if the given relation is a squat diagonal layout for the given
// matrix type and ciphertext semantic shape.
bool isRelationSquatDiagonal(RankedTensorType matrixType, int64_t minSlotCount,
                             const presburger::IntegerRelation& relation);

// Returns true if the given relation is a row-major layout for the given
// vector type and slot size.
bool isRelationRowMajor(RankedTensorType vectorType, int64_t numSlots,
                        const presburger::IntegerRelation& relation);

// Returns true if the relation packs a vector into ciphertext zero with each
// vector element occupying exactly one distinct slot.
bool isOneToOneSingleCiphertextPacking(
    const presburger::IntegerRelation& relation);

// Folds a single-ciphertext vector permutation (as accepted by
// isOneToOneSingleCiphertextPacking) into a matrix layout, returning the matrix
// layout that lets a diagonal matvec consume the un-permuted vector directly.
// `vectorPermutation` maps [col] -> [ct, slot]; `matrixLayout` maps
// [row, col] -> [ct, slot].
presburger::IntegerRelation foldVectorPermutationIntoMatrixLayout(
    const presburger::IntegerRelation& vectorPermutation,
    const presburger::IntegerRelation& matrixLayout);

// Returns true if the given relation is a per-row layout
// for the given matrix type and ciphertext semantic shape.
bool isRelationPerRow(RankedTensorType matrixType, int64_t minSlotCount,
                      presburger::IntegerRelation relation);

// Returns true if the given relation is a bicyclic layout for the given
// matrix type and ciphertext semantic shape.
bool isRelationBicyclic(RankedTensorType matrixType, int64_t numSlots,
                        const presburger::IntegerRelation& relation);

// Returns true if the relation corresponds to the tricyclic layout for the
// tensor type and ciphertext semantic shape.
bool isRelationTricyclic(RankedTensorType tensorType, int64_t numSlots,
                         const presburger::IntegerRelation& relation);

// Returns true if relation is the row-major-assignment MTP layout produced by
// getMultiTileLayoutRelation for the supplied tensor and packing parameters.
bool isRelationMultiTile(
    RankedTensorType tensorType, int64_t rowAxis, int64_t columnAxis,
    int64_t tileRows, int64_t tileColumns, int64_t tilesPerCiphertext,
    int64_t numSlots, const presburger::IntegerRelation& relation);

// Returns a new IntegerRelation that is the same as the given relation, but
// with the given dimensions collapsed. This expects that the reassociation
// indices result in a rank-reduction of the source type (i.e. the collapsed
// dimensions are all unit dimensions).
presburger::IntegerRelation collapseDimensions(
    const presburger::IntegerRelation& relation, RankedTensorType sourceType,
    ArrayRef<ReassociationIndices> reassociation);

// Returns a new IntegerRelation that is the same as the given relation, but
// with the given dimensions expanded. This expects that the reassociation
// indices result in a rank-expansion of the result type (i.e. the expanded
// dimensions are all unit dimensions).
presburger::IntegerRelation expandDimensions(
    const presburger::IntegerRelation& relation, RankedTensorType resultType,
    ArrayRef<ReassociationIndices> reassociation);

// Returns a new relation produced by constraining the index dimensions of
// type varKind to the given relation to the provided values. The fixedValues
// array size should equal the number of variables of type varKind.
presburger::IntegerRelation fixVars(const presburger::IntegerRelation& relation,
                                    ArrayRef<int64_t> fixedValues,
                                    presburger::VarKind varKind);

// Returns a new relation produced by constraining the domain variables of the
// given relation to the provided values.
//
// The fixedValues array should have size equal to the number of domain
// variables in the same order as `relation`. This generally should align with
// the order of the dimensions of the RankedTensorType this relation is laying
// out.
inline presburger::IntegerRelation fixDomainVars(
    const presburger::IntegerRelation& relation,
    ArrayRef<int64_t> fixedValues) {
  return fixVars(relation, fixedValues, presburger::VarKind::Domain);
}

inline presburger::IntegerRelation fixRangeVars(
    const presburger::IntegerRelation& relation,
    ArrayRef<int64_t> fixedValues) {
  return fixVars(relation, fixedValues, presburger::VarKind::Range);
}

struct PointCollector {
  std::vector<std::vector<int64_t>> points;
  isl_ctx* ctx;

  PointCollector() { ctx = isl_ctx_alloc(); }

  ~PointCollector() { isl_ctx_free(ctx); }

  // Delete copy constructor and assignment to avoid double-free
  PointCollector(const PointCollector&) = delete;
  PointCollector& operator=(const PointCollector&) = delete;
};

struct PointPairCollector {
  using Point = std::vector<int64_t>;
  std::vector<std::pair<Point, Point>> points;
  isl_ctx* ctx;
  int domainDims;
  int rangeDims;

  PointPairCollector(int domainDims, int rangeDims)
      : domainDims(domainDims), rangeDims(rangeDims) {
    ctx = isl_ctx_alloc();
  }

  ~PointPairCollector() { isl_ctx_free(ctx); }

  // Delete copy constructor and assignment to avoid double-free
  PointPairCollector(const PointPairCollector&) = delete;
  PointPairCollector& operator=(const PointPairCollector&) = delete;
};

// Get a list of points in the relation by enumerating all possible values.
void enumeratePoints(const presburger::IntegerRelation& relation,
                     PointPairCollector& collector);

// Get a list of points in the range of the relation by enumerating all
// possible values.
void getRangePoints(const presburger::IntegerRelation& relation,
                    PointCollector& collector);

// Sample a point in the range of the relation.
std::vector<int64_t> anyRangePoint(const presburger::IntegerRelation& relation);

// Get a list of ct points that are not in the range of the relation. Assumes
// that the relation is a layout relation mapping input tensor dimensions to a
// 2-D image set of (ct, slot) pairs with the given outputType.
void getCtComplementPoints(const presburger::IntegerRelation& relation,
                           PointCollector& collector,
                           RankedTensorType outputType);

// Collapse a relation with the given reassociation indices. Dimensions that are
// collapsed in a row-major order.
presburger::IntegerRelation getCollapsedRelation(
    RankedTensorType sourceType, RankedTensorType destType,
    ArrayRef<ReassociationIndices> reassociation);

// Get layout relation that corresponds to a tensor::insert_slice op.
//
// `droppedDims` must be indexed over `resultType`'s rank (i.e. one bit per
// destination dimension, matching `tensor::InsertSliceOp::getDroppedDims()`)
// and identify which destination dimensions do not correspond to a dimension
// of `sliceType`. A destination dimension of static size 1 is not always
// dropped: it is retained when `sliceType` also has a corresponding
// dimension there (a rank-preserving insertion), so this cannot be inferred
// from `sizes` alone -- the caller must supply the real answer, typically via
// the inserting op's own `getDroppedDims()`. Returns failure if
// `droppedDims`'s size does not match `resultType`'s rank, or if the number
// of retained (non-dropped) dimensions does not match `sliceType`'s rank.
FailureOr<presburger::IntegerRelation> getSliceInsertionRelation(
    RankedTensorType sliceType, RankedTensorType resultType,
    SmallVector<int64_t> offsets, SmallVector<int64_t> sizes,
    SmallVector<int64_t> strides, const llvm::SmallBitVector& droppedDims);

// Shift a var at pos by a constant offset in an IntegerRelation, i.e. replace
// var with var' = var + offset.
presburger::IntegerRelation shiftVar(
    const presburger::IntegerRelation& relation, unsigned int pos,
    int64_t offset);

// Get layout relation that corresponds to a tensor::pad op.
presburger::IntegerRelation getPaddingRelation(RankedTensorType paddedType,
                                               RankedTensorType unpaddedType,
                                               ArrayRef<int64_t> lowPadding);

// Get layout relation that corresponds to a tensor::extract_slice op.
//
// `droppedDims` must be indexed over `sourceType`'s rank (i.e. one bit per
// source dimension, matching `tensor::ExtractSliceOp::getDroppedDims()`) and
// identify which source dimensions do not survive into `resultType`. A
// source dimension of static size 1 is not always dropped: it is retained
// when `resultType` also has a corresponding dimension there (a
// rank-preserving extraction), so this cannot be inferred from `sizes`
// alone -- the caller must supply the real answer, typically via the
// extracting op's own `getDroppedDims()`. Returns failure if
// `droppedDims`'s size does not match `sourceType`'s rank, or if the number
// of retained (non-dropped) dimensions does not match `resultType`'s rank.
FailureOr<presburger::IntegerRelation> getSliceExtractionRelation(
    RankedTensorType sourceType, RankedTensorType resultType,
    SmallVector<int64_t> offsets, SmallVector<int64_t> sizes,
    SmallVector<int64_t> strides, const llvm::SmallBitVector& droppedDims);

// Returns the balanced multi-tile packing parameters for `taskCount`
// primitive tiles of shape `tileRows x tileColumns`, given `minSlotCount`
// slots per ciphertext:
//
//   capacity           = floor(minSlotCount / (tileRows*tileColumns))
//   numCiphertexts     = ceil(taskCount / capacity)
//   tilesPerCiphertext = ceil(taskCount / numCiphertexts)
//
// This is the single source of truth for the balanced-packing formula that
// both automatic MTP-JKLS batch-matmul selection and general outer-tiled
// GEMM tile scheduling rely on. `tilesPerCiphertext` is defined so that it
// always agrees with the value `ConvertLinalgBatchMatmul::mtpJklsKernel`
// independently recovers from the physical shape
// (`ceil(taskCount / numCiphertexts)`), which is what makes this "balanced"
// rather than a maximum-fill policy -- see the Phase 7 plan-review history
// for the disagreement a maximum-fill formula can produce.
//
// Returns failure for non-positive inputs, for a primitive tile that does
// not fit in one ciphertext, or if any intermediate arithmetic would
// overflow.
struct BalancedMtpPacking {
  int64_t numCiphertexts;
  int64_t tilesPerCiphertext;
};

FailureOr<BalancedMtpPacking> getBalancedMtpPacking(int64_t taskCount,
                                                    int64_t tileRows,
                                                    int64_t tileColumns,
                                                    int64_t minSlotCount);

// Tests whether two layout relations describe the same set of points.
//
// This check is one-sided: `true` means the relations are provably equal, but
// `false` means "not proven equal" rather than "proven unequal".
//
// IntegerRelation::isEqual is deliberately not used: it fails to
// return within 120s on layouts isl decides in single-digit milliseconds.
bool isRelationEqual(const presburger::IntegerRelation& relation1,
                     const presburger::IntegerRelation& relation2);

// Returns true if the given relation is surjective onto the given tensor type.
// This tests that the range set of the relation covers all points of the given
// tensor type. This is used to test if a layout is dense, so that the layout
// materialization can be simplified into a constant splat.
bool isDenseLayout(const presburger::IntegerRelation& relation,
                   RankedTensorType type);

// Returns the number of integer points in the relation. Returns -1 if the
// relation is unbounded of the number of points exceeds the size of an int64_t.
int64_t relationSize(const presburger::IntegerRelation& relation);

}  // namespace heir
}  // namespace mlir

#endif  // LIB_UTILS_LAYOUT_UTILS_H_
