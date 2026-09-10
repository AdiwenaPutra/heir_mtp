#include "lib/Transforms/ConvertToCiphertextSemantics/TypeConversion.h"

#include <cassert>
#include <cstdint>
#include <optional>

#include "lib/Dialect/TensorExt/IR/TensorExtAttributes.h"
#include "llvm/include/llvm/ADT/DynamicAPInt.h"     // from @llvm-project
#include "llvm/include/llvm/Support/Debug.h"        // from @llvm-project
#include "llvm/include/llvm/Support/raw_ostream.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/IntegerRelation.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/PresburgerSpace.h"  // from @llvm-project
#include "mlir/include/mlir/Analysis/Presburger/Simplex.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"   // from @llvm-project
#include "mlir/include/mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"          // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"      // from @llvm-project

#define DEBUG_TYPE "convert-to-ciphertext-semantics"

namespace mlir {
namespace heir {

using presburger::IntegerRelation;
using presburger::VarKind;
using tensor_ext::LayoutAttr;

Type materializeLayout(Type dataType, LayoutAttr attr, int minSlotCount) {
  IntegerRelation rel = attr.getIntegerRelation();
  presburger::Simplex simplex(rel);
  llvm::SmallVector<int64_t> ciphertextSemanticShape;
  unsigned rangeOffset = rel.getVarKindOffset(VarKind::Range);
  for (unsigned varPos = rangeOffset;
       varPos < rel.getVarKindEnd(VarKind::Range) - 1; ++varPos) {
    LLVM_DEBUG({
      llvm::dbgs() << "materializeLayout: computing upper bound for range "
                      "dimension "
                   << varPos - rangeOffset << " (ct), layout=" << attr << "\n";
      llvm::dbgs().flush();
    });
    llvm::SmallVector<llvm::DynamicAPInt> coeffs(rel.getNumVars() + 1,
                                                 llvm::DynamicAPInt(0));
    coeffs[varPos] = llvm::DynamicAPInt(1);
    auto bounds = simplex.computeIntegerBounds(coeffs);
    assert(bounds.second.isBounded() &&
           "No upper bound found for range variable");
    LLVM_DEBUG(llvm::dbgs()
                   << "materializeLayout: upper bound for range dimension "
                   << varPos - rangeOffset << " (ct) = " << *bounds.second
                   << "\n\n";);
    // The number of physical ciphertexts this dimension actually needs is
    // the SPAN of its achievable range (upper - lower + 1), not simply
    // upper + 1. Those coincide whenever the achievable range starts at 0
    // (true for every "primary" layout constructed with an explicit
    // `0 <= ct <= numCiphertexts - 1` bound, e.g. getMultiTileLayoutRelation
    // and getRowMajorLayoutRelation), but a RESTRICTION of a larger
    // multi-ciphertext layout -- e.g. composing a destination's own
    // addressing with a fixed offset, as ConvertToCiphertextSemantics.cpp's
    // secretScalarSecretTensor does when reconciling one sibling
    // tensor.insert_slice among several into a shared multi-ciphertext
    // destination -- can leave this dimension achievable ONLY at a single
    // nonzero constant (lower == upper != 0). Using upper + 1 there
    // silently over-allocates ciphertext-dimension rows the relation itself
    // never populates (row 0 up to lower - 1), materializing a "padded"
    // shape inconsistent with what any conversion that actually preserves
    // physical shape (e.g. a same-ciphertext data permutation) produces.
    // `bounds.first` (from the same computeIntegerBounds(coeffs) call
    // above, coeffs selecting this exact dimension) is already this
    // dimension's own tight lower bound, so no second solve is needed.
    int64_t ciphertextsNeeded = int64_t(*bounds.second) + 1;
    if (bounds.first.isBounded() && int64_t(*bounds.first) > 0) {
      ciphertextsNeeded = int64_t(*bounds.second) - int64_t(*bounds.first) + 1;
    }
    ciphertextSemanticShape.push_back(ciphertextsNeeded);
  }
  // Last dimension is always the slot size. The relation may enforce a tighter
  // bound depending on whether the slots at the end are full, so use the upper
  // bound.
  ciphertextSemanticShape.push_back(minSlotCount);
  return RankedTensorType::get(ciphertextSemanticShape, dataType);
}

Type materializeScalarLayout(Type type, LayoutAttr attr, int minSlotCount) {
  return RankedTensorType::get({1, minSlotCount}, type);
}

Type materializePermutationLayout(Type type, DenseIntElementsAttr permutation,
                                  int minSlotCount) {
  auto tensorType = dyn_cast<RankedTensorType>(type);

  assert(tensorType &&
         "Permutation layout attributes on non-tensor args are not supported");
  assert(tensorType.getShape().size() <= 2 &&
         "Permutation layouts only supports tensor args of max dim-2");

  auto inShape = tensorType.getShape();
  if (inShape.size() == 2)
    return RankedTensorType::get({inShape[0], minSlotCount},
                                 getElementTypeOrSelf(type));

  return RankedTensorType::get({1, minSlotCount}, getElementTypeOrSelf(type));
}

}  // namespace heir
}  // namespace mlir
