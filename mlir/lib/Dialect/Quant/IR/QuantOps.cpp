//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Quant/QuantOps.h"
#include "QuantDialectBytecode.h"
#include "TypeDetail.h"

#include "mlir/Dialect/Quant/QuantTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

using namespace mlir;
using namespace mlir::quant;
using namespace mlir::quant::detail;

#include "mlir/Dialect/Quant/QuantOpsDialect.cpp.inc"

// Verifies that the sub-channel quantization parameters are consistent with
// the given container type. The function checks the following:
//
// - The container type must be a ranked tensor type.
// - Each quantized dimension must be less than the rank of the tensor.
// - The size of each dimension at the quantized dimension must be divisible
//    by the corresponding block size.
// - The scale dimension size at each axis index should match the tensor
//    dimension at the index divided by the corresponding block size.
//
// The `uniformQuantizedSubChannelType` argument provides the sub-channel
// quantization parameters, and the `containerType` argument specifies the
// type of the container holding the quantized data.
namespace {
LogicalResult verifySubChannelQuantization(
    Operation *op,
    UniformQuantizedSubChannelType uniformQuantizedSubChannelType,
    Type containerType) {
  auto tensorType = dyn_cast<TensorType>(containerType);
  if (!tensorType)
    return op->emitError("scalar types may not use sub-channel quantization");

  if (!tensorType.hasRank())
    return op->emitError(
        "tensor containing the sub-channel quantized type must be ranked");

  const SmallVector<std::pair<int32_t, int64_t>> &blockSizeInfo =
      uniformQuantizedSubChannelType.getBlockSizeInfo();
  auto shape = tensorType.getShape();

  // The dimension size of scale for an axis which is not specified as quantized
  // dimension should be 1.
  SmallVector<int64_t> expectedScaleShape(tensorType.getShape().size(), 1);
  for (auto [quantizedDimension, blockSize] : blockSizeInfo) {
    if (quantizedDimension >= tensorType.getRank())
      return op->emitError()
             << "quantized dimension " << quantizedDimension
             << " must be less than tensor rank " << tensorType.getRank();
    if (!tensorType.isDynamicDim(quantizedDimension) &&
        tensorType.getDimSize(quantizedDimension) % blockSize != 0)
      return op->emitError()
             << "tensor dimension size "
             << tensorType.getDimSize(quantizedDimension) << " at axis "
             << quantizedDimension
             << " must be divisible by the corresponding block size "
             << blockSize;
    if (tensorType.isDynamicDim(quantizedDimension))
      expectedScaleShape[quantizedDimension] = ShapedType::kDynamic;
    else
      expectedScaleShape[quantizedDimension] =
          tensorType.getDimSize(quantizedDimension) / blockSize;
  }

  // Block sizes must be greater than 0 and divide the corresponding dimension
  // size. While a block size b must be less than or equal to the corresponding
  // dimension size d, this constraint is implicitly enforced by requiring that
  // d % b == 0 when d != 0.
  //
  // However, a problem arises when d = 0.  The divisibility constraint allows b
  // to be any value, potentially violating the requirement that b <= d.
  // Furthermore, if b is unspecified (implicitly equal to d), it violates the
  // constraint that b > 0.
  //
  // Therefore, we explicitly disallow the case where d = 0 to maintain
  // consistency and avoid these issues.
  if (llvm::find(tensorType.getShape(), 0) != tensorType.getShape().end()) {
    return op->emitError() << "tensor dimension size of zero is not allowed "
                              "with sub-channel quantization";
  }

  auto scaleShape =
      uniformQuantizedSubChannelType.getScales().getType().getShape();
  if (scaleShape.size() != shape.size()) {
    return op->emitError() << "Rank of scales " << scaleShape.size()
                           << " must match "
                           << "the rank of the tensor " << shape.size();
  }

  for (auto [index, scaleDim] : llvm::enumerate(expectedScaleShape)) {
    if (expectedScaleShape[index] != ShapedType::kDynamic &&
        expectedScaleShape[index] != scaleShape[index])
      return op->emitError() << "dimension size " << scaleDim
                             << " of scales tensor at axis " << index
                             << " should match (tensor dimension at axis / "
                                "block sizes at axis) = "
                             << expectedScaleShape[index];
  }

  return success();
}
} // namespace
void QuantizationDialect::initialize() {
  addTypes<AnyQuantizedType, CalibratedQuantizedType, UniformQuantizedType,
           UniformQuantizedPerAxisType, QuantileQuantizedType,
           QuantileQuantizedPerAxisType, UniformQuantizedSubChannelType>();
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/Quant/QuantOps.cpp.inc"
      >();
  addBytecodeInterface(this);
}

OpFoldResult StorageCastOp::fold(FoldAdaptor adaptor) {
  // Matches x -> [scast -> scast] -> y, replacing the second scast with the
  // value of x if the casts invert each other.
  auto srcScastOp = getArg().getDefiningOp<StorageCastOp>();
  if (!srcScastOp || srcScastOp.getArg().getType() != getType())
    return OpFoldResult();
  return srcScastOp.getArg();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/Quant/QuantOps.cpp.inc"
