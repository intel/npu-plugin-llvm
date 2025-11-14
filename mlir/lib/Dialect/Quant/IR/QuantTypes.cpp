//===- QuantOps.cpp - Quantization Type and Ops Implementation --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TypeDetail.h"
#include "mlir/Dialect/Quant/IR/Quant.h"
#include "mlir/Dialect/Quant/IR/QuantTypes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace mlir::quant;
using namespace mlir::quant::detail;

namespace {

// Return the minimum scale representable in a given float type
double getMinScale(Type expressedType) {
  auto floatType = cast<FloatType>(expressedType);
  return APFloat::getSmallest(floatType.getFloatSemantics()).convertToDouble();
}

// Return the maximum scale representable in a given float type
double getMaxScale(Type expressedType) {
  auto floatType = cast<FloatType>(expressedType);
  return APFloat::getLargest(floatType.getFloatSemantics()).convertToDouble();
}

}  // namespace

unsigned QuantizedType::getFlags() const {
  return static_cast<ImplType *>(impl)->flags;
}

bool QuantizedType::classof(Type type) {
  return llvm::isa<QuantDialect>(type.getDialect());
}

LogicalResult
QuantizedType::verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                                unsigned flags, Type storageType,
                                Type expressedType, int64_t storageTypeMin,
                                int64_t storageTypeMax) {
                                 
  bool isSigned =
      (flags & QuantizationFlags::Signed) == QuantizationFlags::Signed;

  // Integral storage type width checks
  if (storageType.isa<IntegerType>()) {
    unsigned integralWidth =
        llvm::dyn_cast<IntegerType>(storageType).getWidth();

    if (integralWidth == 0 || integralWidth > MaxStorageBits)
      return emitError() << "illegal storage type size: " << integralWidth;
  }

  int64_t defaultMin, defaultMax;
  if (storageType.isa<IntegerType>()) {
    const auto width = llvm::dyn_cast<IntegerType>(storageType).getWidth();
    defaultMin = QuantizedType::getDefaultMinimumForInteger(isSigned, width);
    defaultMax = QuantizedType::getDefaultMaximumForInteger(isSigned, width);
  } else if (mlir::isa<Float8E5M2Type>(storageType)) {
    defaultMin = QuantizedType::getDefaultMinimumForF8E5M2();
    defaultMax = QuantizedType::getDefaultMaximumForF8E5M2();
  } else if (mlir::isa<Float8E4M3FNType>(storageType)) {
    defaultMin = QuantizedType::getDefaultMinimumForF8E4M3FN();
    defaultMax = QuantizedType::getDefaultMaximumForF8E4M3FN();
  } else if (mlir::isa<Float4E2M1FNType>(storageType)) {
    defaultMin = QuantizedType::getDefaultMinimumForF4E2M1FN();
    defaultMax = QuantizedType::getDefaultMaximumForF4E2M1FN();
  } else {
    return emitError() << "illegal storage type, supported types are: integral "
                          "types, Float8E4M3FNType, Float8E5M2Type and Float4E2M1FNType ";
  }

  // Verify storageTypeMin and storageTypeMax.
  if (storageTypeMax - storageTypeMin <= 0 || storageTypeMin < defaultMin ||
      storageTypeMax > defaultMax) {
    return emitError() << "illegal storage min and storage max: ("
                       << storageTypeMin << ":" << storageTypeMax << ")";
  }
  return success();
}

Type QuantizedType::getStorageType() const {
  return static_cast<ImplType *>(impl)->storageType;
}

int64_t QuantizedType::getStorageTypeMin() const {
  return static_cast<ImplType *>(impl)->storageTypeMin;
}

int64_t QuantizedType::getStorageTypeMax() const {
  return static_cast<ImplType *>(impl)->storageTypeMax;
}

bool QuantizedType::hasStorageTypeBounds() const {
  unsigned int integralWidth = getStorageTypeIntegralWidth();
  bool isSignedInteger = isSigned();
  int64_t defaultIntegerMin =
      getDefaultMinimumForInteger(isSignedInteger, integralWidth);
  int64_t defaultIntegerMax =
      getDefaultMaximumForInteger(isSignedInteger, integralWidth);
  return defaultIntegerMin != getStorageTypeMin() ||
         defaultIntegerMax != getStorageTypeMax();
}

unsigned QuantizedType::getStorageTypeIntegralWidth() const {
  // NOTE: If ever supporting non-integral storage types, some other scheme
  // for determining the width will be needed.
  return static_cast<ImplType *>(impl)->storageType.getIntOrFloatBitWidth();
}

Type QuantizedType::getExpressedType() const {
  return static_cast<ImplType *>(impl)->expressedType;
}

bool QuantizedType::isCompatibleExpressedType(Type candidateExpressedType) {
  if (llvm::isa<ShapedType>(candidateExpressedType)) {
    return llvm::cast<ShapedType>(candidateExpressedType).getElementType() ==
           getExpressedType();
  }
  return candidateExpressedType == getExpressedType();
}

QuantizedType
QuantizedType::getQuantizedElementType(Type primitiveOrContainerType) {
  if (llvm::isa<ShapedType>(primitiveOrContainerType)) {
    Type elementType =
        llvm::cast<ShapedType>(primitiveOrContainerType).getElementType();
    return llvm::dyn_cast<QuantizedType>(elementType);
  }
  return llvm::dyn_cast<QuantizedType>(primitiveOrContainerType);
}

Type QuantizedType::castFromStorageType(Type candidateType) {
  if (candidateType == getStorageType()) {
    // i.e. i32 -> quant<"uniform[i8:f32]{1.0}">
    return *this;
  }
  if (llvm::isa<RankedTensorType>(candidateType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    return RankedTensorType::get(
        llvm::cast<RankedTensorType>(candidateType).getShape(),
        getStorageType());
  }
  if (llvm::isa<UnrankedTensorType>(candidateType)) {
    // i.e. tensor<i8> -> tensor<!quant<"uniform[i8:f32]{1.0}">>
    return UnrankedTensorType::get(getStorageType());
  }
  if (llvm::isa<VectorType>(candidateType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    return VectorType::get(llvm::cast<VectorType>(candidateType).getShape(),
                           getStorageType());
  }

  return nullptr;
}

Type QuantizedType::castToStorageType(Type quantizedType) {
  if (llvm::isa<QuantizedType>(quantizedType)) {
    // i.e. quant<"uniform[i8:f32]{1.0}"> -> i8
    return llvm::cast<QuantizedType>(quantizedType).getStorageType();
  }
  if (llvm::isa<ShapedType>(quantizedType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    ShapedType sType = llvm::cast<ShapedType>(quantizedType);
    if (!llvm::isa<QuantizedType>(sType.getElementType())) {
      return nullptr;
    }
    Type storageType =
        llvm::cast<QuantizedType>(sType.getElementType()).getStorageType();
    if (llvm::isa<RankedTensorType>(quantizedType)) {
      return RankedTensorType::get(sType.getShape(), storageType);
    }
    if (llvm::isa<UnrankedTensorType>(quantizedType)) {
      return UnrankedTensorType::get(storageType);
    }
    if (llvm::isa<VectorType>(quantizedType)) {
      return VectorType::get(sType.getShape(), storageType);
    }
  }

  return nullptr;
}

Type QuantizedType::castFromExpressedType(Type candidateType) {
  if (candidateType == getExpressedType()) {
    // i.e. f32 -> quant<"uniform[i8:f32]{1.0}">
    return *this;
  }
  if (llvm::isa<ShapedType>(candidateType)) {
    ShapedType candidateShapedType = llvm::cast<ShapedType>(candidateType);
    if (candidateShapedType.getElementType() != getExpressedType()) {
      return nullptr;
    }

    if (llvm::isa<RankedTensorType>(candidateType)) {
      // i.e. tensor<4xf32> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
      return RankedTensorType::get(candidateShapedType.getShape(), *this);
    }
    if (llvm::isa<UnrankedTensorType>(candidateType)) {
      // i.e. tensor<xf32> -> tensor<x!quant<"uniform[i8:f32]{1.0}">>
      return UnrankedTensorType::get(*this);
    }
    if (llvm::isa<VectorType>(candidateType)) {
      // i.e. tensor<4xf32> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
      return VectorType::get(candidateShapedType.getShape(), *this);
    }
  }

  return nullptr;
}

Type QuantizedType::castToExpressedType(Type quantizedType) {
  if (llvm::isa<QuantizedType>(quantizedType)) {
    // i.e. quant<"uniform[i8:f32]{1.0}"> -> f32
    return llvm::cast<QuantizedType>(quantizedType).getExpressedType();
  }
  if (llvm::isa<ShapedType>(quantizedType)) {
    // i.e. tensor<4xi8> -> tensor<4x!quant<"uniform[i8:f32]{1.0}">>
    ShapedType sType = llvm::cast<ShapedType>(quantizedType);
    if (!llvm::isa<QuantizedType>(sType.getElementType())) {
      return nullptr;
    }
    Type expressedType =
        llvm::cast<QuantizedType>(sType.getElementType()).getExpressedType();
    if (llvm::isa<RankedTensorType>(quantizedType)) {
      return RankedTensorType::get(sType.getShape(), expressedType);
    }
    if (llvm::isa<UnrankedTensorType>(quantizedType)) {
      return UnrankedTensorType::get(expressedType);
    }
    if (llvm::isa<VectorType>(quantizedType)) {
      return VectorType::get(sType.getShape(), expressedType);
    }
  }

  return nullptr;
}

Type QuantizedType::castExpressedToStorageType(Type candidateType) {
  Type expressedQuantizedType = castFromExpressedType(candidateType);
  if (!expressedQuantizedType) {
    return nullptr;
  }
  return QuantizedType::castToStorageType(expressedQuantizedType);
}

AnyQuantizedType AnyQuantizedType::get(unsigned flags, Type storageType,
                                       Type expressedType,
                                       int64_t storageTypeMin,
                                       int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   storageTypeMin, storageTypeMax);
}

AnyQuantizedType
AnyQuantizedType::getChecked(function_ref<InFlightDiagnostic()> emitError,
                             unsigned flags, Type storageType,
                             Type expressedType, int64_t storageTypeMin,
                             int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, storageTypeMin,
                          storageTypeMax);
}

LogicalResult
AnyQuantizedType::verifyInvariants(function_ref<InFlightDiagnostic()> emitError,
                                   unsigned flags, Type storageType,
                                   Type expressedType, int64_t storageTypeMin,
                                   int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyInvariants(emitError, flags, storageType,
                                             expressedType, storageTypeMin,
                                             storageTypeMax))) {
    return failure();
  }

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (expressedType && !llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  return success();
}

UniformQuantizedType UniformQuantizedType::get(unsigned flags, Type storageType,
                                               Type expressedType, double scale,
                                               int64_t zeroPoint,
                                               int64_t storageTypeMin,
                                               int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   scale, zeroPoint, storageTypeMin, storageTypeMax);
}

UniformQuantizedType UniformQuantizedType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, scale, zeroPoint,
                          storageTypeMin, storageTypeMax);
}

LogicalResult UniformQuantizedType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyInvariants(emitError, flags, storageType,
                                             expressedType, storageTypeMin,
                                             storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitError() << "uniform quantization requires expressed type";

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  // Verify scale.
  if (std::isinf(scale) || std::isnan(scale))
    return emitError() << "illegal scale: " << scale;

  return success();
}

bool UniformQuantizedType::classof(mlir::Type type) {
  return type.getTypeID() == mlir::TypeID::get<UniformQuantizedType>() ||
         type.getTypeID() == mlir::TypeID::get<QuantileQuantizedType>();
}

double UniformQuantizedType::getScale() const { return getImpl()->scale; }

int64_t UniformQuantizedType::getZeroPoint() const {
  return getImpl()->zeroPoint;
}

UniformQuantizedPerAxisType UniformQuantizedPerAxisType::get(
    unsigned flags, Type storageType, Type expressedType,
    ArrayRef<double> scales, ArrayRef<int64_t> zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   scales, zeroPoints, quantizedDimension, storageTypeMin,
                   storageTypeMax);
}

UniformQuantizedPerAxisType UniformQuantizedPerAxisType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, scales, zeroPoints,
                          quantizedDimension, storageTypeMin, storageTypeMax);
}

LogicalResult UniformQuantizedPerAxisType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyInvariants(emitError, flags, storageType,
                                             expressedType, storageTypeMin,
                                             storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitError() << "uniform quantization requires expressed type";

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  // Ensure that the number of scales and zeroPoints match.
  if (scales.size() != zeroPoints.size())
    return emitError() << "illegal number of scales and zeroPoints: "
                       << scales.size() << ", " << zeroPoints.size();

  // Verify scale.
  for (double scale : scales) {
    if (std::isinf(scale) || std::isnan(scale))
      return emitError() << "illegal scale: " << scale;
  }

  // Verify quantized dimension.
  if (quantizedDimension < 0)
    return emitError() << "illegal quantized dimension: " << quantizedDimension;

  return success();
}

bool UniformQuantizedPerAxisType::classof(mlir::Type type) {
  return type.getTypeID() == mlir::TypeID::get<UniformQuantizedPerAxisType>() ||
         type.getTypeID() == mlir::TypeID::get<QuantileQuantizedPerAxisType>();
}

ArrayRef<double> UniformQuantizedPerAxisType::getScales() const {
  return getImpl()->getScales();
}

ArrayRef<int64_t> UniformQuantizedPerAxisType::getZeroPoints() const {
  return getImpl()->getZeroPoints();
}

int32_t UniformQuantizedPerAxisType::getQuantizedDimension() const {
  return getImpl()->quantizedDimension;
}

UniformQuantizedSubChannelType UniformQuantizedSubChannelType::get(
    unsigned flags, Type storageType, Type expressedType,
    DenseElementsAttr scales, DenseElementsAttr zeroPoints,
    ArrayRef<int32_t> quantizedDimensions, ArrayRef<int64_t> blockSizes,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, expressedType,
                   scales, zeroPoints, quantizedDimensions, blockSizes,
                   storageTypeMin, storageTypeMax);
}

UniformQuantizedSubChannelType UniformQuantizedSubChannelType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, DenseElementsAttr scales,
    DenseElementsAttr zeroPoints, ArrayRef<int32_t> quantizedDimensions,
    ArrayRef<int64_t> blockSizes, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, expressedType, scales, zeroPoints,
                          quantizedDimensions, blockSizes, storageTypeMin,
                          storageTypeMax);
}

LogicalResult UniformQuantizedSubChannelType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type expressedType, DenseElementsAttr scales,
    DenseElementsAttr zeroPoints, ArrayRef<int32_t> quantizedDimensions,
    ArrayRef<int64_t> blockSizes, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  if (failed(QuantizedType::verifyInvariants(emitError, flags, storageType,
                                             expressedType, storageTypeMin,
                                             storageTypeMax))) {
    return failure();
  }

  // Uniform quantization requires fully expressed parameters, including
  // expressed type.
  if (!expressedType)
    return emitError() << "uniform quantization requires expressed type";

  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";

  // Verify scale type to match expressedType.
  if (scales.getType().getElementType() != expressedType) {
    return emitError() << "type of scale values "
                       << scales.getType().getElementType()
                       << " must match the expressed type " << expressedType;
  }

  // Verify zero-point type to match storageType.
  if (zeroPoints.getType().getElementType() != storageType) {
    return emitError() << "type of zero point values "
                       << zeroPoints.getType().getElementType()
                       << " must match the storage type " << storageType;
  }

  // Ensure that the shape of scales and zeroPoints match.
  if (scales.getType().getShape() != zeroPoints.getType().getShape())
    return emitError() << "shape of scales and zeroPoints ("
                       << scales.getType().getShape() << " vs "
                       << zeroPoints.getType().getShape() << ") does not match";

  // Ensure that the number of quantized-dimensions and block-sizes match.
  if (quantizedDimensions.size() != blockSizes.size())
    return emitError() << "number of quantized dimensions and block sizes ("
                       << scales.size() << " vs " << zeroPoints.size()
                       << ") does not match";

  // Verify quantized dimension.
  for (auto quantizedDimension : quantizedDimensions) {
    if (quantizedDimension < 0)
      return emitError() << "illegal quantized dimension: "
                         << quantizedDimension;
  }

  // Verify block sizes.
  for (auto blockSize : blockSizes) {
    if (blockSize <= 0)
      return emitError() << "illegal block size: " << blockSize;
  }

  return success();
}

DenseElementsAttr UniformQuantizedSubChannelType::getScales() const {
  return getImpl()->getScales();
}

DenseElementsAttr UniformQuantizedSubChannelType::getZeroPoints() const {
  return getImpl()->getZeroPoints();
}

ArrayRef<int32_t>
UniformQuantizedSubChannelType::getQuantizedDimensions() const {
  return getImpl()->getQuantizedDimensions();
}

ArrayRef<int64_t> UniformQuantizedSubChannelType::getBlockSizes() const {
  return getImpl()->getBlockSizes();
}

const SmallVector<std::pair<int32_t, int64_t>>
UniformQuantizedSubChannelType::getBlockSizeInfo() const {
  SmallVector<std::pair<int32_t, int64_t>> result;
  result.reserve(getQuantizedDimensions().size());

  for (auto [dim, size] :
       llvm::zip(getQuantizedDimensions(), getBlockSizes())) {
    result.push_back({dim, size});
  }

  return result;
}

QuantileQuantizedType
QuantileQuantizedType::get(unsigned flags, Type storageType, Type quantileType,
                           Type expressedType, ArrayRef<double> quantiles,
                           double scale, int64_t zeroPoint,
                           int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, quantileType,
                   expressedType, quantiles, scale, zeroPoint, storageTypeMin,
                   storageTypeMax);
}

QuantileQuantizedType QuantileQuantizedType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type quantileType, Type expressedType,
    ArrayRef<double> quantiles, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, quantileType, expressedType, quantiles,
                          scale, zeroPoint, storageTypeMin, storageTypeMax);
}
LogicalResult QuantileQuantizedType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type quantileType, Type expressedType,
    ArrayRef<double> quantiles, double scale, int64_t zeroPoint,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(UniformQuantizedType::verifyInvariants(emitError, flags, storageType,
                                          expressedType, scale, zeroPoint,
                                          storageTypeMin, storageTypeMax))) {
    return failure();
  }

  unsigned typeWidth{};
  if (storageType.isa<IntegerType>()) {
    typeWidth = llvm::dyn_cast<IntegerType>(storageType).getWidth();
  } else if (mlir::isa<Float8E5M2Type, Float8E4M3FNType, Float4E2M1FNType>(storageType)) {
    // Float8E5M2Type, Float8E4M3FNType and Float4E2M1FNType derive from FloatType.
    typeWidth = llvm::dyn_cast<FloatType>(storageType).getWidth();
  } else {
    return emitError() << "illegal storage type, supported types are: integral "
                          "types, Float8E4M3FNType, Float8E5M2Type and Float4E2M1FNType ";
  }

  const size_t storageTypeRange = storageTypeMax - storageTypeMin + 1;
  const size_t typeWidthSize = 1 << typeWidth;
  const size_t expectedSize =
      (storageTypeRange < typeWidthSize) && !mlir::isa<FloatType>(storageType) ? storageTypeRange : typeWidthSize;

  const auto quantileArraySize = quantiles.size();
  if (quantileArraySize != expectedSize) {
    return emitError() << "quantiles array size needs to be equal to "
                          "2^(bit_size(storageType)), or (storageTypeMax - "
                          "storageTypeMin + 1) when max and min differ from "
                          "the type limits; expected: "
                       << expectedSize << ", found: " << quantileArraySize;
  }

  // Verify quantiles
  for (double quantile : quantiles) {
    if (std::isinf(quantile) || std::isnan(quantile)) {
      return emitError() << "illegal quantile value: " << quantile;
    }
  }

  return success();
}

bool QuantileQuantizedType::classof(mlir::Type type) {
  return type.getTypeID() == mlir::TypeID::get<QuantileQuantizedType>();
}

Type QuantileQuantizedType::getQuantileType() const {
  return getImpl()->quantileType;
}

unsigned QuantileQuantizedType::getQuantileTypeIntegralWidth() const {
  return getImpl()->getQuantileType().getIntOrFloatBitWidth();
}

ArrayRef<double> QuantileQuantizedType::getQuantiles() const {
  return getImpl()->getQuantiles();
}

QuantileQuantizedPerAxisType QuantileQuantizedPerAxisType::get(
    unsigned flags, Type storageType, Type quantileType, Type expressedType,
    ArrayRef<double> quantiles, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::get(storageType.getContext(), flags, storageType, quantileType,
                   expressedType, quantiles, scales, zeroPoints,
                   quantizedDimension, storageTypeMin, storageTypeMax);
}

QuantileQuantizedPerAxisType QuantileQuantizedPerAxisType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type quantileType, Type expressedType,
    ArrayRef<double> quantiles, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  return Base::getChecked(emitError, storageType.getContext(), flags,
                          storageType, quantileType, expressedType, quantiles,
                          scales, zeroPoints, quantizedDimension,
                          storageTypeMin, storageTypeMax);
}

LogicalResult QuantileQuantizedPerAxisType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, unsigned flags,
    Type storageType, Type quantileType, Type expressedType,
    ArrayRef<double> quantiles, ArrayRef<double> scales,
    ArrayRef<int64_t> zeroPoints, int32_t quantizedDimension,
    int64_t storageTypeMin, int64_t storageTypeMax) {
  if (failed(UniformQuantizedPerAxisType::verifyInvariants(
          emitError, flags, storageType, expressedType, scales, zeroPoints,
          quantizedDimension, storageTypeMin, storageTypeMax))) {
    return failure();
  }

  unsigned typeWidth{};
  if (storageType.isa<IntegerType>()) {
    typeWidth = llvm::dyn_cast<IntegerType>(storageType).getWidth();
  } else if (mlir::isa<Float8E5M2Type, Float8E4M3FNType, Float4E2M1FNType>(storageType)) {
    // Float8E5M2Type, Float8E4M3FNType and Float4E2M1FNType derive from FloatType.
    typeWidth = llvm::dyn_cast<FloatType>(storageType).getWidth();
  } else {
    return emitError() << "illegal storage type, supported types are: integral "
                          "types, Float8E4M3FNType, Float8E5M2Type and Float4E2M1FNType ";
  }

  const size_t storageTypeRange = storageTypeMax - storageTypeMin + 1;
  const size_t typeWidthSize = 1 << typeWidth;
  const size_t expectedSize =
      (storageTypeRange < typeWidthSize) && !mlir::isa<FloatType>(storageType) ? storageTypeRange : typeWidthSize;

  const auto quantileArraySize = quantiles.size();
  if (quantileArraySize != expectedSize) {
    return emitError() << "quantiles array size needs to be equal to "
                          "2^(bit_size(storageType)), or (storageTypeMax - "
                          "storageTypeMin + 1) when max and min differ from "
                          "the type limits; expected: "
                       << expectedSize << ", found: " << quantileArraySize;
  }

  // Verify quantiles
  for (double quantile : quantiles) {
    if (std::isinf(quantile) || std::isnan(quantile)) {
      return emitError() << "illegal quantile value: " << quantile;
    }
  }

  return success();
}

bool QuantileQuantizedPerAxisType::classof(mlir::Type type) {
  return type.getTypeID() == mlir::TypeID::get<QuantileQuantizedPerAxisType>();
}

Type QuantileQuantizedPerAxisType::getQuantileType() const {
  return getImpl()->quantileType;
}

unsigned QuantileQuantizedPerAxisType::getQuantileTypeIntegralWidth() const {
  return getImpl()->getQuantileType().getIntOrFloatBitWidth();
}

ArrayRef<double> QuantileQuantizedPerAxisType::getQuantiles() const {
  return getImpl()->getQuantiles();
}

CalibratedQuantizedType CalibratedQuantizedType::get(Type expressedType,
                                                     double min, double max) {
  return Base::get(expressedType.getContext(), expressedType, min, max);
}

CalibratedQuantizedType CalibratedQuantizedType::getChecked(
    function_ref<InFlightDiagnostic()> emitError, Type expressedType,
    double min, double max) {
  return Base::getChecked(emitError, expressedType.getContext(), expressedType,
                          min, max);
}

LogicalResult CalibratedQuantizedType::verifyInvariants(
    function_ref<InFlightDiagnostic()> emitError, Type expressedType,
    double min, double max) {
  // Verify that the expressed type is floating point.
  // If this restriction is ever eliminated, the parser/printer must be
  // extended.
  if (!llvm::isa<FloatType>(expressedType))
    return emitError() << "expressed type must be floating point";
  if (max <= min)
    return emitError() << "illegal min and max: (" << min << ":" << max << ")";

  return success();
}

double CalibratedQuantizedType::getMin() const { return getImpl()->min; }

double CalibratedQuantizedType::getMax() const { return getImpl()->max; }
