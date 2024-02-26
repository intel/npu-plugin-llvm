//===- BufferizationUtils.cpp ---- Utilities for bufferization-------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements (one-shot) bufferization utilities.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/BufferizationUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

namespace mlir {
Type BufferizerInterface::getTensorTypeFromMemRefType(Type type) const {
  Dialect *dialect = &type.getDialect();
  const auto *handle = getInterfaceFor(dialect);
  assert(handle && "The dialect must have BufferizerInterface implemented.");
  return handle->getTensorTypeFromMemRefType(type);
}
} // namespace mlir
