//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Bufferization/Extensions/AllExtensions.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/BufferizationUtils.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// Default (one-shot) bufferization interface for the builtin dialect
struct BuiltinBufferizerInterface : DialectBufferizerInterface {
  using DialectBufferizerInterface::DialectBufferizerInterface;

  Type getTensorTypeFromMemRefType(Type type) const override {
    return memref::getTensorTypeFromMemRefType(type);
  }
};
} // namespace

void mlir::bufferization::registerAllExtensions(DialectRegistry &registry) {
  registerBufferizerExtensionForBuiltinDialect(registry);
}

void mlir::bufferization::registerBufferizerExtensionForBuiltinDialect(
    DialectRegistry &registry) {
  // default one-shot bufferization interface on *builtin* dialect
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterfaces<BuiltinBufferizerInterface>();
  });
}
