//===- BufferizationUtils.h - One-Shot Bufferization utilities --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines interfaces for various (one-shot) bufferization
// utility methods.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_BUFFERIZATIONUTILS_H
#define MLIR_TRANSFORMS_BUFFERIZATIONUTILS_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectInterface.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// BufferizerInterface
//===----------------------------------------------------------------------===//

/// This is the interface that must be implemented by the dialects of memref /
/// tensor types to be bufferized. Note that the default implementation for the
/// builtin dialect is provided via the bufferization dialect.
struct DialectBufferizerInterface
    : DialectInterface::Base<DialectBufferizerInterface> {
  DialectBufferizerInterface(Dialect *dialect) : Base(dialect) {}

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  /// Hook to customize the behavior of memref -> tensor conversion. Returns a
  /// tensor type for the specified memref type.
  virtual Type getTensorTypeFromMemRefType(Type type) const = 0;
};

/// This wrapper automatically collects DialectBufferizerInterface
/// implementations from all registered dialects.
struct BufferizerInterface
    : DialectInterfaceCollection<DialectBufferizerInterface> {
  using Base::Base;

  /// Dispatches to DialectBufferizerInterface::getTensorTypeFromMemRefType() of
  /// the dialect of the specified type.
  Type getTensorTypeFromMemRefType(Type type) const;
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_BUFFERIZATIONUTILS_H
