//===- AllExtensions.h - Bufferization dialect extensions -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_EXTENSIONS_ALLEXTENSIONS_H_
#define MLIR_DIALECT_BUFFERIZATION_EXTENSIONS_ALLEXTENSIONS_H_

namespace mlir {
class DialectRegistry;

namespace bufferization {
/// Register all extensions of the bufferization dialect. This should generally
/// only be used by tools, or other use cases that really do want *all*
/// extensions of the dialect. All other cases should prefer to instead register
/// the specific extensions they intend to take advantage of.
void registerAllExtensions(DialectRegistry &registry);

/// Register the extension used to support one-shot bufferization for *builtin*
/// types. This is a special function since it extends the Builtin dialect, not
/// Bufferization dialect.
void registerBufferizerExtensionForBuiltinDialect(DialectRegistry &registry);
} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_EXTENSIONS_ALLEXTENSIONS_H_
