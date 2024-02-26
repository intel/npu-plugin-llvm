//===- Passes.h - MLIR Test Dialect pass entrypoints ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TEST_PASSES_H
#define MLIR_DIALECT_TEST_PASSES_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace test {

#define GEN_PASS_DECL
#include "Passes.h.inc"

std::unique_ptr<mlir::Pass> createTestBufferizePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

// special registration API for mlir-opt (that implicitly calls this function)
void registerAllTestDialectPasses();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace test

#endif // MLIR_DIALECT_TEST_PASSES_H
