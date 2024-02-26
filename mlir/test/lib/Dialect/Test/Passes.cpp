//===-- Passes.cpp - MLIR Test Dialect pass definitions ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Passes.h"
#include "TestDialect.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferizationUtils.h"

namespace test {
#define GEN_PASS_DEF_TESTBUFFERIZE
#include "Passes.h.inc"
} // namespace test

using namespace mlir;
using namespace test;

namespace {
// Use this pass to test that one-shot bufferization works in general for custom
// ops and types
//
// TODO: a special pass is used *instead* of -one-shot-bufferize because:
//  * one-shot-bufferize does not support function boundary bufferization with
//    custom tensors / memrefs
//  * one-shot-bufferize would (practically) unconditionally insert tensor
//    copies (with / without analysis) and this insertion procedure does not
//    support custom tensors / memrefs
struct TestBufferizerPass : test::impl::TestBufferizeBase<TestBufferizerPass> {
  using Base::Base;

  void runOnOperation() final {
    bufferization::BufferizationOptions options =
        bufferization::getPartialBufferizationOptions();
    options.opFilter.allowDialect<test::TestDialect>();

    // FIXME: disable tensor copy insertions
    options.copyBeforeWrite = false;
    if (failed(bufferization::bufferizeOp(getOperation(), options,
                                          options.copyBeforeWrite))) {
      signalPassFailure();
    }
  }
};
} // namespace

void test::registerAllTestDialectPasses() { test::registerTestPasses(); }

std::unique_ptr<Pass> test::createTestBufferizePass() {
  return std::make_unique<TestBufferizerPass>();
}
