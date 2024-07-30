//===- PrintIR.cpp - Pass to dump IR on debug stream ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"

namespace mlir {
namespace {

#define GEN_PASS_DEF_PRINTIRPASS
#include "mlir/Transforms/Passes.h.inc"

struct PrintIRPass : public impl::PrintIRPassBase<PrintIRPass> {
  PrintIRPass(const PrintIRPassOptions &options) : PrintIRPassBase(options) {}

  PrintIRPass(const PrintIRPassOptions &options, OpPrintingFlags printingFlags)
      : PrintIRPassBase(options), printingFlags(printingFlags) {}

  void runOnOperation() override {
    if (this->file.empty()) {
      printIRTo(llvm::errs());
      return;
    }

    std::error_code EC;
    llvm::raw_fd_ostream stream(this->file, EC, llvm::sys::fs::OF_Append);

    if (EC) {
      llvm::errs() << "Could not open file: " << EC.message();
      signalPassFailure();
      return;
    }

    printIRTo(stream);
  }

private:
  void printIRTo(llvm::raw_ostream &stream) {
    stream << "// -----// IR Dump";
    if (!this->label.empty())
      stream << " " << this->label;
    stream << " //----- //\n";

    getOperation()->print(stream, printingFlags);
  }

  OpPrintingFlags printingFlags = std::nullopt;
};

} // namespace

std::unique_ptr<Pass> createPrintIRPass(const PrintIRPassOptions &options,
                                        OpPrintingFlags printingFlags) {
  return std::make_unique<PrintIRPass>(options, printingFlags);
}

} // namespace mlir