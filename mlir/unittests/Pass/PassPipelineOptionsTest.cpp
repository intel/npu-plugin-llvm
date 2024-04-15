//===- PassPipelineParserTest.cpp - Pass Parser unit tests ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "gtest/gtest.h"

#include <memory>

using namespace mlir;
using namespace mlir::detail;

namespace {

// these types are used for automatically generated code of pass
using StrPassOpt = ::mlir::Pass::Option<std::string>;
using IntPassOpt = ::mlir::Pass::Option<int>;
using BoolPassOpt = ::mlir::Pass::Option<bool>;

// these types are used for pipeline options that we manually pass to the constructor
using StrOption = mlir::detail::PassOptions::Option<std::string>;
using IntOption = mlir::detail::PassOptions::Option<int>;
using BoolOption = mlir::detail::PassOptions::Option<bool>;

const int intOptDefaultVal = 5;
const bool boolOptDefaultVal = true;

struct SimplePassWithOptions
    : public PassWrapper<SimplePassWithOptions, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimplePassWithOptions)
  
  SimplePassWithOptions() = default;
  SimplePassWithOptions(const SimplePassWithOptions &other) : PassWrapper(other) {}
  
  SimplePassWithOptions(const detail::PassOptions& options) {
      copyOptionValuesFrom(options);
  }

  LogicalResult initialize(MLIRContext *ctx) final {
    return success();
  }

  void runOnOperation() override { }

public:
  StrPassOpt strOpt{*this, "str-opt", ::llvm::cl::desc("string test option"), llvm::cl::init("")};
  IntPassOpt intOpt{*this, "int-opt", ::llvm::cl::desc("int test option"), llvm::cl::init(intOptDefaultVal)};
  BoolPassOpt boolOpt{*this, "bool-opt", ::llvm::cl::desc("bool test option"), llvm::cl::init(boolOptDefaultVal)};
};

TEST(PassPipelineOptionsTest, CopyAllOptions) {
  struct DuplicatedOtions : ::mlir::PassPipelineOptions<DuplicatedOtions> {
    StrOption strOpt{*this, "str-opt", ::llvm::cl::desc("string test option")};
    IntOption intOpt{*this, "int-opt", ::llvm::cl::desc("int test option"), llvm::cl::init(intOptDefaultVal)};
    BoolOption boolOpt{*this, "bool-opt", ::llvm::cl::desc("bool test option"), llvm::cl::init(boolOptDefaultVal)};
  };

  const auto expectedStrVal = "test1";
  const auto expectedIntVal = -intOptDefaultVal;
  const auto expectedBoolVal = !boolOptDefaultVal;

  DuplicatedOtions options;
  options.strOpt.setValue(expectedStrVal);  
  options.intOpt.setValue(expectedIntVal);  
  options.boolOpt.setValue(expectedBoolVal);  

  const auto& pass = std::make_unique<SimplePassWithOptions>(options);

  EXPECT_EQ(pass->strOpt.getValue(), expectedStrVal);
  EXPECT_EQ(pass->intOpt.getValue(), expectedIntVal);
  EXPECT_EQ(pass->boolOpt.getValue(), expectedBoolVal);
}

TEST(PassPipelineOptionsTest, CopyMatchedOptions) {
  struct SomePipelineOptions : ::mlir::PassPipelineOptions<SomePipelineOptions> {
    StrOption strOpt{*this, "str-opt", ::llvm::cl::desc("string test option")};
    IntOption intOpt{*this, "int-opt", ::llvm::cl::desc("int test option")};
    StrOption anotherStrOpt{*this, "another-str-pipeline-opt", 
                      ::llvm::cl::desc("there is no such option in SimplePassWithOptions"), llvm::cl::init("anotherOptVal")};
    IntOption anotherIntOpt{*this, "another-int-pipeline-opt", 
                      ::llvm::cl::desc("there is no such option in SimplePassWithOptions"), llvm::cl::init(10)};
  };

  const auto expectedStrVal = "test2";
  const auto expectedIntVal = -intOptDefaultVal;

  SomePipelineOptions options;
  options.strOpt.setValue(expectedStrVal);  
  options.intOpt.setValue(expectedIntVal);  

  const auto pass = std::make_unique<SimplePassWithOptions>(options);

  EXPECT_EQ(pass->strOpt.getValue(), expectedStrVal);
  EXPECT_EQ(pass->intOpt.getValue(), expectedIntVal);
  EXPECT_EQ(pass->boolOpt.getValue(), boolOptDefaultVal);
}

TEST(PassPipelineOptionsTest, NoMatchedOptions) {
  struct SomePipelineOptions : ::mlir::PassPipelineOptions<SomePipelineOptions> {
    StrOption anotherStrOpt{*this, "another-str-pipeline-opt", 
                      ::llvm::cl::desc("there is no such option in SimplePassWithOptions"), llvm::cl::init("anotherOptVal")};
    IntOption anotherIntOpt{*this, "another-int-pipeline-opt", 
                      ::llvm::cl::desc("there is no such option in SimplePassWithOptions"), llvm::cl::init(10)};
  };

  SomePipelineOptions options;
  const auto pass = std::make_unique<SimplePassWithOptions>(options);

  EXPECT_EQ(pass->strOpt.getValue(), "");
  EXPECT_EQ(pass->intOpt.getValue(), intOptDefaultVal);
  EXPECT_EQ(pass->boolOpt.getValue(), boolOptDefaultVal);
}

} // namespace
