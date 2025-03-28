//===- BufferizationUtils.cpp - Unit Tests for Bufferization Utils         ===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/BufferizationUtils.h"
#include "mlir/Dialect/Bufferization/Extensions/AllExtensions.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/TensorEncoding.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
struct TestTensorAttr : public StringAttr {
  using mlir::StringAttr::StringAttr;

  static bool classof(mlir::Attribute attr) {
    return mlir::isa<mlir::StringAttr>(attr);
  }

  static TestTensorAttr fromStringAttr(StringAttr attr) {
    return mlir::dyn_cast<TestTensorAttr>(attr);
  }
};

class TestTensorEncodingVerifier final
    : public mlir::VerifiableTensorEncoding::ExternalModel<
          TestTensorEncodingVerifier, TestTensorAttr> {
public:
  using ConcreteEntity = mlir::StringAttr;

  mlir::LogicalResult verifyEncoding(
      mlir::Attribute attr, mlir::ArrayRef<int64_t> shape, mlir::Type,
      mlir::function_ref<mlir::InFlightDiagnostic()> emitError) const {
    std::ignore = shape;

    if (mlir::isa<TestTensorAttr>(attr)) {
      return mlir::success();
    }
    return emitError() << "Unknown Tensor enconding: " << attr;
  }
};

struct TestMemRefAttr : public mlir::StringAttr {
  using mlir::StringAttr::StringAttr;

  static bool classof(mlir::Attribute attr) {
    return mlir::isa<mlir::StringAttr>(attr);
  }
};

class TestMemRefAttrLayout final
    : public mlir::MemRefLayoutAttrInterface::ExternalModel<
          TestMemRefAttrLayout, TestMemRefAttr> {
public:
  using ConcreteEntity = mlir::StringAttr;

  bool isIdentity(mlir::Attribute) const { return true; }
  mlir::AffineMap getAffineMap(mlir::Attribute attr) const {
    return mlir::AffineMap::getMultiDimIdentityMap(1, attr.getContext());
  }
  mlir::LogicalResult
  verifyLayout(mlir::Attribute attr, mlir::ArrayRef<int64_t> shape,
               mlir::function_ref<mlir::InFlightDiagnostic()> emitError) const {
    std::ignore = shape;

    if (mlir::isa<TestMemRefAttr>(attr)) {
      return mlir::success();
    }
    return emitError() << "Unknown MemRef layout: " << attr;
  }
};

struct CustomBuiltinBufferizerInterface : DialectBufferizerInterface {
  using DialectBufferizerInterface::DialectBufferizerInterface;

  Type getTensorTypeFromMemRefType(Type type) const override {
    // propagate memref's layout back to tensor's encoding
    if (auto memref = llvm::dyn_cast<MemRefType>(type)) {
      TestTensorAttr encoding = nullptr;
      if (auto layout = llvm::dyn_cast<TestMemRefAttr>(memref.getLayout())) {
        encoding = TestTensorAttr::fromStringAttr(layout);
      }
      return RankedTensorType::get(memref.getShape(), memref.getElementType(),
                                   encoding);
    }

    if (auto memref = llvm::dyn_cast<UnrankedMemRefType>(type)) {
      return UnrankedTensorType::get(memref.getElementType());
    }
    return NoneType::get(type.getContext());
  }
};
} // namespace

class BufferizerInterfaceTest : public ::testing::Test {
protected:
  MLIRContext ctx;

  BufferizerInterfaceTest() {
    ctx.loadDialect<BuiltinDialect>();
    ctx.loadDialect<func::FuncDialect>();
    ctx.loadDialect<bufferization::BufferizationDialect>();

    DialectRegistry registry;
    registry.addExtension(+[](mlir::MLIRContext *ctx, BuiltinDialect *) {
      TestTensorAttr::attachInterface<TestTensorEncodingVerifier>(*ctx);
      TestMemRefAttr::attachInterface<TestMemRefAttrLayout>(*ctx);
    });
    ctx.appendDialectRegistry(registry);
  }
};

const char *const extendedBuiltinsCode = R"mlir(
func.func @foo(%t : tensor<1x2x3xf64, "hello">) -> memref<1x2x3xf64, "hello"> {
    %m = bufferization.to_memref %t : memref<1x2x3xf64, "hello">
    return %m : memref<1x2x3xf64, "hello">
}
)mlir";

TEST_F(BufferizerInterfaceTest, TestDefaultBuiltinBufferizer) {
  DialectRegistry registry;
  bufferization::registerAllExtensions(registry);
  ctx.appendDialectRegistry(registry);

  OwningOpRef<ModuleOp> res =
      parseSourceString<ModuleOp>(extendedBuiltinsCode, &ctx);
  ASSERT_FALSE(res);
}

TEST_F(BufferizerInterfaceTest, TestCustomBuiltinBufferizer) {
  DialectRegistry registry;
  registry.addExtension(+[](MLIRContext *ctx, BuiltinDialect *dialect) {
    dialect->addInterfaces<CustomBuiltinBufferizerInterface>();
  });
  ctx.appendDialectRegistry(registry);

  // MLIR parsing must succeed for tensor with custom encoding and memref with
  // custom layout because a custom DialectBufferizerInterface is used (it would
  // copy the layout back to the encoding during type checking).
  OwningOpRef<ModuleOp> res =
      parseSourceString<ModuleOp>(extendedBuiltinsCode, &ctx);
  ASSERT_TRUE(res);
}
