if(MLIR_DIALECT_XEGPU_ENABLE)
  add_mlir_conversion_library(MLIRVectorToXeGPU
    VectorToXeGPU.cpp

    ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToXeGPU

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRArithDialect
    MLIRMemRefDialect
    MLIRTransforms
    MLIRVectorDialect
    MLIRXeGPUDialect
    )
endif()
