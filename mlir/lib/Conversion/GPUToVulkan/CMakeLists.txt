if(MLIR_DIALECT_SPIRV_ENABLE)
  add_mlir_conversion_library(MLIRGPUToVulkanTransforms
    ConvertLaunchFuncToVulkanCalls.cpp
    ConvertGPULaunchFuncToVulkanLaunchFunc.cpp

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRGPUOps
    MLIRIR
    MLIRLLVMIR
    MLIRPass
    MLIRSPIRV
    MLIRSPIRVSerialization
    MLIRStandard
    MLIRSupport
    MLIRTransforms
    MLIRTranslation
    )
else()
  add_mlir_conversion_library(MLIRGPUToVulkanTransforms
    ConvertLaunchFuncToVulkanCalls.cpp
    ConvertGPULaunchFuncToVulkanLaunchFunc.cpp

    DEPENDS
    MLIRConversionPassIncGen

    LINK_LIBS PUBLIC
    MLIRGPUOps
    MLIRIR
    MLIRLLVMIR
    MLIRPass
    MLIRStandard
    MLIRSupport
    MLIRTransforms
    MLIRTranslation
    )
endif()
