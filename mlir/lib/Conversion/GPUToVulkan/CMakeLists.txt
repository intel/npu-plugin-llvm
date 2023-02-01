if(MLIR_DIALECT_SPIRV_ENABLE)
  add_mlir_conversion_library(MLIRGPUToVulkanTransforms
    ConvertLaunchFuncToVulkanCalls.cpp
    ConvertGPULaunchFuncToVulkanLaunchFunc.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSPIRVDialect
  MLIRSPIRVSerialization
  MLIRSupport
  MLIRTransforms
  MLIRTranslateLib
  )
else()
  add_mlir_conversion_library(MLIRGPUToVulkanTransforms
    ConvertLaunchFuncToVulkanCalls.cpp
    ConvertGPULaunchFuncToVulkanLaunchFunc.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSupport
  MLIRTransforms
  MLIRTranslateLib
  )
endif()
