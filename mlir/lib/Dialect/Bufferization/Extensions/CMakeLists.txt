add_mlir_dialect_library(MLIRBufferizationAllExtensions
  AllExtensions.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Bufferization/Extensions

  LINK_LIBS PUBLIC
  MLIRDialect
  MLIRIR
  MLIRTransformUtils
  )
