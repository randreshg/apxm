# Collect CAPI-specific sources
file(GLOB APXM_CAPI_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

# Add parser sources that were collected by Parser/CMakeLists.txt
list(APPEND APXM_CAPI_SOURCES ${APXM_PARSER_SOURCES})

# Add artifact conversion sources
file(GLOB APXM_ARTIFACT_CONVERSION_SOURCES CONFIGURE_DEPENDS
  ${APXM_MLIR_DIR}/lib/Dialect/AIS/Conversion/Artifact/*.cpp
)
list(APPEND APXM_CAPI_SOURCES ${APXM_ARTIFACT_CONVERSION_SOURCES})


add_library(apxm_compiler_c SHARED ${APXM_CAPI_SOURCES})

target_compile_definitions(apxm_compiler_c
  PRIVATE
  APXM_CAPI_INTERNAL
  APXM_HAS_CPP_DSL=1
)

target_include_directories(apxm_compiler_c
  PUBLIC
  ${APXM_INCLUDE_DIR}
  ${APXM_GENERATED_INCLUDE_DIR}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${LLVM_INCLUDE_DIRS}
  ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(apxm_compiler_c
  PRIVATE
  APXMAISIR
  APXMAISTransforms
  MLIRArithDialect
  MLIRIR
  MLIRFuncDialect
  MLIRParser
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
)

# Link optional libraries if they exist
if(TARGET APXMParser)
  target_link_libraries(apxm_compiler_c PRIVATE APXMParser)
endif()

if(APXM_RUNTIME_RPATHS)
  set_target_properties(apxm_compiler_c
    PROPERTIES
      BUILD_RPATH "${APXM_RUNTIME_RPATHS}"
      INSTALL_RPATH "${APXM_RUNTIME_RPATHS}"
  )
endif()

install(TARGETS apxm_compiler_c
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
)
