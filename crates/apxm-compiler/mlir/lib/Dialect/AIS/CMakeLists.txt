set(APXM_AIS_GENERATED_INCLUDE_DIR ${APXM_GENERATED_INCLUDE_DIR})
file(MAKE_DIRECTORY ${APXM_AIS_GENERATED_INCLUDE_DIR}/ais/Dialect/AIS/IR)
file(MAKE_DIRECTORY ${APXM_AIS_GENERATED_INCLUDE_DIR}/ais/Dialect/AIS/Transforms)

file(RELATIVE_PATH APXM_AIS_TABLEGEN_OUTPUT_PREFIX
  ${CMAKE_CURRENT_BINARY_DIR}
  ${APXM_AIS_GENERATED_INCLUDE_DIR}
)

# Glob TableGen files instead of listing them individually
file(GLOB APXM_AIS_IR_TD_FILES CONFIGURE_DEPENDS
  ${APXM_INCLUDE_DIR}/ais/Dialect/AIS/IR/*.td
)

set(LLVM_TARGET_DEFINITIONS ${APXM_INCLUDE_DIR}/ais/Dialect/AIS/IR/AIS.td)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISDialect.h.inc -gen-dialect-decls -dialect=ais
  DEPENDS ${APXM_AIS_IR_TD_FILES}
)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISDialect.cpp.inc -gen-dialect-defs -dialect=ais
  DEPENDS ${APXM_AIS_IR_TD_FILES}
)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISOps.h.inc -gen-op-decls
  DEPENDS ${APXM_AIS_IR_TD_FILES}
)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISOps.cpp.inc -gen-op-defs
  DEPENDS ${APXM_AIS_IR_TD_FILES}
)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISTypes.h.inc -gen-typedef-decls -typedefs-dialect=ais
  DEPENDS ${APXM_AIS_IR_TD_FILES}
)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ais
  DEPENDS ${APXM_AIS_IR_TD_FILES}
)
add_public_tablegen_target(AISIRIncGen)

set(LLVM_TARGET_DEFINITIONS ${APXM_INCLUDE_DIR}/ais/Dialect/AIS/IR/AISAttributes.td)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISEnums.h.inc -gen-enum-decls)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=ais)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/IR/AISAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=ais)
add_public_tablegen_target(AISAttrIncGen)

set(LLVM_TARGET_DEFINITIONS ${APXM_INCLUDE_DIR}/ais/Dialect/AIS/Transforms/Passes.td)
mlir_tablegen(${APXM_AIS_TABLEGEN_OUTPUT_PREFIX}/ais/Dialect/AIS/Transforms/Passes.h.inc -gen-pass-decls -name AIS)
add_public_tablegen_target(AISPassIncGen)

file(GLOB APXMAISIR_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/IR/*.cpp
)

add_mlir_dialect_library(APXMAISIR
  ${APXMAISIR_SOURCES}

  ADDITIONAL_HEADER_DIRS
  ${APXM_INCLUDE_DIR}
  ${APXM_AIS_GENERATED_INCLUDE_DIR}

  DEPENDS
  AISIRIncGen
  AISAttrIncGen

  LINK_LIBS
  MLIRIR
  MLIRParser
  MLIRSupport
)

file(GLOB APXMAISTRANS_SOURCES CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms/*.cpp
)

add_mlir_library(APXMAISTransforms
  ${APXMAISTRANS_SOURCES}

  ADDITIONAL_HEADER_DIRS
  ${APXM_INCLUDE_DIR}
  ${APXM_AIS_GENERATED_INCLUDE_DIR}

  DEPENDS
  AISPassIncGen
  AISIRIncGen

  LINK_LIBS
  APXMAISIR
  MLIRIR
  MLIRPass
  MLIRTransforms
)

set_target_properties(APXMAISIR PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(APXMAISTransforms PROPERTIES POSITION_INDEPENDENT_CODE ON)
