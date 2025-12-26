cmake_minimum_required(VERSION 3.20.0)
project(apxm-compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(APXM_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(APXM_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# The MLIR directory contains include/ and lib/ (CARTS-style layout).
# Use the source include tree and a build-tree generated include dir.
set(APXM_MLIR_DIR ${APXM_SOURCE_DIR}/mlir)
set(APXM_INCLUDE_DIR ${APXM_MLIR_DIR}/include)
set(APXM_GENERATED_INCLUDE_DIR ${APXM_BINARY_DIR}/include)

# Expose LLVM/MLIR include dirs and AIS include dirs without hardcoding file lists.
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${APXM_INCLUDE_DIR})
include_directories(${APXM_GENERATED_INCLUDE_DIR})

separate_arguments(LLVM_DEFINITIONS_LIST NATIVE_COMMAND ${LLVM_DEFINITIONS})
add_definitions(${LLVM_DEFINITIONS_LIST})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(mlir/lib)
