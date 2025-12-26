/**
 * @file  Types.cpp
 * @brief C++ glue for the public C API types.
 *
 * Owns the single `ApxmCompilerContext` that every other object keeps a
 * pointer to.  The context is just a thin wrapper around an MLIRContext
 * with the AIS, Func and Async dialects pre-loaded.
 */

#include "ais/CAPI/Types.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "ais/Dialect/AIS/IR/AISDialect.h"

extern "C" {

ApxmCompilerContext *apxm_compiler_context_create() {
  return new (std::nothrow) ApxmCompilerContext();
}

void apxm_compiler_context_destroy(ApxmCompilerContext *ctx) {
  delete ctx;
}

void apxm_string_free(char *str) {
  free(str);
}

 } // extern "C"
