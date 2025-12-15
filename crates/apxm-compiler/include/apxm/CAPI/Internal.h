/*
 * @file Internal.h
 * @brief Internal C++ definitions
 *
 * When building the C++ implementation of the C API (internal builds),
 * translation units require the concrete definitions of the opaque types.
 * Define APXM_CAPI_INTERNAL (e.g. in the CMake target for apxm_compiler_c)
 * to enable these definitions. The guards ensure public C consumers still
 * see only opaque handles.
 */
#if defined(__cplusplus) && defined(APXM_CAPI_INTERNAL)

#include <memory>
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "apxm/Dialect/AIS/IR/AISDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"

/* Make the AIS Rust codegen options type available to implementation
 * translation units that include this internal header. This allows files
 * that build the C API (which include the internal header via Types.h)
 * to see the declaration of `mlir::ais::RustCodegenOptions` without
 * adding includes in many individual .cpp files.
 */
#include "apxm/Dialect/AIS/Conversion/Rust/AISToRust.h"

struct ApxmCompilerContext {
  std::unique_ptr<mlir::MLIRContext> mlir_context;

  ApxmCompilerContext() {
    mlir_context = std::make_unique<mlir::MLIRContext>();
    // Register required dialects used across the C API implementation
    mlir_context->loadDialect<mlir::ais::AISDialect>();
    mlir_context->loadDialect<mlir::func::FuncDialect>();
    mlir_context->loadDialect<mlir::async::AsyncDialect>();
    mlir_context->allowUnregisteredDialects(true);
  }
};

struct ApxmModule {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  ApxmCompilerContext* context;

  ApxmModule(ApxmCompilerContext* ctx, mlir::ModuleOp mod)
    : module(mod), context(ctx) {}
};

struct ApxmPassManager {
  std::unique_ptr<mlir::PassManager> pass_manager;
  ApxmCompilerContext* context;
  llvm::SmallPtrSet<void*, 8> registered_passes;

  ApxmPassManager(ApxmCompilerContext* ctx)
    : pass_manager(std::make_unique<mlir::PassManager>(ctx->mlir_context.get())),
      context(ctx) {}
};

#endif // defined(__cplusplus) && defined(APXM_CAPI_INTERNAL)
