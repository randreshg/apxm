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
#include "mlir/IR/Diagnostics.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ais/Dialect/AIS/IR/AISDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

struct ApxmCompilerContext {
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  mlir::DiagnosticEngine::HandlerID diagnostic_handler_id;

  ApxmCompilerContext() {
    mlir_context = std::make_unique<mlir::MLIRContext>();
    // Register required dialects used across the C API implementation
    mlir_context->loadDialect<mlir::ais::AISDialect>();
    mlir_context->loadDialect<mlir::func::FuncDialect>();
    mlir_context->allowUnregisteredDialects(true);

    // Register diagnostic handler to print warnings and errors to stderr
    diagnostic_handler_id = mlir_context->getDiagEngine().registerHandler(
        [](mlir::Diagnostic &diag) {
          auto severity = diag.getSeverity();
          llvm::raw_ostream &os = llvm::errs();

          // Format: severity: message
          switch (severity) {
          case mlir::DiagnosticSeverity::Error:
            os << "error: ";
            break;
          case mlir::DiagnosticSeverity::Warning:
            os << "warning: ";
            break;
          case mlir::DiagnosticSeverity::Note:
            os << "note: ";
            break;
          case mlir::DiagnosticSeverity::Remark:
            os << "remark: ";
            break;
          }

          os << diag.str() << "\n";

          // Print location if available
          auto loc = diag.getLocation();
          if (auto fileLoc = mlir::dyn_cast<mlir::FileLineColLoc>(loc)) {
            os << "  --> " << fileLoc.getFilename() << ":"
               << fileLoc.getLine() << ":" << fileLoc.getColumn() << "\n";
          }

          return mlir::success();
        });
  }

  ~ApxmCompilerContext() {
    if (mlir_context) {
      mlir_context->getDiagEngine().eraseHandler(diagnostic_handler_id);
    }
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
