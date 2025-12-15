/*
 * @file MLIRGen.cpp
 * @brief Core infrastructure for MLIR generation
 *
 * MLIRGen does the following:
 * - Initializes the MLIR context and builder.
 * - Loads required dialects.
 * - Provides methods to generate MLIR operations and types.
 */

#include "apxm/Parser/MLIR/MLIRGen.h"
#include "apxm/Parser/MLIR/MLIRGenExpressions.h"
#include "apxm/Parser/MLIR/MLIRGenStatement.h"
#include "apxm/Dialect/AIS/IR/AISDialect.h"
#include "apxm/Dialect/AIS/IR/AISOps.h"
#include "apxm/Dialect/AIS/IR/AISTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace apxm::parser;
using namespace mlir;
using namespace mlir::ais;

MLIRGen::MLIRGen(mlir::MLIRContext &context, llvm::SourceMgr &sourceMgr)
    : context(context), builder(&context), srcMgr(sourceMgr) {
  // Load required dialects
  context.loadDialect<mlir::ais::AISDialect>();
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.loadDialect<mlir::scf::SCFDialect>();
}

mlir::Location MLIRGen::getLocation(parser::Location loc) {
  if (!loc.isValid())
    return builder.getUnknownLoc();

  // Convert SMLoc to file:line:col using SourceMgr
  unsigned bufferId = srcMgr.FindBufferContainingLoc(loc.getStart());
  if (bufferId == 0)
    return builder.getUnknownLoc();

  auto [line, col] = srcMgr.getLineAndColumn(loc.getStart(), bufferId);
  auto buffer = srcMgr.getMemoryBuffer(bufferId);
  llvm::StringRef filename = buffer->getBufferIdentifier();

  // Create FileLineColLoc for precise source tracking
  return mlir::FileLineColLoc::get(&context, filename, line, col);
}

mlir::Type MLIRGen::getMLIRType(llvm::StringRef typeName) {
  // Map DSL type names to MLIR types
  auto i64Type = builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&context, i64Type);

  return llvm::StringSwitch<mlir::Type>(typeName)
      .Case("token", tokenType)
      .Case("string", tokenType)
      .Case("handle", mlir::ais::HandleType::get(&context, mlir::ais::MemorySpace::STM))
      .Case("goal", mlir::ais::GoalType::get(&context, /*priority=*/1))
      .Case("number", builder.getF64Type())
      .Case("float", builder.getF64Type())
      .Case("f64", builder.getF64Type())
      .Case("int", builder.getI64Type())
      .Case("i64", builder.getI64Type())
      .Case("bool", builder.getI1Type())
      .Default(tokenType); // Default to token type for unknown types
}

mlir::Value MLIRGen::generateExpression(Expr *expr) {
  return MLIRGenExpressions::generateExpression(*this, expr);
}

mlir::LogicalResult MLIRGen::generateStatement(Stmt *stmt) {
  return MLIRGenStatements::generateStatement(*this, stmt);
}

mlir::ais::MemorySpace MLIRGen::toMemorySpace(llvm::StringRef tier) const {
  return llvm::StringSwitch<mlir::ais::MemorySpace>(tier)
      .Case("stm", mlir::ais::MemorySpace::STM)
      .Case("ltm", mlir::ais::MemorySpace::LTM)
      .Case("episodic", mlir::ais::MemorySpace::Episodic)
      .Default(mlir::ais::MemorySpace::STM);
}
