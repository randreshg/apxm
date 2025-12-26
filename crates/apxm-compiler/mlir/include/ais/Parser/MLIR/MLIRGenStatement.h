/*
 * @file MLIRGenStatements.h
 * @brief MLIR statement generation interface
 *
 * This file provides an interface for generating MLIR statements from AST nodes.
 */

#ifndef APXM_PARSER_MLIR_MLIRGENSTATEMENTS_H
#define APXM_PARSER_MLIR_MLIRGENSTATEMENTS_H

#include "ais/Parser/MLIR/MLIRGenForwards.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>

namespace apxm::parser {

/// MLIR statement generation interface
class MLIRGenStatements {
public:
  /// Generate statement with proper dispatch
  static mlir::LogicalResult generateStatement(MLIRGen &gen, Stmt *stmt);

  /// Convert value to boolean type with proper coercion
  static mlir::Value coerceToBool(MLIRGen &gen, mlir::Value value, mlir::Location loc);

private:
  friend class MLIRGen;

  // Statement generation methods
  static mlir::LogicalResult generateLetStmt(MLIRGen &gen, LetStmt *stmt);
  static mlir::LogicalResult generateReturnStmt(MLIRGen &gen, ReturnStmt *stmt);
  static mlir::LogicalResult generateIfStmt(MLIRGen &gen, IfStmt *stmt);
  static mlir::LogicalResult generateParallelStmt(MLIRGen &gen, ParallelStmt *stmt);
  static mlir::LogicalResult generateLoopStmt(MLIRGen &gen, LoopStmt *stmt);
  static mlir::LogicalResult generateTryCatchStmt(MLIRGen &gen, TryCatchStmt *stmt);
};

} // namespace apxm::parser

#endif // APXM_PARSER_MLIR_MLIRGENSTATEMENTS_H
