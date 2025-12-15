/*
 * @file MLIRGenExpressions.h
 * @brief MLIR expression generation interface
 *
 * This file exposes different expression generation methods for MLIR.
 */

#ifndef APXM_PARSER_MLIR_MLIRGENEXPRESSIONS_H
#define APXM_PARSER_MLIR_MLIRGENEXPRESSIONS_H

#include "apxm/Parser/MLIR/MLIRGenForwards.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace apxm::parser {

/// MLIR expression generation interface
class MLIRGenExpressions {
public:
  /// Generate expression with proper dispatch
  static mlir::Value generateExpression(MLIRGen &gen, Expr *expr);

private:
  friend class MLIRGen;

  // Expression generation methods
  static mlir::Value generateStringLiteral(MLIRGen &gen, StringLiteralExpr *expr);
  static mlir::Value generateNumberLiteral(MLIRGen &gen, NumberLiteralExpr *expr);
  static mlir::Value generateBooleanLiteral(MLIRGen &gen, BooleanLiteralExpr *expr);
  static mlir::Value generateNullLiteral(MLIRGen &gen, NullLiteralExpr *expr);
  static mlir::Value generateVarExpr(MLIRGen &gen, VarExpr *expr);
  static mlir::Value generateArrayExpr(MLIRGen &gen, ArrayExpr *expr);
  static mlir::Value generateBinaryExpr(MLIRGen &gen, BinaryExpr *expr);
  static mlir::Value generateUnaryExpr(MLIRGen &gen, UnaryExpr *expr);
  static mlir::Value generateAssignmentExpr(MLIRGen &gen, AssignmentExpr *expr);
  static mlir::Value generateMemberAccess(MLIRGen &gen, MemberAccessExpr *expr);
  static mlir::Value generatePlanExpr(MLIRGen &gen, PlanExpr *expr);
  static mlir::Value generateReflectExpr(MLIRGen &gen, ReflectExpr *expr);
  static mlir::Value generateVerifyExpr(MLIRGen &gen, VerifyExpr *expr);
  static mlir::Value generateExecuteExpr(MLIRGen &gen, ExecuteExpr *expr);
  static mlir::Value generateCommunicateExpr(MLIRGen &gen, CommunicateExpr *expr);
  static mlir::Value generateWaitAllExpr(MLIRGen &gen, WaitAllExpr *expr);
  static mlir::Value generateMergeExpr(MLIRGen &gen, MergeExpr *expr);

  /// Convert value to target type with proper conversion
  static mlir::Value convertNumericValue(MLIRGen &gen, mlir::Value value, mlir::Type targetType,
                                        mlir::Location loc);

  /// Infer appropriate numeric type from operands
  static mlir::Type inferNumericType(mlir::Value lhs, mlir::Value rhs);

  /// Generate memory access operation
  static mlir::Value generateMemoryAccess(MLIRGen &gen, llvm::StringRef store, mlir::Location loc);
};

} // namespace apxm::parser

#endif // APXM_PARSER_MLIR_MLIRGENEXPRESSIONS_H
