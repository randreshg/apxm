/*
 * @file MLIRGenOperations.h
 * @brief Operation Generation Interface
 *
 * This file provides an interface for generating MLIR operations based on the
 * provided arguments and location information.
 *
 * LLM Operations (ask/think/reason):
 *   Three distinct ops for critical path analysis. Each has different latency:
 *   - ask:    LOW latency  - simple Q&A
 *   - think:  HIGH latency - extended thinking
 *   - reason: MEDIUM latency - structured reasoning
 */

#ifndef APXM_PARSER_MLIR_MLIRGENOPERATIONS_H
#define APXM_PARSER_MLIR_MLIRGENOPERATIONS_H

#include "ais/Parser/MLIR/MLIRGenForwards.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <memory>

namespace apxm::parser {

/// MLIR operation generation interface
class MLIRGenOperations {
public:
  /// Generate call expression with proper operation dispatch
  static mlir::Value generateCallExpr(MLIRGen &gen, CallExpr *expr);

  /// Public factory methods
  static mlir::Value generateVerifyOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                     mlir::Location loc);
  static mlir::Value generateCommunicateOp(MLIRGen &gen,
                                          llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                          mlir::Location loc,
                                          llvm::SmallVectorImpl<mlir::Value> &contextArgs);
  static mlir::Value generateWaitAllOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                      mlir::Location loc);
  static mlir::Value generateMergeOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                    mlir::Location loc);

private:
  friend class MLIRGen;

  // Operation factory methods
  static mlir::Value generateQMemOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                   mlir::Location loc);
  static mlir::Value generateUMemOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                   mlir::Location loc);
  static mlir::Value generateInvOp(MLIRGen &gen, llvm::StringRef callee,
                                  llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                  mlir::Location loc);

  // LLM Operations - three distinct ops for critical path analysis
  static mlir::Value generateAskOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                   mlir::Location loc);    // LOW latency
  static mlir::Value generateThinkOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                     mlir::Location loc);  // HIGH latency
  static mlir::Value generateReasonOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                      mlir::Location loc); // MEDIUM latency

  static mlir::Value generateReflectOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                      mlir::Location loc,
                                      llvm::SmallVectorImpl<mlir::Value> &contextArgs);
  static mlir::Value generatePlanOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                   mlir::Location loc,
                                   llvm::SmallVectorImpl<mlir::Value> &contextArgs);
  static mlir::Value generateExcOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                  mlir::Location loc,
                                  llvm::SmallVectorImpl<mlir::Value> &contextArgs);
  static mlir::Value generatePrintOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                    mlir::Location loc,
                                    llvm::SmallVectorImpl<mlir::Value> &contextArgs);

  /// Extract string literal from expression if possible
  static llvm::StringRef extractStringArg(Expr *expr);
};

} // namespace apxm::parser

#endif // APXM_PARSER_MLIR_MLIRGENOPERATIONS_H
