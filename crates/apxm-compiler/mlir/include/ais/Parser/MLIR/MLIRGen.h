/*
 * @file MLIRGen.h
 * @brief MLIR generation from APXM AST
 *
 * This file is responsible for generating MLIR IR from APXM AST.
 * This phase helps in translating the high-level APXM language constructs into a
 * low-level representation that can be executed by the MLIR runtime.
 */

#ifndef APXM_PARSER_MLIR_MLIRGEN_H
#define APXM_PARSER_MLIR_MLIRGEN_H

#include "ais/Parser/MLIR/MLIRGenForwards.h"
#include "ais/Dialect/AIS/IR/AISTypes.h"
#include "ais/Parser/AST/Declaration.h"
#include "ais/Parser/AST/Expression.h"
#include "ais/Parser/AST/Statement.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/SourceMgr.h"
#include <memory>

namespace apxm::parser {

/// MLIR IR generator from APXM AST
class MLIRGen {
public:
  MLIRGen(mlir::MLIRContext &context, llvm::SourceMgr &sourceMgr);

  /// Generate MLIR module from agent declaration
  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp> generateModule(AgentDecl *agent);

  /// Generate MLIR module from multiple agent declarations
  [[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp> generateModuleFromAgents(
      const std::vector<std::unique_ptr<AgentDecl>> &agents);

  void emitError(parser::Location loc, llvm::StringRef message);
  void emitError(mlir::Location loc, llvm::StringRef message);

private:
  mlir::MLIRContext &context;
  mlir::OpBuilder builder;
  llvm::SourceMgr &srcMgr;

  // Symbol table for variable resolution
  using SymbolTable = llvm::ScopedHashTable<llvm::StringRef, mlir::Value>;
  SymbolTable symbolTable;
  llvm::StringMap<mlir::ais::MemorySpace> memorySpaces;
  llvm::StringSet<> flowSymbols;
  llvm::StringMap<mlir::FunctionType> flowTypes;

  //===--------------------------------------------------------------------===//
  // Declaration Generation
  //===--------------------------------------------------------------------===//

  mlir::LogicalResult generateAgentDecl(mlir::ModuleOp module, AgentDecl *agent);
  mlir::LogicalResult generateFlowDecl(mlir::ModuleOp module, FlowDecl *flow, llvm::StringRef agentName);
  mlir::LogicalResult generateOnEventDecl(mlir::ModuleOp module, OnEventDecl *handler,
                                         llvm::StringRef symbolName, bool returnsValue);

  //===--------------------------------------------------------------------===//
  // Statement Generation
  //===--------------------------------------------------------------------===//

  mlir::LogicalResult generateStatement(Stmt *stmt);
  mlir::LogicalResult generateLetStmt(LetStmt *stmt);
  mlir::LogicalResult generateReturnStmt(ReturnStmt *stmt);
  mlir::LogicalResult generateIfStmt(IfStmt *stmt);
  mlir::LogicalResult generateParallelStmt(ParallelStmt *stmt);
  mlir::LogicalResult generateLoopStmt(LoopStmt *stmt);
  mlir::LogicalResult generateTryCatchStmt(TryCatchStmt *stmt);

  //===--------------------------------------------------------------------===//
  // Expression Generation
  //===--------------------------------------------------------------------===//

  mlir::Value generateExpression(Expr *expr);
  mlir::Value generateStringLiteral(StringLiteralExpr *expr);
  mlir::Value generateNumberLiteral(NumberLiteralExpr *expr);
  mlir::Value generateBooleanLiteral(BooleanLiteralExpr *expr);
  mlir::Value generateNullLiteral(NullLiteralExpr *expr);
  mlir::Value generateVarExpr(VarExpr *expr);
  mlir::Value generateCallExpr(CallExpr *expr);
  mlir::Value generateArrayExpr(ArrayExpr *expr);
  mlir::Value generateBinaryExpr(BinaryExpr *expr);
  mlir::Value generateUnaryExpr(UnaryExpr *expr);
  mlir::Value generateAssignmentExpr(AssignmentExpr *expr);
  mlir::Value generateMemberAccess(MemberAccessExpr *expr);

  //===--------------------------------------------------------------------===//
  // Special Operation Generation
  //===--------------------------------------------------------------------===//

  mlir::Value generatePlanExpr(PlanExpr *expr);
  mlir::Value generateReflectExpr(ReflectExpr *expr);
  mlir::Value generateVerifyExpr(VerifyExpr *expr);
  mlir::Value generateExecuteExpr(ExecuteExpr *expr);
  mlir::Value generateCommunicateExpr(CommunicateExpr *expr);
  mlir::Value generateWaitAllExpr(WaitAllExpr *expr);
  mlir::Value generateMergeExpr(MergeExpr *expr);

  //===--------------------------------------------------------------------===//
  // Operation Factories
  //===--------------------------------------------------------------------===//

  mlir::Value generateQMemOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc);
  mlir::Value generateUMemOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc);
  mlir::Value generateInvOp(llvm::StringRef callee, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                           mlir::Location loc);
  mlir::Value generateRsnOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc);
  mlir::Value generateReflectOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc,
                               llvm::SmallVector<mlir::Value, 4> &contextArgs);
  mlir::Value generatePlanOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc,
                            llvm::SmallVector<mlir::Value, 4> &contextArgs);
  mlir::Value generateVerifyOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc);
  mlir::Value generateExcOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc,
                           llvm::SmallVector<mlir::Value, 4> &contextArgs);
  mlir::Value generateCommunicateOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc,
                                   llvm::SmallVector<mlir::Value, 4> &contextArgs);
  mlir::Value generateWaitAllOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc);
  mlir::Value generateMergeOp(llvm::ArrayRef<std::unique_ptr<Expr>> args, mlir::Location loc);

  //===--------------------------------------------------------------------===//
  // Utilities
  //===--------------------------------------------------------------------===//

  [[nodiscard]] mlir::Location getLocation(parser::Location loc);
  [[nodiscard]] mlir::Type getMLIRType(llvm::StringRef typeName);
  [[nodiscard]] llvm::StringRef extractStringArg(Expr *expr);
  [[nodiscard]] mlir::Value coerceToBool(mlir::Value value, mlir::Location loc);
  [[nodiscard]] mlir::Value convertNumericValue(mlir::Value value, mlir::Type targetType,
                                               mlir::Location loc);
  [[nodiscard]] mlir::Type inferNumericType(mlir::Value &lhs, mlir::Value &rhs);
  [[nodiscard]] mlir::ais::MemorySpace toMemorySpace(llvm::StringRef tier) const;
  [[nodiscard]] mlir::Value generateMemoryAccess(llvm::StringRef store, mlir::Location loc);
  [[nodiscard]] std::string stringifyMetadataExpr(Expr *expr) const;
  [[nodiscard]] bool blockHasReturn(llvm::ArrayRef<std::unique_ptr<Stmt>> body) const;
  [[nodiscard]] bool isFlowSymbol(llvm::StringRef name) const;
  [[nodiscard]] mlir::FunctionType getFlowFunctionType(llvm::StringRef name) const;
  mlir::Value generateFlowCall(llvm::StringRef callee, CallExpr *expr, mlir::Location loc);

  friend class MLIRGenStatements;
  friend class MLIRGenOperations;
  friend class MLIRGenDeclarations;
  friend class MLIRGenExpressions;
};

} // namespace apxm::parser

#endif // APXM_PARSER_MLIR_MLIRGEN_H
