/*
 * @file MLIRGenStatements.cpp
 * @brief MLIR statement generation implementation
 *
 * This file contains the implementation of the MLIR statement generation logic.
 */

#include "apxm/Parser/MLIR/MLIRGen.h"
#include "apxm/Parser/MLIR/MLIRGenStatement.h"
#include "apxm/Dialect/AIS/IR/AISDialect.h"
#include "apxm/Dialect/AIS/IR/AISOps.h"
#include "apxm/Dialect/AIS/IR/AISTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace apxm::parser;
using namespace mlir;
using namespace mlir::ais;

mlir::LogicalResult MLIRGenStatements::generateStatement(MLIRGen &gen, Stmt *stmt) {
  return llvm::TypeSwitch<Stmt *, mlir::LogicalResult>(stmt)
      .Case<LetStmt>([&](auto *s) { return generateLetStmt(gen, s); })
      .Case<ReturnStmt>([&](auto *s) { return generateReturnStmt(gen, s); })
      .Case<IfStmt>([&](auto *s) { return generateIfStmt(gen, s); })
      .Case<ParallelStmt>([&](auto *s) { return generateParallelStmt(gen, s); })
      .Case<LoopStmt>([&](auto *s) { return generateLoopStmt(gen, s); })
      .Case<TryCatchStmt>([&](auto *s) { return generateTryCatchStmt(gen, s); })
      .Case<ExprStmt>([&](auto *s) {
        gen.generateExpression(s->getExpr());
        return mlir::success();
      })
      .Default([&](auto *) {
        gen.emitError(gen.getLocation(stmt->getLocation()), "Unsupported statement type");
        return mlir::failure();
      });
}

mlir::LogicalResult MLIRGenStatements::generateLetStmt(MLIRGen &gen, LetStmt *stmt) {
  auto value = gen.generateExpression(stmt->getInitExpr());
  if (!value) return mlir::failure();

  gen.symbolTable.insert(stmt->getVarName(), value);
  return mlir::success();
}

mlir::LogicalResult MLIRGenStatements::generateReturnStmt(MLIRGen &gen, ReturnStmt *stmt) {
  mlir::Location loc = gen.getLocation(stmt->getLocation());

  if (stmt->getReturnExpr()) {
    auto value = gen.generateExpression(stmt->getReturnExpr());
    if (!value) return mlir::failure();

    gen.builder.create<mlir::func::ReturnOp>(loc, value);
  } else {
    gen.builder.create<mlir::func::ReturnOp>(loc);
  }

  return mlir::success();
}

mlir::LogicalResult MLIRGenStatements::generateIfStmt(MLIRGen &gen, IfStmt *stmt) {
  mlir::Location loc = gen.getLocation(stmt->getLocation());

  auto condition = gen.generateExpression(stmt->getCondition());
  if (!condition) return mlir::failure();

  // Convert condition to boolean
  mlir::Value condBool = coerceToBool(gen, condition, loc);
  bool hasElse = !stmt->getElseStmts().empty();

  // Create scf.if operation
  auto ifOp = gen.builder.create<mlir::scf::IfOp>(loc, /*resultTypes=*/TypeRange{}, condBool, hasElse);

  // Generate then block
  {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> scope(gen.symbolTable);
    gen.builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

    for (const auto &thenStmt : stmt->getThenStmts()) {
      if (failed(generateStatement(gen, thenStmt.get()))) {
        return mlir::failure();
      }
    }

    gen.builder.create<mlir::scf::YieldOp>(loc);
  }

  // Generate else block if present
  if (hasElse) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> scope(gen.symbolTable);
    gen.builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

    for (const auto &elseStmt : stmt->getElseStmts()) {
      if (failed(generateStatement(gen, elseStmt.get()))) {
        return mlir::failure();
      }
    }

    gen.builder.create<mlir::scf::YieldOp>(loc);
  }

  gen.builder.setInsertionPointAfter(ifOp);
  return mlir::success();
}

mlir::LogicalResult MLIRGenStatements::generateParallelStmt(MLIRGen &gen, ParallelStmt *stmt) {
  // Create unique parallel group identifier
  llvm::SmallString<64> parallelGroup;
  parallelGroup.append("parallel_");
  parallelGroup.append(std::to_string(reinterpret_cast<uintptr_t>(stmt)));
  auto parallelGroupAttr = gen.builder.getStringAttr(parallelGroup);

  // Generate statements with parallel group attribute
  for (const auto &bodyStmt : stmt->getBody()) {
    if (failed(generateStatement(gen, bodyStmt.get()))) {
      return mlir::failure();
    }

    // Mark last operation with parallel group attribute
    if (auto *lastOp = gen.builder.getInsertionBlock()->getTerminator()) {
      lastOp->setAttr("ais.parallel_group", parallelGroupAttr);
    } else if (!gen.builder.getInsertionBlock()->empty()) {
      gen.builder.getInsertionBlock()->back().setAttr("ais.parallel_group", parallelGroupAttr);
    }
  }

  return mlir::success();
}

mlir::LogicalResult MLIRGenStatements::generateLoopStmt(MLIRGen &gen, LoopStmt *stmt) {
  mlir::Location loc = gen.getLocation(stmt->getLocation());

  // Generate collection expression
  auto collection = gen.generateExpression(stmt->getCollection());
  if (!collection) return mlir::failure();

  // Create unique loop label
  llvm::SmallString<64> loopLabel;
  loopLabel.append("loop_");
  loopLabel.append(stmt->getVarName());
  loopLabel.append("_");
  loopLabel.append(std::to_string(reinterpret_cast<uintptr_t>(stmt)));

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  // Create loop operations
  auto loopStart = gen.builder.create<mlir::ais::LoopStartOp>(
      loc, tokenType, collection, gen.builder.getStringAttr(loopLabel));

  // Generate loop body with scoped symbol table
  {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> scope(gen.symbolTable);
    gen.symbolTable.insert(stmt->getVarName(), loopStart.getState());

    for (const auto &bodyStmt : stmt->getBody()) {
      if (failed(generateStatement(gen, bodyStmt.get()))) {
        return mlir::failure();
      }
    }
  }

  gen.builder.create<mlir::ais::LoopEndOp>(loc, tokenType, loopStart.getState());
  return mlir::success();
}

mlir::LogicalResult MLIRGenStatements::generateTryCatchStmt(MLIRGen &gen, TryCatchStmt *stmt) {
  mlir::Location loc = gen.getLocation(stmt->getLocation());

  // Create try/catch markers
  auto tryLabel = gen.builder.getStringAttr("try_block");
  auto catchLabel = gen.builder.getStringAttr("catch_block");

  gen.builder.create<mlir::ais::TryCatchOp>(loc, tryLabel, catchLabel);

  // Generate try body
  {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> scope(gen.symbolTable);
    for (const auto &tryStmt : stmt->getTryBody()) {
      if (failed(generateStatement(gen, tryStmt.get()))) {
        return mlir::failure();
      }
    }
  }

  // Generate catch body if present
  if (!stmt->getCatchBody().empty()) {
    llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> scope(gen.symbolTable);
    for (const auto &catchStmt : stmt->getCatchBody()) {
      if (failed(generateStatement(gen, catchStmt.get()))) {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

mlir::Value MLIRGenStatements::coerceToBool(MLIRGen &gen, mlir::Value value, mlir::Location loc) {
  auto type = value.getType();
  if (type.isInteger(1)) return value;

  if (llvm::isa<IntegerType>(type)) {
    auto intType = llvm::cast<IntegerType>(type);
    auto zero = gen.builder.create<arith::ConstantOp>(loc, gen.builder.getIntegerAttr(intType, 0));
    return gen.builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, value, zero);
  }

  if (llvm::isa<FloatType>(type)) {
    auto floatType = llvm::cast<FloatType>(type);
    auto zero = gen.builder.create<arith::ConstantOp>(loc, gen.builder.getFloatAttr(floatType, 0.0));
    return gen.builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::UNE, value, zero);
  }

  // Non-numeric values are always truthy
  return gen.builder.create<arith::ConstantOp>(loc, gen.builder.getBoolAttr(true));
}
