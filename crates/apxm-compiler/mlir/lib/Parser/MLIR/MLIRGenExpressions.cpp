/*
 * @file MLIRGenExpressions.cpp
 * @brief Expression Generation Implementation
 *
 * This file contains the implementation of the expression generation logic for the MLIR code generator.
 */

#include "ais/Parser/MLIR/MLIRGen.h"
#include "ais/Parser/MLIR/MLIRGenExpressions.h"
#include "ais/Parser/MLIR/MLIRGenStatement.h"
#include "ais/Parser/MLIR/MLIRGenOperations.h"
#include "ais/Common/Constants.h"
#include "ais/Dialect/AIS/IR/AISDialect.h"
#include "ais/Dialect/AIS/IR/AISOps.h"
#include "ais/Dialect/AIS/IR/AISTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace apxm::parser;
using namespace mlir;
using namespace mlir::ais;
using namespace apxm::constants;

mlir::Value MLIRGenExpressions::generateExpression(MLIRGen &gen, Expr *expr) {
  return llvm::TypeSwitch<Expr *, mlir::Value>(expr)
      .Case<StringLiteralExpr>([&](auto *e) { return generateStringLiteral(gen, e); })
      .Case<NumberLiteralExpr>([&](auto *e) { return generateNumberLiteral(gen, e); })
      .Case<BooleanLiteralExpr>([&](auto *e) { return generateBooleanLiteral(gen, e); })
      .Case<NullLiteralExpr>([&](auto *e) { return generateNullLiteral(gen, e); })
      .Case<VarExpr>([&](auto *e) { return generateVarExpr(gen, e); })
      .Case<CallExpr>([&](auto *e) { return MLIRGenOperations::generateCallExpr(gen, e); })
      .Case<ArrayExpr>([&](auto *e) { return generateArrayExpr(gen, e); })
      .Case<BinaryExpr>([&](auto *e) { return generateBinaryExpr(gen, e); })
      .Case<UnaryExpr>([&](auto *e) { return generateUnaryExpr(gen, e); })
      .Case<AssignmentExpr>([&](auto *e) { return generateAssignmentExpr(gen, e); })
      .Case<MemberAccessExpr>([&](auto *e) { return generateMemberAccess(gen, e); })
      .Case<PlanExpr>([&](auto *e) { return generatePlanExpr(gen, e); })
      .Case<ReflectExpr>([&](auto *e) { return generateReflectExpr(gen, e); })
      .Case<VerifyExpr>([&](auto *e) { return generateVerifyExpr(gen, e); })
      .Case<ExecuteExpr>([&](auto *e) { return generateExecuteExpr(gen, e); })
      .Case<CommunicateExpr>([&](auto *e) { return generateCommunicateExpr(gen, e); })
      .Case<WaitAllExpr>([&](auto *e) { return generateWaitAllExpr(gen, e); })
      .Case<MergeExpr>([&](auto *e) { return generateMergeExpr(gen, e); })
      .Default([&](auto *) {
        gen.emitError(gen.getLocation(expr->getLocation()), "Unsupported expression type");
        return mlir::Value();
      });
}

mlir::Value MLIRGenExpressions::generateStringLiteral(MLIRGen &gen, StringLiteralExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::ConstStrOp>(loc, tokenType,
                                                  gen.builder.getStringAttr(expr->getValue()));
}

mlir::Value MLIRGenExpressions::generateNumberLiteral(MLIRGen &gen, NumberLiteralExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  return gen.builder.create<arith::ConstantOp>(loc, gen.builder.getF64FloatAttr(expr->getValue()));
}

mlir::Value MLIRGenExpressions::generateBooleanLiteral(MLIRGen &gen, BooleanLiteralExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  return gen.builder.create<arith::ConstantOp>(loc, gen.builder.getBoolAttr(expr->getValue()));
}

mlir::Value MLIRGenExpressions::generateNullLiteral(MLIRGen &gen, NullLiteralExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  return gen.builder.create<arith::ConstantOp>(loc, gen.builder.getI64IntegerAttr(0));
}

mlir::Value MLIRGenExpressions::generateVarExpr(MLIRGen &gen, VarExpr *expr) {
  auto value = gen.symbolTable.lookup(expr->getName());
  if (!value) {
    gen.emitError(gen.getLocation(expr->getLocation()),
                 "Undefined variable: " + expr->getName().str());
    return nullptr;
  }
  return value;
}

mlir::Value MLIRGenExpressions::generateArrayExpr(MLIRGen &gen, ArrayExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  llvm::SmallVector<mlir::Value, 4> elements;

  for (const auto &elemExpr : expr->getElements()) {
    auto elem = gen.generateExpression(elemExpr.get());
    if (!elem) return nullptr;
    elements.push_back(elem);
  }

  // For empty arrays, return a constant zero
  if (elements.empty()) {
    return gen.builder.create<arith::ConstantOp>(loc, gen.builder.getI64IntegerAttr(0));
  }

  // Merge all elements
  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::MergeOp>(loc, tokenType, elements);
}

mlir::Value MLIRGenExpressions::generateBinaryExpr(MLIRGen &gen, BinaryExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  auto lhs = gen.generateExpression(expr->getLHS());
  auto rhs = gen.generateExpression(expr->getRHS());
  if (!lhs || !rhs) return nullptr;

  auto op = expr->getOperator();
  switch (op) {
  case BinaryExpr::Operator::And:
  case BinaryExpr::Operator::Or: {
    auto lhsBool = MLIRGenStatements::coerceToBool(gen, lhs, loc);
    auto rhsBool = MLIRGenStatements::coerceToBool(gen, rhs, loc);
    if (op == BinaryExpr::Operator::And) {
      return gen.builder.create<arith::AndIOp>(loc, lhsBool, rhsBool).getResult();
    } else {
      return gen.builder.create<arith::OrIOp>(loc, lhsBool, rhsBool).getResult();
    }
  }

  case BinaryExpr::Operator::Equal:
  case BinaryExpr::Operator::NotEqual:
  case BinaryExpr::Operator::Less:
  case BinaryExpr::Operator::LessEqual:
  case BinaryExpr::Operator::Greater:
  case BinaryExpr::Operator::GreaterEqual: {
    auto targetType = inferNumericType(lhs, rhs);
    if (!targetType) return nullptr;

    lhs = convertNumericValue(gen, lhs, targetType, loc);
    rhs = convertNumericValue(gen, rhs, targetType, loc);

    if (llvm::isa<FloatType>(targetType)) {
      int opCode = static_cast<int>(op);
      arith::CmpFPredicate predicate = llvm::StringSwitch<arith::CmpFPredicate>(
          llvm::Twine(opCode).str())
          .Case("0", arith::CmpFPredicate::OEQ)
          .Case("1", arith::CmpFPredicate::UNE)
          .Case("2", arith::CmpFPredicate::OLT)
          .Case("3", arith::CmpFPredicate::OLE)
          .Case("4", arith::CmpFPredicate::OGT)
          .Case("5", arith::CmpFPredicate::OGE)
          .Default(arith::CmpFPredicate::OEQ);

      return gen.builder.create<arith::CmpFOp>(loc, predicate, lhs, rhs);
    }

    int opCode = static_cast<int>(op);
    arith::CmpIPredicate predicate = llvm::StringSwitch<arith::CmpIPredicate>(
        llvm::Twine(opCode).str())
        .Case("0", arith::CmpIPredicate::eq)
        .Case("1", arith::CmpIPredicate::ne)
        .Case("2", arith::CmpIPredicate::slt)
        .Case("3", arith::CmpIPredicate::sle)
        .Case("4", arith::CmpIPredicate::sgt)
        .Case("5", arith::CmpIPredicate::sge)
        .Default(arith::CmpIPredicate::eq);

    return gen.builder.create<arith::CmpIOp>(loc, predicate, lhs, rhs);
  }

  case BinaryExpr::Operator::Add:
  case BinaryExpr::Operator::Sub:
  case BinaryExpr::Operator::Mul:
  case BinaryExpr::Operator::Div:
  case BinaryExpr::Operator::Mod: {
    auto targetType = inferNumericType(lhs, rhs);
    if (!targetType) return nullptr;

    lhs = convertNumericValue(gen, lhs, targetType, loc);
    rhs = convertNumericValue(gen, rhs, targetType, loc);

    if (llvm::isa<FloatType>(targetType)) {
      switch (op) {
      case BinaryExpr::Operator::Add: return gen.builder.create<arith::AddFOp>(loc, lhs, rhs);
      case BinaryExpr::Operator::Sub: return gen.builder.create<arith::SubFOp>(loc, lhs, rhs);
      case BinaryExpr::Operator::Mul: return gen.builder.create<arith::MulFOp>(loc, lhs, rhs);
      case BinaryExpr::Operator::Div: return gen.builder.create<arith::DivFOp>(loc, lhs, rhs);
      case BinaryExpr::Operator::Mod: return gen.builder.create<arith::RemFOp>(loc, lhs, rhs);
      default: break;
      }
    } else {
      switch (op) {
      case BinaryExpr::Operator::Add: return gen.builder.create<arith::AddIOp>(loc, lhs, rhs);
      case BinaryExpr::Operator::Sub: return gen.builder.create<arith::SubIOp>(loc, lhs, rhs);
      case BinaryExpr::Operator::Mul: return gen.builder.create<arith::MulIOp>(loc, lhs, rhs);
      case BinaryExpr::Operator::Div: return gen.builder.create<arith::DivSIOp>(loc, lhs, rhs);
      case BinaryExpr::Operator::Mod: return gen.builder.create<arith::RemSIOp>(loc, lhs, rhs);
      default: break;
      }
    }
    break;
  }
  }

  gen.emitError(loc, "Unsupported binary operation");
  return nullptr;
}

mlir::Value MLIRGenExpressions::generateUnaryExpr(MLIRGen &gen, UnaryExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  auto operand = gen.generateExpression(expr->getOperand());
  if (!operand) return nullptr;

  switch (expr->getOperator()) {
  case UnaryExpr::Operator::Negate: {
    auto type = operand.getType();
    if (llvm::isa<FloatType>(type)) return gen.builder.create<arith::NegFOp>(loc, operand);

    if (llvm::isa<IntegerType>(type)) {
      auto intType = llvm::cast<IntegerType>(type);
      auto zero = gen.builder.create<arith::ConstantOp>(loc, gen.builder.getIntegerAttr(intType, 0));
      return gen.builder.create<arith::SubIOp>(loc, zero, operand);
    }

    if (type.isIndex()) {
      auto i64Type = gen.builder.getI64Type();
      auto casted = gen.builder.create<arith::IndexCastOp>(loc, i64Type, operand);
      auto zero = gen.builder.create<arith::ConstantOp>(loc, gen.builder.getIntegerAttr(i64Type, 0));
      auto neg = gen.builder.create<arith::SubIOp>(loc, zero, casted);
      return gen.builder.create<arith::IndexCastOp>(loc, type, neg);
    }
    break;
  }

  case UnaryExpr::Operator::Not: {
    auto boolValue = MLIRGenStatements::coerceToBool(gen, operand, loc);
    auto trueConst = gen.builder.create<arith::ConstantOp>(loc, gen.builder.getBoolAttr(true));
    return gen.builder.create<arith::XOrIOp>(loc, boolValue, trueConst);
  }
  }

  gen.emitError(loc, "Unsupported unary operation");
  return nullptr;
}

mlir::Value MLIRGenExpressions::generateAssignmentExpr(MLIRGen &gen, AssignmentExpr *expr) {
  auto value = gen.generateExpression(expr->getRHS());
  if (!value) return nullptr;

  if (auto *varExpr = dyn_cast<VarExpr>(expr->getLHS())) {
    gen.symbolTable.insert(varExpr->getName(), value);
    return value;
  }

  gen.emitError(gen.getLocation(expr->getLocation()), "Assignment to non-variable");
  return nullptr;
}

mlir::Value MLIRGenExpressions::generateMemberAccess(MLIRGen &gen, MemberAccessExpr *expr) {
  if (auto *varExpr = dyn_cast<VarExpr>(expr->getObject())) {
    if (varExpr->getName() == "mem") {
      return generateMemoryAccess(gen, expr->getMember(), gen.getLocation(expr->getLocation()));
    }
  }

  gen.emitError(gen.getLocation(expr->getLocation()), "Unsupported member access");
  return nullptr;
}

mlir::Value MLIRGenExpressions::generatePlanExpr(MLIRGen &gen, PlanExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  llvm::SmallVector<mlir::Value, 4> contextValues;

  for (const auto &ctx : expr->getContext()) {
    auto value = gen.generateExpression(ctx.get());
    if (!value) return nullptr;
    contextValues.push_back(value);
  }

  auto goalType = mlir::ais::GoalType::get(&gen.context, /*priority=*/1);
  return gen.builder.create<mlir::ais::PlanOp>(loc, goalType,
                                             gen.builder.getStringAttr(expr->getGoal()),
                                             contextValues);
}

mlir::Value MLIRGenExpressions::generateReflectExpr(MLIRGen &gen, ReflectExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  llvm::SmallVector<mlir::Value, 4> contextValues;

  for (const auto &ctx : expr->getContext()) {
    auto value = gen.generateExpression(ctx.get());
    if (!value) return nullptr;
    contextValues.push_back(value);
  }

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);
  return gen.builder.create<mlir::ais::ReflectOp>(
      loc, tokenType, gen.builder.getStringAttr(expr->getTraceId()), contextValues);
}

mlir::Value MLIRGenExpressions::generateVerifyExpr(MLIRGen &gen, VerifyExpr *expr) {
  return MLIRGenOperations::generateVerifyOp(gen, llvm::ArrayRef<std::unique_ptr<Expr>>{},
                                           gen.getLocation(expr->getLocation()));
}

mlir::Value MLIRGenExpressions::generateExecuteExpr(MLIRGen &gen, ExecuteExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  llvm::SmallVector<mlir::Value, 4> contextValues;

  for (const auto &ctx : expr->getContext()) {
    auto value = gen.generateExpression(ctx.get());
    if (!value) return nullptr;
    contextValues.push_back(value);
  }

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);
  return gen.builder.create<mlir::ais::ExcOp>(loc, tokenType,
                                            gen.builder.getStringAttr(expr->getCode()),
                                            contextValues);
}

mlir::Value MLIRGenExpressions::generateCommunicateExpr(MLIRGen &gen, CommunicateExpr *expr) {
  llvm::SmallVector<mlir::Value, 4> contextValues;
  return MLIRGenOperations::generateCommunicateOp(gen, llvm::ArrayRef<std::unique_ptr<Expr>>{},
                                                 gen.getLocation(expr->getLocation()),
                                                 contextValues);
}

mlir::Value MLIRGenExpressions::generateWaitAllExpr(MLIRGen &gen, WaitAllExpr *expr) {
  return MLIRGenOperations::generateWaitAllOp(gen, expr->getTokens(),
                                            gen.getLocation(expr->getLocation()));
}

mlir::Value MLIRGenExpressions::generateMergeExpr(MLIRGen &gen, MergeExpr *expr) {
  return MLIRGenOperations::generateMergeOp(gen, expr->getValues(),
                                           gen.getLocation(expr->getLocation()));
}

mlir::Value MLIRGenExpressions::convertNumericValue(MLIRGen &gen, mlir::Value value,
                                                  mlir::Type targetType, mlir::Location loc) {
  if (!value || value.getType() == targetType) return value;

  auto sourceType = value.getType();
  if (llvm::isa<FloatType>(targetType)) {
    if (llvm::isa<FloatType>(sourceType))
      return gen.builder.create<arith::ExtFOp>(loc, targetType, value);

    if (llvm::isa<IntegerType>(sourceType))
      return gen.builder.create<arith::SIToFPOp>(loc, targetType, value);

    if (sourceType.isIndex()) {
      auto i64Type = gen.builder.getI64Type();
      auto casted = gen.builder.create<arith::IndexCastOp>(loc, i64Type, value);
      return gen.builder.create<arith::SIToFPOp>(loc, targetType, casted);
    }
  } else if (llvm::isa<IntegerType>(targetType)) {
    if (sourceType.isIndex())
      return gen.builder.create<arith::IndexCastOp>(loc, targetType, value);

    if (llvm::isa<IntegerType>(sourceType)) {
      auto targetInt = llvm::cast<IntegerType>(targetType);
      auto sourceInt = llvm::cast<IntegerType>(sourceType);
      unsigned targetWidth = targetInt.getWidth();
      unsigned sourceWidth = sourceInt.getWidth();

      if (targetWidth >= sourceWidth)
        return gen.builder.create<arith::ExtSIOp>(loc, targetType, value);
      return gen.builder.create<arith::TruncIOp>(loc, targetType, value);
    }

    if (llvm::isa<FloatType>(sourceType))
      return gen.builder.create<arith::FPToSIOp>(loc, targetType, value);
  }

  return value;
}

mlir::Type MLIRGenExpressions::inferNumericType(mlir::Value lhs, mlir::Value rhs) {
  auto lhsType = lhs.getType();
  auto rhsType = rhs.getType();

  if (llvm::isa<FloatType>(lhsType)) return lhsType;
  if (llvm::isa<FloatType>(rhsType)) return rhsType;

  if (llvm::isa<IntegerType>(lhsType)) return lhsType;
  if (llvm::isa<IntegerType>(rhsType)) return rhsType;

  if (lhsType.isIndex() || rhsType.isIndex())
    return mlir::IntegerType::get(lhs.getContext(), 64);

  return mlir::Type();
}

mlir::Value MLIRGenExpressions::generateMemoryAccess(MLIRGen &gen, llvm::StringRef store,
                                                   mlir::Location loc) {
  auto it = gen.memorySpaces.find(store);
  mlir::ais::MemorySpace space = it != gen.memorySpaces.end()
                               ? it->second : mlir::ais::MemorySpace::STM;

  auto handleType = mlir::ais::HandleType::get(&gen.context, space);
  auto queryAttr = gen.builder.getStringAttr(store);
  auto sidAttr = gen.builder.getStringAttr(session::DEFAULT_SID);
  auto spaceAttr = gen.builder.getStringAttr(mlir::ais::stringifyMemorySpace(space));

  return gen.builder.create<mlir::ais::QMemOp>(loc, handleType, queryAttr, sidAttr, spaceAttr,
                                             /*limit=*/nullptr);
}
