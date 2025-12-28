/*
 * @file MLIRGenOperations.cpp
 * @brief MLIR operation generation implementation
 *
 * This file contains the implementation of MLIR operation generation.
 */

#include "ais/Parser/MLIR/MLIRGen.h"
#include "ais/Parser/MLIR/MLIRGenOperations.h"
#include "ais/Common/Constants.h"
#include "ais/Dialect/AIS/IR/AISDialect.h"
#include "ais/Dialect/AIS/IR/AISOps.h"
#include "ais/Dialect/AIS/IR/AISTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/StringSwitch.h"

using namespace apxm::parser;
using namespace mlir;
using namespace mlir::ais;
using namespace apxm::constants;

mlir::Value MLIRGenOperations::generateCallExpr(MLIRGen &gen, CallExpr *expr) {
  mlir::Location loc = gen.getLocation(expr->getLocation());
  llvm::StringRef callee = expr->getCallee();

  // Generate non-string arguments
  llvm::SmallVector<mlir::Value, 4> args;
  for (const auto &argExpr : expr->getArgs()) {
    if (!llvm::isa<StringLiteralExpr>(argExpr.get())) {
      auto arg = gen.generateExpression(argExpr.get());
      if (!arg) return nullptr;
      args.push_back(arg);
    }
  }

  // Map callee to operation kind
  enum class CallKind {
    QMem, UMem, Invoke, Reason, Reflect, Plan, Verify, Execute, Communicate, WaitAll, Merge, Unknown
  };

  CallKind kind = llvm::StringSwitch<CallKind>(callee)
      .Cases("query_memory", "qmem", "mem", CallKind::QMem)
      .Cases("update_memory", "umem", CallKind::UMem)
      .Cases("invoke", "inv", "tool", CallKind::Invoke)
      .Cases("reason", "rsn", "think", "llm", CallKind::Reason)
      .Case("reflect", CallKind::Reflect)
      .Case("plan", CallKind::Plan)
      .Case("verify", CallKind::Verify)
      .Cases("execute", "exc", "exec", CallKind::Execute)
      .Cases("communicate", "talk", CallKind::Communicate)
      .Cases("wait_all", "wait", CallKind::WaitAll)
      .Case("merge", CallKind::Merge)
      .Default(CallKind::Unknown);

  // Dispatch to appropriate operation generator
  switch (kind) {
  case CallKind::QMem: return generateQMemOp(gen, expr->getArgs(), loc);
  case CallKind::UMem: return generateUMemOp(gen, expr->getArgs(), loc);
  case CallKind::Invoke: return generateInvOp(gen, callee, expr->getArgs(), loc);
  case CallKind::Reason: return generateRsnOp(gen, expr->getArgs(), loc);
  case CallKind::Reflect: return generateReflectOp(gen, expr->getArgs(), loc, args);
  case CallKind::Plan: return generatePlanOp(gen, expr->getArgs(), loc, args);
  case CallKind::Verify: return generateVerifyOp(gen, expr->getArgs(), loc);
  case CallKind::Execute: return generateExcOp(gen, expr->getArgs(), loc, args);
  case CallKind::Communicate: return generateCommunicateOp(gen, expr->getArgs(), loc, args);
  case CallKind::WaitAll: return generateWaitAllOp(gen, expr->getArgs(), loc);
  case CallKind::Merge: return generateMergeOp(gen, expr->getArgs(), loc);
  case CallKind::Unknown: break;
  }

  if (gen.isFlowSymbol(callee)) {
    return gen.generateFlowCall(callee, expr, loc);
  }

  // Default: capability invocation
  llvm::StringRef params = expr->getArgs().empty() ? "{}" : "{\"args\":[]}";
  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::InvOp>(loc, tokenType, gen.builder.getStringAttr(callee),
                                             gen.builder.getStringAttr(params));
}

llvm::StringRef MLIRGenOperations::extractStringArg(Expr *expr) {
  if (auto strLit = llvm::dyn_cast<StringLiteralExpr>(expr)) {
    return strLit->getValue();
  }
  return "";
}

mlir::Value MLIRGenOperations::generateQMemOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                             mlir::Location loc) {
  llvm::StringRef query = args.empty() ? "" : extractStringArg(args[0].get());
  llvm::StringRef space = args.size() > 1 ? extractStringArg(args[1].get())
                                         : llvm::StringRef(memory::DEFAULT_SPACE);
  llvm::StringRef sid = session::DEFAULT_SID;

  auto memSpace = [&]() {
    if (space == memory::LTM) return mlir::ais::MemorySpace::LTM;
    if (space == memory::EPISODIC) return mlir::ais::MemorySpace::Episodic;
    return mlir::ais::MemorySpace::STM;
  }();

  auto resultHandleType = mlir::ais::HandleType::get(&gen.context, memSpace);
  auto spaceStr = space.empty() ? std::string(memory::DEFAULT_SPACE) : space.str();
  auto spaceAttr = gen.builder.getStringAttr(spaceStr);

  return gen.builder.create<mlir::ais::QMemOp>(loc, resultHandleType,
                                              gen.builder.getStringAttr(query),
                                              gen.builder.getStringAttr(sid),
                                              spaceAttr, /*limit=*/nullptr);
}

mlir::Value MLIRGenOperations::generateUMemOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                             mlir::Location loc) {
  llvm::StringRef space = args.size() > 1 ? extractStringArg(args[1].get())
                                         : llvm::StringRef(memory::DEFAULT_SPACE);

  mlir::Value value = args.empty() ? nullptr : gen.generateExpression(args[0].get());
  if (!value) {
    value = gen.builder.create<mlir::arith::ConstantOp>(loc, gen.builder.getI64IntegerAttr(0));
  }

  gen.builder.create<mlir::ais::UMemOp>(loc, value, gen.builder.getStringAttr(space));
  return value;
}

mlir::Value MLIRGenOperations::generateInvOp(MLIRGen &gen, llvm::StringRef callee,
                                           llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                           mlir::Location loc) {
  llvm::StringRef capability = args.empty() ? callee : extractStringArg(args[0].get());
  if (capability.empty()) capability = callee;

  llvm::StringRef params = args.size() > 1 ? extractStringArg(args[1].get())
                                          : llvm::StringRef(data::EMPTY_JSON);

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::InvOp>(loc, tokenType,
                                            gen.builder.getStringAttr(capability),
                                            gen.builder.getStringAttr(params));
}

mlir::Value MLIRGenOperations::generateRsnOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                           mlir::Location loc) {
  llvm::SmallVector<mlir::Value, 4> contextArgs;
  llvm::StringRef templateStr;

  if (!args.empty()) {
    // Check if the first arg is a simple string literal or an expression
    if (auto strLit = llvm::dyn_cast<StringLiteralExpr>(args[0].get())) {
      // Simple case: string literal becomes the template
      templateStr = strLit->getValue();
    } else {
      // Complex case: expression (e.g., "prefix" + var + "suffix")
      // Generate the expression and pass as context
      // Use empty template - the context operand becomes the prompt
      templateStr = "";
      if (auto arg = gen.generateExpression(args[0].get())) {
        contextArgs.push_back(arg);
      }
    }
  }

  // Process additional arguments as context
  for (size_t i = 1; i < args.size(); ++i) {
    if (!llvm::isa<StringLiteralExpr>(args[i].get())) {
      if (auto arg = gen.generateExpression(args[i].get())) {
        contextArgs.push_back(arg);
      }
    }
  }

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::RsnOp>(loc, tokenType,
                                            gen.builder.getStringAttr(templateStr),
                                            contextArgs);
}

mlir::Value MLIRGenOperations::generateReflectOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                                mlir::Location loc,
                                                llvm::SmallVectorImpl<mlir::Value> &contextArgs) {
  llvm::StringRef traceId = args.empty() ? "" : extractStringArg(args[0].get());

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::ReflectOp>(loc, tokenType,
                                                 gen.builder.getStringAttr(traceId),
                                                 contextArgs);
}

mlir::Value MLIRGenOperations::generatePlanOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                             mlir::Location loc,
                                             llvm::SmallVectorImpl<mlir::Value> &contextArgs) {
  auto goalType = mlir::ais::GoalType::get(&gen.context, /*priority=*/1);
  llvm::StringRef goal = args.empty() ? "" : extractStringArg(args[0].get());

  return gen.builder.create<mlir::ais::PlanOp>(loc, goalType,
                                             gen.builder.getStringAttr(goal),
                                             contextArgs);
}

mlir::Value MLIRGenOperations::generateVerifyOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                               mlir::Location loc) {
  if (args.size() < 2) {
    gen.emitError(loc, "verify requires at least two arguments");
    return nullptr;
  }

  mlir::Value claim = gen.generateExpression(args[0].get());
  mlir::Value evidence = gen.generateExpression(args[1].get());
  if (!claim || !evidence) return nullptr;

  llvm::StringRef templateStr = args.size() > 2 ? extractStringArg(args[2].get()) : "";

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::VerifyOp>(loc, tokenType, claim, evidence,
                                                gen.builder.getStringAttr(templateStr));
}

mlir::Value MLIRGenOperations::generateExcOp(MLIRGen &gen, llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                            mlir::Location loc,
                                            llvm::SmallVectorImpl<mlir::Value> &contextArgs) {
  llvm::StringRef code = args.empty() ? "" : extractStringArg(args[0].get());

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::ExcOp>(loc, tokenType,
                                            gen.builder.getStringAttr(code),
                                            contextArgs);
}

mlir::Value MLIRGenOperations::generateCommunicateOp(MLIRGen &gen,
                                                    llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                                    mlir::Location loc,
                                                    llvm::SmallVectorImpl<mlir::Value> &contextArgs) {
  llvm::StringRef recipient = args.empty() ? "" : extractStringArg(args[0].get());
  llvm::StringRef protocol = args.size() > 1 ? extractStringArg(args[1].get()) : "";

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::CommunicateOp>(
      loc, tokenType, gen.builder.getStringAttr(recipient),
      protocol.empty() ? nullptr : gen.builder.getStringAttr(protocol),
      /*payload=*/nullptr, contextArgs);
}

mlir::Value MLIRGenOperations::generateWaitAllOp(MLIRGen &gen,
                                                llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                                mlir::Location loc) {
  llvm::SmallVector<mlir::Value, 4> tokenArgs;
  for (const auto &argExpr : args) {
    if (auto arg = gen.generateExpression(argExpr.get())) {
      tokenArgs.push_back(arg);
    }
  }

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::WaitAllOp>(loc, tokenType, tokenArgs);
}

mlir::Value MLIRGenOperations::generateMergeOp(MLIRGen &gen,
                                              llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                              mlir::Location loc) {
  llvm::SmallVector<mlir::Value, 4> tokenArgs;
  for (const auto &argExpr : args) {
    if (auto arg = gen.generateExpression(argExpr.get())) {
      tokenArgs.push_back(arg);
    }
  }

  auto i64Type = gen.builder.getI64Type();
  auto tokenType = mlir::ais::TokenType::get(&gen.context, i64Type);

  return gen.builder.create<mlir::ais::MergeOp>(loc, tokenType, tokenArgs);
}
