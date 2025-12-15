/*
 * @file MLIRGenDeclarations.cpp
 * @brief Implementation for agent, flow, and event handler generation.
 *
 * This file contains the implementation for generating MLIR declarations:
 * - Agent declaration
 * - Flow declaration
 * - Event handler declaration
 */

#include "apxm/Parser/MLIR/MLIRGen.h"
#include "apxm/Dialect/AIS/IR/AISDialect.h"
#include "apxm/Dialect/AIS/IR/AISOps.h"
#include "apxm/Dialect/AIS/IR/AISTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

using namespace apxm::parser;
using namespace mlir;
using namespace mlir::ais;

mlir::OwningOpRef<mlir::ModuleOp> MLIRGen::generateModule(AgentDecl *agent) {
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

  if (failed(generateAgentDecl(module, agent))) {
    module->print(llvm::errs());
    module->erase();
    return nullptr;
  }

  if (failed(mlir::verify(module.getOperation()))) {
    module->print(llvm::errs());
    module->erase();
    return nullptr;
  }

  return module;
}

void MLIRGen::emitError(parser::Location loc, llvm::StringRef message) {
  emitError(getLocation(loc), message);
}

void MLIRGen::emitError(mlir::Location loc, llvm::StringRef message) {
  mlir::emitError(loc) << message;
}

mlir::LogicalResult MLIRGen::generateAgentDecl(mlir::ModuleOp module, AgentDecl *agent) {
  builder.setInsertionPointToEnd(module.getBody());

  mlir::Location agentLoc = getLocation(agent->getLocation());
  auto agentOp =
      builder.create<mlir::ais::AgentOp>(agentLoc, builder.getStringAttr(agent->getName()));

  auto makeDictAttr = [&](llvm::ArrayRef<mlir::NamedAttribute> attrs) {
    return mlir::DictionaryAttr::get(&context, attrs);
  };

  auto toLocationAttr = [&](const parser::Location &loc) -> mlir::NamedAttribute {
    auto locStr = builder.getStringAttr(loc.toString(srcMgr));
    return builder.getNamedAttr("location", locStr);
  };

  // Generate memory declarations
  memorySpaces.clear();
  llvm::SmallVector<mlir::Attribute, 4> memoryAttrs;
  for (const auto &memory : agent->getMemoryDecls()) {
    memorySpaces[memory->getName()] = toMemorySpace(memory->getTier());

    llvm::SmallVector<mlir::NamedAttribute, 3> fields;
    fields.push_back(builder.getNamedAttr("name", builder.getStringAttr(memory->getName())));
    fields.push_back(builder.getNamedAttr("tier", builder.getStringAttr(memory->getTier())));
    fields.push_back(toLocationAttr(memory->getLocation()));
    memoryAttrs.push_back(makeDictAttr(fields));
  }
  agentOp->setAttr("memories", builder.getArrayAttr(memoryAttrs));

  // Generate capability declarations
  llvm::SmallVector<mlir::Attribute, 4> capabilityAttrs;
  for (const auto &capability : agent->getCapabilityDecls()) {
    llvm::SmallVector<mlir::Attribute, 4> paramAttrs;
    for (const auto &param : capability->getParams()) {
      paramAttrs.push_back(makeDictAttr({
          builder.getNamedAttr("name", builder.getStringAttr(param.first)),
          builder.getNamedAttr("type", builder.getStringAttr(param.second)),
      }));
    }

    llvm::SmallVector<mlir::NamedAttribute, 4> fields;
    fields.push_back(builder.getNamedAttr("name", builder.getStringAttr(capability->getName())));
    fields.push_back(builder.getNamedAttr("params", builder.getArrayAttr(paramAttrs)));
    fields.push_back(
        builder.getNamedAttr("return_type", builder.getStringAttr(capability->getReturnType())));
    fields.push_back(toLocationAttr(capability->getLocation()));
    capabilityAttrs.push_back(makeDictAttr(fields));
  }
  agentOp->setAttr("capabilities", builder.getArrayAttr(capabilityAttrs));

  // Generate belief declarations
  llvm::SmallVector<mlir::Attribute, 4> beliefAttrs;
  for (const auto &belief : agent->getBeliefDecls()) {
    llvm::SmallVector<mlir::NamedAttribute, 4> fields;
    fields.push_back(builder.getNamedAttr("name", builder.getStringAttr(belief->getName())));
    fields.push_back(builder.getNamedAttr(
        "source", builder.getStringAttr(stringifyMetadataExpr(belief->getSource()))));
    fields.push_back(toLocationAttr(belief->getLocation()));
    beliefAttrs.push_back(makeDictAttr(fields));
  }
  agentOp->setAttr("beliefs", builder.getArrayAttr(beliefAttrs));

  // Generate goal declarations
  llvm::SmallVector<mlir::Attribute, 4> goalAttrs;
  for (const auto &goal : agent->getGoalDecls()) {
    llvm::SmallVector<mlir::NamedAttribute, 4> fields;
    fields.push_back(builder.getNamedAttr("name", builder.getStringAttr(goal->getName())));
    if (!goal->getDescription().empty()) {
      fields.push_back(
          builder.getNamedAttr("description", builder.getStringAttr(goal->getDescription())));
    }
    fields.push_back(
        builder.getNamedAttr("priority", builder.getI64IntegerAttr(goal->getPriority())));
    fields.push_back(toLocationAttr(goal->getLocation()));
    goalAttrs.push_back(makeDictAttr(fields));
  }
  agentOp->setAttr("goals", builder.getArrayAttr(goalAttrs));

  // Generate event handler declarations
  llvm::SmallVector<mlir::Attribute, 4> handlerAttrs;
  unsigned handlerIndex = 0;
  auto sanitizeEventName = [](llvm::StringRef eventType) -> std::string {
    std::string result;
    result.reserve(eventType.size());
    for (char c : eventType) {
      char lowered = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
      if (std::isalnum(static_cast<unsigned char>(lowered)) || lowered == '_') {
        result.push_back(lowered);
      } else {
        result.push_back('_');
      }
    }
    return result;
  };

  for (const auto &handler : agent->getOnEventDecls()) {
    bool returnsValue = blockHasReturn(handler->getBody());
    auto symbolName = llvm::formatv("{0}__on_{1}_{2}", agent->getName(),
                                    sanitizeEventName(handler->getEventType()), handlerIndex)
                          .str();

    if (failed(generateOnEventDecl(module, handler.get(), symbolName, returnsValue)))
      return mlir::failure();

    llvm::SmallVector<mlir::Attribute, 4> paramAttrs;
    for (const auto &param : handler->getParams()) {
      llvm::StringRef typeName = param.second;
      if (typeName.empty())
        typeName = "token";
      paramAttrs.push_back(makeDictAttr({
          builder.getNamedAttr("name", builder.getStringAttr(param.first)),
          builder.getNamedAttr("type", builder.getStringAttr(typeName)),
      }));
    }

    llvm::SmallVector<mlir::NamedAttribute, 6> fields;
    fields.push_back(builder.getNamedAttr("event", builder.getStringAttr(handler->getEventType())));
    fields.push_back(
        builder.getNamedAttr("symbol", mlir::SymbolRefAttr::get(&context, symbolName)));
    fields.push_back(builder.getNamedAttr("returns_value", builder.getBoolAttr(returnsValue)));
    if (!paramAttrs.empty())
      fields.push_back(builder.getNamedAttr("params", builder.getArrayAttr(paramAttrs)));

    if (!handler->getConditions().empty()) {
      llvm::SmallVector<mlir::Attribute, 4> conditionAttrs;
      for (const auto &cond : handler->getConditions()) {
        conditionAttrs.push_back(builder.getStringAttr(stringifyMetadataExpr(cond.get())));
      }
      fields.push_back(builder.getNamedAttr("conditions", builder.getArrayAttr(conditionAttrs)));
    }

    if (!handler->getContextExprs().empty()) {
      llvm::SmallVector<mlir::Attribute, 4> contextAttrs;
      for (const auto &ctx : handler->getContextExprs()) {
        contextAttrs.push_back(builder.getStringAttr(stringifyMetadataExpr(ctx.get())));
      }
      fields.push_back(builder.getNamedAttr("context", builder.getArrayAttr(contextAttrs)));
    }

    fields.push_back(toLocationAttr(handler->getLocation()));
    handlerAttrs.push_back(makeDictAttr(fields));
    ++handlerIndex;
  }

  if (!handlerAttrs.empty())
    agentOp->setAttr("handlers", builder.getArrayAttr(handlerAttrs));

  // Generate flow declarations as functions
  for (const auto &flow : agent->getFlowDecls()) {
    if (failed(generateFlowDecl(module, flow.get())))
      return mlir::failure();
  }

  return mlir::success();
}

std::string MLIRGen::stringifyMetadataExpr(Expr *expr) const {
  if (!expr)
    return "";

  auto joinExprList = [&](llvm::ArrayRef<std::unique_ptr<Expr>> elements) -> std::string {
    std::string text;
    for (size_t i = 0; i < elements.size(); ++i) {
      if (i)
        text.append(", ");
      text.append(stringifyMetadataExpr(elements[i].get()));
    }
    return text;
  };

  // Use RTTI for expression classification
  return llvm::TypeSwitch<Expr *, std::string>(expr)
      .Case<MemberAccessExpr>([&](auto *member) {
        std::string object;
        if (auto *var = llvm::dyn_cast<VarExpr>(member->getObject())) {
          object = var->getName().str();
        } else {
          object = stringifyMetadataExpr(member->getObject());
        }
        if (!object.empty())
          return object + "." + member->getMember().str();
        return member->getMember().str();
      })
      .Case<SubscriptExpr>([&](auto *subscript) {
        return stringifyMetadataExpr(subscript->getBase()) + "[" +
               stringifyMetadataExpr(subscript->getIndex()) + "]";
      })
      .Case<VarExpr>([&](auto *var) { return var->getName().str(); })
      .Case<CallExpr>([&](auto *call) {
        std::string text = call->getCallee().str() + "(";
        text.append(joinExprList(call->getArgs()));
        text.push_back(')');
        return text;
      })
      .Case<StringLiteralExpr>([&](auto *str) { return str->getValue().str(); })
      .Case<NumberLiteralExpr>([&](auto *num) {
        return llvm::formatv("{0}", num->getValue()).str();
      })
      .Case<BooleanLiteralExpr>([&](auto *boolExpr) {
        return boolExpr->getValue() ? "true" : "false";
      })
      .Case<NullLiteralExpr>([&](auto *) { return "null"; })
      .Case<BinaryExpr>([&](auto *binary) {
        auto opToStr = [](BinaryExpr::Operator op) -> llvm::StringRef {
          switch (op) {
          case BinaryExpr::Operator::Add: return "+";
          case BinaryExpr::Operator::Sub: return "-";
          case BinaryExpr::Operator::Mul: return "*";
          case BinaryExpr::Operator::Div: return "/";
          case BinaryExpr::Operator::Mod: return "%";
          case BinaryExpr::Operator::And: return "&&";
          case BinaryExpr::Operator::Or: return "||";
          case BinaryExpr::Operator::Equal: return "==";
          case BinaryExpr::Operator::NotEqual: return "!=";
          case BinaryExpr::Operator::Less: return "<";
          case BinaryExpr::Operator::LessEqual: return "<=";
          case BinaryExpr::Operator::Greater: return ">";
          case BinaryExpr::Operator::GreaterEqual: return ">=";
          }
          return "?";
        };
        return "(" + stringifyMetadataExpr(binary->getLHS()) + " " +
               opToStr(binary->getOperator()).str() + " " +
               stringifyMetadataExpr(binary->getRHS()) + ")";
      })
      .Case<UnaryExpr>([&](auto *unary) {
        auto symbol = unary->getOperator() == UnaryExpr::Operator::Not ? "!" : "-";
        return symbol + stringifyMetadataExpr(unary->getOperand());
      })
      .Case<PlanExpr>([&](auto *plan) {
        std::string text = "plan(\"" + plan->getGoal().str() + "\"";
        if (!plan->getContext().empty()) {
          text.append(", ");
          text.append(joinExprList(plan->getContext()));
        }
        text.push_back(')');
        return text;
      })
      .Case<ReflectExpr>([&](auto *reflect) {
        std::string text = "reflect(\"" + reflect->getTraceId().str() + "\"";
        if (!reflect->getContext().empty()) {
          text.append(", ");
          text.append(joinExprList(reflect->getContext()));
        }
        text.push_back(')');
        return text;
      })
      .Case<VerifyExpr>([&](auto *verify) {
        std::string text = "verify(" + stringifyMetadataExpr(verify->getClaim()) + ", " +
                           stringifyMetadataExpr(verify->getEvidence());
        if (!verify->getTemplate().empty()) {
          text.append(", \"");
          text.append(verify->getTemplate().str());
          text.append("\"");
        }
        text.push_back(')');
        return text;
      })
      .Case<ExecuteExpr>([&](auto *exec) {
        std::string text = "exec(\"" + exec->getCode().str() + "\"";
        if (!exec->getContext().empty()) {
          text.append(", ");
          text.append(joinExprList(exec->getContext()));
        }
        text.push_back(')');
        return text;
      })
      .Case<CommunicateExpr>([&](auto *communicate) {
        std::string text = "communicate(\"" + communicate->getRecipient().str() + "\", \"" +
                           communicate->getProtocol().str() + "\"";
        if (!communicate->getAttachments().empty()) {
          text.append(", ");
          text.append(joinExprList(communicate->getAttachments()));
        }
        text.push_back(')');
        return text;
      })
      .Case<WaitAllExpr>([&](auto *waitAll) {
        return "wait(" + joinExprList(waitAll->getTokens()) + ")";
      })
      .Case<MergeExpr>([&](auto *merge) {
        return "merge(" + joinExprList(merge->getValues()) + ")";
      })
      .Default([&](auto *) { return "<expr>"; });
}

mlir::LogicalResult MLIRGen::generateFlowDecl(mlir::ModuleOp module, FlowDecl *flow) {
  mlir::Location loc = getLocation(flow->getLocation());

  // Ensure functions are always inserted at module scope
  builder.setInsertionPointToEnd(module.getBody());

  // Build function type
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (const auto &[paramName, paramType] : flow->getParams()) {
    argTypes.push_back(getMLIRType(paramType));
  }

  mlir::Type resultType = getMLIRType(flow->getReturnType());
  auto funcType = builder.getFunctionType(argTypes, resultType);

  // Create function
  auto funcOp = builder.create<mlir::func::FuncOp>(loc, flow->getName(), funcType);
  funcOp.setPrivate();

  // Create entry block
  auto &entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  // Declare parameters in symbol table
  SymbolTable::ScopeTy varScope(symbolTable);
  for (size_t i = 0; i < flow->getParams().size(); ++i) {
    const auto &[paramName, paramType] = flow->getParams()[i];
    symbolTable.insert(paramName, entryBlock.getArgument(i));
  }

  // Generate statements
  for (const auto &stmt : flow->getBody()) {
    if (failed(generateStatement(stmt.get())))
      return mlir::failure();
  }

  // Add implicit return if no explicit return
  if (entryBlock.empty() || !isa<mlir::func::ReturnOp>(entryBlock.back())) {
    if (llvm::isa<mlir::NoneType>(resultType)) {
      builder.create<mlir::func::ReturnOp>(loc);
    } else {
      // Create default value for return type
      mlir::Value defaultValue;
      if (llvm::isa<mlir::FloatType>(resultType)) {
        defaultValue = builder.create<mlir::arith::ConstantOp>(
            loc, resultType, builder.getFloatAttr(resultType, 0.0));
      } else if (llvm::isa<mlir::IntegerType>(resultType)) {
        defaultValue = builder.create<mlir::arith::ConstantOp>(
            loc, resultType, builder.getIntegerAttr(resultType, 0));
      } else {
        // For token types, create empty string constant
        auto tokenType = mlir::ais::TokenType::get(&context, builder.getI64Type());
        auto constStr = builder.create<mlir::ais::ConstStrOp>(
            loc, tokenType, builder.getStringAttr(""));
        defaultValue = constStr.getResult();
      }
      builder.create<mlir::func::ReturnOp>(loc, defaultValue);
    }
  }

  return mlir::success();
}

mlir::LogicalResult MLIRGen::generateOnEventDecl(mlir::ModuleOp module, OnEventDecl *handler,
                                                 llvm::StringRef symbolName, bool returnsValue) {
  mlir::Location loc = getLocation(handler->getLocation());
  builder.setInsertionPointToEnd(module.getBody());

  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (const auto &[paramName, paramType] : handler->getParams()) {
    argTypes.push_back(getMLIRType(paramType));
  }

  llvm::SmallVector<mlir::Type, 1> resultTypes;
  mlir::Type tokenType;
  if (returnsValue) {
    tokenType = mlir::ais::TokenType::get(&context, builder.getI64Type());
    resultTypes.push_back(tokenType);
  }

  auto funcType = builder.getFunctionType(argTypes, resultTypes);
  auto funcOp = builder.create<mlir::func::FuncOp>(loc, symbolName, funcType);
  funcOp.setPrivate();

  auto &entryBlock = *funcOp.addEntryBlock();
  builder.setInsertionPointToStart(&entryBlock);

  SymbolTable::ScopeTy varScope(symbolTable);
  for (size_t i = 0; i < handler->getParams().size(); ++i) {
    const auto &param = handler->getParams()[i];
    symbolTable.insert(param.first, entryBlock.getArgument(i));
  }

  for (const auto &stmt : handler->getBody()) {
    if (failed(generateStatement(stmt.get())))
      return mlir::failure();
  }

  // Ensure proper termination
  if (entryBlock.empty() || !entryBlock.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    if (returnsValue) {
      auto placeholder = builder.create<mlir::ais::ConstStrOp>(
          loc, llvm::cast<mlir::ais::TokenType>(tokenType), builder.getStringAttr(""));
      builder.create<mlir::func::ReturnOp>(loc, placeholder.getResult());
    } else {
      builder.create<mlir::func::ReturnOp>(loc);
    }
  }

  return mlir::success();
}

bool MLIRGen::blockHasReturn(llvm::ArrayRef<std::unique_ptr<Stmt>> body) const {
  for (const auto &stmt : body) {
    if (llvm::isa<ReturnStmt>(stmt.get()))
      return true;

    // Recursively check compound statements
    if (auto *ifStmt = llvm::dyn_cast<IfStmt>(stmt.get())) {
      if (blockHasReturn(ifStmt->getThenStmts()) || blockHasReturn(ifStmt->getElseStmts()))
        return true;
      continue;
    }

    if (auto *parallelStmt = llvm::dyn_cast<ParallelStmt>(stmt.get())) {
      if (blockHasReturn(parallelStmt->getBody()))
        return true;
      continue;
    }

    if (auto *loopStmt = llvm::dyn_cast<LoopStmt>(stmt.get())) {
      if (blockHasReturn(loopStmt->getBody()))
        return true;
      continue;
    }

    if (auto *tryStmt = llvm::dyn_cast<TryCatchStmt>(stmt.get())) {
      if (blockHasReturn(tryStmt->getTryBody()) || blockHasReturn(tryStmt->getCatchBody()))
        return true;
      continue;
    }
  }

  return false;
}
