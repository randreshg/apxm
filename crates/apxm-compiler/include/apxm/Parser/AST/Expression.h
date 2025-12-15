/*
 * @file Expression.h
 * @brief Everything that can be evaluated to a value inside APXM.
 *
 * The hierarchy rooted in Expr covers:
 * - Literals (string, number, boolean, null)
 * - Access (variable, call, member, subscript, array)
 * - Arithmetic/Logic operators (binary, unary, assignment)
 * - BDI-specific forms (plan, reflect, verify, execute, communicate, waitAll, merge)
 *
 * Every node is immutable after creation, owns its children via unique_ptr
 * and exposes const/non-const accessors for traversal and mutation passes.
 */

#ifndef APXM_PARSER_EXPRESSION_H
#define APXM_PARSER_EXPRESSION_H

#include "apxm/Parser/AST/ASTNode.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>

namespace apxm::parser {

class Expr : public ASTNode {
protected:
  Expr(Kind kind, Location loc) : ASTNode(kind, loc) {}

public:
  static bool classof(const ASTNode *node) {
    const Kind kind = node->getKind();
    return kind >= Kind::StringLiteralExpr && kind <= Kind::MergeExpr;
  }
};

//===----------------------------------------------------------------------===//
// Literal Expressions
//===----------------------------------------------------------------------===//

class StringLiteralExpr final : public Expr {
  const std::string value;

public:
  StringLiteralExpr(Location loc, llvm::StringRef value)
      : Expr(Kind::StringLiteralExpr, loc), value(value.str()) {
    assert(!value.empty() && "String literal cannot be empty");
  }

  llvm::StringRef getValue() const noexcept { return value; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::StringLiteralExpr;
  }
};

class NumberLiteralExpr final : public Expr {
  const double value;

public:
  NumberLiteralExpr(Location loc, double value)
      : Expr(Kind::NumberLiteralExpr, loc), value(value) {}

  double getValue() const noexcept { return value; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::NumberLiteralExpr;
  }
};

class BooleanLiteralExpr final : public Expr {
  const bool value;

public:
  BooleanLiteralExpr(Location loc, bool value)
      : Expr(Kind::BooleanLiteralExpr, loc), value(value) {}

  bool getValue() const noexcept { return value; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::BooleanLiteralExpr;
  }
};

class NullLiteralExpr final : public Expr {
public:
  explicit NullLiteralExpr(Location loc) : Expr(Kind::NullLiteralExpr, loc) {}

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::NullLiteralExpr;
  }
};

//===----------------------------------------------------------------------===//
// Variable and Call Expressions
//===----------------------------------------------------------------------===//

class VarExpr final : public Expr {
  const std::string name;

public:
  VarExpr(Location loc, llvm::StringRef name)
      : Expr(Kind::VarExpr, loc), name(name.str()) {
    assert(!name.empty() && "Variable name cannot be empty");
  }

  llvm::StringRef getName() const noexcept { return name; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::VarExpr;
  }
};

class CallExpr final : public Expr {
  const std::string callee;
  llvm::SmallVector<std::unique_ptr<Expr>, 4> args;

public:
  CallExpr(Location loc, llvm::StringRef callee,
           llvm::MutableArrayRef<std::unique_ptr<Expr>> args)
      : Expr(Kind::CallExpr, loc), callee(callee.str()) {
    assert(!callee.empty() && "Callee name cannot be empty");

    this->args.reserve(args.size());
    for (auto &&arg : args) {
      this->args.push_back(std::move(arg));
    }
  }

  llvm::StringRef getCallee() const noexcept { return callee; }
  llvm::ArrayRef<std::unique_ptr<Expr>> getArgs() const noexcept {
    return {args.data(), args.size()};
  }

  llvm::SmallVector<std::unique_ptr<Expr>, 4> takeArgs() noexcept {
    return std::move(args);
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::CallExpr;
  }
};

//===----------------------------------------------------------------------===//
// Access Expressions
//===----------------------------------------------------------------------===//

class MemberAccessExpr final : public Expr {
  std::unique_ptr<Expr> object;
  const std::string member;

public:
  MemberAccessExpr(Location loc, std::unique_ptr<Expr> object, llvm::StringRef member)
      : Expr(Kind::MemberAccessExpr, loc), object(std::move(object)), member(member.str()) {
    assert(!member.empty() && "Member name cannot be empty");
    assert(this->object && "Object expression cannot be null");
  }

  const Expr *getObject() const noexcept { return object.get(); }
  Expr *getObject() noexcept { return object.get(); }
  llvm::StringRef getMember() const noexcept { return member; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::MemberAccessExpr;
  }
};

class SubscriptExpr final : public Expr {
  std::unique_ptr<Expr> base;
  std::unique_ptr<Expr> index;

public:
  SubscriptExpr(Location loc, std::unique_ptr<Expr> base, std::unique_ptr<Expr> index)
      : Expr(Kind::SubscriptExpr, loc), base(std::move(base)), index(std::move(index)) {
    assert(this->base && "Base expression cannot be null");
    assert(this->index && "Index expression cannot be null");
  }

  const Expr *getBase() const noexcept { return base.get(); }
  Expr *getBase() noexcept { return base.get(); }
  const Expr *getIndex() const noexcept { return index.get(); }
  Expr *getIndex() noexcept { return index.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::SubscriptExpr;
  }
};

class ArrayExpr final : public Expr {
  llvm::SmallVector<std::unique_ptr<Expr>, 4> elements;

public:
  ArrayExpr(Location loc, llvm::MutableArrayRef<std::unique_ptr<Expr>> elements)
      : Expr(Kind::ArrayExpr, loc) {
    this->elements.reserve(elements.size());
    for (auto &&elem : elements) {
      this->elements.push_back(std::move(elem));
    }
  }

  llvm::ArrayRef<std::unique_ptr<Expr>> getElements() const noexcept {
    return {elements.data(), elements.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::ArrayExpr;
  }
};

//===----------------------------------------------------------------------===//
// Operator Expressions
//===----------------------------------------------------------------------===//

class BinaryExpr final : public Expr {
public:
  enum class Operator {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual
  };

private:
  const Operator op;
  std::unique_ptr<Expr> lhs;
  std::unique_ptr<Expr> rhs;

public:
  BinaryExpr(Location loc, Operator op, std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs)
      : Expr(Kind::BinaryExpr, loc), op(op), lhs(std::move(lhs)), rhs(std::move(rhs)) {
    assert(this->lhs && "Left operand cannot be null");
    assert(this->rhs && "Right operand cannot be null");
  }

  Operator getOperator() const noexcept { return op; }
  const Expr *getLHS() const noexcept { return lhs.get(); }
  Expr *getLHS() noexcept { return lhs.get(); }
  const Expr *getRHS() const noexcept { return rhs.get(); }
  Expr *getRHS() noexcept { return rhs.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::BinaryExpr;
  }
};

class UnaryExpr final : public Expr {
public:
  enum class Operator { Negate, Not };

private:
  const Operator op;
  std::unique_ptr<Expr> operand;

public:
  UnaryExpr(Location loc, Operator op, std::unique_ptr<Expr> operand)
      : Expr(Kind::UnaryExpr, loc), op(op), operand(std::move(operand)) {
    assert(this->operand && "Operand cannot be null");
  }

  Operator getOperator() const noexcept { return op; }
  const Expr *getOperand() const noexcept { return operand.get(); }
  Expr *getOperand() noexcept { return operand.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::UnaryExpr;
  }
};

class AssignmentExpr final : public Expr {
  std::unique_ptr<Expr> lhs;
  std::unique_ptr<Expr> rhs;

public:
  AssignmentExpr(Location loc, std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs)
      : Expr(Kind::AssignmentExpr, loc), lhs(std::move(lhs)), rhs(std::move(rhs)) {
    assert(this->lhs && "Left operand cannot be null");
    assert(this->rhs && "Right operand cannot be null");
  }

  const Expr *getLHS() const noexcept { return lhs.get(); }
  Expr *getLHS() noexcept { return lhs.get(); }
  const Expr *getRHS() const noexcept { return rhs.get(); }
  Expr *getRHS() noexcept { return rhs.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::AssignmentExpr;
  }
};

//===----------------------------------------------------------------------===//
// Special Operation Expressions
//===----------------------------------------------------------------------===//

class PlanExpr final : public Expr {
  const std::string goal;
  llvm::SmallVector<std::unique_ptr<Expr>, 4> context;

public:
  PlanExpr(Location loc, llvm::StringRef goal,
           llvm::MutableArrayRef<std::unique_ptr<Expr>> context)
      : Expr(Kind::PlanExpr, loc), goal(goal.str()) {
    assert(!goal.empty() && "Goal cannot be empty");

    this->context.reserve(context.size());
    for (auto &&ctx : context) {
      this->context.push_back(std::move(ctx));
    }
  }

  llvm::StringRef getGoal() const noexcept { return goal; }
  llvm::ArrayRef<std::unique_ptr<Expr>> getContext() const noexcept {
    return {context.data(), context.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::PlanExpr;
  }
};

class ReflectExpr final : public Expr {
  const std::string traceId;
  llvm::SmallVector<std::unique_ptr<Expr>, 4> context;

public:
  ReflectExpr(Location loc, llvm::StringRef traceId,
              llvm::MutableArrayRef<std::unique_ptr<Expr>> context)
      : Expr(Kind::ReflectExpr, loc), traceId(traceId.str()) {
    assert(!traceId.empty() && "Trace ID cannot be empty");

    this->context.reserve(context.size());
    for (auto &&ctx : context) {
      this->context.push_back(std::move(ctx));
    }
  }

  llvm::StringRef getTraceId() const noexcept { return traceId; }
  llvm::ArrayRef<std::unique_ptr<Expr>> getContext() const noexcept {
    return {context.data(), context.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::ReflectExpr;
  }
};

class VerifyExpr final : public Expr {
  std::unique_ptr<Expr> claim;
  std::unique_ptr<Expr> evidence;
  const std::string templateStr;

public:
  VerifyExpr(Location loc, std::unique_ptr<Expr> claim, std::unique_ptr<Expr> evidence,
             llvm::StringRef templateStr)
      : Expr(Kind::VerifyExpr, loc), claim(std::move(claim)), evidence(std::move(evidence)),
        templateStr(templateStr.str()) {
    assert(this->claim && "Claim cannot be null");
    assert(this->evidence && "Evidence cannot be null");
    assert(!templateStr.empty() && "Template string cannot be empty");
  }

  const Expr *getClaim() const noexcept { return claim.get(); }
  Expr *getClaim() noexcept { return claim.get(); }
  const Expr *getEvidence() const noexcept { return evidence.get(); }
  Expr *getEvidence() noexcept { return evidence.get(); }
  llvm::StringRef getTemplate() const noexcept { return templateStr; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::VerifyExpr;
  }
};

class ExecuteExpr final : public Expr {
  const std::string code;
  llvm::SmallVector<std::unique_ptr<Expr>, 4> context;

public:
  ExecuteExpr(Location loc, llvm::StringRef code,
              llvm::MutableArrayRef<std::unique_ptr<Expr>> context)
      : Expr(Kind::ExecuteExpr, loc), code(code.str()) {
    assert(!code.empty() && "Code string cannot be empty");

    this->context.reserve(context.size());
    for (auto &&ctx : context) {
      this->context.push_back(std::move(ctx));
    }
  }

  llvm::StringRef getCode() const noexcept { return code; }
  llvm::ArrayRef<std::unique_ptr<Expr>> getContext() const noexcept {
    return {context.data(), context.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::ExecuteExpr;
  }
};

class CommunicateExpr final : public Expr {
  const std::string recipient;
  const std::string protocol;
  llvm::SmallVector<std::unique_ptr<Expr>, 4> attachments;

public:
  CommunicateExpr(Location loc, llvm::StringRef recipient, llvm::StringRef protocol,
                  llvm::MutableArrayRef<std::unique_ptr<Expr>> attachments)
      : Expr(Kind::CommunicateExpr, loc), recipient(recipient.str()),
        protocol(protocol.str()) {
    assert(!recipient.empty() && "Recipient cannot be empty");
    assert(!protocol.empty() && "Protocol cannot be empty");

    this->attachments.reserve(attachments.size());
    for (auto &&attach : attachments) {
      this->attachments.push_back(std::move(attach));
    }
  }

  llvm::StringRef getRecipient() const noexcept { return recipient; }
  llvm::StringRef getProtocol() const noexcept { return protocol; }
  llvm::ArrayRef<std::unique_ptr<Expr>> getAttachments() const noexcept {
    return {attachments.data(), attachments.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::CommunicateExpr;
  }
};

class WaitAllExpr final : public Expr {
  llvm::SmallVector<std::unique_ptr<Expr>, 4> tokens;

public:
  WaitAllExpr(Location loc, llvm::MutableArrayRef<std::unique_ptr<Expr>> tokens)
      : Expr(Kind::WaitAllExpr, loc) {
    this->tokens.reserve(tokens.size());
    for (auto &&token : tokens) {
      this->tokens.push_back(std::move(token));
    }
  }

  llvm::ArrayRef<std::unique_ptr<Expr>> getTokens() const noexcept {
    return {tokens.data(), tokens.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::WaitAllExpr;
  }
};

class MergeExpr final : public Expr {
  llvm::SmallVector<std::unique_ptr<Expr>, 4> values;

public:
  MergeExpr(Location loc, llvm::MutableArrayRef<std::unique_ptr<Expr>> values)
      : Expr(Kind::MergeExpr, loc) {
    this->values.reserve(values.size());
    for (auto &&value : values) {
      this->values.push_back(std::move(value));
    }
  }

  llvm::ArrayRef<std::unique_ptr<Expr>> getValues() const noexcept {
    return {values.data(), values.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::MergeExpr;
  }
};

} // namespace apxm::parser

#endif // APXM_PARSER_EXPRESSION_H
