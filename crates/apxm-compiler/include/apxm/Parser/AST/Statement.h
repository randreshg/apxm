/*
 * @file Statement.h
 * @brief Control-flow constructs that appear inside APXM flows & capabilities.
 *
 * The Stmt hierarchy models:
 * - Local binding (let)
 * - Control flow (if, loop, try/catch)
 * - Concurrency (parallel)
 * - Flow exit (return)
 * - Expression statements whose value is discarded.
 *
 * All children are owned uniquely, ensuring the AST is a DAG that can be
 * walked freely by passes without worrying about shared subtrees.
 */

#ifndef APXM_PARSER_STATEMENT_H
#define APXM_PARSER_STATEMENT_H

#include "apxm/Parser/AST/ASTNode.h"
#include "apxm/Parser/AST/Expression.h"
#include "llvm/ADT/ArrayRef.h"
#include <optional>
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <vector>

namespace apxm::parser {

class Stmt : public ASTNode {
protected:
  Stmt(Kind kind, Location loc) : ASTNode(kind, loc) {}

public:
  static bool classof(const ASTNode *node) {
    const Kind kind = node->getKind();
    return kind >= Kind::LetStmt && kind <= Kind::TryCatchStmt;
  }
};

class LetStmt final : public Stmt {
  const std::string varName;
  std::optional<std::string> typeAnnotation;
  std::unique_ptr<Expr> initExpr;

public:
  LetStmt(Location loc, llvm::StringRef varName, std::optional<llvm::StringRef> typeAnnotation,
          std::unique_ptr<Expr> initExpr)
      : Stmt(Kind::LetStmt, loc), varName(varName.str()), initExpr(std::move(initExpr)) {
    assert(!varName.empty() && "Variable name cannot be empty");
    assert(this->initExpr && "Initializer cannot be null");

    if (typeAnnotation) {
      this->typeAnnotation = typeAnnotation->str();
    }
  }

  llvm::StringRef getVarName() const noexcept { return varName; }
  const std::optional<std::string> &getTypeAnnotation() const noexcept { return typeAnnotation; }
  const Expr *getInitExpr() const noexcept { return initExpr.get(); }
  Expr *getInitExpr() noexcept { return initExpr.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::LetStmt;
  }
};

class ReturnStmt final : public Stmt {
  std::unique_ptr<Expr> returnExpr;

public:
  ReturnStmt(Location loc, std::unique_ptr<Expr> returnExpr)
      : Stmt(Kind::ReturnStmt, loc), returnExpr(std::move(returnExpr)) {
    assert(this->returnExpr && "Return expression cannot be null");
  }

  const Expr *getReturnExpr() const noexcept { return returnExpr.get(); }
  Expr *getReturnExpr() noexcept { return returnExpr.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::ReturnStmt;
  }
};

class IfStmt final : public Stmt {
  std::unique_ptr<Expr> condition;
  llvm::SmallVector<std::unique_ptr<Stmt>, 4> thenStmts;
  llvm::SmallVector<std::unique_ptr<Stmt>, 4> elseStmts;

public:
  IfStmt(Location loc, std::unique_ptr<Expr> condition,
         llvm::MutableArrayRef<std::unique_ptr<Stmt>> thenStmts,
         llvm::MutableArrayRef<std::unique_ptr<Stmt>> elseStmts = {})
      : Stmt(Kind::IfStmt, loc), condition(std::move(condition)) {
    assert(this->condition && "Condition cannot be null");

    this->thenStmts.reserve(thenStmts.size());
    for (auto &&stmt : thenStmts) {
      this->thenStmts.push_back(std::move(stmt));
    }

    this->elseStmts.reserve(elseStmts.size());
    for (auto &&stmt : elseStmts) {
      this->elseStmts.push_back(std::move(stmt));
    }
  }

  const Expr *getCondition() const noexcept { return condition.get(); }
  Expr *getCondition() noexcept { return condition.get(); }
  llvm::ArrayRef<std::unique_ptr<Stmt>> getThenStmts() const noexcept {
    return {thenStmts.data(), thenStmts.size()};
  }
  llvm::ArrayRef<std::unique_ptr<Stmt>> getElseStmts() const noexcept {
    return {elseStmts.data(), elseStmts.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::IfStmt;
  }
};

class ExprStmt final : public Stmt {
  std::unique_ptr<Expr> expr;

public:
  ExprStmt(Location loc, std::unique_ptr<Expr> expr)
      : Stmt(Kind::ExprStmt, loc), expr(std::move(expr)) {
    assert(this->expr && "Expression cannot be null");
  }

  const Expr *getExpr() const noexcept { return expr.get(); }
  Expr *getExpr() noexcept { return expr.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::ExprStmt;
  }
};

class ParallelStmt final : public Stmt {
  llvm::SmallVector<std::unique_ptr<Stmt>, 4> body;

public:
  ParallelStmt(Location loc, llvm::MutableArrayRef<std::unique_ptr<Stmt>> body)
      : Stmt(Kind::ParallelStmt, loc) {
    this->body.reserve(body.size());
    for (auto &&stmt : body) {
      this->body.push_back(std::move(stmt));
    }
  }

  llvm::ArrayRef<std::unique_ptr<Stmt>> getBody() const noexcept {
    return {body.data(), body.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::ParallelStmt;
  }
};

class LoopStmt final : public Stmt {
  const std::string varName;
  std::unique_ptr<Expr> collection;
  llvm::SmallVector<std::unique_ptr<Stmt>, 4> body;

public:
  LoopStmt(Location loc, llvm::StringRef varName, std::unique_ptr<Expr> collection,
           llvm::MutableArrayRef<std::unique_ptr<Stmt>> body)
      : Stmt(Kind::LoopStmt, loc), varName(varName.str()), collection(std::move(collection)) {
    assert(!varName.empty() && "Variable name cannot be empty");
    assert(this->collection && "Collection cannot be null");

    this->body.reserve(body.size());
    for (auto &&stmt : body) {
      this->body.push_back(std::move(stmt));
    }
  }

  llvm::StringRef getVarName() const noexcept { return varName; }
  const Expr *getCollection() const noexcept { return collection.get(); }
  Expr *getCollection() noexcept { return collection.get(); }
  llvm::ArrayRef<std::unique_ptr<Stmt>> getBody() const noexcept {
    return {body.data(), body.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::LoopStmt;
  }
};

class TryCatchStmt final : public Stmt {
  llvm::SmallVector<std::unique_ptr<Stmt>, 4> tryBody;
  llvm::SmallVector<std::unique_ptr<Stmt>, 4> catchBody;

public:
  TryCatchStmt(Location loc, llvm::MutableArrayRef<std::unique_ptr<Stmt>> tryBody,
               llvm::MutableArrayRef<std::unique_ptr<Stmt>> catchBody)
      : Stmt(Kind::TryCatchStmt, loc) {
    this->tryBody.reserve(tryBody.size());
    for (auto &&stmt : tryBody) {
      this->tryBody.push_back(std::move(stmt));
    }

    this->catchBody.reserve(catchBody.size());
    for (auto &&stmt : catchBody) {
      this->catchBody.push_back(std::move(stmt));
    }
  }

  llvm::ArrayRef<std::unique_ptr<Stmt>> getTryBody() const noexcept {
    return {tryBody.data(), tryBody.size()};
  }
  llvm::ArrayRef<std::unique_ptr<Stmt>> getCatchBody() const noexcept {
    return {catchBody.data(), catchBody.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::TryCatchStmt;
  }
};

} // namespace apxm::parser

#endif // APXM_PARSER_STATEMENT_H
