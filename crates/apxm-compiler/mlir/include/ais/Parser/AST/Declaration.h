/*
 * @file Declaration.h
 * @brief AST nodes that introduce names into an APXM.
 *
 * This header file contains every top-level definition an agent can contain:
 * - BDI artefacts (beliefs, goals, event handlers)
 * - Structural artefacts (memories, capabilities, flows)
 * - The Agent itself, which aggregates all of the above
 *
 * Each node is immutable after construction, carries its source location,
 * and participates in the RTTI hierarchy.
 */

#ifndef APXM_PARSER_DECLARATION_H
#define APXM_PARSER_DECLARATION_H

#include "ais/Parser/AST/ASTNode.h"
#include "ais/Parser/AST/Expression.h"
#include "ais/Parser/AST/Statement.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <utility>

namespace apxm::parser {

//===----------------------------------------------------------------------===//
// BDI Declarations
//===----------------------------------------------------------------------===//

class BeliefDecl final : public ASTNode {
  const std::string name;
  std::unique_ptr<Expr> source;

public:
  BeliefDecl(Location loc, llvm::StringRef name, std::unique_ptr<Expr> source)
      : ASTNode(Kind::BeliefDecl, loc), name(name.str()), source(std::move(source)) {
    assert(!name.empty() && "Belief name cannot be empty");
  }

  llvm::StringRef getName() const noexcept { return name; }
  const Expr *getSource() const noexcept { return source.get(); }
  Expr *getSource() noexcept { return source.get(); }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::BeliefDecl;
  }
};

class GoalDecl final : public ASTNode {
  const std::string name;
  const std::string description;
  const int priority;

public:
  GoalDecl(Location loc, llvm::StringRef name, llvm::StringRef description, int priority = 1)
      : ASTNode(Kind::GoalDecl, loc), name(name.str()), description(description.str()),
        priority(priority) {
    assert(!name.empty() && "Goal name cannot be empty");
    assert(priority > 0 && "Priority must be positive");
  }

  llvm::StringRef getName() const noexcept { return name; }
  llvm::StringRef getDescription() const noexcept { return description; }
  int getPriority() const noexcept { return priority; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::GoalDecl;
  }
};

class OnEventDecl final : public ASTNode {
  const std::string eventType;
  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  llvm::SmallVector<std::unique_ptr<Expr>, 4> conditions;
  llvm::SmallVector<std::unique_ptr<Expr>, 4> contextExprs;
  llvm::SmallVector<std::unique_ptr<Stmt>, 4> body;

public:
  OnEventDecl(Location loc, llvm::StringRef eventType,
              llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> params,
              llvm::MutableArrayRef<std::unique_ptr<Expr>> conditions,
              llvm::MutableArrayRef<std::unique_ptr<Expr>> contextExprs,
              llvm::MutableArrayRef<std::unique_ptr<Stmt>> body)
      : ASTNode(Kind::OnEventDecl, loc), eventType(eventType.str()) {
    assert(!eventType.empty() && "Event type cannot be empty");

    this->params.reserve(params.size());
    for (const auto &param : params) {
      this->params.emplace_back(param.first.str(), param.second.str());
    }

    this->conditions.reserve(conditions.size());
    for (auto &&cond : conditions) {
      this->conditions.push_back(std::move(cond));
    }

    this->contextExprs.reserve(contextExprs.size());
    for (auto &&ctx : contextExprs) {
      this->contextExprs.push_back(std::move(ctx));
    }

    this->body.reserve(body.size());
    for (auto &&stmt : body) {
      this->body.push_back(std::move(stmt));
    }
  }

  llvm::StringRef getEventType() const noexcept { return eventType; }
  llvm::ArrayRef<std::pair<std::string, std::string>> getParams() const noexcept {
    return params;
  }
  llvm::ArrayRef<std::unique_ptr<Expr>> getConditions() const noexcept {
    return {conditions.data(), conditions.size()};
  }
  llvm::ArrayRef<std::unique_ptr<Expr>> getContextExprs() const noexcept {
    return {contextExprs.data(), contextExprs.size()};
  }
  llvm::ArrayRef<std::unique_ptr<Stmt>> getBody() const noexcept {
    return {body.data(), body.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::OnEventDecl;
  }
};

//===----------------------------------------------------------------------===//
// Structural Declarations
//===----------------------------------------------------------------------===//

class MemoryDecl final : public ASTNode {
  const std::string name;
  const std::string tier; // STM, LTM, Episodic

public:
  MemoryDecl(Location loc, llvm::StringRef name, llvm::StringRef tier)
      : ASTNode(Kind::MemoryDecl, loc), name(name.str()), tier(tier.str()) {
    assert(!name.empty() && "Memory name cannot be empty");
    assert(!tier.empty() && "Memory tier cannot be empty");
  }

  llvm::StringRef getName() const noexcept { return name; }
  llvm::StringRef getTier() const noexcept { return tier; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::MemoryDecl;
  }
};

class CapabilityDecl final : public ASTNode {
  const std::string name;
  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  const std::string returnType;

public:
  CapabilityDecl(Location loc, llvm::StringRef name,
                 llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> params,
                 llvm::StringRef returnType)
      : ASTNode(Kind::CapabilityDecl, loc), name(name.str()), returnType(returnType.str()) {
    assert(!name.empty() && "Capability name cannot be empty");
    assert(!returnType.empty() && "Return type cannot be empty");

    this->params.reserve(params.size());
    for (const auto &param : params) {
      this->params.emplace_back(param.first.str(), param.second.str());
    }
  }

  llvm::StringRef getName() const noexcept { return name; }
  llvm::ArrayRef<std::pair<std::string, std::string>> getParams() const noexcept {
    return params;
  }
  llvm::StringRef getReturnType() const noexcept { return returnType; }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::CapabilityDecl;
  }
};

class FlowDecl final : public ASTNode {
  const std::string name;
  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  const std::string returnType;
  const bool isEntry;  // @entry annotation marker
  llvm::SmallVector<std::unique_ptr<Stmt>, 8> body;

public:
  FlowDecl(Location loc, llvm::StringRef name,
           llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> params,
           llvm::StringRef returnType,
           bool isEntry,
           llvm::MutableArrayRef<std::unique_ptr<Stmt>> body)
      : ASTNode(Kind::FlowDecl, loc), name(name.str()), returnType(returnType.str()),
        isEntry(isEntry) {
    assert(!name.empty() && "Flow name cannot be empty");
    assert(!returnType.empty() && "Return type cannot be empty");

    this->params.reserve(params.size());
    for (const auto &param : params) {
      this->params.emplace_back(param.first.str(), param.second.str());
    }

    this->body.reserve(body.size());
    for (auto &&stmt : body) {
      this->body.push_back(std::move(stmt));
    }
  }

  llvm::StringRef getName() const noexcept { return name; }
  llvm::ArrayRef<std::pair<std::string, std::string>> getParams() const noexcept {
    return params;
  }
  llvm::StringRef getReturnType() const noexcept { return returnType; }
  bool isEntryFlow() const noexcept { return isEntry; }
  llvm::ArrayRef<std::unique_ptr<Stmt>> getBody() const noexcept {
    return {body.data(), body.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::FlowDecl;
  }
};

class AgentDecl final : public ASTNode {
  const std::string name;
  llvm::SmallVector<std::unique_ptr<MemoryDecl>, 4> memoryDecls;
  llvm::SmallVector<std::unique_ptr<CapabilityDecl>, 4> capabilityDecls;
  llvm::SmallVector<std::unique_ptr<FlowDecl>, 4> flowDecls;
  llvm::SmallVector<std::unique_ptr<BeliefDecl>, 4> beliefDecls;
  llvm::SmallVector<std::unique_ptr<GoalDecl>, 4> goalDecls;
  llvm::SmallVector<std::unique_ptr<OnEventDecl>, 4> onEventDecls;

public:
  AgentDecl(Location loc, llvm::StringRef name,
            llvm::MutableArrayRef<std::unique_ptr<MemoryDecl>> memoryDecls,
            llvm::MutableArrayRef<std::unique_ptr<CapabilityDecl>> capabilityDecls,
            llvm::MutableArrayRef<std::unique_ptr<FlowDecl>> flowDecls,
            llvm::MutableArrayRef<std::unique_ptr<BeliefDecl>> beliefDecls = {},
            llvm::MutableArrayRef<std::unique_ptr<GoalDecl>> goalDecls = {},
            llvm::MutableArrayRef<std::unique_ptr<OnEventDecl>> onEventDecls = {})
      : ASTNode(Kind::AgentDecl, loc), name(name.str()) {
    assert(!name.empty() && "Agent name cannot be empty");

    this->memoryDecls.reserve(memoryDecls.size());
    for (auto &&decl : memoryDecls) {
      this->memoryDecls.push_back(std::move(decl));
    }

    this->capabilityDecls.reserve(capabilityDecls.size());
    for (auto &&decl : capabilityDecls) {
      this->capabilityDecls.push_back(std::move(decl));
    }

    this->flowDecls.reserve(flowDecls.size());
    for (auto &&decl : flowDecls) {
      this->flowDecls.push_back(std::move(decl));
    }

    this->beliefDecls.reserve(beliefDecls.size());
    for (auto &&decl : beliefDecls) {
      this->beliefDecls.push_back(std::move(decl));
    }

    this->goalDecls.reserve(goalDecls.size());
    for (auto &&decl : goalDecls) {
      this->goalDecls.push_back(std::move(decl));
    }

    this->onEventDecls.reserve(onEventDecls.size());
    for (auto &&decl : onEventDecls) {
      this->onEventDecls.push_back(std::move(decl));
    }
  }

  llvm::StringRef getName() const noexcept { return name; }
  llvm::ArrayRef<std::unique_ptr<MemoryDecl>> getMemoryDecls() const noexcept {
    return {memoryDecls.data(), memoryDecls.size()};
  }
  llvm::ArrayRef<std::unique_ptr<CapabilityDecl>> getCapabilityDecls() const noexcept {
    return {capabilityDecls.data(), capabilityDecls.size()};
  }
  llvm::ArrayRef<std::unique_ptr<FlowDecl>> getFlowDecls() const noexcept {
    return {flowDecls.data(), flowDecls.size()};
  }
  llvm::ArrayRef<std::unique_ptr<BeliefDecl>> getBeliefDecls() const noexcept {
    return {beliefDecls.data(), beliefDecls.size()};
  }
  llvm::ArrayRef<std::unique_ptr<GoalDecl>> getGoalDecls() const noexcept {
    return {goalDecls.data(), goalDecls.size()};
  }
  llvm::ArrayRef<std::unique_ptr<OnEventDecl>> getOnEventDecls() const noexcept {
    return {onEventDecls.data(), onEventDecls.size()};
  }

  static bool classof(const ASTNode *node) {
    return node->getKind() == Kind::AgentDecl;
  }
};

} // namespace apxm::parser

#endif // APXM_PARSER_DECLARATION_H
