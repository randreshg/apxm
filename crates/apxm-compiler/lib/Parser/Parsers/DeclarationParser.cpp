/*
 * @file DeclarationParser.cpp
 * @brief Implementation of declaration parsing.
 *
 * This file contains the implementation of the declaration parsing functionality.
 * It provides methods to parse different types of declarations such as:
 * - Agent declaration
 * - Memory declaration
 * - Capability declaration
 * - Flow declaration
 * - Belief declaration
 * - Goal declaration
 * - On-event declaration
 */

#include "apxm/Parser/Parsers/DeclarationParser.h"
#include "apxm/Parser/Parsers/ExpressionParser.h"
#include "apxm/Parser/Parsers/StatementParser.h"
#include "apxm/Parser/Utils/Container.h"
#include "apxm/Parser/Utils/Token.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include <utility>

namespace {

llvm::SmallVector<std::pair<llvm::StringRef, llvm::StringRef>, 4>
makeStringRefPairs(const llvm::SmallVector<std::pair<std::string, std::string>, 4> &pairs) {
  llvm::SmallVector<std::pair<llvm::StringRef, llvm::StringRef>, 4> refs;
  refs.reserve(pairs.size());
  for (const auto &entry : pairs) {
    refs.emplace_back(entry.first, entry.second);
  }
  return refs;
}

} // namespace
#include <memory>

using namespace apxm::parser;

std::unique_ptr<AgentDecl> DeclarationParser::parseAgent() {
  if (!expect(TokenKind::kw_agent)) {
    synchronize();
    return nullptr;
  }

  Location nameLoc = getCurrentLocation();
  if (!expect(TokenKind::identifier)) {
    synchronize();
    return nullptr;
  }

  llvm::StringRef agentName = peek().spelling;
  advance();

  if (!expect(TokenKind::l_brace)) {
    synchronize();
    return nullptr;
  }

  // Parse declarations
  llvm::SmallVector<std::unique_ptr<MemoryDecl>, 4> memoryDecls;
  llvm::SmallVector<std::unique_ptr<CapabilityDecl>, 4> capabilityDecls;
  llvm::SmallVector<std::unique_ptr<FlowDecl>, 4> flowDecls;
  llvm::SmallVector<std::unique_ptr<BeliefDecl>, 4> beliefDecls;
  llvm::SmallVector<std::unique_ptr<GoalDecl>, 4> goalDecls;
  llvm::SmallVector<std::unique_ptr<OnEventDecl>, 4> onEventDecls;

  while (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
    if (peek(TokenKind::kw_memory)) {
      if (auto decl = parseMemoryDecl()) {
        memoryDecls.push_back(std::move(decl));
      }
    } else if (peek(TokenKind::kw_beliefs)) {
      if (!expect(TokenKind::kw_beliefs)) continue;
      if (!expect(TokenKind::l_brace)) continue;

      while (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
        if (auto decl = parseBeliefDecl()) {
          beliefDecls.push_back(std::move(decl));
        } else {
          synchronize();
        }
      }
      if (!expect(TokenKind::r_brace)) {
        synchronize();
      }
    } else if (peek(TokenKind::kw_goals)) {
      if (!expect(TokenKind::kw_goals)) continue;
      if (!expect(TokenKind::l_brace)) continue;

      while (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
        if (auto decl = parseGoalDecl()) {
          goalDecls.push_back(std::move(decl));
        } else {
          synchronize();
        }
      }
      if (!expect(TokenKind::r_brace)) {
        synchronize();
      }
    } else if (peek(TokenKind::kw_capability)) {
      if (auto decl = parseCapabilityDecl()) {
        capabilityDecls.push_back(std::move(decl));
      }
    } else if (peek(TokenKind::kw_flow)) {
      if (auto decl = parseFlowDecl()) {
        flowDecls.push_back(std::move(decl));
      }
    } else if (peek(TokenKind::kw_on)) {
      if (auto decl = parseOnEventDecl()) {
        onEventDecls.push_back(std::move(decl));
      }
    } else {
      emitError(getCurrentLocation(), "Unexpected declaration");
      synchronize();
    }
  }

  if (!expect(TokenKind::r_brace)) {
    synchronize();
    return nullptr;
  }

  // Deduplicate and validate containers
  deduplicateAndValidate(memoryDecls);
  deduplicateAndValidate(capabilityDecls);
  deduplicateAndValidate(flowDecls);
  deduplicateAndValidate(beliefDecls);
  deduplicateAndValidate(goalDecls);
  deduplicateAndValidate(onEventDecls);

  return std::make_unique<AgentDecl>(nameLoc, agentName, memoryDecls, capabilityDecls,
                                    flowDecls, beliefDecls, goalDecls, onEventDecls);
}

std::unique_ptr<MemoryDecl> DeclarationParser::parseMemoryDecl() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_memory)) return nullptr;
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = peek().spelling;
  advance();

  if (!expect(TokenKind::colon)) return nullptr;

  TokenKind tierKind = peek().kind;
  llvm::StringRef tier;
  if (tierKind == TokenKind::kw_STM || tierKind == TokenKind::kw_LTM ||
      tierKind == TokenKind::kw_Episodic) {
    tier = peek().spelling;
    advance();
  } else {
    emitError(getCurrentLocation(), "Expected memory tier (STM, LTM, or Episodic)");
    return nullptr;
  }

  if (!expect(TokenKind::semicolon)) return nullptr;

  return std::make_unique<MemoryDecl>(loc, name, tier);
}

std::unique_ptr<CapabilityDecl> DeclarationParser::parseCapabilityDecl() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_capability)) return nullptr;
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = peek().spelling;
  advance();

  if (!expect(TokenKind::l_paren)) return nullptr;

  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  if (!parseNamedTypeList(TokenKind::r_paren, "parameter", params, true)) {
    return nullptr;
  }

  if (!expect(TokenKind::arrow)) return nullptr;

  if (!peek(TokenKind::identifier) && !peek(TokenKind::kw_string) &&
      !peek(TokenKind::kw_number) && !peek(TokenKind::kw_bool) &&
      !peek(TokenKind::kw_json) && !peek(TokenKind::kw_void)) {
    emitError(getCurrentLocation(), "Expected return type");
    return nullptr;
  }

  llvm::StringRef returnType = peek().spelling;
  advance();

  if (!expect(TokenKind::semicolon)) return nullptr;

  auto paramRefs = makeStringRefPairs(params);
  return std::make_unique<CapabilityDecl>(loc, name, paramRefs, returnType);
}

std::unique_ptr<FlowDecl> DeclarationParser::parseFlowDecl() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_flow)) return nullptr;
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = peek().spelling;
  advance();

  if (!expect(TokenKind::l_paren)) return nullptr;

  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  if (!parseNamedTypeList(TokenKind::r_paren, "parameter", params, true)) {
    return nullptr;
  }

  if (!expect(TokenKind::arrow)) return nullptr;

  if (!peek(TokenKind::identifier) && !peek(TokenKind::kw_string) &&
      !peek(TokenKind::kw_number) && !peek(TokenKind::kw_bool) &&
      !peek(TokenKind::kw_json) && !peek(TokenKind::kw_void)) {
    emitError(getCurrentLocation(), "Expected return type");
    return nullptr;
  }

  llvm::StringRef returnType = peek().spelling;
  advance();

  if (!expect(TokenKind::l_brace)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 8> body;
  if (!stmtParser.parseStatementBlock(body)) {
    return nullptr;
  }

  if (!expect(TokenKind::r_brace)) return nullptr;

  auto paramRefs = makeStringRefPairs(params);
  return std::make_unique<FlowDecl>(loc, name, paramRefs, returnType, body);
}

std::unique_ptr<BeliefDecl> DeclarationParser::parseBeliefDecl() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = peek().spelling;
  advance();

  if (!expect(TokenKind::equal)) return nullptr;

  auto source = exprParser.parseExpression();
  if (!source) {
    synchronize();
    return nullptr;
  }

  if (!expect(TokenKind::semicolon)) return nullptr;

  return std::make_unique<BeliefDecl>(loc, name, std::move(source));
}

std::unique_ptr<GoalDecl> DeclarationParser::parseGoalDecl() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = peek().spelling;
  advance();

  int priority = 1;
  llvm::StringRef description;

  if (consume(TokenKind::colon)) {
    if (peek(TokenKind::string_literal)) {
      description = peek().spelling;
      advance();
    } else {
      emitError(getCurrentLocation(), "Expected goal description string");
      return nullptr;
    }
  }

  if (consume(TokenKind::l_paren)) {
    if (peek(TokenKind::number_literal)) {
      if (auto value = getNumericValue(peek())) {
        priority = static_cast<int>(*value);
      }
    }
    if (!expect(TokenKind::r_paren)) return nullptr;
  }

  if (!expect(TokenKind::semicolon)) return nullptr;

  return std::make_unique<GoalDecl>(loc, name, description, priority);
}

std::unique_ptr<OnEventDecl> DeclarationParser::parseOnEventDecl() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_on)) return nullptr;

  if (!peek(TokenKind::identifier) && !peek(TokenKind::string_literal)) {
    emitError(getCurrentLocation(), "Expected event type");
    return nullptr;
  }

  llvm::StringRef eventType = peek().spelling;
  advance();

  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  if (consume(TokenKind::l_paren)) {
    if (!parseNamedTypeList(TokenKind::r_paren, "parameter", params, true)) {
      return nullptr;
    }
  }

  llvm::SmallVector<std::unique_ptr<Expr>, 4> conditions;
  if (consume(TokenKind::kw_when)) {
    if (!expect(TokenKind::l_paren)) return nullptr;

    if (!exprParser.parseExpressionList(TokenKind::r_paren, conditions)) {
      return nullptr;
    }
  }

  llvm::SmallVector<std::unique_ptr<Expr>, 4> contextExprs;
  if (consume(TokenKind::kw_with)) {
    if (!expect(TokenKind::l_paren)) return nullptr;

    if (!exprParser.parseExpressionList(TokenKind::r_paren, contextExprs)) {
      return nullptr;
    }
  }

  if (!expect(TokenKind::l_brace)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 8> body;
  if (!stmtParser.parseStatementBlock(body)) {
    return nullptr;
  }

  if (!expect(TokenKind::r_brace)) return nullptr;

  auto paramRefs = makeStringRefPairs(params);
  return std::make_unique<OnEventDecl>(loc, eventType, paramRefs, conditions,
                                      contextExprs, body);
}

bool DeclarationParser::parseNamedTypeList(TokenKind endToken, llvm::StringRef elementDesc,
                                         llvm::SmallVectorImpl<std::pair<std::string, std::string>> &storage,
                                         bool allowTypeKeywords) {
  while (!peek(endToken) && !peek(TokenKind::eof)) {
    if (!peek(TokenKind::identifier)) {
      auto message = (llvm::Twine("Expected ") + elementDesc + " name").str();
      emitError(getCurrentLocation(), message);
      return false;
    }

    llvm::StringRef name = peek().spelling;
    advance();

    if (!expect(TokenKind::colon)) {
      return false;
    }

    if (!peek(TokenKind::identifier)) {
      if (allowTypeKeywords &&
          (peek(TokenKind::kw_string) || peek(TokenKind::kw_number) ||
           peek(TokenKind::kw_bool) || peek(TokenKind::kw_json) ||
           peek(TokenKind::kw_agent_type) || peek(TokenKind::kw_capability_type) ||
           peek(TokenKind::kw_goal_type) || peek(TokenKind::kw_handle) ||
           peek(TokenKind::kw_result) || peek(TokenKind::kw_response) ||
           peek(TokenKind::kw_context_type) || peek(TokenKind::kw_void))) {
        // Allow type keywords as types
      } else {
        auto message = (llvm::Twine("Expected ") + elementDesc + " type").str();
        emitError(getCurrentLocation(), message);
        return false;
      }
    }

    llvm::StringRef type = peek().spelling;
    advance();

    storage.emplace_back(name.str(), type.str());

    if (!peek(endToken) && !consume(TokenKind::comma)) {
      emitError(getCurrentLocation(), "Expected ',' or '" +
               std::to_string(static_cast<char>(endToken)) + "'");
      return false;
    }
  }

  return expect(endToken);
}

bool DeclarationParser::parsePatternFieldList(llvm::SmallVectorImpl<std::pair<std::string, std::string>> &storage) {
  while (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
    if (!peek(TokenKind::identifier)) {
      emitError(getCurrentLocation(), "Expected field name");
      return false;
    }

    llvm::StringRef name = peek().spelling;
    advance();

    if (!expect(TokenKind::colon)) {
      return false;
    }

    if (!peek(TokenKind::identifier) && !peek(TokenKind::string_literal) &&
        !peek(TokenKind::number_literal) && !peek(TokenKind::kw_true) &&
        !peek(TokenKind::kw_false) && !peek(TokenKind::kw_null)) {
      emitError(getCurrentLocation(), "Expected field value");
      return false;
    }

    llvm::StringRef value = peek().spelling;
    advance();

    storage.emplace_back(name.str(), value.str());

    if (!peek(TokenKind::r_brace) && !consume(TokenKind::comma)) {
      emitError(getCurrentLocation(), "Expected ',' or '}'");
      return false;
    }
  }

  return expect(TokenKind::r_brace);
}
