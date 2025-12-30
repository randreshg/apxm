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

#include "ais/Parser/AST/Statement.h"
#include "ais/Parser/Parsers/DeclarationParser.h"
#include "ais/Parser/Parsers/ExpressionParser.h"
#include "ais/Parser/Parsers/StatementParser.h"
#include "ais/Parser/Utils/Container.h"
#include "ais/Parser/Utils/Token.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include <memory>
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

using namespace apxm::parser;

std::unique_ptr<AgentDecl> DeclarationParser::parseAgent() {
  if (!expect(TokenKind::kw_agent)) {
    synchronize();
    return nullptr;
  }

  Location nameLoc = getCurrentLocation();
  Token nameTok = peek();
  if (!expect(TokenKind::identifier)) {
    synchronize();
    return nullptr;
  }

  llvm::StringRef agentName = nameTok.spelling;

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
      advance();
      if (peek(TokenKind::l_brace)) {
        advance();
        while (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
          if (peek(TokenKind::comma)) {
            advance();
            continue;
          }
          if (auto decl = parseMemoryDecl()) {
            memoryDecls.push_back(std::move(decl));
          } else {
            synchronize();
          }
          consume(TokenKind::comma);
          consume(TokenKind::semicolon);
        }
        if (!expect(TokenKind::r_brace)) {
          synchronize();
        }
      } else {
        if (auto decl = parseMemoryDecl()) {
          memoryDecls.push_back(std::move(decl));
        }
        if (!consume(TokenKind::semicolon)) {
          expect(TokenKind::semicolon);
        }
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
        consume(TokenKind::comma);
        consume(TokenKind::semicolon);
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
        consume(TokenKind::comma);
        consume(TokenKind::semicolon);
      }
      if (!expect(TokenKind::r_brace)) {
        synchronize();
      }
    } else if (peek(TokenKind::kw_capability)) {
      if (auto decl = parseCapabilityDecl()) {
        capabilityDecls.push_back(std::move(decl));
      }
    } else if (peek(TokenKind::kw_flow) || peek(TokenKind::at_sign)) {
      // Handle both `flow name { }` and `@entry flow name { }`
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
      if (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
        advance();
      }
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

  // Validate exactly one @entry flow per agent
  int entryCount = 0;
  for (const auto &flow : flowDecls) {
    if (flow->isEntryFlow()) {
      entryCount++;
      if (entryCount > 1) {
        emitError(flow->getLocation(),
                  "Multiple @entry flows declared. Only one allowed per agent.");
      }
    }
  }

  return std::make_unique<AgentDecl>(nameLoc, agentName, memoryDecls, capabilityDecls,
                                    flowDecls, beliefDecls, goalDecls, onEventDecls);
}

std::unique_ptr<MemoryDecl> DeclarationParser::parseMemoryDecl() {
  Location loc = getCurrentLocation();
  Token nameTok = peek();
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = nameTok.spelling;

  if (!expect(TokenKind::colon)) return nullptr;

  TokenKind tierKind = peek().kind;
  llvm::StringRef tier;
  if (tierKind == TokenKind::kw_STM || tierKind == TokenKind::kw_LTM ||
      tierKind == TokenKind::kw_Episodic || tierKind == TokenKind::identifier) {
    tier = peek().spelling;
    advance();
  } else {
    emitError(getCurrentLocation(), "Expected memory tier (STM, LTM, or Episodic)");
    return nullptr;
  }

  return std::make_unique<MemoryDecl>(loc, name, tier);
}

std::unique_ptr<CapabilityDecl> DeclarationParser::parseCapabilityDecl() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_capability)) return nullptr;
  Token nameTok = peek();
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = nameTok.spelling;

  if (!expect(TokenKind::l_paren)) return nullptr;

  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  if (!parseNamedTypeList(TokenKind::r_paren, "parameter", params, true)) {
    return nullptr;
  }

  if (!expect(TokenKind::arrow)) return nullptr;

  if (!peek(TokenKind::identifier) && !peek(TokenKind::kw_string) &&
      !peek(TokenKind::kw_number) && !peek(TokenKind::kw_bool) &&
      !peek(TokenKind::kw_json) && !peek(TokenKind::kw_void) &&
      !peek(TokenKind::kw_token)) {
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

  // Check for @entry annotation
  bool isEntry = false;
  if (consume(TokenKind::at_sign)) {
    if (!expect(TokenKind::kw_entry)) {
      emitError(getCurrentLocation(), "Expected 'entry' after '@'");
      return nullptr;
    }
    isEntry = true;
  }

  if (!expect(TokenKind::kw_flow)) return nullptr;
  Token nameTok = peek();
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = nameTok.spelling;

  // Parameters and return type are optional
  // Supports both: `flow main { }` and `flow main() -> token { }`
  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  llvm::StringRef returnType = "void";

  if (consume(TokenKind::l_paren)) {
    if (!parseNamedTypeList(TokenKind::r_paren, "parameter", params, true)) {
      return nullptr;
    }

    if (consume(TokenKind::arrow)) {
      if (!peek(TokenKind::identifier) && !peek(TokenKind::kw_string) &&
          !peek(TokenKind::kw_number) && !peek(TokenKind::kw_bool) &&
          !peek(TokenKind::kw_json) && !peek(TokenKind::kw_void) &&
          !peek(TokenKind::kw_token)) {
        emitError(getCurrentLocation(), "Expected return type");
        return nullptr;
      }
      returnType = peek().spelling;
      advance();
    }
  }

  llvm::SmallVector<std::unique_ptr<Stmt>, 8> body;
  if (!peek(TokenKind::l_brace)) {
    emitError(getCurrentLocation(), "Expected '{' to start flow body");
    return nullptr;
  }
  if (!stmtParser.parseStatementBlock(body)) {
    return nullptr;
  }

  auto paramRefs = makeStringRefPairs(params);
  return std::make_unique<FlowDecl>(loc, name, paramRefs, returnType, isEntry, body);
}

std::unique_ptr<BeliefDecl> DeclarationParser::parseBeliefDecl() {
  Location loc = getCurrentLocation();
  Token nameTok = peek();
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = nameTok.spelling;

  if (!consume(TokenKind::colon)) {
    if (!expect(TokenKind::equal)) return nullptr;
  }

  if (consume(TokenKind::kw_from)) {
    // Optional 'from' keyword for readability.
  }

  auto source = exprParser.parseExpression();
  if (!source) {
    synchronize();
    return nullptr;
  }

  return std::make_unique<BeliefDecl>(loc, name, std::move(source));
}

std::unique_ptr<GoalDecl> DeclarationParser::parseGoalDecl() {
  Location loc = getCurrentLocation();
  Token nameTok = peek();
  if (!expect(TokenKind::identifier)) return nullptr;

  llvm::StringRef name = nameTok.spelling;

  int priority = 1;
  std::string description;

  if (consume(TokenKind::l_paren)) {
    while (!peek(TokenKind::r_paren) && !peek(TokenKind::eof)) {
      Token keyTok = peek();
      if (!expect(TokenKind::identifier)) return nullptr;
      llvm::StringRef key = keyTok.spelling;

      if (!expect(TokenKind::colon)) return nullptr;

      if (key.equals_insensitive("priority")) {
        if (!peek(TokenKind::number_literal)) {
          emitError(getCurrentLocation(), "Expected numeric priority value");
          return nullptr;
        }
        if (auto value = getNumericValue(peek())) {
          priority = static_cast<int>(*value);
        }
        advance();
      } else if (key.equals_insensitive("description")) {
        if (!peek(TokenKind::string_literal)) {
          emitError(getCurrentLocation(), "Expected string description");
          return nullptr;
        }
        if (auto value = getStringValue(peek())) {
          description = *value;
        }
        advance();
      } else {
        emitError(getCurrentLocation(), "Unknown goal attribute");
        return nullptr;
      }

      if (!consume(TokenKind::comma))
        break;
    }
    if (!expect(TokenKind::r_paren)) return nullptr;
  } else if (consume(TokenKind::colon)) {
    if (peek(TokenKind::string_literal)) {
      if (auto value = getStringValue(peek())) {
        description = *value;
      }
      advance();
    } else {
      emitError(getCurrentLocation(), "Expected goal description string");
      return nullptr;
    }
    if (peek(TokenKind::identifier) && peek().spelling == "priority") {
      advance();
      if (!peek(TokenKind::number_literal)) {
        emitError(getCurrentLocation(), "Expected numeric priority value");
        return nullptr;
      }
      if (auto value = getNumericValue(peek())) {
        priority = static_cast<int>(*value);
      }
      advance();
    } else if (consume(TokenKind::l_paren)) {
      if (peek(TokenKind::number_literal)) {
        if (auto value = getNumericValue(peek())) {
          priority = static_cast<int>(*value);
        }
        advance();
      }
      if (!expect(TokenKind::r_paren)) return nullptr;
    }
  } else if (consume(TokenKind::l_paren)) {
    if (peek(TokenKind::number_literal)) {
      if (auto value = getNumericValue(peek())) {
        priority = static_cast<int>(*value);
      }
      advance();
    }
    if (!expect(TokenKind::r_paren)) return nullptr;
  }

  return std::make_unique<GoalDecl>(loc, name, description, priority);
}

std::unique_ptr<OnEventDecl> DeclarationParser::parseOnEventDecl() {
  Location loc = getCurrentLocation();
  if (!expect(TokenKind::kw_on)) return nullptr;

  if (!peek(TokenKind::identifier)) {
    emitError(getCurrentLocation(), "Expected event type");
    return nullptr;
  }

  std::string eventType = peek().spelling.str();
  advance();
  while (consume(TokenKind::dot)) {
    if (!peek(TokenKind::identifier)) {
      emitError(getCurrentLocation(), "Expected identifier after '.' in event type");
      return nullptr;
    }
    eventType.push_back('.');
    eventType.append(peek().spelling.str());
    advance();
  }

  llvm::SmallVector<std::pair<std::string, std::string>, 4> params;
  if (!expect(TokenKind::l_brace)) return nullptr;
  while (!peek(TokenKind::r_brace) && !peek(TokenKind::eof)) {
    if (peek(TokenKind::comma)) {
      advance();
      continue;
    }
    Token fieldTok = peek();
    if (!expect(TokenKind::identifier)) return nullptr;
    std::string fieldName = fieldTok.spelling.str();
    std::string fieldType = "token";
    if (consume(TokenKind::colon)) {
      if (!peek(TokenKind::identifier)) {
        emitError(getCurrentLocation(), "Expected field type");
        return nullptr;
      }
      fieldType = peek().spelling.str();
      advance();
    }
    params.emplace_back(std::move(fieldName), std::move(fieldType));
    consume(TokenKind::comma);
    consume(TokenKind::semicolon);
  }
  if (!expect(TokenKind::r_brace)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Expr>, 4> conditions;
  if (consume(TokenKind::kw_if)) {
    auto guard = exprParser.parseExpression();
    if (!guard) {
      synchronize();
      return nullptr;
    }
    conditions.push_back(std::move(guard));
  }

  if (!expect(TokenKind::fat_arrow)) return nullptr;

  llvm::SmallVector<std::unique_ptr<Stmt>, 8> body;
  if (peek(TokenKind::l_brace)) {
    if (!stmtParser.parseStatementBlock(body)) {
      return nullptr;
    }
  } else {
    auto expr = exprParser.parseExpression();
    if (!expr) {
      synchronize();
      return nullptr;
    }
    auto retLoc = expr->getLocation();
    auto ret = std::make_unique<ReturnStmt>(retLoc, std::move(expr));
    body.push_back(std::move(ret));
  }

  llvm::SmallVector<std::unique_ptr<Expr>, 4> contextExprs;
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
