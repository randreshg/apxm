/*
 * @file Parser.cpp
 * @brief Implementation of the Parser class.
 *
 * This file contains the implementation of the Parser class.
 */


#include "ais/Parser/Core/Parser.h"
#include "ais/Dialect/AIS/Support/AISDebug.h"

APXM_AIS_DEBUG_SETUP(parser)

using namespace apxm::parser;
using namespace mlir::ais;

Parser::Parser(Lexer &lexer)
    : lexer(lexer), hadErrorFlag(false), exprParser(lexer, hadErrorFlag),
      stmtParser(lexer, exprParser, hadErrorFlag),
      declParser(lexer, stmtParser, exprParser, hadErrorFlag) {}

std::unique_ptr<AgentDecl> Parser::parseAgent() {
  return declParser.parseAgent();
}
