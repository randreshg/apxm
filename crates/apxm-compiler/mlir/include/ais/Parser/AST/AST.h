/*
 * @file AST.h
 * @brief Abstract Syntax Tree (AST) for APXM.
 *
 * The AST hierarchy models:
 * - ASTNode: Base class for all AST nodes.
 * - Declaration: Represents variable declarations.
 * - Expression: Represents expressions.
 * - Statement: Represents control-flow constructs.
 *
 * The AST is designed to be extensible and flexible, allowing for easy
 * integration of new language features and optimizations.
 */

#ifndef APXM_PARSER_AST_H
#define APXM_PARSER_AST_H

#include "ais/Parser/AST/ASTNode.h"
#include "ais/Parser/AST/Declaration.h"
#include "ais/Parser/AST/Expression.h"
#include "ais/Parser/AST/Statement.h"

#endif // APXM_PARSER_AST_H
