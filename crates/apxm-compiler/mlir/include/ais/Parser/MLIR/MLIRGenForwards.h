/*
 * @file MLIRGenForwards.h
 * @brief Forward declarations for MLIR generation classes and AST nodes
 *
 * Centralized location for all forward declarations used by MLIR generation
 * to reduce duplication across MLIRGen, MLIRGenExpressions, MLIRGenStatements,
 * and MLIRGenOperations headers.
 */

#ifndef APXM_PARSER_MLIR_MLIRGENFORWARDS_H
#define APXM_PARSER_MLIR_MLIRGENFORWARDS_H

namespace apxm::parser {

// Forward declarations for AST nodes
class AgentDecl;
class FlowDecl;
class OnEventDecl;
class Stmt;
class Expr;
class CallExpr;
class LetStmt;
class ReturnStmt;
class IfStmt;
class ParallelStmt;
class LoopStmt;
class TryCatchStmt;
class ExprStmt;
class StringLiteralExpr;
class NumberLiteralExpr;
class BooleanLiteralExpr;
class NullLiteralExpr;
class VarExpr;
class ArrayExpr;
class BinaryExpr;
class UnaryExpr;
class AssignmentExpr;
class MemberAccessExpr;
class PlanExpr;
class ReflectExpr;
class VerifyExpr;
class ExecuteExpr;
class CommunicateExpr;
class WaitAllExpr;
class MergeExpr;
class SubscriptExpr;

// Forward declarations for MLIR generation helper classes
class MLIRGen;
class MLIRGenStatements;
class MLIRGenOperations;
class MLIRGenDeclarations;
class MLIRGenExpressions;

} // namespace apxm::parser

#endif // APXM_PARSER_MLIR_MLIRGENFORWARDS_H
