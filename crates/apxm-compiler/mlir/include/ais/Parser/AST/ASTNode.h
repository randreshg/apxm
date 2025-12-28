/*
 * @file ASTNode.h
 * @brief Base infrastructure for the APXM abstract-syntax tree.
 *
 * This header defines:
 * - The root ASTNode class and its LLVM-style RTTI
 * - The exhaustive node-kind enumeration used by the ASTNode class
 * - The ASTVisitor interface for AST traversal
 */

#ifndef APXM_PARSER_ASTNODE_H
#define APXM_PARSER_ASTNODE_H

#include "ais/Parser/Utils/Location.h"
#include "llvm/Support/Casting.h"
#include <cstdint>

namespace apxm::parser {

/// Base class for all AST nodes with LLVM-style RTTI
class ASTNode {
public:
  enum class Kind {
    // Declarations
    AgentDecl,
    MemoryDecl,
    CapabilityDecl,
    FlowDecl,
    BeliefDecl,
    GoalDecl,
    OnEventDecl,

    // Expressions
    StringLiteralExpr,
    NumberLiteralExpr,
    BooleanLiteralExpr,
    NullLiteralExpr,
    VarExpr,
    CallExpr,
    BinaryExpr,
    UnaryExpr,
    AssignmentExpr,
    ArrayExpr,
    MemberAccessExpr,
    SubscriptExpr,
    PlanExpr,
    ReflectExpr,
    VerifyExpr,
    ExecuteExpr,
    CommunicateExpr,
    WaitAllExpr,
    MergeExpr,
    FlowCallExpr,

    // Statements
    LetStmt,
    ReturnStmt,
    IfStmt,
    ExprStmt,
    ParallelStmt,
    LoopStmt,
    TryCatchStmt,
    SwitchStmt
  };

protected:
  const Kind kind;
  const Location loc;

  ASTNode(Kind kind, Location loc) : kind(kind), loc(loc) {}

public:
  virtual ~ASTNode() = default;

  Kind getKind() const noexcept { return kind; }
  Location getLocation() const noexcept { return loc; }

  /// LLVM-style RTTI support
  static bool classof(const ASTNode *) { return true; }
};

/// Visitor interface for AST traversal
//
// Forward declarations for AST node classes (file scope).
// These enable the visitor interfaces below to reference the concrete
// node types without requiring each translation unit to include all
// individual node headers.
class AgentDecl;
class MemoryDecl;
class CapabilityDecl;
class FlowDecl;
class BeliefDecl;
class GoalDecl;
class OnEventDecl;

class StringLiteralExpr;
class NumberLiteralExpr;
class BooleanLiteralExpr;
class NullLiteralExpr;
class VarExpr;
class CallExpr;
class BinaryExpr;
class UnaryExpr;
class AssignmentExpr;
class ArrayExpr;
class MemberAccessExpr;
class SubscriptExpr;
class PlanExpr;
class ReflectExpr;
class VerifyExpr;
class ExecuteExpr;
class CommunicateExpr;
class WaitAllExpr;
class MergeExpr;
class FlowCallExpr;

class LetStmt;
class ReturnStmt;
class IfStmt;
class ExprStmt;
class ParallelStmt;
class LoopStmt;
class TryCatchStmt;
class SwitchStmt;

class ASTVisitor {
public:
  virtual ~ASTVisitor() = default;



#define HANDLE_NODE(CLASS) virtual void visit##CLASS(CLASS &node) = 0;
  HANDLE_NODE(AgentDecl)
  HANDLE_NODE(MemoryDecl)
  HANDLE_NODE(CapabilityDecl)
  HANDLE_NODE(FlowDecl)
  HANDLE_NODE(BeliefDecl)
  HANDLE_NODE(GoalDecl)
  HANDLE_NODE(OnEventDecl)
  HANDLE_NODE(StringLiteralExpr)
  HANDLE_NODE(NumberLiteralExpr)
  HANDLE_NODE(BooleanLiteralExpr)
  HANDLE_NODE(NullLiteralExpr)
  HANDLE_NODE(VarExpr)
  HANDLE_NODE(CallExpr)
  HANDLE_NODE(BinaryExpr)
  HANDLE_NODE(UnaryExpr)
  HANDLE_NODE(AssignmentExpr)
  HANDLE_NODE(ArrayExpr)
  HANDLE_NODE(MemberAccessExpr)
  HANDLE_NODE(SubscriptExpr)
  HANDLE_NODE(PlanExpr)
  HANDLE_NODE(ReflectExpr)
  HANDLE_NODE(VerifyExpr)
  HANDLE_NODE(ExecuteExpr)
  HANDLE_NODE(CommunicateExpr)
  HANDLE_NODE(WaitAllExpr)
  HANDLE_NODE(MergeExpr)
  HANDLE_NODE(FlowCallExpr)
  HANDLE_NODE(LetStmt)
  HANDLE_NODE(ReturnStmt)
  HANDLE_NODE(IfStmt)
  HANDLE_NODE(ExprStmt)
  HANDLE_NODE(ParallelStmt)
  HANDLE_NODE(LoopStmt)
  HANDLE_NODE(TryCatchStmt)
  HANDLE_NODE(SwitchStmt)
#undef HANDLE_NODE
};

/// Const visitor interface
class ConstASTVisitor {
public:
  virtual ~ConstASTVisitor() = default;

#define HANDLE_NODE(CLASS) virtual void visit##CLASS(const CLASS &node) = 0;
  HANDLE_NODE(AgentDecl)
  HANDLE_NODE(MemoryDecl)
  HANDLE_NODE(CapabilityDecl)
  HANDLE_NODE(FlowDecl)
  HANDLE_NODE(BeliefDecl)
  HANDLE_NODE(GoalDecl)
  HANDLE_NODE(OnEventDecl)
  HANDLE_NODE(StringLiteralExpr)
  HANDLE_NODE(NumberLiteralExpr)
  HANDLE_NODE(BooleanLiteralExpr)
  HANDLE_NODE(NullLiteralExpr)
  HANDLE_NODE(VarExpr)
  HANDLE_NODE(CallExpr)
  HANDLE_NODE(BinaryExpr)
  HANDLE_NODE(UnaryExpr)
  HANDLE_NODE(AssignmentExpr)
  HANDLE_NODE(ArrayExpr)
  HANDLE_NODE(MemberAccessExpr)
  HANDLE_NODE(SubscriptExpr)
  HANDLE_NODE(PlanExpr)
  HANDLE_NODE(ReflectExpr)
  HANDLE_NODE(VerifyExpr)
  HANDLE_NODE(ExecuteExpr)
  HANDLE_NODE(CommunicateExpr)
  HANDLE_NODE(WaitAllExpr)
  HANDLE_NODE(MergeExpr)
  HANDLE_NODE(LetStmt)
  HANDLE_NODE(ReturnStmt)
  HANDLE_NODE(IfStmt)
  HANDLE_NODE(ExprStmt)
  HANDLE_NODE(ParallelStmt)
  HANDLE_NODE(LoopStmt)
  HANDLE_NODE(TryCatchStmt)
  HANDLE_NODE(SwitchStmt)
  HANDLE_NODE(FlowCallExpr)
#undef HANDLE_NODE
};

} // namespace apxm::parser

#endif // APXM_PARSER_ASTNODE_H
