/**
 * @file  AISOps.cpp
 * @brief Verifiers, folders and canonicalisers for every AIS operation.
 *
 * The file is organised by operation groups (memory, reasoning, sync, etc.).
 * Each verifier enforces the invariants that TableGen cannot express (valid
 * memory-space names, non-empty strings, compatible types, etc.).
 *
 * Canonicalisation patterns are local and SSA-preserving:
 *   - wait_all / merge: identity, flatten, dedup
 *   - rsn: deduplicate context operands
 *
 * All patterns are guarded by the standard `hasVerifier` / `hasCanonicalizer`
 * flags declared in AISOps.td, so they run only when requested.
 */

#include "ais/Dialect/AIS/IR/AISOps.h"
#include "ais/Common/Constants.h"
#include "ais/Dialect/AIS/IR/AISTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::ais;

namespace {
/// Helper to verify that a value is of a specific type.
template <typename T>
LogicalResult verifyType(Operation *op, Value value, StringRef errorMsg) {
  if (!llvm::isa<T>(value.getType()))
    return op->emitOpError(errorMsg);
  return success();
}

/// Helper to verify that a range of values are of specific types.
template <typename... Ts>
LogicalResult verifyTypes(Operation *op, ValueRange values, StringRef errorMsg) {
  if (!llvm::all_of(values, [](Value operand) {
        auto type = operand.getType();
        return (llvm::isa<Ts>(type) || ...);
      }))
    return op->emitOpError(errorMsg);
  return success();
}
}  // namespace

//===----------------------------------------------------------------------===//
// QMemOp - Query Memory
//===----------------------------------------------------------------------===//

LogicalResult QMemOp::verify() {
  if (getSid().empty())
    return emitOpError("sid must be non-empty");

  // Check space attribute is valid
  StringRef space = getSpace();
  if (space != apxm::constants::memory::STM && space != apxm::constants::memory::LTM &&
      space != apxm::constants::memory::EPISODIC)
    return emitOpError("space must be 'stm', 'ltm', or 'episodic'");

  // Check result is HandleType
  if (failed(verifyType<HandleType>(*this, getResult(), "result must be !ais.handle type")))
    return failure();

  // Check memory space consistency if specified
  auto handleType = llvm::cast<HandleType>(getResult().getType());
  auto expectedSpace = symbolizeMemorySpace(space);
  if (expectedSpace && handleType.getSpace() != *expectedSpace)
    return emitOpError("result handle space does not match operation space attribute");

  if (auto limit = getLimit()) {
    if (*limit <= 0)
      return emitOpError("limit must be positive if specified");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// UMemOp - Update Memory
//===----------------------------------------------------------------------===//

LogicalResult UMemOp::verify() {
  // Check space attribute is valid
  StringRef space = getSpace();
  if (space != apxm::constants::memory::STM && space != apxm::constants::memory::LTM &&
      space != apxm::constants::memory::EPISODIC)
    return emitOpError("space must be 'stm', 'ltm', or 'episodic'");

  // Check value operand is TokenType
  if (failed(verifyType<TokenType>(*this, getValue(), "value operand must be !ais.token type")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// InvOp - Invoke Capability
//===----------------------------------------------------------------------===//

LogicalResult InvOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Check capability name is non-empty
  if (getCapability().empty())
    return emitOpError("capability name cannot be empty");

  // Check params_json is non-empty (at minimum should be "{}")
  if (getParamsJson().empty())
    return emitOpError("params_json cannot be empty (use \"{}\" for no params)");

  return success();
}

//===----------------------------------------------------------------------===//
// RsnOp - Reasoning with LLM
//===----------------------------------------------------------------------===//

LogicalResult RsnOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Must have either context or template (or both)
  if (getContext().empty() && getTemplateStr().empty())
    return emitOpError("reasoning operation must have at least context or template");

  // Check all context operands are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getContext(),
          "context operands must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

  // Check inner_plan region if non-empty
  if (!getInnerPlan().empty()) {
    // Verify region has at least one block
    if (getInnerPlan().getBlocks().empty())
      return emitOpError("inner_plan region cannot be empty if specified");

    // Check each block has terminator
    if (!llvm::all_of(getInnerPlan(), [](Block &block) {
          return !block.empty() && block.back().hasTrait<OpTrait::IsTerminator>();
        }))
      return emitOpError("inner_plan blocks must have terminators");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// PlanOp - Plan New Subgraph
//===----------------------------------------------------------------------===//

LogicalResult PlanOp::verify() {
  // Check result is GoalType
  if (failed(verifyType<GoalType>(*this, getResult(), "result must be !ais.goal type")))
    return failure();

  // Check all context operands are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getContext(),
          "context operands must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ReflectOp - Reflect Over Episodic Traces
//===----------------------------------------------------------------------===//

LogicalResult ReflectOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Check all context operands are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getContext(),
          "context operands must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// VerifyOp - Verify Claim Against Evidence
//===----------------------------------------------------------------------===//

LogicalResult VerifyOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Check claim operand
  auto claimType = getClaim().getType();
  if (!llvm::isa<TokenType>(claimType) && !llvm::isa<HandleType>(claimType) &&
      !llvm::isa<GoalType>(claimType))
    return emitOpError("claim operand must be !ais.token, !ais.handle, or !ais.goal type");

  // Check evidence operand
  auto evidenceType = getEvidence().getType();
  if (!llvm::isa<TokenType>(evidenceType) && !llvm::isa<HandleType>(evidenceType) &&
      !llvm::isa<GoalType>(evidenceType))
    return emitOpError("evidence operand must be !ais.token, !ais.handle, or !ais.goal type");

  return success();
}

//===----------------------------------------------------------------------===//
// ExcOp - Execute Sandboxed Code
//===----------------------------------------------------------------------===//

LogicalResult ExcOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Check all context operands are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getContext(),
          "context operands must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// WaitAllOp - Barrier Waiting for Tokens
//===----------------------------------------------------------------------===//

LogicalResult WaitAllOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Check all operands are tokens
  if (failed(verifyTypes<TokenType>(*this, getTokens(), "all operands must be !ais.token types")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// MergeOp - Merge Tokens
//===----------------------------------------------------------------------===//

LogicalResult MergeOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Check all operands are tokens
  if (failed(verifyTypes<TokenType>(*this, getTokens(), "all operands must be !ais.token types")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Simplify wait_all with single input: wait_all(%x) -> %x
struct WaitAllSingleInput : public OpRewritePattern<WaitAllOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitAllOp op, PatternRewriter &rewriter) const override {
    auto tokens = op.getTokens();

    // wait_all with single input is identity
    if (tokens.size() == 1) {
      rewriter.replaceOp(op, tokens[0]);
      return success();
    }

    // wait_all with no inputs - replace with constant
    if (tokens.empty()) {
      // Create a trivial token (this case is rare but handle it)
      return failure();  // Keep as-is for empty wait_all
    }

    return failure();
  }
};

/// Flatten nested wait_all: wait_all(wait_all(a,b), c) -> wait_all(a,b,c)
struct WaitAllFlatten : public OpRewritePattern<WaitAllOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitAllOp op, PatternRewriter &rewriter) const override {

    // Define flattening condition: token comes from single-use WaitAllOp
    auto shouldFlatten = [](Value token) -> bool {
        auto waitOp = token.getDefiningOp<WaitAllOp>();
        return waitOp && waitOp->hasOneUse();
    };

    // Early exit if not tokens qualify for flattening
    if (llvm::none_of(op.getTokens(), shouldFlatten)) {
        return failure();
    }

    // Flatten qualifying tokens by splicing their operands
    SmallVector<Value>  newTokens;
    for (Value token: op.getTokens()) {
        if (shouldFlatten(token)) {
            auto innerWait = token.getDefiningOp<WaitAllOp>();
            llvm::append_range(newTokens, innerWait.getTokens());
        } else {
            newTokens.push_back(token);
        }
    }

    // Create replacement operation
    rewriter.replaceOpWithNewOp<WaitAllOp>(op, op.getResult().getType(), newTokens);
    return success();
  }
};

/// Remove duplicate tokens from wait_all
struct WaitAllDedup : public OpRewritePattern<WaitAllOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(WaitAllOp op, PatternRewriter &rewriter) const override {
    SmallVector<Value> uniqueTokens;
    llvm::SmallPtrSet<Value, 8> seen;

    for (Value token : op.getTokens()) {
      if (seen.insert(token).second) {
        uniqueTokens.push_back(token);
      }
    }

    if (uniqueTokens.size() == op.getTokens().size())
      return failure();

    rewriter.replaceOpWithNewOp<WaitAllOp>(op, op.getResult().getType(), uniqueTokens);
    return success();
  }
};

/// Simplify merge with single input: merge(%x) -> %x
struct MergeSingleInput : public OpRewritePattern<MergeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MergeOp op, PatternRewriter &rewriter) const override {
    auto tokens = op.getTokens();

    // merge with single input is identity
    if (tokens.size() == 1) {
      rewriter.replaceOp(op, tokens[0]);
      return success();
    }

    return failure();
  }
};

/// Flatten nested merge: merge(merge(a,b), c) -> merge(a,b,c)
struct MergeFlatten : public OpRewritePattern<MergeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MergeOp op, PatternRewriter &rewriter) const override {
    SmallVector<Value> newTokens;
    bool changed = false;

    for (Value token : op.getTokens()) {
      if (auto innerMerge = token.getDefiningOp<MergeOp>()) {
        // If the inner merge has only this use, we can flatten
        if (innerMerge->hasOneUse()) {
          newTokens.append(innerMerge.getTokens().begin(), innerMerge.getTokens().end());
          changed = true;
          continue;
        }
      }
      newTokens.push_back(token);
    }

    if (!changed)
      return failure();

    rewriter.replaceOpWithNewOp<MergeOp>(op, op.getResult().getType(), newTokens);
    return success();
  }
};

/// Remove duplicate tokens from merge
struct MergeDedup : public OpRewritePattern<MergeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MergeOp op, PatternRewriter &rewriter) const override {
    SmallVector<Value> uniqueTokens;
    llvm::SmallPtrSet<Value, 8> seen;

    for (Value token : op.getTokens()) {
      if (seen.insert(token).second) {
        uniqueTokens.push_back(token);
      }
    }

    if (uniqueTokens.size() == op.getTokens().size())
      return failure();

    rewriter.replaceOpWithNewOp<MergeOp>(op, op.getResult().getType(), uniqueTokens);
    return success();
  }
};

/// Deduplicate context operands in RsnOp: rsn(%a, %b, %a) -> rsn(%a, %b)
struct RsnDedupContext : public OpRewritePattern<RsnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RsnOp op, PatternRewriter &rewriter) const override {
    // Define attributes that should never be copied to the new operation
    const auto shouldSkipAttr = [](StringRef attrName) {
        return attrName == "template_str" || attrName == "operandSegmentSizes";
    };

    // Collect deduplicated context tokens while preserving order
    SmallVector<Value> uniqueContext;
    llvm::SmallDenseSet<Value, 8> seen;

    llvm::copy_if(op.getContext(), std::back_inserter(uniqueContext),
        [&](Value ctx) {return seen.insert(ctx).second; });

    // No transformation needed if no duplicates found
    if (uniqueContext.size() == op.getContext().size())
      return failure();

    // Create new RsnOp with deduplicated context
    auto newOp = rewriter.create<RsnOp>(op.getLoc(), op.getResult().getType(),
                                        op.getTemplateStrAttr(), uniqueContext);

    // Copy inner_plan region if present
    if (!op.getInnerPlan().empty()) {
      IRMapping mapper;
      op.getInnerPlan().cloneInto(&newOp.getInnerPlan(), mapper);
    }

    // Copy other attributes (excluding template_str which is already set)
    for (const NamedAttribute &attr : op -> getAttrs()) {
        if (!shouldSkipAttr(attr.getName())) {
            newOp->setAttr(attr.getName(), attr.getValue());
        }
    }

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

/// Remove empty context if template is self-contained
struct RsnFoldEmptyContext : public OpRewritePattern<RsnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(RsnOp op, PatternRewriter &rewriter) const override {
    // Only apply if context is empty but we have a template
    if (!op.getContext().empty() || op.getTemplateStrAttr().getValue().empty())
      return failure();

    // Already optimal - no change needed (this is just a verification pattern)
    return failure();
  }
};

}  // namespace

//===----------------------------------------------------------------------===//
// FenceOp - Memory Fence
//===----------------------------------------------------------------------===//

LogicalResult FenceOp::verify() {
  // No special constraints beyond what tablegen enforces
  return success();
}

//===----------------------------------------------------------------------===//
// JumpOp - Unconditional Jump
//===----------------------------------------------------------------------===//

LogicalResult JumpOp::verify() {
  // Label existence checked at link/lower time
  // No compile-time constraints
  return success();
}

//===----------------------------------------------------------------------===//
// BranchOnValueOp - Conditional Branch
//===----------------------------------------------------------------------===//

LogicalResult BranchOnValueOp::verify() {
  // Check condition operand
  auto condType = getCondition().getType();
  if (!llvm::isa<TokenType>(condType) && !llvm::isa<HandleType>(condType))
    return emitOpError("condition operand must be !ais.token or !ais.handle type");

  return success();
}

//===----------------------------------------------------------------------===//
// LoopStartOp - Begin Bounded Loop
//===----------------------------------------------------------------------===//

LogicalResult LoopStartOp::verify() {
  // Check count operand
  auto countType = getCount().getType();
  if (!llvm::isa<TokenType>(countType) && !llvm::isa<HandleType>(countType))
    return emitOpError("count operand must be !ais.token or !ais.handle type");

  // Check state result
  if (failed(verifyType<TokenType>(*this, getState(), "state result must be !ais.token type")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// LoopEndOp - End Bounded Loop
//===----------------------------------------------------------------------===//

LogicalResult LoopEndOp::verify() {
  // Check state operand
  if (failed(verifyType<TokenType>(*this, getState(), "state operand must be !ais.token type")))
    return failure();

  // Check result
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TryCatchOp - Try/Catch Region Markers
//===----------------------------------------------------------------------===//

LogicalResult TryCatchOp::verify() {
  // Label existence checked at link/lower time
  // No compile-time constraints
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp - Return from Subgraph
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // Check value operand
  if (failed(verifyType<TokenType>(*this, getValue(), "value operand must be !ais.token type")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ErrOp - Error Handling
//===----------------------------------------------------------------------===//

LogicalResult ErrOp::verify() {
  // Check input operand if present
  if (getInput()) {
    if (failed(verifyType<TokenType>(*this, getInput(), "input operand must be !ais.token type")))
      return failure();
  }

  // Check result
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// CommunicateOp - Multi-agent Communication
//===----------------------------------------------------------------------===//

LogicalResult CommunicateOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Check all attachment operands are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getAttachments(),
          "attachment operands must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen Generated Class Definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "ais/Dialect/AIS/IR/AISOps.cpp.inc"

namespace mlir {
namespace ais {

//===----------------------------------------------------------------------===//
// ConstStrOp Folder
//===----------------------------------------------------------------------===//

/// Fold constant strings - enables CSE to deduplicate identical strings
OpFoldResult ConstStrOp::fold(FoldAdaptor adaptor) {
  // Return the value attribute to enable constant propagation
  return getValueAttr();
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns (must be after GET_OP_CLASSES)
//===----------------------------------------------------------------------===//

void WaitAllOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<WaitAllSingleInput, WaitAllFlatten, WaitAllDedup>(context);
}

void MergeOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<MergeSingleInput, MergeFlatten, MergeDedup>(context);
}

void RsnOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<RsnDedupContext, RsnFoldEmptyContext>(context);
}

}  // namespace ais
}  // namespace mlir
