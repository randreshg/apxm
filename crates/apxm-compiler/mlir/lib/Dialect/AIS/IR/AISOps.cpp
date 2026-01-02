/**
 * @file  AISOps.cpp
 * @brief Verifiers, folders and canonicalisers for every AIS operation.
 *
 * The file is organised by operation groups (memory, LLM ops, sync, etc.).
 * Each verifier enforces the invariants that TableGen cannot express (valid
 * memory-space names, non-empty strings, compatible types, etc.).
 *
 * LLM Operations (ask/think/reason):
 *   Three distinct ops for critical path analysis. Each has different latency:
 *   - ask:    LOW latency  - simple Q&A
 *   - think:  HIGH latency - extended thinking
 *   - reason: MEDIUM latency - structured reasoning (has canonicalizer)
 *
 * Canonicalisation patterns are local and SSA-preserving:
 *   - wait_all / merge: identity, flatten, dedup
 *   - reason: deduplicate context operands
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
// AskOp - Simple Q&A LLM Call (LOW Latency)
//===----------------------------------------------------------------------===//

LogicalResult AskOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Must have template OR context operands (dynamic prompts use context)
  if (getTemplateStr().empty() && getContext().empty())
    return emitOpError("ask operation requires a template or context operands");

  // Check all context operands are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getContext(),
          "context operands must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ThinkOp - Extended Thinking LLM Call (HIGH Latency)
//===----------------------------------------------------------------------===//

LogicalResult ThinkOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Must have template OR context operands (dynamic prompts use context)
  if (getTemplateStr().empty() && getContext().empty())
    return emitOpError("think operation requires a template or context operands");

  // Check all context operands are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getContext(),
          "context operands must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// ReasonOp - Structured Reasoning LLM Call (MEDIUM Latency)
//===----------------------------------------------------------------------===//

LogicalResult ReasonOp::verify() {
  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
    return failure();

  // Must have either context or template (or both)
  if (getContext().empty() && getTemplateStr().empty())
    return emitOpError("reason operation must have at least context or template");

  // Check all context operands are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getContext(),
          "context operands must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

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

/// Deduplicate context operands in ReasonOp: reason(%a, %b, %a) -> reason(%a, %b)
struct ReasonDedupContext : public OpRewritePattern<ReasonOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ReasonOp op, PatternRewriter &rewriter) const override {
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

    // Create new ReasonOp with deduplicated context
    auto newOp = rewriter.create<ReasonOp>(op.getLoc(), op.getResult().getType(),
                                           op.getTemplateStrAttr(), uniqueContext);

    // Copy other attributes (excluding template_str which is already set)
    for (const NamedAttribute &attr : op->getAttrs()) {
        if (!shouldSkipAttr(attr.getName())) {
            newOp->setAttr(attr.getName(), attr.getValue());
        }
    }

    rewriter.replaceOp(op, newOp.getResult());
    return success();
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
// SwitchOp - Multi-way Branch with Case Regions
//===----------------------------------------------------------------------===//

LogicalResult SwitchOp::verify() {
  // Check discriminant is TokenType
  if (failed(verifyType<TokenType>(*this, getDiscriminant(),
                                   "discriminant operand must be !ais.token type")))
    return failure();

  // Check result is TokenType if present (optional)
  if (getResult()) {
    if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
      return failure();
  }

  // Regions: N case regions + 1 default region (last)
  // case_labels has N entries corresponding to first N regions
  size_t numLabels = getCaseLabels().size();
  size_t totalRegions = getRegions().size();
  if (totalRegions == 0)
    return emitOpError("switch must have at least a default region");

  // Switch must have at least one case (besides default)
  if (numLabels == 0)
    return emitOpError("switch must have at least one case label");

  size_t numCaseRegions = totalRegions - 1;  // Last region is default
  if (numLabels != numCaseRegions)
    return emitOpError() << "expected " << numLabels << " case regions but got " << numCaseRegions;

  // Check that case_labels contains only string attributes
  for (auto label : getCaseLabels()) {
    if (!llvm::isa<StringAttr>(label))
      return emitOpError("case_labels must contain only string attributes");
  }

  // Check each case region terminates with yield
  bool hasResult = getResult() != nullptr;
  for (auto &region : getCaseRegions()) {
    if (region.empty())
      return emitOpError("case region cannot be empty");
    Block &block = region.front();
    if (block.empty() || !llvm::isa<YieldOp>(block.getTerminator()))
      return emitOpError("case region must terminate with ais.yield");

    // Check yield consistency with switch result
    auto yieldOp = llvm::cast<YieldOp>(block.getTerminator());
    bool yieldHasValue = yieldOp.getValue() != nullptr;
    if (hasResult && !yieldHasValue)
      return emitOpError("switch has result but case yield has no value");
    if (!hasResult && yieldHasValue)
      return emitOpError("switch has no result but case yield has value");
  }

  // Check default region (last region) terminates with yield
  if (getDefaultRegion().empty())
    return emitOpError("default region cannot be empty");
  Block &defaultBlock = getDefaultRegion().front();
  if (defaultBlock.empty() || !llvm::isa<YieldOp>(defaultBlock.getTerminator()))
    return emitOpError("default region must terminate with ais.yield");

  // Check default yield consistency
  auto defaultYield = llvm::cast<YieldOp>(defaultBlock.getTerminator());
  bool defaultHasValue = defaultYield.getValue() != nullptr;
  if (hasResult && !defaultHasValue)
    return emitOpError("switch has result but default yield has no value");
  if (!hasResult && defaultHasValue)
    return emitOpError("switch has no result but default yield has value");

  return success();
}

void SwitchOp::print(OpAsmPrinter &p) {
  p << " " << getDiscriminant() << " : " << getDiscriminant().getType();

  // Print case regions with labels
  auto labels = getCaseLabels();
  size_t idx = 0;
  for (auto &region : getCaseRegions()) {
    p.printNewline();
    p << "    case " << labels[idx] << " ";
    p.printRegion(region, /*printEntryBlockArgs=*/false);
    ++idx;
  }

  // Print default region (last region)
  p.printNewline();
  p << "    default ";
  p.printRegion(getDefaultRegion(), /*printEntryBlockArgs=*/false);

  // Only print result type if switch has a result
  if (getResult()) {
    p << " -> " << getResult().getType();
  }
  p.printOptionalAttrDict((*this)->getAttrs(), {"case_labels"});
}

ParseResult SwitchOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse discriminant
  OpAsmParser::UnresolvedOperand discriminant;
  Type discriminantType;
  if (parser.parseOperand(discriminant) ||
      parser.parseColonType(discriminantType) ||
      parser.resolveOperand(discriminant, discriminantType, result.operands))
    return failure();

  // Parse cases
  SmallVector<Attribute> caseLabels;

  while (succeeded(parser.parseOptionalKeyword("case"))) {
    StringAttr label;
    if (parser.parseAttribute(label))
      return failure();
    caseLabels.push_back(label);

    auto *region = result.addRegion();
    if (parser.parseRegion(*region, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
  }

  // Parse default
  if (parser.parseKeyword("default"))
    return failure();
  auto *defaultRegion = result.addRegion();
  if (parser.parseRegion(*defaultRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();

  // Add case_labels attribute
  result.addAttribute("case_labels", parser.getBuilder().getArrayAttr(caseLabels));

  // Parse optional result type (-> type)
  if (succeeded(parser.parseOptionalArrow())) {
    Type resultType;
    if (parser.parseType(resultType))
      return failure();
    result.addTypes(resultType);
  }

  return parser.parseOptionalAttrDict(result.attributes);
}

//===----------------------------------------------------------------------===//
// FlowCallOp - Cross-Agent Flow Invocation
//===----------------------------------------------------------------------===//

LogicalResult FlowCallOp::verify() {
  // Check agent_name is not empty
  if (getAgentName().empty())
    return emitOpError("agent_name cannot be empty");

  // Check flow_name is not empty
  if (getFlowName().empty())
    return emitOpError("flow_name cannot be empty");

  // Check all args are tokens, handles, or goals
  if (failed(verifyTypes<TokenType, HandleType, GoalType>(
          *this, getArgs(),
          "args must be !ais.token, !ais.handle, or !ais.goal types")))
    return failure();

  // Check result is TokenType
  if (failed(verifyType<TokenType>(*this, getResult(), "result must be !ais.token type")))
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

void ReasonOp::getCanonicalizationPatterns(RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ReasonDedupContext>(context);
}

}  // namespace ais
}  // namespace mlir
