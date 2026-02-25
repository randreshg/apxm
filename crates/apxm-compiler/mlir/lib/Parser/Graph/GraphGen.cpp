/*
 * @file GraphGen.cpp
 * @brief AST-to-ApxmGraph canonical lowering for AIS DSL.
 */

#include "ais/Parser/Graph/GraphGen.h"
#include "ais/Common/Constants.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

using namespace apxm::parser;

namespace {

namespace graph_json {
constexpr llvm::StringLiteral NAME = "name";
constexpr llvm::StringLiteral NODES = "nodes";
constexpr llvm::StringLiteral EDGES = "edges";
constexpr llvm::StringLiteral PARAMETERS = "parameters";
constexpr llvm::StringLiteral METADATA = "metadata";
constexpr llvm::StringLiteral ID = "id";
constexpr llvm::StringLiteral OP = "op";
constexpr llvm::StringLiteral ATTRIBUTES = "attributes";
constexpr llvm::StringLiteral FROM = "from";
constexpr llvm::StringLiteral TO = "to";
constexpr llvm::StringLiteral DEPENDENCY = "dependency";
constexpr llvm::StringLiteral TYPE_NAME = "type_name";
constexpr llvm::StringLiteral IS_ENTRY = "is_entry";
constexpr llvm::StringLiteral SOURCE_AGENT = "source_agent";
constexpr llvm::StringLiteral SOURCE_FLOW = "source_flow";
constexpr llvm::StringLiteral DATA = "Data";
constexpr llvm::StringLiteral CONTROL = "Control";
} // namespace graph_json

namespace graph_attrs {
constexpr llvm::StringLiteral VALUE = "value";
constexpr llvm::StringLiteral TEMPLATE_STR = "template_str";
constexpr llvm::StringLiteral QUERY = "query";
constexpr llvm::StringLiteral SPACE = "space";
constexpr llvm::StringLiteral CAPABILITY = "capability";
constexpr llvm::StringLiteral PARAMS_JSON = "params_json";
constexpr llvm::StringLiteral GOAL = "goal";
constexpr llvm::StringLiteral TRACE_ID = "trace_id";
constexpr llvm::StringLiteral AGENT_NAME = "agent_name";
constexpr llvm::StringLiteral FLOW_NAME = "flow_name";
} // namespace graph_attrs

namespace graph_ops {
constexpr llvm::StringLiteral CONST_STR = "CONST_STR";
constexpr llvm::StringLiteral ASK = "ASK";
constexpr llvm::StringLiteral THINK = "THINK";
constexpr llvm::StringLiteral REASON = "REASON";
constexpr llvm::StringLiteral QMEM = "QMEM";
constexpr llvm::StringLiteral UMEM = "UMEM";
constexpr llvm::StringLiteral INV = "INV";
constexpr llvm::StringLiteral PLAN = "PLAN";
constexpr llvm::StringLiteral REFLECT = "REFLECT";
constexpr llvm::StringLiteral VERIFY = "VERIFY";
constexpr llvm::StringLiteral MERGE = "MERGE";
constexpr llvm::StringLiteral WAIT_ALL = "WAIT_ALL";
} // namespace graph_ops

struct ValueRef {
  enum class Kind { Node, Param } kind = Kind::Node;
  uint64_t nodeId = 0;
  std::string paramName;

  static ValueRef node(uint64_t id) {
    ValueRef out;
    out.kind = Kind::Node;
    out.nodeId = id;
    return out;
  }

  static ValueRef param(llvm::StringRef name) {
    ValueRef out;
    out.kind = Kind::Param;
    out.paramName = name.str();
    return out;
  }
};

struct GraphNodeRecord {
  uint64_t id;
  std::string name;
  std::string op;
  llvm::json::Object attributes;
};

struct GraphEdgeRecord {
  uint64_t from;
  uint64_t to;
  std::string dependency;
};

struct FlowHandle {
  const AgentDecl *agent = nullptr;
  const FlowDecl *flow = nullptr;
};

struct FlowContext {
  const AgentDecl *agent = nullptr;
  llvm::StringMap<ValueRef> vars;
  llvm::StringSet<> paramNames;
  std::optional<uint64_t> lastNode;
  llvm::SmallPtrSet<const FlowDecl *, 8> activeFlows;
};

enum class CallKind {
  Ask,
  Think,
  Reason,
  QMem,
  UMem,
  Inv,
  Plan,
  Reflect,
  Verify,
  Merge,
  WaitAll,
  Unknown,
};

struct TemplateBuild {
  std::string text;
  llvm::SmallVector<ValueRef, 4> refs;
};

CallKind classifyCall(llvm::StringRef callee) {
  std::string lower = callee.lower();
  return llvm::StringSwitch<CallKind>(lower)
      .Case("ask", CallKind::Ask)
      .Case("think", CallKind::Think)
      .Case("reason", CallKind::Reason)
      .Cases("query_memory", "qmem", CallKind::QMem)
      .Case("mem", CallKind::QMem)
      .Cases("update_memory", "umem", CallKind::UMem)
      .Cases("invoke", "inv", "tool", CallKind::Inv)
      .Case("plan", CallKind::Plan)
      .Case("reflect", CallKind::Reflect)
      .Case("verify", CallKind::Verify)
      .Case("merge", CallKind::Merge)
      .Cases("wait", "wait_all", CallKind::WaitAll)
      .Default(CallKind::Unknown);
}

std::string normalizeMemorySpace(llvm::StringRef value) {
  std::string lower = value.trim().lower();
  if (lower == apxm::constants::memory::LTM)
    return apxm::constants::memory::LTM.str();
  if (lower == apxm::constants::memory::EPISODIC)
    return apxm::constants::memory::EPISODIC.str();
  return apxm::constants::memory::STM.str();
}

std::string makeQualifiedFlowKey(llvm::StringRef agent, llvm::StringRef flow) {
  std::string key;
  key.reserve(agent.size() + 2 + flow.size());
  key.append(agent.begin(), agent.end());
  key.append("::");
  key.append(flow.begin(), flow.end());
  return key;
}

class GraphLowering {
public:
  explicit GraphLowering(std::optional<GraphGenError> &errorSlot)
      : errorSlot(errorSlot) {}

  std::optional<std::string>
  run(const std::vector<std::unique_ptr<AgentDecl>> &agents) {
    if (!indexAgents(agents))
      return std::nullopt;

    if (!selectEntryFlow(agents))
      return std::nullopt;

    FlowContext entryCtx;
    entryCtx.agent = entryAgent;
    entryCtx.activeFlows.insert(entryFlow);
    for (const auto &param : entryFlow->getParams()) {
      entryCtx.paramNames.insert(param.first);
      entryCtx.vars.insert({param.first, ValueRef::param(param.first)});
      parameters.push_back({param.first, param.second});
    }

    std::optional<ValueRef> flowResult;
    if (!lowerFlowBody(entryFlow, entryCtx, flowResult))
      return std::nullopt;

    if (flowResult && flowResult->kind == ValueRef::Kind::Param) {
      llvm::json::Object attrs;
      attrs[graph_attrs::TEMPLATE_STR.str()] = "{0}";
      uint64_t materialized = addNode("return", graph_ops::ASK, std::move(attrs));
      if (!attachDependencies(materialized, llvm::SmallVector<ValueRef, 1>{*flowResult},
                              entryCtx)) {
        return std::nullopt;
      }
    }

    if (nodes.empty()) {
      llvm::json::Object attrs;
      attrs[graph_attrs::VALUE.str()] = "";
      addNode("empty", graph_ops::CONST_STR, std::move(attrs));
    }

    return emitJson();
  }

private:
  std::optional<GraphGenError> &errorSlot;

  std::unordered_map<std::string, const AgentDecl *> agentsByName;
  std::unordered_map<std::string, FlowHandle> flowsByQualifiedName;

  const AgentDecl *entryAgent = nullptr;
  const FlowDecl *entryFlow = nullptr;

  llvm::SmallVector<std::pair<std::string, std::string>, 8> parameters;
  llvm::SmallVector<GraphNodeRecord, 16> nodes;
  llvm::SmallVector<GraphEdgeRecord, 16> edges;
  uint64_t nextNodeId = 1;

  void fail(Location loc, llvm::Twine message) {
    if (errorSlot)
      return;
    errorSlot = GraphGenError{loc, message.str()};
  }

  bool indexAgents(const std::vector<std::unique_ptr<AgentDecl>> &agents) {
    for (const auto &agent : agents) {
      if (!agent)
        continue;

      std::string agentName = agent->getName().str();
      agentsByName.insert({agentName, agent.get()});
      for (const auto &flow : agent->getFlowDecls()) {
        if (!flow)
          continue;
        std::string key = makeQualifiedFlowKey(agent->getName(), flow->getName());
        flowsByQualifiedName.insert({key, FlowHandle{agent.get(), flow.get()}});
      }
    }
    return true;
  }

  bool selectEntryFlow(const std::vector<std::unique_ptr<AgentDecl>> &agents) {
    llvm::SmallVector<FlowHandle, 4> entryFlows;
    llvm::SmallVector<FlowHandle, 8> allFlows;

    for (const auto &agent : agents) {
      if (!agent)
        continue;
      for (const auto &flow : agent->getFlowDecls()) {
        if (!flow)
          continue;
        FlowHandle handle{agent.get(), flow.get()};
        allFlows.push_back(handle);
        if (flow->isEntryFlow())
          entryFlows.push_back(handle);
      }
    }

    if (entryFlows.size() > 1) {
      fail(entryFlows.front().flow->getLocation(),
           "multiple @entry flows found; canonical graph lowering requires exactly one");
      return false;
    }

    if (entryFlows.empty()) {
      if (allFlows.size() == 1) {
        entryAgent = allFlows.front().agent;
        entryFlow = allFlows.front().flow;
        return true;
      }
      fail(Location(), "no @entry flow found; annotate one flow with @entry");
      return false;
    }

    entryAgent = entryFlows.front().agent;
    entryFlow = entryFlows.front().flow;
    return true;
  }

  FlowHandle lookupFlow(llvm::StringRef agentName, llvm::StringRef flowName) const {
    std::string key = makeQualifiedFlowKey(agentName, flowName);
    auto it = flowsByQualifiedName.find(key);
    if (it == flowsByQualifiedName.end())
      return {};
    return it->second;
  }

  uint64_t addNode(llvm::StringRef preferredName, llvm::StringRef op,
                   llvm::json::Object attributes) {
    uint64_t id = nextNodeId++;
    std::string name = preferredName.empty() ? op.lower() : preferredName.str();
    nodes.push_back(GraphNodeRecord{id, name, op.str(), std::move(attributes)});
    return id;
  }

  void addEdge(uint64_t from, uint64_t to, llvm::StringRef dependency) {
    if (from == to)
      return;
    edges.push_back(GraphEdgeRecord{from, to, dependency.str()});
  }

  bool attachDependencies(uint64_t nodeId, llvm::ArrayRef<ValueRef> refs,
                          FlowContext &ctx) {
    std::set<uint64_t> explicitNodeInputs;
    bool hasParamRefs = false;

    for (const ValueRef &ref : refs) {
      if (ref.kind == ValueRef::Kind::Node) {
        explicitNodeInputs.insert(ref.nodeId);
        addEdge(ref.nodeId, nodeId, graph_json::DATA);
      } else {
        hasParamRefs = true;
      }
    }

    if (hasParamRefs && !explicitNodeInputs.empty()) {
      fail(Location(), "mixing parameter-derived and node-derived context in one operation is "
                       "not supported yet");
      return false;
    }

    if (ctx.lastNode && *ctx.lastNode != nodeId &&
        explicitNodeInputs.find(*ctx.lastNode) == explicitNodeInputs.end()) {
      addEdge(*ctx.lastNode, nodeId, graph_json::CONTROL);
    }

    ctx.lastNode = nodeId;
    return true;
  }

  std::string exprToText(const Expr *expr) {
    if (!expr)
      return "";

    return llvm::TypeSwitch<const Expr *, std::string>(expr)
        .Case<StringLiteralExpr>([](auto *e) { return e->getValue().str(); })
        .Case<NumberLiteralExpr>([](auto *e) {
          return llvm::formatv("{0}", e->getValue()).str();
        })
        .Case<BooleanLiteralExpr>([](auto *e) { return e->getValue() ? "true" : "false"; })
        .Case<NullLiteralExpr>([](auto *) { return "null"; })
        .Case<VarExpr>([](auto *e) { return e->getName().str(); })
        .Case<BinaryExpr>([&](auto *e) {
          if (e->getOperator() == BinaryExpr::Operator::Add) {
            return exprToText(e->getLHS()) + exprToText(e->getRHS());
          }
          return std::string("<binary>");
        })
        .Default([](auto *) { return std::string("<expr>"); });
  }

  llvm::json::Value exprToJsonLiteral(const Expr *expr) {
    if (!expr)
      return llvm::json::Value(nullptr);

    return llvm::TypeSwitch<const Expr *, llvm::json::Value>(expr)
        .Case<StringLiteralExpr>([](auto *e) { return llvm::json::Value(e->getValue().str()); })
        .Case<NumberLiteralExpr>([](auto *e) { return llvm::json::Value(e->getValue()); })
        .Case<BooleanLiteralExpr>([](auto *e) { return llvm::json::Value(e->getValue()); })
        .Case<NullLiteralExpr>([](auto *) { return llvm::json::Value(nullptr); })
        .Case<VarExpr>([](auto *e) { return llvm::json::Value("$" + e->getName().str()); })
        .Default([&](auto *) { return llvm::json::Value(exprToText(expr)); });
  }

  std::string serializeArgsAsJson(llvm::ArrayRef<std::unique_ptr<Expr>> args) {
    llvm::json::Array serializedArgs;
    for (const auto &arg : args) {
      serializedArgs.push_back(exprToJsonLiteral(arg.get()));
    }

    llvm::json::Object obj;
    obj["args"] = std::move(serializedArgs);
    return llvm::formatv("{0}", llvm::json::Value(std::move(obj))).str();
  }

  std::optional<ValueRef> lookupVariable(const VarExpr *expr, FlowContext &ctx) {
    auto it = ctx.vars.find(expr->getName());
    if (it == ctx.vars.end()) {
      fail(expr->getLocation(),
           llvm::Twine("undefined variable in graph lowering: ") + expr->getName());
      return std::nullopt;
    }
    return it->second;
  }

  bool appendTemplateReference(const ValueRef &value, TemplateBuild &out) {
    size_t index = out.refs.size();
    out.text.append("{");
    out.text.append(std::to_string(index));
    out.text.append("}");
    out.refs.push_back(value);
    return true;
  }

  bool buildTemplateExpr(const Expr *expr, FlowContext &ctx, TemplateBuild &out) {
    if (!expr) {
      out.text.append("{0}");
      return true;
    }

    if (auto *str = llvm::dyn_cast<StringLiteralExpr>(expr)) {
      out.text.append(str->getValue().str());
      return true;
    }

    if (auto *var = llvm::dyn_cast<VarExpr>(expr)) {
      auto binding = lookupVariable(var, ctx);
      if (!binding)
        return false;
      return appendTemplateReference(*binding, out);
    }

    if (auto *binary = llvm::dyn_cast<BinaryExpr>(expr)) {
      if (binary->getOperator() == BinaryExpr::Operator::Add) {
        return buildTemplateExpr(binary->getLHS(), ctx, out) &&
               buildTemplateExpr(binary->getRHS(), ctx, out);
      }
    }

    auto value = lowerExprToValue(expr, ctx, "template_expr");
    if (!value)
      return false;
    return appendTemplateReference(*value, out);
  }

  std::optional<ValueRef> lowerCallExpr(const CallExpr *expr, FlowContext &ctx,
                                        llvm::StringRef preferredName) {
    llvm::StringRef callee = expr->getCallee();

    // Local flow call syntax: flowName(args)
    if (ctx.agent) {
      auto localFlow = lookupFlow(ctx.agent->getName(), callee);
      if (localFlow.flow) {
        return inlineFlowCall(localFlow.agent, localFlow.flow, expr->getArgs(), ctx);
      }
    }

    CallKind kind = classifyCall(callee);

    auto requireValueRef = [&](const Expr *argExpr,
                               llvm::StringRef hint) -> std::optional<ValueRef> {
      return lowerExprToValue(argExpr, ctx, hint);
    };

    switch (kind) {
    case CallKind::Ask:
    case CallKind::Think:
    case CallKind::Reason: {
      TemplateBuild templateBuild;
      if (!expr->getArgs().empty()) {
        if (!buildTemplateExpr(expr->getArgs().front().get(), ctx, templateBuild))
          return std::nullopt;
      }
      if (templateBuild.text.empty())
        templateBuild.text = "{0}";

      for (size_t i = 1; i < expr->getArgs().size(); ++i) {
        auto value = requireValueRef(expr->getArgs()[i].get(), "context");
        if (!value)
          return std::nullopt;
        templateBuild.refs.push_back(*value);
      }

      llvm::StringRef op = graph_ops::ASK;
      if (kind == CallKind::Think)
        op = graph_ops::THINK;
      else if (kind == CallKind::Reason)
        op = graph_ops::REASON;

      llvm::json::Object attrs;
      attrs[graph_attrs::TEMPLATE_STR.str()] = templateBuild.text;
      uint64_t nodeId = addNode(preferredName.empty() ? callee : preferredName, op,
                                std::move(attrs));
      if (!attachDependencies(nodeId, templateBuild.refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    case CallKind::QMem: {
      std::string query;
      std::string space = apxm::constants::memory::DEFAULT_SPACE.str();
      if (!expr->getArgs().empty())
        query = exprToText(expr->getArgs()[0].get());
      if (expr->getArgs().size() > 1)
        space = normalizeMemorySpace(exprToText(expr->getArgs()[1].get()));

      llvm::json::Object attrs;
      attrs[graph_attrs::QUERY.str()] = query;
      attrs[graph_attrs::SPACE.str()] = space;

      uint64_t nodeId = addNode(preferredName.empty() ? "qmem" : preferredName,
                                graph_ops::QMEM, std::move(attrs));
      if (!attachDependencies(nodeId, {}, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    case CallKind::UMem: {
      llvm::SmallVector<ValueRef, 2> refs;
      std::string space = apxm::constants::memory::DEFAULT_SPACE.str();
      if (!expr->getArgs().empty()) {
        auto value = requireValueRef(expr->getArgs()[0].get(), "umem_value");
        if (!value)
          return std::nullopt;
        refs.push_back(*value);
      }
      if (expr->getArgs().size() > 1)
        space = normalizeMemorySpace(exprToText(expr->getArgs()[1].get()));

      llvm::json::Object attrs;
      attrs[graph_attrs::SPACE.str()] = space;
      uint64_t nodeId = addNode(preferredName.empty() ? "umem" : preferredName,
                                graph_ops::UMEM, std::move(attrs));
      if (!attachDependencies(nodeId, refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    case CallKind::Inv:
    case CallKind::Unknown: {
      std::string capability;
      llvm::ArrayRef<std::unique_ptr<Expr>> args = expr->getArgs();

      if (kind == CallKind::Inv) {
        if (!args.empty()) {
          capability = exprToText(args.front().get());
          args = args.drop_front();
        } else {
          capability = callee.str();
        }
      } else {
        capability = callee.str();
      }

      llvm::json::Object attrs;
      attrs[graph_attrs::CAPABILITY.str()] = capability;
      attrs[graph_attrs::PARAMS_JSON.str()] = serializeArgsAsJson(args);

      uint64_t nodeId = addNode(preferredName.empty() ? capability : preferredName,
                                graph_ops::INV, std::move(attrs));
      if (!attachDependencies(nodeId, {}, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    case CallKind::Plan: {
      llvm::SmallVector<ValueRef, 4> refs;
      std::string goal = "goal";
      if (!expr->getArgs().empty()) {
        if (auto *str = llvm::dyn_cast<StringLiteralExpr>(expr->getArgs()[0].get())) {
          goal = str->getValue().str();
        } else {
          auto value = requireValueRef(expr->getArgs()[0].get(), "goal");
          if (!value)
            return std::nullopt;
          refs.push_back(*value);
        }
      }
      for (size_t i = 1; i < expr->getArgs().size(); ++i) {
        auto value = requireValueRef(expr->getArgs()[i].get(), "plan_context");
        if (!value)
          return std::nullopt;
        refs.push_back(*value);
      }

      llvm::json::Object attrs;
      attrs[graph_attrs::GOAL.str()] = goal;
      uint64_t nodeId = addNode(preferredName.empty() ? "plan" : preferredName,
                                graph_ops::PLAN, std::move(attrs));
      if (!attachDependencies(nodeId, refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    case CallKind::Reflect: {
      llvm::SmallVector<ValueRef, 4> refs;
      std::string traceId = "trace";
      if (!expr->getArgs().empty()) {
        if (auto *str = llvm::dyn_cast<StringLiteralExpr>(expr->getArgs()[0].get())) {
          traceId = str->getValue().str();
        } else {
          auto value = requireValueRef(expr->getArgs()[0].get(), "reflect_trace");
          if (!value)
            return std::nullopt;
          refs.push_back(*value);
        }
      }
      for (size_t i = 1; i < expr->getArgs().size(); ++i) {
        auto value = requireValueRef(expr->getArgs()[i].get(), "reflect_context");
        if (!value)
          return std::nullopt;
        refs.push_back(*value);
      }

      llvm::json::Object attrs;
      attrs[graph_attrs::TRACE_ID.str()] = traceId;
      uint64_t nodeId = addNode(preferredName.empty() ? "reflect" : preferredName,
                                graph_ops::REFLECT, std::move(attrs));
      if (!attachDependencies(nodeId, refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    case CallKind::Verify: {
      if (expr->getArgs().size() < 2) {
        fail(expr->getLocation(), "verify requires at least two arguments");
        return std::nullopt;
      }

      llvm::SmallVector<ValueRef, 2> refs;
      auto claim = requireValueRef(expr->getArgs()[0].get(), "verify_claim");
      if (!claim)
        return std::nullopt;
      refs.push_back(*claim);

      auto evidence = requireValueRef(expr->getArgs()[1].get(), "verify_evidence");
      if (!evidence)
        return std::nullopt;
      refs.push_back(*evidence);

      std::string templateStr = "Verify claim against evidence";
      if (expr->getArgs().size() > 2) {
        if (auto *str = llvm::dyn_cast<StringLiteralExpr>(expr->getArgs()[2].get()))
          templateStr = str->getValue().str();
      }

      llvm::json::Object attrs;
      attrs[graph_attrs::TEMPLATE_STR.str()] = templateStr;
      uint64_t nodeId = addNode(preferredName.empty() ? "verify" : preferredName,
                                graph_ops::VERIFY, std::move(attrs));
      if (!attachDependencies(nodeId, refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    case CallKind::Merge:
    case CallKind::WaitAll: {
      llvm::SmallVector<ValueRef, 4> refs;
      for (const auto &arg : expr->getArgs()) {
        auto value = requireValueRef(arg.get(), "sync_value");
        if (!value)
          return std::nullopt;
        refs.push_back(*value);
      }

      llvm::StringRef op = kind == CallKind::Merge ? graph_ops::MERGE : graph_ops::WAIT_ALL;
      uint64_t nodeId = addNode(preferredName.empty() ? op.lower() : preferredName, op, {});
      if (!attachDependencies(nodeId, refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }
    }

    fail(expr->getLocation(), "unsupported call during graph lowering");
    return std::nullopt;
  }

  std::optional<ValueRef> lowerFlowCallExpr(const FlowCallExpr *expr, FlowContext &ctx) {
    auto flow = lookupFlow(expr->getAgentName(), expr->getFlowName());
    if (!flow.flow) {
      fail(expr->getLocation(),
           llvm::Twine("unknown flow target: ") + expr->getAgentName() + "." +
               expr->getFlowName());
      return std::nullopt;
    }

    return inlineFlowCall(flow.agent, flow.flow, expr->getArgs(), ctx);
  }

  std::optional<ValueRef> inlineFlowCall(const AgentDecl *targetAgent,
                                         const FlowDecl *targetFlow,
                                         llvm::ArrayRef<std::unique_ptr<Expr>> args,
                                         FlowContext &callerCtx) {
    if (callerCtx.activeFlows.contains(targetFlow)) {
      fail(targetFlow->getLocation(),
           llvm::Twine("recursive flow call detected for '") + targetFlow->getName() + "'");
      return std::nullopt;
    }

    llvm::SmallVector<std::optional<ValueRef>, 4> argValues;
    argValues.reserve(args.size());
    for (const auto &arg : args) {
      argValues.push_back(lowerExprToValue(arg.get(), callerCtx, "flow_arg"));
      if (!argValues.back())
        return std::nullopt;
    }

    FlowContext child;
    child.agent = targetAgent;
    child.lastNode = callerCtx.lastNode;
    child.activeFlows = callerCtx.activeFlows;
    child.activeFlows.insert(targetFlow);

    size_t argIndex = 0;
    for (const auto &param : targetFlow->getParams()) {
      child.paramNames.insert(param.first);
      ValueRef binding = ValueRef::param(param.first);
      if (argIndex < argValues.size() && argValues[argIndex]) {
        binding = *argValues[argIndex];
      }
      child.vars.insert({param.first, binding});
      ++argIndex;
    }

    std::optional<ValueRef> flowResult;
    if (!lowerFlowBody(targetFlow, child, flowResult))
      return std::nullopt;

    callerCtx.lastNode = child.lastNode;
    return flowResult;
  }

  std::optional<ValueRef> lowerExprToValue(const Expr *expr, FlowContext &ctx,
                                           llvm::StringRef preferredName) {
    if (!expr) {
      fail(Location(), "null expression in graph lowering");
      return std::nullopt;
    }

    if (auto *var = llvm::dyn_cast<VarExpr>(expr)) {
      return lookupVariable(var, ctx);
    }

    if (auto *call = llvm::dyn_cast<CallExpr>(expr)) {
      return lowerCallExpr(call, ctx, preferredName);
    }

    if (auto *flowCall = llvm::dyn_cast<FlowCallExpr>(expr)) {
      return lowerFlowCallExpr(flowCall, ctx);
    }

    if (auto *str = llvm::dyn_cast<StringLiteralExpr>(expr)) {
      llvm::json::Object attrs;
      attrs[graph_attrs::VALUE.str()] = str->getValue().str();
      uint64_t nodeId = addNode(preferredName.empty() ? "const" : preferredName,
                                graph_ops::CONST_STR, std::move(attrs));
      if (!attachDependencies(nodeId, {}, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    if (auto *num = llvm::dyn_cast<NumberLiteralExpr>(expr)) {
      llvm::json::Object attrs;
      attrs[graph_attrs::VALUE.str()] = llvm::formatv("{0}", num->getValue()).str();
      uint64_t nodeId = addNode(preferredName.empty() ? "const" : preferredName,
                                graph_ops::CONST_STR, std::move(attrs));
      if (!attachDependencies(nodeId, {}, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    if (auto *boolean = llvm::dyn_cast<BooleanLiteralExpr>(expr)) {
      llvm::json::Object attrs;
      attrs[graph_attrs::VALUE.str()] = boolean->getValue() ? "true" : "false";
      uint64_t nodeId = addNode(preferredName.empty() ? "const" : preferredName,
                                graph_ops::CONST_STR, std::move(attrs));
      if (!attachDependencies(nodeId, {}, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    if (llvm::isa<NullLiteralExpr>(expr)) {
      llvm::json::Object attrs;
      attrs[graph_attrs::VALUE.str()] = "null";
      uint64_t nodeId = addNode(preferredName.empty() ? "const" : preferredName,
                                graph_ops::CONST_STR, std::move(attrs));
      if (!attachDependencies(nodeId, {}, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    if (auto *binary = llvm::dyn_cast<BinaryExpr>(expr)) {
      if (binary->getOperator() != BinaryExpr::Operator::Add) {
        fail(binary->getLocation(),
             "only string concatenation (+) is supported for canonical graph lowering");
        return std::nullopt;
      }

      TemplateBuild build;
      if (!buildTemplateExpr(binary, ctx, build))
        return std::nullopt;

      if (build.refs.empty()) {
        llvm::json::Object attrs;
        attrs[graph_attrs::VALUE.str()] = build.text;
        uint64_t nodeId = addNode(preferredName.empty() ? "const" : preferredName,
                                  graph_ops::CONST_STR, std::move(attrs));
        if (!attachDependencies(nodeId, {}, ctx))
          return std::nullopt;
        return ValueRef::node(nodeId);
      }

      llvm::json::Object attrs;
      attrs[graph_attrs::TEMPLATE_STR.str()] = build.text;
      uint64_t nodeId = addNode(preferredName.empty() ? "concat" : preferredName,
                                graph_ops::ASK, std::move(attrs));
      if (!attachDependencies(nodeId, build.refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    if (auto *plan = llvm::dyn_cast<PlanExpr>(expr)) {
      llvm::SmallVector<ValueRef, 4> refs;
      for (const auto &ctxExpr : plan->getContext()) {
        auto value = lowerExprToValue(ctxExpr.get(), ctx, "plan_context");
        if (!value)
          return std::nullopt;
        refs.push_back(*value);
      }

      llvm::json::Object attrs;
      attrs[graph_attrs::GOAL.str()] = plan->getGoal().str();
      uint64_t nodeId = addNode(preferredName.empty() ? "plan" : preferredName,
                                graph_ops::PLAN, std::move(attrs));
      if (!attachDependencies(nodeId, refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    if (auto *reflect = llvm::dyn_cast<ReflectExpr>(expr)) {
      llvm::SmallVector<ValueRef, 4> refs;
      for (const auto &ctxExpr : reflect->getContext()) {
        auto value = lowerExprToValue(ctxExpr.get(), ctx, "reflect_context");
        if (!value)
          return std::nullopt;
        refs.push_back(*value);
      }

      llvm::json::Object attrs;
      attrs[graph_attrs::TRACE_ID.str()] = reflect->getTraceId().str();
      uint64_t nodeId = addNode(preferredName.empty() ? "reflect" : preferredName,
                                graph_ops::REFLECT, std::move(attrs));
      if (!attachDependencies(nodeId, refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    if (auto *verify = llvm::dyn_cast<VerifyExpr>(expr)) {
      llvm::SmallVector<ValueRef, 2> refs;
      auto claim = lowerExprToValue(verify->getClaim(), ctx, "verify_claim");
      if (!claim)
        return std::nullopt;
      refs.push_back(*claim);

      auto evidence = lowerExprToValue(verify->getEvidence(), ctx, "verify_evidence");
      if (!evidence)
        return std::nullopt;
      refs.push_back(*evidence);

      llvm::json::Object attrs;
      attrs[graph_attrs::TEMPLATE_STR.str()] = verify->getTemplate().str();
      uint64_t nodeId = addNode(preferredName.empty() ? "verify" : preferredName,
                                graph_ops::VERIFY, std::move(attrs));
      if (!attachDependencies(nodeId, refs, ctx))
        return std::nullopt;
      return ValueRef::node(nodeId);
    }

    fail(expr->getLocation(), "unsupported expression in graph lowering");
    return std::nullopt;
  }

  bool lowerStatement(const Stmt *stmt, FlowContext &ctx,
                      std::optional<ValueRef> &flowReturn) {
    if (llvm::isa<IfStmt>(stmt) || llvm::isa<ParallelStmt>(stmt) || llvm::isa<LoopStmt>(stmt) ||
        llvm::isa<TryCatchStmt>(stmt) || llvm::isa<SwitchStmt>(stmt)) {
      fail(stmt->getLocation(),
           "structured-control statements are not yet supported by canonical graph lowering");
      return false;
    }

    return llvm::TypeSwitch<const Stmt *, bool>(stmt)
        .Case<LetStmt>([&](auto *letStmt) {
          auto value = lowerExprToValue(letStmt->getInitExpr(), ctx, letStmt->getVarName());
          if (!value)
            return false;
          ctx.vars[letStmt->getVarName()] = *value;
          return true;
        })
        .Case<ExprStmt>([&](auto *exprStmt) {
          return lowerExprToValue(exprStmt->getExpr(), ctx, "expr").has_value();
        })
        .Case<ReturnStmt>([&](auto *returnStmt) {
          auto value = lowerExprToValue(returnStmt->getReturnExpr(), ctx, "return");
          if (!value)
            return false;
          flowReturn = *value;
          return true;
        })
        .Default([&](auto *) {
          fail(stmt->getLocation(), "unsupported statement in graph lowering");
          return false;
        });
  }

  bool lowerFlowBody(const FlowDecl *flow, FlowContext &ctx,
                     std::optional<ValueRef> &flowReturn) {
    for (const auto &stmt : flow->getBody()) {
      if (!lowerStatement(stmt.get(), ctx, flowReturn))
        return false;
      if (flowReturn)
        break;
    }

    if (!flowReturn && ctx.lastNode) {
      flowReturn = ValueRef::node(*ctx.lastNode);
    }

    if (!flowReturn) {
      fail(flow->getLocation(),
           llvm::Twine("flow '") + flow->getName() + "' does not produce a value");
      return false;
    }

    return true;
  }

  std::optional<std::string> emitJson() {
    llvm::json::Object root;

    std::string graphName =
        llvm::formatv("{0}_{1}", entryAgent->getName(), entryFlow->getName()).str();
    root[graph_json::NAME.str()] = graphName;

    llvm::json::Array nodeArray;
    nodeArray.reserve(nodes.size());
    for (auto &node : nodes) {
      llvm::json::Object obj;
      obj[graph_json::ID.str()] = node.id;
      obj[graph_json::NAME.str()] = node.name;
      obj[graph_json::OP.str()] = node.op;
      obj[graph_json::ATTRIBUTES.str()] = std::move(node.attributes);
      nodeArray.push_back(std::move(obj));
    }
    root[graph_json::NODES.str()] = std::move(nodeArray);

    llvm::json::Array edgeArray;
    edgeArray.reserve(edges.size());
    for (const auto &edge : edges) {
      llvm::json::Object obj;
      obj[graph_json::FROM.str()] = edge.from;
      obj[graph_json::TO.str()] = edge.to;
      obj[graph_json::DEPENDENCY.str()] = edge.dependency;
      edgeArray.push_back(std::move(obj));
    }
    root[graph_json::EDGES.str()] = std::move(edgeArray);

    llvm::json::Array paramsArray;
    paramsArray.reserve(parameters.size());
    for (const auto &param : parameters) {
      llvm::json::Object obj;
      obj[graph_json::NAME.str()] = param.first;
      obj[graph_json::TYPE_NAME.str()] = param.second;
      paramsArray.push_back(std::move(obj));
    }
    root[graph_json::PARAMETERS.str()] = std::move(paramsArray);

    llvm::json::Object metadata;
    metadata[graph_json::IS_ENTRY.str()] = true;
    metadata[graph_json::SOURCE_AGENT.str()] = entryAgent->getName().str();
    metadata[graph_json::SOURCE_FLOW.str()] = entryFlow->getName().str();
    root[graph_json::METADATA.str()] = std::move(metadata);

    return llvm::formatv("{0:2}", llvm::json::Value(std::move(root))).str();
  }
};

} // namespace

GraphGen::GraphGen(llvm::SourceMgr &sourceMgr) : srcMgr(sourceMgr) {}

std::optional<std::string>
GraphGen::generateEntryGraphJson(const std::vector<std::unique_ptr<AgentDecl>> &agents) {
  (void)srcMgr;
  lastError.reset();
  GraphLowering lowering(lastError);
  return lowering.run(agents);
}
