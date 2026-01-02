#include "ais/Dialect/AIS/Conversion/Artifact/ArtifactEmitter.h"

#include "ais/Dialect/AIS/IR/AISOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AttributeSupport.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::ais;
using llvm::dyn_cast;
using llvm::isa;

namespace {

// Operation kinds for artifact serialization
// LLM ops (Ask/Think/Reason) are markers for runtime config lookup
enum class OperationKind : uint32_t {
  Inv = 0,
  Ask = 1,     // LOW latency LLM op (was Rsn)
  QMem = 2,
  UMem = 3,
  Plan = 4,
  WaitAll = 5,
  Merge = 6,
  Fence = 7,
  Exc = 8,
  Communicate = 9,
  Reflect = 10,
  Verify = 11,
  Err = 12,
  ReturnOp = 13,
  Jump = 14,
  BranchOnValue = 15,
  LoopStart = 16,
  LoopEnd = 17,
  TryCatch = 18,
  ConstStr = 19,
  Switch = 20,
  FlowCall = 21,
  Print = 22,
  Think = 23,  // HIGH latency LLM op
  Reason = 24, // MEDIUM latency LLM op
};

enum class DependencyKind : uint8_t {
  Data = 0,
  Effect = 1,
  Control = 2,
};

struct ArtifactValue;

struct ArtifactValue {
  enum class Kind : uint8_t {
    Null = 0,
    Bool = 1,
    Integer = 2,
    Float = 3,
    String = 4,
    Array = 5,
    Object = 6,
    Token = 7,
  };

  Kind kind = Kind::Null;
  bool boolValue = false;
  int64_t intValue = 0;
  double floatValue = 0.0;
  std::string stringValue;
  std::vector<ArtifactValue> arrayValues;
  std::vector<std::pair<std::string, ArtifactValue>> objectValues;
  uint64_t tokenId = 0;

  static ArtifactValue null() { return ArtifactValue(); }

  static ArtifactValue boolean(bool value) {
    ArtifactValue v;
    v.kind = Kind::Bool;
    v.boolValue = value;
    return v;
  }

  static ArtifactValue integer(int64_t value) {
    ArtifactValue v;
    v.kind = Kind::Integer;
    v.intValue = value;
    return v;
  }

  static ArtifactValue floating(double value) {
    ArtifactValue v;
    v.kind = Kind::Float;
    v.floatValue = value;
    return v;
  }

  static ArtifactValue string(std::string value) {
    ArtifactValue v;
    v.kind = Kind::String;
    v.stringValue = std::move(value);
    return v;
  }

  static ArtifactValue array(std::vector<ArtifactValue> values) {
    ArtifactValue v;
    v.kind = Kind::Array;
    v.arrayValues = std::move(values);
    return v;
  }

  static ArtifactValue
  object(std::vector<std::pair<std::string, ArtifactValue>> entries) {
    ArtifactValue v;
    v.kind = Kind::Object;
    v.objectValues = std::move(entries);
    return v;
  }

  static ArtifactValue token(uint64_t value) {
    ArtifactValue v;
    v.kind = Kind::Token;
    v.tokenId = value;
    return v;
  }
};

struct ArtifactNodeMetadata {
  uint32_t priority = 0;
  std::optional<uint64_t> estimatedLatency;
};

struct ArtifactNode {
  uint64_t id = 0;
  OperationKind opType = OperationKind::Inv;
  std::vector<std::pair<std::string, ArtifactValue>> attributes;
  std::vector<uint64_t> inputTokens;
  std::vector<uint64_t> outputTokens;
  ArtifactNodeMetadata metadata;
};

struct ArtifactEdge {
  uint64_t from = 0;
  uint64_t to = 0;
  uint64_t token = 0;
  DependencyKind dependency = DependencyKind::Data;
};

struct FlowParameter {
  std::string name;
  std::string typeName;
};

struct ArtifactDag {
  std::vector<ArtifactNode> nodes;
  std::vector<ArtifactEdge> edges;
  std::vector<uint64_t> entryNodes;
  std::vector<uint64_t> exitNodes;
  std::string moduleName;
  bool isEntry = false; // True if this flow is marked with @entry
  std::vector<FlowParameter> parameters; // Entry flow parameters
};

// Forward declaration
void computeEntryAndExit(const ArtifactDag &dag, std::vector<uint64_t> &entry,
                         std::vector<uint64_t> &exit);

struct DagBuildState {
  uint64_t nextTokenId = 1;
  uint64_t nextNodeId = 1;
  llvm::DenseMap<Value, uint64_t> tokenIds;
  llvm::DenseMap<Value, uint64_t> producerNodeIds;
  std::vector<ArtifactEdge> edges;

  uint64_t getTokenId(Value value) {
    auto it = tokenIds.find(value);
    if (it != tokenIds.end())
      return it->second;

    uint64_t id = nextTokenId++;
    tokenIds.insert({value, id});
    return id;
  }

  void addEdge(uint64_t producerNodeId, uint64_t consumerNodeId,
               uint64_t tokenId,
               DependencyKind dependency = DependencyKind::Data) {
    ArtifactEdge edge;
    edge.from = producerNodeId;
    edge.to = consumerNodeId;
    edge.token = tokenId;
    edge.dependency = dependency;
    edges.push_back(std::move(edge));
  }
};

class ArtifactSerializer {
public:
  void writeU8(uint8_t value) { buffer.push_back(value); }

  void writeU32(uint32_t value) {
    for (int i = 0; i < 4; ++i)
      buffer.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
  }

  void writeU64(uint64_t value) {
    for (int i = 0; i < 8; ++i)
      buffer.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
  }

  void writeI64(int64_t value) { writeU64(static_cast<uint64_t>(value)); }

  void writeF64(double value) {
    static_assert(sizeof(double) == sizeof(uint64_t), "Unexpected double size");
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(double));
    writeU64(bits);
  }

  void writeBool(bool value) { writeU8(value ? 1 : 0); }

  void writeString(StringRef value) {
    writeU64(value.size());
    buffer.insert(buffer.end(), value.begin(), value.end());
  }

  void writeValue(const ArtifactValue &value) {
    writeU8(static_cast<uint8_t>(value.kind));
    switch (value.kind) {
    case ArtifactValue::Kind::Null:
      break;
    case ArtifactValue::Kind::Bool:
      writeBool(value.boolValue);
      break;
    case ArtifactValue::Kind::Integer:
      writeI64(value.intValue);
      break;
    case ArtifactValue::Kind::Float:
      writeF64(value.floatValue);
      break;
    case ArtifactValue::Kind::String:
      writeString(value.stringValue);
      break;
    case ArtifactValue::Kind::Array:
      writeU64(value.arrayValues.size());
      for (const auto &element : value.arrayValues)
        writeValue(element);
      break;
    case ArtifactValue::Kind::Object:
      writeU64(value.objectValues.size());
      for (const auto &entry : value.objectValues) {
        writeString(entry.first);
        writeValue(entry.second);
      }
      break;
    case ArtifactValue::Kind::Token:
      writeU64(value.tokenId);
      break;
    }
  }

  void writeNode(const ArtifactNode &node) {
    writeU64(node.id);
    writeU32(static_cast<uint32_t>(node.opType));

    writeU64(node.attributes.size());
    for (const auto &attr : node.attributes) {
      writeString(attr.first);
      writeValue(attr.second);
    }

    writeU64(node.inputTokens.size());
    for (uint64_t token : node.inputTokens)
      writeU64(token);

    writeU64(node.outputTokens.size());
    for (uint64_t token : node.outputTokens)
      writeU64(token);

    writeU32(node.metadata.priority);
    writeBool(node.metadata.estimatedLatency.has_value());
    if (node.metadata.estimatedLatency)
      writeU64(*node.metadata.estimatedLatency);
  }

  void writeDag(const ArtifactDag &dag) {
    writeString(dag.moduleName);
    writeBool(dag.isEntry); // @entry flow marker

    // Write parameter metadata for entry flows
    writeU64(dag.parameters.size());
    for (const auto &param : dag.parameters) {
      writeString(param.name);
      writeString(param.typeName);
    }

    writeU64(dag.nodes.size());
    for (const auto &node : dag.nodes)
      writeNode(node);

    writeU64(dag.edges.size());
    for (const auto &edge : dag.edges) {
      writeU64(edge.from);
      writeU64(edge.to);
      writeU64(edge.token);
      writeU8(static_cast<uint8_t>(edge.dependency));
    }

    writeU64(dag.entryNodes.size());
    for (uint64_t id : dag.entryNodes)
      writeU64(id);

    writeU64(dag.exitNodes.size());
    for (uint64_t id : dag.exitNodes)
      writeU64(id);
  }

  std::vector<uint8_t> takeBuffer() { return std::move(buffer); }

private:
  std::vector<uint8_t> buffer;
};

std::optional<OperationKind> mapOperation(Operation *op) {
  return TypeSwitch<Operation *, std::optional<OperationKind>>(op)
      .Case<ConstStrOp>([](auto) { return OperationKind::ConstStr; })
      .Case<QMemOp>([](auto) { return OperationKind::QMem; })
      .Case<UMemOp>([](auto) { return OperationKind::UMem; })
      .Case<InvOp>([](auto) { return OperationKind::Inv; })
      .Case<AskOp>([](auto) { return OperationKind::Ask; })
      .Case<ThinkOp>([](auto) { return OperationKind::Think; })
      .Case<ReasonOp>([](auto) { return OperationKind::Reason; })
      .Case<ReflectOp>([](auto) { return OperationKind::Reflect; })
      .Case<VerifyOp>([](auto) { return OperationKind::Verify; })
      .Case<PlanOp>([](auto) { return OperationKind::Plan; })
      .Case<ExcOp>([](auto) { return OperationKind::Exc; })
      .Case<PrintOp>([](auto) { return OperationKind::Print; })
      .Case<WaitAllOp>([](auto) { return OperationKind::WaitAll; })
      .Case<MergeOp>([](auto) { return OperationKind::Merge; })
      .Case<FenceOp>([](auto) { return OperationKind::Fence; })
      .Case<CommunicateOp>([](auto) { return OperationKind::Communicate; })
      .Case<ErrOp>([](auto) { return OperationKind::Err; })
      .Case<ReturnOp>([](auto) { return OperationKind::ReturnOp; })
      .Case<func::ReturnOp>([](auto) { return OperationKind::ReturnOp; })
      .Case<JumpOp>([](auto) { return OperationKind::Jump; })
      .Case<BranchOnValueOp>([](auto) { return OperationKind::BranchOnValue; })
      .Case<LoopStartOp>([](auto) { return OperationKind::LoopStart; })
      .Case<LoopEndOp>([](auto) { return OperationKind::LoopEnd; })
      .Case<SwitchOp>([](auto) { return OperationKind::Switch; })
      .Case<FlowCallOp>([](auto) { return OperationKind::FlowCall; })
      .Case<TryCatchOp>([](auto) { return OperationKind::TryCatch; })
      .Case<YieldOp>([](auto) {
        return std::nullopt;
      }) // Skip yield - it's a region terminator
      .Default([](Operation *) { return std::nullopt; });
}

/// Check if an operation is a region terminator that should be skipped
bool isRegionTerminator(Operation *op) { return isa<YieldOp>(op); }

ArtifactValue convertAttribute(Attribute attr) {
  if (!attr)
    return ArtifactValue::null();

  if (auto strAttr = dyn_cast<StringAttr>(attr))
    return ArtifactValue::string(strAttr.getValue().str());

  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    if (intAttr.getType().isInteger(1))
      return ArtifactValue::boolean(!intAttr.getValue().isZero());
    return ArtifactValue::integer(intAttr.getInt());
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(attr))
    return ArtifactValue::floating(floatAttr.getValueAsDouble());

  if (isa<UnitAttr>(attr))
    return ArtifactValue::boolean(true);

  if (auto arrayAttr = dyn_cast<ArrayAttr>(attr)) {
    std::vector<ArtifactValue> values;
    values.reserve(arrayAttr.size());
    for (Attribute element : arrayAttr)
      values.push_back(convertAttribute(element));
    return ArtifactValue::array(std::move(values));
  }

  if (auto dictAttr = dyn_cast<DictionaryAttr>(attr)) {
    std::vector<std::pair<std::string, ArtifactValue>> entries;
    entries.reserve(dictAttr.size());
    for (NamedAttribute named : dictAttr.getValue()) {
      entries.emplace_back(named.getName().str(),
                           convertAttribute(named.getValue()));
    }
    return ArtifactValue::object(std::move(entries));
  }

  std::string fallback;
  llvm::raw_string_ostream os(fallback);
  attr.print(os);
  os.flush();
  return ArtifactValue::string(fallback);
}

// Forward declarations for mutual recursion
LogicalResult emitNode(Operation *op, DagBuildState &state, ArtifactDag &dag);

/// Emit a region as a sub-DAG serialized to an ArtifactValue
/// Returns an object with nodes, edges, entry_nodes, exit_nodes
LogicalResult emitRegionAsSubDag(Region &region, DagBuildState &parentState,
                                 ArtifactValue &result) {
  // Create a fresh state for the sub-DAG with unique node IDs
  // Use a separate ID space starting from a high offset to avoid conflicts
  DagBuildState subState;
  subState.nextNodeId = parentState.nextNodeId;
  subState.nextTokenId = parentState.nextTokenId;

  ArtifactDag subDag;

  for (Block &block : region) {
    for (Operation &op : block) {
      // Skip yield operations - they just mark the exit value
      if (isRegionTerminator(&op))
        continue;

      if (failed(emitNode(&op, subState, subDag)))
        return failure();
    }
  }

  // Update parent state with consumed IDs
  parentState.nextNodeId = subState.nextNodeId;
  parentState.nextTokenId = subState.nextTokenId;

  subDag.edges = std::move(subState.edges);

  // Compute entry and exit nodes for the sub-DAG
  std::vector<uint64_t> entry, exit;
  computeEntryAndExit(subDag, entry, exit);

  // Serialize the sub-DAG as an object value
  std::vector<std::pair<std::string, ArtifactValue>> subDagObj;

  // Serialize nodes
  std::vector<ArtifactValue> nodesArray;
  for (const auto &node : subDag.nodes) {
    std::vector<std::pair<std::string, ArtifactValue>> nodeObj;
    nodeObj.emplace_back("id", ArtifactValue::integer(node.id));
    nodeObj.emplace_back(
        "op_type", ArtifactValue::integer(static_cast<int64_t>(node.opType)));

    std::vector<std::pair<std::string, ArtifactValue>> attrsObj;
    for (const auto &attr : node.attributes) {
      attrsObj.emplace_back(attr.first, attr.second);
    }
    nodeObj.emplace_back("attributes",
                         ArtifactValue::object(std::move(attrsObj)));

    std::vector<ArtifactValue> inputTokens;
    for (uint64_t t : node.inputTokens) {
      inputTokens.push_back(ArtifactValue::integer(t));
    }
    nodeObj.emplace_back("input_tokens",
                         ArtifactValue::array(std::move(inputTokens)));

    std::vector<ArtifactValue> outputTokens;
    for (uint64_t t : node.outputTokens) {
      outputTokens.push_back(ArtifactValue::integer(t));
    }
    nodeObj.emplace_back("output_tokens",
                         ArtifactValue::array(std::move(outputTokens)));

    nodesArray.push_back(ArtifactValue::object(std::move(nodeObj)));
  }
  subDagObj.emplace_back("nodes", ArtifactValue::array(std::move(nodesArray)));

  // Serialize edges
  std::vector<ArtifactValue> edgesArray;
  for (const auto &edge : subDag.edges) {
    std::vector<std::pair<std::string, ArtifactValue>> edgeObj;
    edgeObj.emplace_back("from", ArtifactValue::integer(edge.from));
    edgeObj.emplace_back("to", ArtifactValue::integer(edge.to));
    edgeObj.emplace_back("token", ArtifactValue::integer(edge.token));
    edgeObj.emplace_back(
        "dependency",
        ArtifactValue::integer(static_cast<int64_t>(edge.dependency)));
    edgesArray.push_back(ArtifactValue::object(std::move(edgeObj)));
  }
  subDagObj.emplace_back("edges", ArtifactValue::array(std::move(edgesArray)));

  // Entry and exit nodes
  std::vector<ArtifactValue> entryArray, exitArray;
  for (uint64_t id : entry) {
    entryArray.push_back(ArtifactValue::integer(id));
  }
  for (uint64_t id : exit) {
    exitArray.push_back(ArtifactValue::integer(id));
  }
  subDagObj.emplace_back("entry_nodes",
                         ArtifactValue::array(std::move(entryArray)));
  subDagObj.emplace_back("exit_nodes",
                         ArtifactValue::array(std::move(exitArray)));

  result = ArtifactValue::object(std::move(subDagObj));
  return success();
}

void computeEntryAndExit(const ArtifactDag &dag, std::vector<uint64_t> &entry,
                         std::vector<uint64_t> &exit) {
  llvm::DenseMap<uint64_t, uint64_t> incoming;
  llvm::DenseMap<uint64_t, uint64_t> outgoing;

  for (const auto &node : dag.nodes) {
    incoming.try_emplace(node.id, 0);
    outgoing.try_emplace(node.id, 0);
  }

  for (const auto &edge : dag.edges) {
    incoming[edge.to]++;
    outgoing[edge.from]++;
  }

  for (const auto &node : dag.nodes) {
    if (incoming.lookup(node.id) == 0)
      entry.push_back(node.id);
    if (outgoing.lookup(node.id) == 0)
      exit.push_back(node.id);
  }
}

LogicalResult emitNode(Operation *op, DagBuildState &state, ArtifactDag &dag) {
  // Skip region terminators - they're handled by their parent ops
  if (isRegionTerminator(op))
    return success();

  auto kind = mapOperation(op);
  if (!kind)
    return op->emitError("Unsupported AIS operation for artifact emission"),
           failure();

  ArtifactNode node;
  node.id = state.nextNodeId++;
  node.opType = *kind;

  for (NamedAttribute named : op->getAttrs()) {
    StringRef key = named.getName().getValue();
    // Translate attribute names to match runtime expectations
    StringRef emitKey = key;
    if (key == "parameters") emitKey = "params";
    if (key == "space") emitKey = "memory_tier";  // MLIR uses space, runtime expects memory_tier
    node.attributes.emplace_back(emitKey.str(),
                                 convertAttribute(named.getValue()));
  }

  // UMemOp needs a key attribute - auto-generate from node ID
  if (isa<UMemOp>(op)) {
    node.attributes.emplace_back("key",
        ArtifactValue::string("mem_" + std::to_string(node.id)));
  }

  // ReasonOp supports inner planning for structured reasoning
  // Note: PlanOp has its own handler (plan.rs) that manages inner plans differently
  if (isa<ais::ReasonOp>(op)) {
    node.attributes.emplace_back("inner_plan_supported",
                                 ArtifactValue::boolean(true));
  }

  // Special handling for SwitchOp: emit regions as sub-DAGs
  if (auto switchOp = dyn_cast<SwitchOp>(op)) {
    // Emit case regions as sub-DAGs
    std::vector<ArtifactValue> caseRegionsArray;
    for (Region &caseRegion : switchOp.getCaseRegions()) {
      ArtifactValue subDag;
      if (failed(emitRegionAsSubDag(caseRegion, state, subDag)))
        return failure();
      caseRegionsArray.push_back(std::move(subDag));
    }
    node.attributes.emplace_back(
        "case_regions", ArtifactValue::array(std::move(caseRegionsArray)));

    // Emit default region as sub-DAG
    ArtifactValue defaultSubDag;
    if (failed(emitRegionAsSubDag(switchOp.getDefaultRegion(), state,
                                  defaultSubDag)))
      return failure();
    node.attributes.emplace_back("default_region", std::move(defaultSubDag));
  }

  for (Value operand : op->getOperands()) {
    uint64_t tokenId = state.getTokenId(operand);
    node.inputTokens.push_back(tokenId);
    auto producerIt = state.producerNodeIds.find(operand);
    if (producerIt != state.producerNodeIds.end()) {
      state.addEdge(producerIt->second, node.id, tokenId);
    }
  }

  for (Value result : op->getResults()) {
    uint64_t tokenId = state.getTokenId(result);
    node.outputTokens.push_back(tokenId);
    state.producerNodeIds.insert({result, node.id});
  }

  dag.nodes.push_back(std::move(node));
  return success();
}

LogicalResult emitFunction(func::FuncOp func,
                           const ArtifactEmitOptions &options,
                           ArtifactDag &dag) {
  DagBuildState state;

  for (Block &block : func) {
    for (Operation &op : block) {
      if (failed(emitNode(&op, state, dag)))
        return failure();
    }
  }

  dag.edges = std::move(state.edges);

  dag.entryNodes.clear();
  dag.exitNodes.clear();
  computeEntryAndExit(dag, dag.entryNodes, dag.exitNodes);

  if (!options.moduleName.empty())
    dag.moduleName = options.moduleName;
  else
    dag.moduleName = func.getName().str();

  // Check if this flow is marked as @entry (set by parser)
  dag.isEntry = func->hasAttr("ais.entry");

  // Extract parameter metadata from function arguments
  for (unsigned i = 0; i < func.getNumArguments(); ++i) {
    FlowParameter param;
    if (auto nameAttr = func.getArgAttrOfType<StringAttr>(i, "ais.param_name")) {
      param.name = nameAttr.getValue().str();
    } else {
      param.name = "arg" + std::to_string(i);
    }
    if (auto typeAttr = func.getArgAttrOfType<StringAttr>(i, "ais.param_type")) {
      param.typeName = typeAttr.getValue().str();
    } else {
      param.typeName = "any";
    }
    dag.parameters.push_back(std::move(param));
  }

  return success();
}

} // namespace

ArtifactEmitter::ArtifactEmitter(const ArtifactEmitOptions &options)
    : options(options) {}

LogicalResult ArtifactEmitter::emitModule(ModuleOp module) {
  buffer.clear();

  auto funcs = module.getOps<func::FuncOp>();
  if (funcs.empty()) {
    module.emitError("Module does not contain any functions to emit");
    return failure();
  }

  // Collect ALL DAGs (one per function) for multi-agent support
  std::vector<ArtifactDag> dags;
  for (auto func : funcs) {
    ArtifactDag dag;
    if (failed(emitFunction(func, options, dag)))
      return failure();
    dags.push_back(std::move(dag));
  }

  // Serialize multi-DAG format (version 3)
  ArtifactSerializer serializer;
  serializer.writeU32(3); // Wire format version 3 = multi-DAG
  serializer.writeU64(dags.size());
  for (const auto &dag : dags) {
    serializer.writeDag(dag);
  }

  buffer = serializer.takeBuffer();
  return success();
}
