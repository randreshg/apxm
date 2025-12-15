/*
 * @file RustEmitter.cpp
 * @brief Implementation of Rust code emission from AIS MLIR
 *
 * The emission works as follows:
 * 1. Convert AIS MLIR operations to Rust code.
 * 2. Generate Rust code for each operation.
 * 3. Emit the generated Rust code to a file.
 */

#include "apxm/Dialect/AIS/Conversion/Rust/RustEmitter.h"
#include "apxm/Dialect/AIS/IR/AISOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::ais;

namespace {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

std::string escapeRustString(StringRef str) {
  llvm::SmallString<64> escaped;
  escaped.reserve(str.size() * 2);

  for (char c : str) {
    switch (c) {
    case '\\': escaped.append("\\\\"); break;
    case '"':  escaped.append("\\\""); break;
    case '\n': escaped.append("\\n"); break;
    case '\t': escaped.append("\\t"); break;
    case '\r': escaped.append("\\r"); break;
    default:
      if (static_cast<unsigned char>(c) < 32) {
        escaped.append("\\u{");
        llvm::raw_svector_ostream(escaped) << llvm::format_hex_no_prefix(
            static_cast<unsigned>(c), 2, false);
        escaped.append("}");
      } else {
        escaped.push_back(c);
      }
      break;
    }
  }

  return escaped.str().str();
}

std::string joinStrings(ArrayRef<std::string> parts, StringRef separator) {
  if (parts.empty()) return {};

  size_t totalSize = (parts.size() - 1) * separator.size();
  for (const auto &part : parts) {
    totalSize += part.size();
  }

  std::string result;
  result.reserve(totalSize);

  result.append(parts[0]);
  for (size_t i = 1; i < parts.size(); ++i) {
    result.append(separator);
    result.append(parts[i]);
  }

  return result;
}

//===----------------------------------------------------------------------===//
// Attribute Emission
//===----------------------------------------------------------------------===//

std::string emitAttrValue(Attribute attr) {
  if (!attr) return std::string("Value::Null");

  return TypeSwitch<Attribute, std::string>(attr)
      .Case([](StringAttr str) {
        return llvm::formatv("Value::String(\"{0}\".to_string())",
                           escapeRustString(str.getValue())).str();
      })
      .Case([](IntegerAttr i) {
        return llvm::formatv("Value::Number(Number::from({0}i64))", i.getInt()).str();
      })
      .Case([](FloatAttr f) {
        return llvm::formatv("Value::Number(Number::from({0}))", f.getValueAsDouble()).str();
      })
      .Case([](UnitAttr) {
        return std::string("Value::Bool(true)");
      })
      .Case([](ArrayAttr arr) {
        llvm::SmallVector<std::string, 8> elements;
        for (Attribute a : arr) {
          elements.push_back(emitAttrValue(a));
        }
        return llvm::formatv("Value::Array(vec![{0}])", joinStrings(elements, ", ")).str();
      })
      .Case([](DictionaryAttr dict) {
        llvm::SmallVector<std::string, 8> pairs;
        for (NamedAttribute named : dict.getValue()) {
          std::string key = escapeRustString(named.getName().getValue());
          std::string value = emitAttrValue(named.getValue());
          pairs.push_back(llvm::formatv("(\"{0}\".to_string(), {1})", key, value).str());
        }

        if (pairs.empty()) {
          return std::string("Value::Object(HashMap::new())");
        }
        return llvm::formatv("Value::Object(HashMap::from([{0}]))",
                           joinStrings(pairs, ", ")).str();
      })
      .Default([&](Attribute) {
        std::string fallback;
        llvm::raw_string_ostream os(fallback);
        attr.print(os);
        return llvm::formatv("Value::String(\"{0}\".to_string())",
                           escapeRustString(fallback)).str();
      });
}

//===----------------------------------------------------------------------===//
// DAG Construction State
//===----------------------------------------------------------------------===//

struct DagBuildState {
  uint64_t nextTokenId = 1;
  uint64_t nextNodeId = 1;
  DenseMap<Value, uint64_t> tokenIds;
  DenseMap<Value, uint64_t> producerNodeIds;
  SmallVector<std::string, 64> edgeDeclarations;

  uint64_t getTokenId(Value value) {
    auto it = tokenIds.find(value);
    if (it != tokenIds.end()) {
      return it->second;
    }

    uint64_t id = nextTokenId++;
    tokenIds.insert({value, id});
    return id;
  }

  void addEdgeDeclaration(StringRef declaration) {
    edgeDeclarations.push_back(declaration.str());
  }

  void emitEdges(llvm::raw_ostream &os, unsigned indentSize) const {
    for (StringRef edge : edgeDeclarations) {
      os.indent(indentSize) << edge << '\n';
    }
  }
};

//===----------------------------------------------------------------------===//
// Operation Type Mapping
//===----------------------------------------------------------------------===//

std::optional<StringRef> mapAisOpToRustType(Operation *op) {
  return TypeSwitch<Operation *, std::optional<StringRef>>(op)
      .Case<ConstStrOp>([](auto) { return "AISOperationType::ConstStr"; })
      .Case<QMemOp>([](auto) { return "AISOperationType::QMem"; })
      .Case<UMemOp>([](auto) { return "AISOperationType::UMem"; })
      .Case<InvOp>([](auto) { return "AISOperationType::Inv"; })
      .Case<RsnOp>([](auto) { return "AISOperationType::Rsn"; })
      .Case<ReflectOp>([](auto) { return "AISOperationType::Reflect"; })
      .Case<VerifyOp>([](auto) { return "AISOperationType::Verify"; })
      .Case<PlanOp>([](auto) { return "AISOperationType::Plan"; })
      .Case<ExcOp>([](auto) { return "AISOperationType::Exc"; })
      .Case<WaitAllOp>([](auto) { return "AISOperationType::WaitAll"; })
      .Case<MergeOp>([](auto) { return "AISOperationType::Merge"; })
      .Case<FenceOp>([](auto) { return "AISOperationType::Fence"; })
      .Case<CommunicateOp>([](auto) { return "AISOperationType::Communicate"; })
      .Case<ErrOp>([](auto) { return "AISOperationType::Err"; })
      .Case<ReturnOp>([](auto) { return "AISOperationType::Return"; })
      .Case<JumpOp>([](auto) { return "AISOperationType::Jump"; })
      .Case<BranchOnValueOp>([](auto) { return "AISOperationType::BranchOnValue"; })
      .Case<LoopStartOp>([](auto) { return "AISOperationType::LoopStart"; })
      .Case<LoopEndOp>([](auto) { return "AISOperationType::LoopEnd"; })
      .Case<TryCatchOp>([](auto) { return "AISOperationType::TryCatch"; })
      .Default([](Operation *) { return std::nullopt; });
}

//===----------------------------------------------------------------------===//
// Edge Generation
//===----------------------------------------------------------------------===//

void emitEdgesForOperand(Operation *op, uint64_t consumerNodeId,
                        DagBuildState &state, llvm::raw_ostream &os) {
  for (Value operand : op->getOperands()) {
    auto producerIt = state.producerNodeIds.find(operand);
    if (producerIt == state.producerNodeIds.end()) {
      continue;
    }

    uint64_t tokenId = state.getTokenId(operand);
    uint64_t producerNodeId = producerIt->second;

    state.addEdgeDeclaration(
        llvm::formatv("dag.edges.push(Edge::new({0}u64, {1}u64, {2}u64, DependencyType::Data));",
                     producerNodeId, consumerNodeId, tokenId)
            .str());
  }
}

//===----------------------------------------------------------------------===//
// Node Construction
//===----------------------------------------------------------------------===//

LogicalResult emitNode(Operation *op, DagBuildState &state,
                      llvm::raw_ostream &os, unsigned indentSize,
                      const RustCodegenOptions &options) {
  auto opType = mapAisOpToRustType(op);
  if (!opType) {
    return op->emitError("unsupported operation type for Rust emission");
  }

  uint64_t nodeId = state.nextNodeId++;
  os.indent(indentSize) << "{ // node " << nodeId << "\n";

  // Node creation
  os.indent(indentSize + options.indentSize)
     << "let mut node = Node::new(" << nodeId << "u64, " << *opType << ");\n";

  // Attributes with key normalization
  for (NamedAttribute named : op->getAttrs()) {
    StringRef key = named.getName().getValue();
    StringRef emitKey = key == "parameters" ? "params" : key;

    std::string valueExpr = emitAttrValue(named.getValue());
    os.indent(indentSize + options.indentSize)
       << "node.set_attribute(\"" << escapeRustString(emitKey)
       << "\".to_string(), " << valueExpr << ");\n";
  }

  // Inputs
  for (Value operand : op->getOperands()) {
    uint64_t tokenId = state.getTokenId(operand);
    os.indent(indentSize + options.indentSize)
       << "node.add_input_token(" << tokenId << "u64);\n";
  }

  // Outputs
  for (Value result : op->getResults()) {
    uint64_t tokenId = state.getTokenId(result);
    os.indent(indentSize + options.indentSize)
       << "node.add_output_token(" << tokenId << "u64);\n";
    state.producerNodeIds.insert({result, nodeId});
  }

  // Add to DAG
  os.indent(indentSize + options.indentSize) << "dag.nodes.push(node);\n";
  os.indent(indentSize) << "}\n";

  // Edges
  emitEdgesForOperand(op, nodeId, state, os);

  return success();
}

//===----------------------------------------------------------------------===//
// Function Emission
//===----------------------------------------------------------------------===//

LogicalResult emitFunction(func::FuncOp func, DagBuildState &state,
                          llvm::raw_ostream &os, const RustCodegenOptions &options) {
  std::string builderName = options.moduleName.empty()
                          ? ("build_" + func.getName().str())
                          : ("build_" + options.moduleName);

  os << "pub fn " << builderName << "() -> ExecutionDag {\n";
  os.indent(options.indentSize) << "let mut dag = ExecutionDag::new();\n";

  // DAG metadata
  if (!options.moduleName.empty()) {
    os.indent(options.indentSize) << "dag.metadata.name = Some(\""
       << escapeRustString(options.moduleName) << "\".to_string());\n";
  } else {
    os.indent(options.indentSize) << "dag.metadata.name = Some(\""
       << escapeRustString(func.getName()) << "\".to_string());\n";
  }
  os << '\n';

  // Nodes initialization
  os.indent(options.indentSize) << "// Nodes\n";
  os.indent(options.indentSize) << "dag.nodes = Vec::new();\n\n";

  // Emit all operations
  for (Block &block : func) {
    for (Operation &op : block) {
      if (failed(emitNode(&op, state, os, options.indentSize, options))) {
        return failure();
      }
    }
  }

  // Edges initialization
  os << '\n';
  os.indent(options.indentSize) << "// Edges\n";
  os.indent(options.indentSize) << "dag.edges = Vec::new();\n";
  state.emitEdges(os, options.indentSize);
  os << '\n';

  // Entry/exit nodes
  os.indent(options.indentSize) << "dag.entry_nodes = dag.find_entry_nodes();\n";
  os.indent(options.indentSize) << "dag.exit_nodes = dag.find_exit_nodes();\n";
  os.indent(options.indentSize) << "dag\n";
  os << "}\n\n";

  return success();
}

//===----------------------------------------------------------------------===//
// Module Emission
//===----------------------------------------------------------------------===//

LogicalResult emitModule(ModuleOp module, llvm::raw_ostream &os,
                        const RustCodegenOptions &options) {
  // Header
  os << "// Generated by APXM Compiler\n";
  os << "// This file is generated. Do not edit manually.\n\n";
  os << "use std::collections::HashMap;\n";
  os << "use apxm_core::types::{AISOperationType, DependencyType, Edge, ExecutionDag, Node, NodeMetadata, Number, Value};\n";
  os << "use apxm_runtime::{Runtime, RuntimeResult};\n";
  os << "use apxm_runtime::config::RuntimeConfig;\n\n";

  // Function builders
  DagBuildState state;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (failed(emitFunction(func, state, os, options))) {
      return failure();
    }
  }

  // Main function if requested
  if (options.emitMainFunction) {
    std::string entryBuilder = options.moduleName.empty()
                             ? "build_main"
                             : ("build_" + options.moduleName);

    os << "#[tokio::main]\n";
    os << "async fn main() -> RuntimeResult<()> {\n";
    os.indent(options.indentSize) << "let dag = " << entryBuilder << "();\n";
    os.indent(options.indentSize) << "let runtime = Runtime::new(RuntimeConfig::default());\n";
    os.indent(options.indentSize) << "let _report = runtime.execute(dag).await?;\n";
    os.indent(options.indentSize) << "Ok(())\n";
    os << "}\n";
  }

  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// RustEmitter Implementation
//===----------------------------------------------------------------------===//

class RustEmitter::Impl {
public:
  explicit Impl(llvm::raw_ostream &os, const RustCodegenOptions &options)
      : os(os), options(options) {}

  LogicalResult emitModule(ModuleOp module) {
    return ::emitModule(module, os, options);
  }

private:
  llvm::raw_ostream &os;
  RustCodegenOptions options;
};

RustEmitter::RustEmitter(llvm::raw_ostream &os, const RustCodegenOptions &options)
    : impl(std::make_unique<Impl>(os, options)) {}

LogicalResult RustEmitter::emitModule(ModuleOp module) {
  return impl->emitModule(module);
}

RustEmitter::~RustEmitter() = default;
