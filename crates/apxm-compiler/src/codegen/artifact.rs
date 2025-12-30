use std::collections::HashMap;

use crate::api::module::invalid_input_error;
use apxm_core::error::compiler::CompilerError;
use apxm_core::types::execution::{DagMetadata, ExecutionDag};
use apxm_core::types::values::{Number, Value};
use apxm_core::types::{AISOperationType, DependencyType, Edge, Node, NodeMetadata};

const WIRE_VERSION: u32 = 3;

/// Parse multiple DAGs from wire format v3 (multi-DAG support)
pub fn parse_wire_dags(bytes: &[u8]) -> Result<Vec<ExecutionDag>, CompilerError> {
    let mut reader = BinaryReader::new(bytes);
    let version = reader.read_u32()?;

    if version != WIRE_VERSION {
        return Err(invalid_input_error(format!(
            "Unsupported artifact version {version}, expected {WIRE_VERSION}"
        )));
    }

    let num_dags = reader.read_u64()? as usize;
    let mut dags = Vec::with_capacity(num_dags);

    for _ in 0..num_dags {
        dags.push(parse_single_dag(&mut reader)?);
    }

    Ok(dags)
}

/// Parse a single DAG from the reader (shared logic)
fn parse_single_dag(reader: &mut BinaryReader) -> Result<ExecutionDag, CompilerError> {
    let module_name = reader.read_string()?;
    let is_entry = reader.read_bool()?;

    let node_count = reader.read_u64()? as usize;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(read_node(reader)?);
    }

    let edge_count = reader.read_u64()? as usize;
    let mut edges = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let from = reader.read_u64()?;
        let to = reader.read_u64()?;
        let token_id = reader.read_u64()?;
        let dependency = match reader.read_u8()? {
            0 => DependencyType::Data,
            1 => DependencyType::Effect,
            2 => DependencyType::Control,
            other => {
                return Err(invalid_input_error(format!(
                    "Unknown dependency kind {other} in artifact"
                )));
            }
        };
        edges.push(Edge {
            from,
            to,
            token_id,
            dependency_type: dependency,
        });
    }

    let entry_count = reader.read_u64()? as usize;
    let mut entry_nodes = Vec::with_capacity(entry_count);
    for _ in 0..entry_count {
        entry_nodes.push(reader.read_u64()?);
    }

    let exit_count = reader.read_u64()? as usize;
    let mut exit_nodes = Vec::with_capacity(exit_count);
    for _ in 0..exit_count {
        exit_nodes.push(reader.read_u64()?);
    }

    let metadata = DagMetadata {
        name: if module_name.is_empty() {
            None
        } else {
            Some(module_name)
        },
        is_entry,
    };

    Ok(ExecutionDag {
        nodes,
        edges,
        entry_nodes,
        exit_nodes,
        metadata,
    })
}

/// Parse wire format and return the @entry DAG (or first DAG if no @entry)
/// Legacy function for backward compatibility with single-DAG consumers
pub fn parse_wire_dag(bytes: &[u8]) -> Result<ExecutionDag, CompilerError> {
    let dags = parse_wire_dags(bytes)?;

    // Prefer @entry DAG, fall back to first
    dags.iter()
        .find(|d| d.metadata.is_entry)
        .cloned()
        .or_else(|| dags.into_iter().next())
        .ok_or_else(|| invalid_input_error("No DAGs found in artifact"))
}

fn read_node(reader: &mut BinaryReader) -> Result<Node, CompilerError> {
    let id = reader.read_u64()?;
    let op_index = reader.read_u32()? as usize;
    let op_type = OP_KIND_MAP
        .get(op_index)
        .ok_or_else(|| invalid_input_error(format!("Unknown operation kind index {op_index}")))?;

    let attr_count = reader.read_u64()? as usize;
    let mut attributes = HashMap::with_capacity(attr_count);
    for _ in 0..attr_count {
        let key = reader.read_string()?;
        let value = read_value(reader)?;
        attributes.insert(key, value);
    }

    let input_count = reader.read_u64()? as usize;
    let mut input_tokens = Vec::with_capacity(input_count);
    for _ in 0..input_count {
        input_tokens.push(reader.read_u64()?);
    }

    let output_count = reader.read_u64()? as usize;
    let mut output_tokens = Vec::with_capacity(output_count);
    for _ in 0..output_count {
        output_tokens.push(reader.read_u64()?);
    }

    let priority = reader.read_u32()?;
    let has_latency = reader.read_bool()?;
    let estimated_latency = if has_latency {
        Some(reader.read_u64()?)
    } else {
        None
    };

    Ok(Node {
        id,
        op_type: op_type.clone(),
        attributes,
        input_tokens,
        output_tokens,
        metadata: NodeMetadata {
            priority,
            estimated_latency,
        },
    })
}

fn read_value(reader: &mut BinaryReader) -> Result<Value, CompilerError> {
    match reader.read_u8()? {
        0 => Ok(Value::Null),
        1 => Ok(Value::Bool(reader.read_bool()?)),
        2 => Ok(Value::Number(Number::Integer(reader.read_i64()?))),
        3 => Ok(Value::Number(Number::Float(reader.read_f64()?))),
        4 => Ok(Value::String(reader.read_string()?)),
        5 => {
            let len = reader.read_u64()? as usize;
            let mut values = Vec::with_capacity(len);
            for _ in 0..len {
                values.push(read_value(reader)?);
            }
            Ok(Value::Array(values))
        }
        6 => {
            let len = reader.read_u64()? as usize;
            let mut entries = HashMap::with_capacity(len);
            for _ in 0..len {
                let key = reader.read_string()?;
                let value = read_value(reader)?;
                entries.insert(key, value);
            }
            Ok(Value::Object(entries))
        }
        7 => Ok(Value::Token(reader.read_u64()?)),
        other => Err(invalid_input_error(format!(
            "Unknown value kind {other} in artifact"
        ))),
    }
}

struct BinaryReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> BinaryReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn read_exact(&mut self, count: usize) -> Result<&'a [u8], CompilerError> {
        if self.pos + count > self.bytes.len() {
            return Err(invalid_input_error(
                "Unexpected end of artifact wire payload".to_string(),
            ));
        }
        let slice = &self.bytes[self.pos..self.pos + count];
        self.pos += count;
        Ok(slice)
    }

    fn read_u8(&mut self) -> Result<u8, CompilerError> {
        Ok(self.read_exact(1)?[0])
    }

    fn read_bool(&mut self) -> Result<bool, CompilerError> {
        Ok(self.read_u8()? != 0)
    }

    fn read_u32(&mut self) -> Result<u32, CompilerError> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.read_exact(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, CompilerError> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64, CompilerError> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64, CompilerError> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.read_exact(8)?);
        Ok(f64::from_le_bytes(buf))
    }

    fn read_string(&mut self) -> Result<String, CompilerError> {
        let len = self.read_u64()? as usize;
        let bytes = self.read_exact(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| {
            invalid_input_error("Invalid UTF-8 string in artifact payload".to_string())
        })
    }
}

const OP_KIND_MAP: [AISOperationType; 22] = [
    AISOperationType::Inv,
    AISOperationType::Rsn,
    AISOperationType::QMem,
    AISOperationType::UMem,
    AISOperationType::Plan,
    AISOperationType::WaitAll,
    AISOperationType::Merge,
    AISOperationType::Fence,
    AISOperationType::Exc,
    AISOperationType::Communicate,
    AISOperationType::Reflect,
    AISOperationType::Verify,
    AISOperationType::Err,
    AISOperationType::Return,
    AISOperationType::Jump,
    AISOperationType::BranchOnValue,
    AISOperationType::LoopStart,
    AISOperationType::LoopEnd,
    AISOperationType::TryCatch,
    AISOperationType::ConstStr,
    AISOperationType::Switch,
    AISOperationType::FlowCall,
];
