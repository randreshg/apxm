//! CodeGen operations are responsible for generating Rust code from MLIR operations.
pub mod base;
pub mod control_flow;
pub mod literals;

use crate::codegen::emitter::RustEmitter;
use apxm_core::error::compiler::Result;
use std::collections::HashMap;

use apxm_core::types::OperationMetadata;
use apxm_core::types::operation_definitions::{INV, MERGE, QMEM, RSN, UMEM, WAIT_ALL};
use base::OperationEmit;

pub struct OperationEmitter {
    next_op_id: usize,
    value_map: HashMap<String, String>,
}

impl OperationEmitter {
    pub fn new() -> Self {
        Self {
            next_op_id: 0,
            value_map: HashMap::new(),
        }
    }

    fn next_id(&mut self) -> String {
        base::generate_op_id(&mut self.next_op_id)
    }

    fn emit_descriptor(
        &mut self,
        emitter: &mut RustEmitter,
        descriptor: &OperationMetadata,
        args: HashMap<&'static str, String>,
    ) -> Result<String> {
        let op_id = self.next_id();
        descriptor.emit(emitter, &op_id, &args)?;
        Ok(op_id)
    }

    pub fn register_value(&mut self, mlir_name: String, rust_name: String) {
        self.value_map.insert(mlir_name, rust_name);
    }

    pub fn get_value(&self, mlir_name: &str) -> Option<&str> {
        self.value_map.get(mlir_name).map(|s| s.as_str())
    }

    pub fn emit_qmem(
        &mut self,
        emitter: &mut RustEmitter,
        query: &str,
        tier: Option<&str>,
    ) -> Result<String> {
        let mut args = HashMap::new();
        args.insert("query", query.to_string());
        if let Some(t) = tier {
            args.insert("memory_tier", t.to_string());
        }
        self.emit_descriptor(emitter, &QMEM, args)
    }

    pub fn emit_umem(
        &mut self,
        emitter: &mut RustEmitter,
        key: &str,
        value: &str,
        tier: Option<&str>,
    ) -> Result<String> {
        let mut args = HashMap::new();
        args.insert("key", key.to_string());
        args.insert("value", value.to_string());
        if let Some(t) = tier {
            args.insert("memory_tier", t.to_string());
        }
        self.emit_descriptor(emitter, &UMEM, args)
    }

    pub fn emit_rsn(
        &mut self,
        emitter: &mut RustEmitter,
        prompt: &str,
        model: Option<&str>,
        context: Option<&str>,
    ) -> Result<String> {
        let mut args = HashMap::new();
        args.insert("prompt", prompt.to_string());
        if let Some(m) = model {
            args.insert("model", m.to_string());
        }
        if let Some(c) = context {
            args.insert("context", c.to_string());
        }
        self.emit_descriptor(emitter, &RSN, args)
    }

    pub fn emit_inv(
        &mut self,
        emitter: &mut RustEmitter,
        capability: &str,
        parameters: Option<&str>,
    ) -> Result<String> {
        let mut args = HashMap::new();
        args.insert("capability", capability.to_string());
        if let Some(p) = parameters {
            args.insert("parameters", p.to_string());
        }
        self.emit_descriptor(emitter, &INV, args)
    }

    pub fn emit_wait_all(
        &mut self,
        emitter: &mut RustEmitter,
        tokens: &[String],
    ) -> Result<String> {
        let mut args = HashMap::new();
        args.insert("tokens", literals::format_array(tokens));
        self.emit_descriptor(emitter, &WAIT_ALL, args)
    }

    pub fn emit_merge(&mut self, emitter: &mut RustEmitter, tokens: &[String]) -> Result<String> {
        let mut args = HashMap::new();
        args.insert("tokens", literals::format_array(tokens));
        self.emit_descriptor(emitter, &MERGE, args)
    }

    pub fn emit_branch_on_value(
        &mut self,
        emitter: &mut RustEmitter,
        condition: &str,
        then_label: &str,
        else_label: &str,
    ) -> Result<String> {
        control_flow::emit_branch_on_value(emitter, condition, then_label, else_label)?;
        Ok(self.next_id())
    }

    pub fn emit_else_branch(&mut self, emitter: &mut RustEmitter, label: &str) -> Result<()> {
        control_flow::emit_else_branch(emitter, label)
    }

    pub fn emit_end_branch(&mut self, emitter: &mut RustEmitter) -> Result<()> {
        control_flow::emit_end_branch(emitter)
    }

    pub fn emit_loop_start(
        &mut self,
        emitter: &mut RustEmitter,
        max_iterations: &str,
    ) -> Result<String> {
        control_flow::emit_loop_start(emitter, max_iterations)?;
        Ok(self.next_id())
    }

    pub fn emit_loop_end(&mut self, emitter: &mut RustEmitter) -> Result<()> {
        control_flow::emit_loop_end(emitter)
    }

    pub fn emit_return(
        &mut self,
        emitter: &mut RustEmitter,
        value: Option<&str>,
    ) -> Result<String> {
        control_flow::emit_return(emitter, value)?;
        Ok(self.next_id())
    }

    pub fn emit_const_str(&mut self, value: &str) -> Result<String> {
        Ok(literals::escape_string(value))
    }

    pub fn reset(&mut self) {
        self.next_op_id = 0;
        self.value_map.clear();
    }
}

impl Default for OperationEmitter {
    fn default() -> Self {
        Self::new()
    }
}
