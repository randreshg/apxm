//! Control flow operations for the Rust code generator.

use crate::codegen::emitter::RustEmitter;
use apxm_core::error::compiler::Result;

pub fn emit_branch_on_value(
    emitter: &mut RustEmitter,
    condition: &str,
    then_label: &str,
    else_label: &str,
) -> Result<()> {
    emitter.emit_comment(&format!(
        "Branch: if {} then {} else {}",
        condition, then_label, else_label
    ));
    emitter.emit_line(&format!("if {} {{", condition));
    emitter.indent();
    emitter.emit_comment(&format!("Then: {}", then_label));
    Ok(())
}

pub fn emit_else_branch(emitter: &mut RustEmitter, label: &str) -> Result<()> {
    emitter.dedent();
    emitter.emit_line("} else {");
    emitter.indent();
    emitter.emit_comment(&format!("Else: {}", label));
    Ok(())
}

pub fn emit_end_branch(emitter: &mut RustEmitter) -> Result<()> {
    emitter.dedent();
    emitter.emit_line("}");
    emitter.blank_line();
    Ok(())
}

pub fn emit_loop_start(emitter: &mut RustEmitter, max_iterations: &str) -> Result<()> {
    emitter.emit_line(&format!("for iteration in 0..{} {{", max_iterations));
    emitter.indent();
    Ok(())
}

pub fn emit_loop_end(emitter: &mut RustEmitter) -> Result<()> {
    emitter.dedent();
    emitter.emit_line("}");
    emitter.blank_line();
    Ok(())
}

pub fn emit_return(emitter: &mut RustEmitter, value: Option<&str>) -> Result<()> {
    if let Some(value) = value {
        emitter.emit_line(&format!("return Ok({});", value));
    } else {
        emitter.emit_line("return Ok(());");
    }
    Ok(())
}
