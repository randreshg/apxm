//! Literals codegen for:
//! - String
//! - Integer
//! - Float
//! - Boolean
//! - Array

pub use apxm_core::types::operation_definitions::CONST_STR;

pub fn escape_string(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

pub fn format_integer(n: i64) -> String {
    n.to_string()
}

pub fn format_float(f: f64) -> String {
    format!("{:.6}", f)
}

pub fn format_boolean(b: bool) -> String {
    b.to_string()
}

pub fn format_array(elements: &[String]) -> String {
    format!("vec![{}]", elements.join(", "))
}
