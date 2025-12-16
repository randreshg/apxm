//! Unified value representation for APXM.
//!
//! The `Value` enum provides a dynamic type system for the runtime, supporting
//! primitives (null, bool, number, string), structured data (array, object),
//! and system-specific types (token).

use crate::types::{Number, TokenId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// Runtime error type for value operations.
#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// A unified value type for the APXM runtime.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Value {
    /// Null value.
    Null,
    /// Boolean value.
    Bool(bool),
    /// Numeric value (integer or float).
    Number(Number),
    /// String value.
    String(String),
    /// Array of values.
    Array(Vec<Value>),
    /// Key-value map.
    Object(HashMap<String, Value>),
    /// Token reference (not serializable to standard JSON).
    Token(TokenId),
}

impl Value {
    /// Checks if the value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Attempts to get the value as a boolean.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Attempts to get the value as a number reference.
    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Value::Number(n) => Some(n),
            _ => None,
        }
    }

    /// Attempts to get the value as a string reference.
    pub fn as_string(&self) -> Option<&String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Attempts to get the value as a u64 (if it is a non-negative integer or float).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::Number(Number::Integer(i)) => {
                if *i >= 0 {
                    Some(*i as u64)
                } else {
                    None
                }
            }
            Value::Number(Number::Float(f)) => {
                if *f >= 0.0 {
                    Some(*f as u64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Attempts to get the value as an array reference.
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    /// Attempts to get the value as an object reference.
    pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    /// Attempts to get the value as a token ID.
    pub fn as_token_id(&self) -> Option<TokenId> {
        match self {
            Value::Token(id) => Some(*id),
            _ => None,
        }
    }

    /// Attempts to get the value as a boolean (alias for `as_boolean`).
    pub fn as_bool(&self) -> Option<bool> {
        self.as_boolean()
    }

    /// Converts the value to a JSON value.
    ///
    /// Token variants cannot be serialized.
    ///
    /// # Errors
    ///
    /// Returns an error if the value contains a Token variant.
    pub fn to_json(&self) -> Result<serde_json::Value, RuntimeError> {
        match self {
            Value::Null => Ok(serde_json::Value::Null),
            Value::Bool(b) => Ok(serde_json::Value::Bool(*b)),
            Value::Number(n) => {
                let json_num = match n {
                    Number::Integer(i) => serde_json::Number::from(*i).into(),
                    Number::Float(f) => serde_json::Number::from_f64(*f)
                        .ok_or_else(|| {
                            RuntimeError::Serialization("Invalid float value".to_string())
                        })?
                        .into(),
                };
                Ok(json_num)
            }
            Value::String(s) => Ok(serde_json::Value::String(s.clone())),
            Value::Array(a) => {
                let json_array: Result<Vec<_>, _> = a.iter().map(|v| v.to_json()).collect();
                Ok(serde_json::Value::Array(json_array?))
            }
            Value::Object(o) => {
                let mut map = serde_json::Map::new();
                for (k, v) in o {
                    map.insert(k.clone(), v.to_json()?);
                }
                Ok(serde_json::Value::Object(map))
            }
            Value::Token(_) => Err(RuntimeError::Serialization(
                "Cannot serialize Token to JSON".to_string(),
            )),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{b}"),
            Value::Number(n) => write!(f, "{n}"),
            Value::String(s) => write!(f, "\"{s}\""),
            Value::Array(arr) => {
                write!(f, "[")?;
                let mut first = true;
                for v in arr {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "{v}")?;
                    first = false;
                }
                write!(f, "]")
            }
            Value::Object(obj) => {
                write!(f, "{{")?;
                let mut first = true;
                for (k, v) in obj {
                    if !first {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{k}\": {v}")?;
                    first = false;
                }
                write!(f, "}}")
            }
            Value::Token(id) => write!(f, "<token:{}>", id),
        }
    }
}

impl From<bool> for Value {
    fn from(value: bool) -> Self {
        Value::Bool(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        Value::Number(Number::Integer(value))
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Number(Number::Float(value))
    }
}

impl From<String> for Value {
    fn from(value: String) -> Self {
        Value::String(value)
    }
}

impl From<&str> for Value {
    fn from(value: &str) -> Self {
        Value::String(value.to_string())
    }
}

impl From<Vec<Value>> for Value {
    fn from(value: Vec<Value>) -> Self {
        Value::Array(value)
    }
}

impl From<HashMap<String, Value>> for Value {
    fn from(value: HashMap<String, Value>) -> Self {
        Value::Object(value)
    }
}

impl From<Number> for Value {
    fn from(value: Number) -> Self {
        Value::Number(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null() {
        let v = Value::Null;
        assert!(v.is_null());
        assert_eq!(format!("{}", v), "null");
    }

    #[test]
    fn test_bool() {
        let v = Value::Bool(true);
        assert_eq!(v.as_boolean(), Some(true));
        assert_eq!(v.as_bool(), Some(true));
        assert_eq!(format!("{}", v), "true");
    }

    #[test]
    fn test_number() {
        let v = Value::Number(Number::Integer(42));
        assert_eq!(v.as_number(), Some(&Number::Integer(42)));
        assert_eq!(v.as_u64(), Some(42));
        assert_eq!(format!("{}", v), "42");
    }

    #[test]
    fn test_string() {
        let v = Value::String("hello".to_string());
        assert_eq!(v.as_string(), Some(&"hello".to_string()));
        assert_eq!(format!("{}", v), "\"hello\"");
    }

    #[test]
    fn test_array() {
        let v = Value::Array(vec![
            Value::Number(Number::Integer(1)),
            Value::Number(Number::Integer(2)),
        ]);
        assert!(v.as_array().is_some());
        assert_eq!(format!("{}", v), "[1, 2]");
    }

    #[test]
    fn test_object() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), Value::String("value".to_string()));
        let v = Value::Object(map);
        assert!(v.as_object().is_some());
        assert_eq!(format!("{}", v), "{\"key\": \"value\"}");
    }

    #[test]
    fn test_token() {
        let id = 1;
        let v = Value::Token(id);
        assert_eq!(v.as_token_id(), Some(id));
        assert!(format!("{}", v).starts_with("<token:"));
    }

    #[test]
    fn test_from_bool() {
        let v: Value = true.into();
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn test_from_i64() {
        let v: Value = 42i64.into();
        assert_eq!(v, Value::Number(Number::Integer(42)));
    }

    #[test]
    fn test_from_f64() {
        let v: Value = std::f64::consts::PI.into();
        match v {
            Value::Number(Number::Float(f)) => {
                assert!((f - std::f64::consts::PI).abs() < f64::EPSILON)
            }
            _ => panic!("Expected float"),
        }
    }

    #[test]
    fn test_from_string() {
        let v: Value = "test".to_string().into();
        assert_eq!(v, Value::String("test".to_string()));
    }

    #[test]
    fn test_from_str() {
        let v: Value = "test".into();
        assert_eq!(v, Value::String("test".to_string()));
    }

    #[test]
    fn test_to_json_null() {
        let v = Value::Null;
        assert_eq!(v.to_json().unwrap(), serde_json::Value::Null);
    }

    #[test]
    fn test_to_json_bool() {
        let v = Value::Bool(true);
        assert_eq!(v.to_json().unwrap(), serde_json::Value::Bool(true));
    }

    #[test]
    fn test_to_json_number() {
        let v = Value::Number(Number::Integer(42));
        assert_eq!(v.to_json().unwrap(), serde_json::json!(42));
    }

    #[test]
    fn test_to_json_string() {
        let v = Value::String("test".to_string());
        assert_eq!(
            v.to_json().unwrap(),
            serde_json::Value::String("test".to_string())
        );
    }

    #[test]
    fn test_to_json_array() {
        let v = Value::Array(vec![Value::Number(Number::Integer(1))]);
        assert_eq!(v.to_json().unwrap(), serde_json::json!([1]));
    }

    #[test]
    fn test_to_json_object() {
        let mut map = HashMap::new();
        map.insert("k".to_string(), Value::Number(Number::Integer(1)));
        let v = Value::Object(map);
        assert_eq!(v.to_json().unwrap(), serde_json::json!({"k": 1}));
    }

    #[test]
    fn test_to_json_token_error() {
        let v = Value::Token(1);
        assert!(v.to_json().is_err());
    }

    #[test]
    fn test_serialization() {
        let v = Value::Number(Number::Integer(42));
        let json = serde_json::to_string(&v).unwrap();
        assert!(json.contains("42"));
    }

    #[test]
    fn test_deserialization() {
        let json = "42";
        let v: Value = serde_json::from_str(json).unwrap();
        assert_eq!(v, Value::Number(Number::Integer(42)));
    }

    #[test]
    fn test_round_trip_serialization() {
        let v = Value::Array(vec![
            Value::Null,
            Value::Bool(true),
            Value::Number(Number::Integer(42)),
            Value::String("test".to_string()),
        ]);
        let json = serde_json::to_string(&v).unwrap();
        let v2: Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v, v2);
    }
}
