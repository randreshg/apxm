//! AIS Value Types
//!
//! Contains the unified value representation for AIS operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

/// Type alias for token identifiers.
pub type TokenId = u64;

/// Type alias for node identifiers.
pub type NodeId = u64;

/// Runtime error type for value operations.
#[derive(Error, Debug)]
pub enum ValueError {
    #[error("Serialization error: {0}")]
    Serialization(String),
}

/// Numeric value type.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Number {
    /// Integer value.
    Integer(i64),
    /// Floating-point value.
    Float(f64),
}

impl fmt::Display for Number {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Number::Integer(i) => write!(f, "{}", i),
            Number::Float(fl) => write!(f, "{}", fl),
        }
    }
}

impl Number {
    /// Gets the value as an f64.
    pub fn as_f64(&self) -> f64 {
        match self {
            Number::Integer(i) => *i as f64,
            Number::Float(f) => *f,
        }
    }

    /// Gets the value as an i64, if it is an integer.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Number::Integer(i) => Some(*i),
            Number::Float(_) => None,
        }
    }
}

impl From<i64> for Number {
    fn from(v: i64) -> Self {
        Number::Integer(v)
    }
}

impl From<f64> for Number {
    fn from(v: f64) -> Self {
        Number::Float(v)
    }
}

/// A unified value type for AIS operations.
///
/// Supports primitives (null, bool, number, string), structured data (array, object),
/// and system-specific types (token reference).
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
    /// Token reference (dataflow token ID).
    Token(TokenId),
}

impl Value {
    /// Checks if the value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// Attempts to get the value as a boolean.
    pub fn as_bool(&self) -> Option<bool> {
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
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    /// Attempts to get the value as an i64.
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::Number(Number::Integer(i)) => Some(*i),
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

    /// Attempts to get the value as a boolean (alias for `as_bool`).
    pub fn as_boolean(&self) -> Option<bool> {
        self.as_bool()
    }

    /// Attempts to get the value as a String reference.
    pub fn as_string(&self) -> Option<&String> {
        match self {
            Value::String(s) => Some(s),
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

    /// Converts the value to a JSON value.
    pub fn to_json(&self) -> Result<serde_json::Value, ValueError> {
        match self {
            Value::Null => Ok(serde_json::Value::Null),
            Value::Bool(b) => Ok(serde_json::Value::Bool(*b)),
            Value::Number(n) => match n {
                Number::Integer(i) => Ok(serde_json::json!(*i)),
                Number::Float(f) => serde_json::Number::from_f64(*f)
                    .map(serde_json::Value::Number)
                    .ok_or_else(|| ValueError::Serialization("Invalid float".to_string())),
            },
            Value::String(s) => Ok(serde_json::Value::String(s.clone())),
            Value::Array(a) => {
                let arr: Result<Vec<_>, _> = a.iter().map(|v| v.to_json()).collect();
                Ok(serde_json::Value::Array(arr?))
            }
            Value::Object(o) => {
                let mut map = serde_json::Map::new();
                for (k, v) in o {
                    map.insert(k.clone(), v.to_json()?);
                }
                Ok(serde_json::Value::Object(map))
            }
            Value::Token(_) => Err(ValueError::Serialization(
                "Cannot serialize Token to JSON".to_string(),
            )),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Null => write!(f, "null"),
            Value::Bool(b) => write!(f, "{}", b),
            Value::Number(n) => write!(f, "{}", n),
            Value::String(s) => write!(f, "\"{}\"", s),
            Value::Array(arr) => {
                write!(f, "[")?;
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", v)?;
                }
                write!(f, "]")
            }
            Value::Object(obj) => {
                write!(f, "{{")?;
                for (i, (k, v)) in obj.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "\"{}\": {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Token(id) => write!(f, "<token:{}>", id),
        }
    }
}

// Convenient From implementations
impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}

impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::Number(Number::Integer(v))
    }
}

impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::Number(Number::Float(v))
    }
}

impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::String(v)
    }
}

impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::String(v.to_string())
    }
}

impl From<Vec<Value>> for Value {
    fn from(v: Vec<Value>) -> Self {
        Value::Array(v)
    }
}

impl From<HashMap<String, Value>> for Value {
    fn from(v: HashMap<String, Value>) -> Self {
        Value::Object(v)
    }
}

impl From<Number> for Value {
    fn from(v: Number) -> Self {
        Value::Number(v)
    }
}

impl TryFrom<serde_json::Value> for Value {
    type Error = ValueError;

    fn try_from(value: serde_json::Value) -> Result<Self, Self::Error> {
        match value {
            serde_json::Value::Null => Ok(Value::Null),
            serde_json::Value::Bool(b) => Ok(Value::Bool(b)),
            serde_json::Value::Number(num) => {
                if let Some(i) = num.as_i64() {
                    return Ok(Value::Number(Number::Integer(i)));
                }

                if let Some(u) = num.as_u64() {
                    if u <= i64::MAX as u64 {
                        return Ok(Value::Number(Number::Integer(u as i64)));
                    }
                    return Ok(Value::Number(Number::Float(u as f64)));
                }

                if let Some(f) = num.as_f64() {
                    return Ok(Value::Number(Number::Float(f)));
                }

                Err(ValueError::Serialization(
                    "Unsupported numeric value".to_string(),
                ))
            }
            serde_json::Value::String(s) => Ok(Value::String(s)),
            serde_json::Value::Array(arr) => {
                let converted = arr
                    .into_iter()
                    .map(Value::try_from)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Value::Array(converted))
            }
            serde_json::Value::Object(obj) => {
                let mut map = HashMap::with_capacity(obj.len());
                for (key, value) in obj {
                    map.insert(key, Value::try_from(value)?);
                }
                Ok(Value::Object(map))
            }
        }
    }
}
