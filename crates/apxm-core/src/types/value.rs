//! Unified value representation for APXM.
//!
//! The `Value` enum provides a unified representation for all values in the APXM system,
//! supporting primitive types, complex structures, and special values like tokens.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::{Number, TokenId};

/// Represents a unified value that can be various types.
///
/// This enum allows APXM to handle all value types uniformly while preserving
/// type information for type-safe operations.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Value {
    // Null value (no data).
    Null,
    // Boolean value.
    Bool(bool),
    // Numeric value (integer or float).
    Number(Number),
    // String value.
    String(String),
    // Array of values.
    Array(Vec<Value>),
    // Object/map of string keys to values.
    Object(HashMap<String, Value>),
    // Token reference (by ID).
    Token(TokenId),
}

impl Value {
    // Checks if the value is null.
    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    // Attempts to get the value as a boolean.
    pub fn as_boolean(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    // Attempts to get the value as a number referece.
    pub fn as_number(&self) -> Option<&Number> {
        match self {
            Value::Number(n) => Some(n),
            _ => None,
        }
    }

    // Attempts to get the value as a string reference.
    pub fn as_string(&self) -> Option<&String> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    // Attempts to get the value as an array reference.
    pub fn as_array(&self) -> Option<&Vec<Value>> {
        match self {
            Value::Array(a) => Some(a),
            _ => None,
        }
    }

    // Attempts to get the value as an object reference.
    pub fn as_object(&self) -> Option<&HashMap<String, Value>> {
        match self {
            Value::Object(o) => Some(o),
            _ => None,
        }
    }

    // Attempts to get the value as a token ID.
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
    pub fn to_json(&self) -> Result<serde_json::Value, String> {
        match self {
            Value::Null => Ok(serde_json::Value::Null),
            Value::Bool(b) => Ok(serde_json::Value::Bool(*b)),
            Value::Number(n) => {
                let json_num = match n {
                    Number::Integer(i) => serde_json::Number::from(*i).into(),
                    Number::Float(f) => serde_json::Number::from_f64(*f)
                        .ok_or_else(|| "Invalid float value".to_string())?
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
            Value::Token(_) => Err("Cannot serialize Token to JSON".to_string()),
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
        Value::Number(Number::from(value))
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Number(Number::from(value))
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
        let val = Value::Null;
        assert!(val.is_null());
        assert_eq!(val.as_bool(), None);
    }

    #[test]
    fn test_bool() {
        let val = Value::Bool(true);
        assert!(!val.is_null());
        assert_eq!(val.as_bool(), Some(true));
        assert_eq!(Value::Bool(false).as_bool(), Some(false));
    }

    #[test]
    fn test_number() {
        let val = Value::Number(Number::Integer(42));
        assert_eq!(val.as_number(), Some(&Number::Integer(42)));
    }

    #[test]
    fn test_string() {
        let val = Value::String("hello".to_string());
        assert_eq!(val.as_string(), Some(&"hello".to_string()));
    }

    #[test]
    fn test_array() {
        let val = Value::Array(vec![Value::Bool(true), Value::Number(Number::Integer(42))]);
        let arr = val
            .as_array()
            .expect("value should provide array reference");
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_object() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), Value::String("value".to_string()));
        let val = Value::Object(map);
        let obj = val
            .as_object()
            .expect("value should provide object reference");
        assert_eq!(obj.len(), 1);
    }

    #[test]
    fn test_token() {
        let val = Value::Token(123);
        assert_eq!(val.as_token_id(), Some(123));
    }

    #[test]
    fn test_from_bool() {
        let val = Value::from(true);
        assert!(matches!(val, Value::Bool(true)));
    }

    #[test]
    fn test_from_i64() {
        let val = Value::from(42i64);
        assert!(matches!(val, Value::Number(Number::Integer(42))));
    }

    #[test]
    fn test_from_f64() {
        let val = Value::from(3.14);
        assert!(matches!(val, Value::Number(Number::Float(f)) if (f - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn test_from_string() {
        let val = Value::from("hello".to_string());
        assert_eq!(val.as_string(), Some(&"hello".to_string()));
    }

    #[test]
    fn test_from_str() {
        let val = Value::from("hello");
        assert_eq!(val.as_string(), Some(&"hello".to_string()));
    }

    #[test]
    fn test_to_json_null() {
        let val = Value::Null;
        let json = val.to_json().expect("serialize null value");
        assert!(json.is_null());
    }

    #[test]
    fn test_to_json_bool() {
        let val = Value::Bool(true);
        let json = val.to_json().expect("serialize bool value");
        assert_eq!(json.as_bool(), Some(true));
    }

    #[test]
    fn test_to_json_number() {
        let val = Value::Number(Number::Integer(42));
        let json = val.to_json().expect("serialize number value");
        assert_eq!(json.as_i64(), Some(42));
    }

    #[test]
    fn test_to_json_string() {
        let val = Value::String("hello".to_string());
        let json = val.to_json().expect("serialize string value");
        assert_eq!(json.as_str(), Some("hello"));
    }

    #[test]
    fn test_to_json_array() {
        let val = Value::Array(vec![Value::Bool(true), Value::Number(Number::Integer(42))]);
        let json = val.to_json().expect("serialize array value");
        assert!(json.is_array());
        let arr = json.as_array().expect("array JSON expected");
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn test_to_json_object() {
        let mut map = HashMap::new();
        map.insert("key".to_string(), Value::String("value".to_string()));
        let val = Value::Object(map);
        let json = val.to_json().expect("serialize object value");
        assert!(json.is_object());
    }

    #[test]
    fn test_to_json_token_error() {
        let val = Value::Token(123);
        assert!(val.to_json().is_err());
    }

    #[test]
    fn test_serialization() {
        let val = Value::Array(vec![
            Value::Bool(true),
            Value::Number(Number::Integer(42)),
            Value::String("hello".to_string()),
        ]);
        let json = serde_json::to_string(&val).expect("serialize value via serde");
        assert!(json.contains("Array"));
    }

    #[test]
    fn test_deserialization() {
        let json = r#"{"type":"Bool","value":true}"#;
        let val: Value = serde_json::from_str(json).expect("deserialize value via serde");
        assert_eq!(val.as_boolean(), Some(true));
    }

    #[test]
    fn test_round_trip_serialization() {
        let original = Value::Array(vec![Value::Bool(true), Value::Number(Number::Integer(42))]);
        let json =
            serde_json::to_string(&original).expect("serialize value for round-trip testing");
        let deserialized: Value =
            serde_json::from_str(&json).expect("deserialize value for round-trip testing");
        assert_eq!(original, deserialized);
    }
}
