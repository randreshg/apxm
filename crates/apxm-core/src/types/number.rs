//! Numeric value representation for APXM.
//!
//! The `Number` enum represents numeric values that can be either integers or floating point
//! numbers.
//! This allows the system to handle both types uniformly while preserving type information.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum Number {
    // Integer value (64-bit signed integer).
    Integer(i64),
    // Floating-point value (64-bit float).
    Float(f64),
}

impl Number {
    /// Converts the number to a `f64` value.
    ///
    /// Integers are converted to floats without loss of precision.
    ///
    /// # Examples
    ///
    /// ```
    /// use apxm_core::types::Number;
    ///
    /// let int = Number::Integer(42);
    /// assert_eq!(int.as_f64(), 42.0);
    ///
    /// let float = Number::Float(3.14);
    /// assert_eq!(float.as_f64(), 3.14);
    /// ```
    pub fn as_f64(&self) -> f64 {
        match self {
            Number::Integer(i) => *i as f64,
            Number::Float(f) => *f,
        }
    }
}

impl From<i64> for Number {
    /// Creates a `Number::Integer` from an `i64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use apxm_core::types::Number;
    ///
    /// let num = Number::from(42i64);
    /// assert!(matches!(num, Number::Integer(42)));
    /// ```
    fn from(value: i64) -> Self {
        Number::Integer(value)
    }
}

impl From<i32> for Number {
    /// Creates a `Number::Integer` from a `i32`.
    fn from(value: i32) -> Self {
        Number::Integer(value as i64)
    }
}

impl From<u64> for Number {
    /// Creates a `Number::Integer` from `u64`.
    fn from(value: u64) -> Self {
        Number::Integer(value as i64)
    }
}

impl From<f64> for Number {
    /// Creates a `Number::Float` from `f64`.
    fn from(value: f64) -> Self {
        Number::Float(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_integer_creation() {
        let num = Number::Integer(42);
        assert!(matches!(num, Number::Integer(42)));
    }

    #[test]
    fn test_float_creation() {
        let num = Number::Float(3.14);
        assert!(matches!(num, Number::Float(f) if (f - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn test_as_f64_integer() {
        let num = Number::Integer(42);
        assert_eq!(num.as_f64(), 42.0);
    }

    #[test]
    fn test_as_f64_float() {
        let num = Number::Float(3.14);
        assert!((num.as_f64() - 3.14).abs() < f64::EPSILON);
    }

    #[test]
    fn test_from_i64() {
        let num = Number::from(42i64);
        assert!(matches!(num, Number::Integer(42)));
    }

    #[test]
    fn test_from_f64() {
        let num = Number::from(3.14);
        assert!(matches!(num, Number::Float(f) if (f - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn test_from_i32() {
        let num = Number::from(42i32);
        assert!(matches!(num, Number::Integer(42)));
    }

    #[test]
    fn test_serialization_integer() {
        let num = Number::Integer(42);
        let json = serde_json::to_string(&num).unwrap();
        assert!(json.contains("Integer"));
        assert!(json.contains("42"));
    }

    #[test]
    fn test_serialization_float() {
        let num = Number::Float(3.14);
        let json = serde_json::to_string(&num).unwrap();
        assert!(json.contains("Float"));
        assert!(json.contains("3.14"));
    }

    #[test]
    fn test_deserialization_integer() {
        let json = r#"{"type":"Integer","value":42}"#;
        let num: Number = serde_json::from_str(json).unwrap();
        assert!(matches!(num, Number::Integer(42)));
    }

    #[test]
    fn test_deserialization_float() {
        let json = r#"{"type":"Float","value":3.14}"#;
        let num: Number = serde_json::from_str(json).unwrap();
        assert!(matches!(num, Number::Float(f) if (f - 3.14).abs() < f64::EPSILON));
    }

    #[test]
    fn test_round_trip_serialization_integer() {
        let original = Number::Integer(42);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Number = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_round_trip_serialization_float() {
        let original = Number::Float(3.14);
        let json = serde_json::to_string(&original).unwrap();
        let deserialized: Number = serde_json::from_str(&json).unwrap();
        assert_eq!(original, deserialized);
    }

    #[test]
    fn test_clone() {
        let num = Number::Integer(42);
        let cloned = num.clone();
        assert_eq!(num, cloned);
    }

    #[test]
    fn test_debug() {
        let num = Number::Integer(42);
        let debug_str = format!("{:?}", num);
        assert!(debug_str.contains("Integer"));
        assert!(debug_str.contains("42"));
    }

    #[test]
    fn test_equality() {
        let num1 = Number::Integer(42);
        let num2 = Number::Integer(42);
        let num3 = Number::Integer(43);
        assert_eq!(num1, num2);
        assert_ne!(num1, num3);
    }
}
