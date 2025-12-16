//! Type-safe identifier newtypes for domain entities.

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! define_id {
    ($name:ident, $inner:ty, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub $inner);

        impl $name {
            pub const fn new(id: $inner) -> Self {
                Self(id)
            }

            pub const fn get(&self) -> $inner {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<$inner> for $name {
            fn from(id: $inner) -> Self {
                Self::new(id)
            }
        }

        impl From<$name> for $inner {
            fn from(id: $name) -> Self {
                id.0
            }
        }
    };
}

macro_rules! define_string_id {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }

            pub fn generate() -> Self {
                let timestamp = uuid::Timestamp::now(uuid::NoContext);
                Self(uuid::Uuid::new_v7(timestamp).to_string())
            }

            pub fn as_str(&self) -> &str {
                &self.0
            }

            pub fn into_string(self) -> String {
                self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }

        impl From<String> for $name {
            fn from(id: String) -> Self {
                Self::new(id)
            }
        }

        impl From<&str> for $name {
            fn from(id: &str) -> Self {
                Self::new(id)
            }
        }

        impl From<$name> for String {
            fn from(id: $name) -> Self {
                id.0
            }
        }

        impl AsRef<str> for $name {
            fn as_ref(&self) -> &str {
                &self.0
            }
        }
    };
}

define_id!(NodeIdType, u64, "Unique identifier for a DAG node");
define_id!(TokenIdType, u64, "Unique identifier for a dataflow token");
define_id!(OpIdType, u64, "Unique identifier for an operation");
define_id!(GoalIdType, u64, "Unique identifier for an AAM goal");

define_string_id!(ExecutionId, "Unique identifier for an execution context");
define_string_id!(SessionId, "Unique identifier for a chat session");
define_string_id!(TraceId, "Unique identifier for an execution trace");
define_string_id!(CapabilityName, "Unique identifier for a capability");
define_string_id!(MessageId, "Unique identifier for a chat message");
define_string_id!(CheckpointId, "Unique identifier for a session checkpoint");
