pub mod bash;
pub mod read;
pub mod web_search;
pub mod write;

pub use bash::{BashCapability, BashConfig};
pub use read::{ReadCapability, ReadConfig};
pub use web_search::{SearchDepth, SearchWebCapability, SearchWebConfig};
pub use write::{WriteCapability, WriteConfig};

use apxm_core::error::RuntimeError;
use apxm_runtime::CapabilitySystem;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for APxM standard tools.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[derive(Default)]
pub struct ToolsConfig {
    #[serde(default)]
    pub bash: BashConfig,
    #[serde(default)]
    pub read: ReadConfig,
    #[serde(default)]
    pub write: WriteConfig,
    #[serde(default)]
    pub search_web: SearchWebConfig,
}


/// Register the standard APxM capabilities with the runtime capability system.
pub fn register_standard_tools(
    capability_system: &CapabilitySystem,
    config: &ToolsConfig,
) -> Result<(), RuntimeError> {
    if config.bash.enabled {
        capability_system.register(Arc::new(BashCapability::with_config(config.bash.clone())))?;
    }
    if config.read.enabled {
        capability_system.register(Arc::new(ReadCapability::with_config(config.read.clone())))?;
    }
    if config.write.enabled {
        capability_system.register(Arc::new(WriteCapability::with_config(config.write.clone())))?;
    }
    if config.search_web.enabled {
        capability_system.register(Arc::new(SearchWebCapability::with_config(
            config.search_web.clone(),
        )))?;
    }
    Ok(())
}
