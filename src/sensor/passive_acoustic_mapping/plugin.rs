//! PAM plugin for physics simulation

use crate::error::KwaversResult;
use crate::physics::plugin::{PhysicsPlugin, PluginContext, PluginMetadata, PluginState};
use crate::physics::field_mapping::UnifiedFieldType;
use ndarray::Array3;
use std::fmt;

/// PAM plugin for passive acoustic mapping
pub struct PAMPlugin {
    mapper: super::PassiveAcousticMapper,
    metadata: PluginMetadata,
    state: PluginState,
}

impl PAMPlugin {
    /// Create a new PAM plugin
    pub fn new(config: super::PAMConfig, geometry: super::ArrayGeometry) -> KwaversResult<Self> {
        let mapper = super::PassiveAcousticMapper::new(config, geometry)?;
        
        Ok(Self {
            mapper,
            metadata: PluginMetadata {
                name: "Passive Acoustic Mapping".to_string(),
                version: "1.0.0".to_string(),
                description: "Maps cavitation fields using passive acoustic emissions".to_string(),
                author: "Kwavers Team".to_string(),
                dependencies: vec![],
            },
            state: PluginState::Idle,
        })
    }
}

impl fmt::Debug for PAMPlugin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PAMPlugin")
            .field("metadata", &self.metadata)
            .field("state", &self.state)
            .finish()
    }
}

impl PhysicsPlugin for PAMPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn state(&self) -> PluginState {
        self.state.clone()
    }
    
    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Custom("sensor_data".to_string())]
    }
    
    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Custom("cavitation_map".to_string())]
    }
}