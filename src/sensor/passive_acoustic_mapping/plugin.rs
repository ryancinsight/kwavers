//! PAM plugin for physics simulation

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::medium::Medium;
use crate::physics::field_mapping::UnifiedFieldType;
use crate::physics::plugin::{PhysicsPlugin, PluginContext, PluginMetadata, PluginState};
use ndarray::{Array3, Array4};
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
                id: "pam".to_string(),
                name: "Passive Acoustic Mapping".to_string(),
                version: "1.0.0".to_string(),
                description: "Maps cavitation fields using passive acoustic emissions".to_string(),
                author: "Kwavers Team".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Created,
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

impl crate::physics::plugin::Plugin for PAMPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state.clone()
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure] // Use pressure field as sensor data
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![] // PAM doesn't modify simulation fields, it extracts information
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        dt: f64,
        _t: f64,
        _context: &PluginContext,
    ) -> KwaversResult<()> {
        self.state = PluginState::Running;

        // Extract pressure field as sensor data
        let pressure_idx = UnifiedFieldType::Pressure.index();
        let shape = fields.shape();

        // Create a view of the pressure field
        let mut sensor_data = Array3::zeros((shape[1], shape[2], shape[3]));
        for ix in 0..shape[1] {
            for iy in 0..shape[2] {
                for iz in 0..shape[3] {
                    sensor_data[[ix, iy, iz]] = fields[[pressure_idx, ix, iy, iz]];
                }
            }
        }

        // Process with PAM
        let sample_rate = dt.recip();
        let cavitation_map = self.mapper.process(&sensor_data, sample_rate)?;

        // Store cavitation intensity in the chemical concentration field as a proxy
        // This allows visualization and analysis of cavitation activity
        let chem_idx = UnifiedFieldType::ChemicalConcentration.index();
        for ix in 0..cavitation_map.shape()[0] {
            for iy in 0..cavitation_map.shape()[1] {
                for iz in 0..cavitation_map.shape()[2] {
                    // Store the first frequency band as cavitation intensity
                    if iz == 0 {
                        fields[[chem_idx, ix, iy, 0]] = cavitation_map[[ix, iy, iz]];
                    }
                }
            }
        }

        Ok(())
    }
}
