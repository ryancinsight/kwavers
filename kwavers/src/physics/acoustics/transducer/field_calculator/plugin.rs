//! TransducerFieldCalculatorPlugin struct and Plugin trait impl

use super::geometry::TransducerGeometry;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::plugin::{PluginMetadata, PluginState};
use ndarray::Array3;
use std::collections::HashMap;

/// Multi-Element Transducer Field Calculator Plugin
#[derive(Debug)]
pub struct TransducerFieldCalculatorPlugin {
    pub(crate) metadata: PluginMetadata,
    pub(crate) state: PluginState,
    /// Transducer geometry definitions
    pub(crate) transducer_geometries: Vec<TransducerGeometry>,
    /// Spatial impulse response cache
    pub(crate) sir_cache: HashMap<String, Array3<f64>>,
}

impl TransducerFieldCalculatorPlugin {
    /// Create new FOCUS-compatible transducer field calculator
    #[must_use]
    pub fn new(transducer_geometries: Vec<TransducerGeometry>) -> Self {
        Self {
            metadata: PluginMetadata {
                id: "focus_transducer_calculator".to_string(),
                name: "FOCUS Transducer Field Calculator".to_string(),
                version: "1.0.0".to_string(),
                author: "Kwavers Team".to_string(),
                description: "Multi-element transducer field calculation with FOCUS compatibility"
                    .to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            transducer_geometries,
            sir_cache: HashMap::new(),
        }
    }
}

impl crate::domain::plugin::Plugin for TransducerFieldCalculatorPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<crate::domain::field::mapping::UnifiedFieldType> {
        vec![]
    }

    fn provided_fields(&self) -> Vec<crate::domain::field::mapping::UnifiedFieldType> {
        vec![crate::domain::field::mapping::UnifiedFieldType::Pressure]
    }

    fn update(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        _dt: f64,
        t: f64,
        _context: &mut crate::domain::plugin::PluginContext<'_>,
    ) -> KwaversResult<()> {
        use crate::domain::field::mapping::UnifiedFieldType;

        let frequency = 1e6;

        let pressure_field = self.calculate_pressure_field(frequency, grid, medium)?;

        let time_factor = (2.0 * std::f64::consts::PI * frequency * t).sin();
        let modulated_field = pressure_field.mapv(|p| p * time_factor);

        let mut pressure_slice =
            fields.index_axis_mut(ndarray::Axis(0), UnifiedFieldType::Pressure.index());
        pressure_slice.assign(&modulated_field);

        Ok(())
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        self.state = PluginState::Initialized;
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        self.state = PluginState::Finalized;
        self.sir_cache.clear();
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        self.sir_cache.clear();
        self.state = PluginState::Created;
        Ok(())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
