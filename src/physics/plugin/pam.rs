use crate::core::error::KwaversResult;
use crate::domain::field::mapping::UnifiedFieldType;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::sensor::passive_acoustic_mapping::{
    ArrayGeometry, PAMConfig, PassiveAcousticMapper,
};
use crate::physics::plugin::{PluginContext, PluginMetadata, PluginState};
use ndarray::{Array3, Array4};

#[derive(Debug)]
pub struct PAMPlugin {
    mapper: PassiveAcousticMapper,
    metadata: PluginMetadata,
    state: PluginState,
}

impl PAMPlugin {
    pub fn new(config: PAMConfig, geometry: ArrayGeometry) -> KwaversResult<Self> {
        let mapper = PassiveAcousticMapper::new(config, geometry)?;

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

impl crate::physics::plugin::Plugin for PAMPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![]
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
        _context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        self.state = PluginState::Running;

        let pressure_idx = UnifiedFieldType::Pressure.index();
        let shape = fields.shape();

        let mut sensor_data = Array3::zeros((shape[1], shape[2], shape[3]));
        for ix in 0..shape[1] {
            for iy in 0..shape[2] {
                for iz in 0..shape[3] {
                    sensor_data[[ix, iy, iz]] = fields[[pressure_idx, ix, iy, iz]];
                }
            }
        }

        let sample_rate = dt.recip();
        let cavitation_map = self.mapper.process(&sensor_data, sample_rate)?;

        let chem_idx = UnifiedFieldType::ChemicalConcentration.index();
        for ix in 0..cavitation_map.shape()[0] {
            for iy in 0..cavitation_map.shape()[1] {
                for iz in 0..cavitation_map.shape()[2] {
                    if iz == 0 {
                        fields[[chem_idx, ix, iy, 0]] = cavitation_map[[ix, iy, iz]];
                    }
                }
            }
        }

        Ok(())
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
