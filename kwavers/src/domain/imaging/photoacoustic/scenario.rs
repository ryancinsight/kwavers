use super::config::{
    PhotoacousticExecutionConfig, PhotoacousticReconstructionConfig, PhotoacousticSolverConfig,
};
use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::optical_map::OpticalPropertyMap;

#[derive(Debug, Clone)]
pub struct PhotoacousticScenario {
    pub grid: Grid,
    pub wavelength_nm: f64,
    pub optical_map: OpticalPropertyMap,
    pub config: PhotoacousticSolverConfig,
    pub sensor_positions_m: Vec<[f64; 3]>,
}

impl PhotoacousticScenario {
    pub fn new(
        grid: Grid,
        wavelength_nm: f64,
        optical_map: OpticalPropertyMap,
        config: PhotoacousticSolverConfig,
        sensor_positions_m: Vec<[f64; 3]>,
    ) -> KwaversResult<Self> {
        if sensor_positions_m.is_empty() {
            return Err(KwaversError::InvalidInput(
                "photoacoustic scenario requires at least one sensor".to_string(),
            ));
        }
        if config.incident_fluence_j_m2 <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "incident fluence must be positive".to_string(),
            ));
        }
        if config.pulse_duration_s <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "pulse duration must be positive".to_string(),
            ));
        }
        Ok(Self {
            grid,
            wavelength_nm,
            optical_map,
            config,
            sensor_positions_m,
        })
    }

    #[must_use]
    pub fn execution_config(&self) -> PhotoacousticExecutionConfig {
        PhotoacousticExecutionConfig {
            solver: self.config.clone(),
            optical_map: self.optical_map.clone(),
            reconstruction: PhotoacousticReconstructionConfig {
                grid_size: [self.grid.nx, self.grid.ny, self.grid.nz],
                sound_speed_m_s: self.config.acoustic.speed_of_sound_m_s,
                sensor_positions_m: self.sensor_positions_m.clone(),
            },
            wavelength_nm: self.wavelength_nm,
        }
    }
}
