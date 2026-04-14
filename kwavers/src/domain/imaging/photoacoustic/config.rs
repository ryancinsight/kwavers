use crate::domain::medium::optical_map::OpticalPropertyMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpticalModel {
    Diffusion,
    MonteCarlo,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IlluminationGeometry {
    PencilBeam {
        origin_m: [f64; 3],
        direction: [f64; 3],
    },
    IsotropicPoint {
        origin_m: [f64; 3],
    },
}

#[derive(Debug, Clone)]
pub struct MonteCarloModelConfig {
    pub photon_count: usize,
    pub max_steps: usize,
    pub russian_roulette_threshold: f64,
    pub russian_roulette_survival: f64,
    pub boundary_reflection: bool,
}

impl Default for MonteCarloModelConfig {
    fn default() -> Self {
        Self {
            photon_count: 100_000,
            max_steps: 10_000,
            russian_roulette_threshold: 1e-3,
            russian_roulette_survival: 0.1,
            boundary_reflection: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ThermoelasticProperties {
    pub density_kg_m3: f64,
    pub sound_speed_m_s: f64,
    pub specific_heat_j_kgk: f64,
    pub thermal_conductivity_w_mk: f64,
}

impl ThermoelasticProperties {
    #[must_use]
    pub fn thermal_diffusivity_m2_s(&self) -> f64 {
        self.thermal_conductivity_w_mk / (self.density_kg_m3 * self.specific_heat_j_kgk)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PhotoacousticAcousticConfig {
    pub speed_of_sound_m_s: f64,
    pub cfl_factor: f64,
    pub num_time_steps: usize,
    pub snapshot_interval: usize,
}

#[derive(Debug, Clone)]
pub struct PhotoacousticReconstructionConfig {
    pub grid_size: [usize; 3],
    pub sound_speed_m_s: f64,
    pub sensor_positions_m: Vec<[f64; 3]>,
}

#[derive(Debug, Clone)]
pub struct PhotoacousticSolverConfig {
    pub optical_model: OpticalModel,
    pub illumination: IlluminationGeometry,
    pub pulse_duration_s: f64,
    pub incident_fluence_j_m2: f64,
    pub acoustic: PhotoacousticAcousticConfig,
    pub thermoelastic: ThermoelasticProperties,
    pub monte_carlo: MonteCarloModelConfig,
}

#[derive(Debug, Clone)]
pub struct PhotoacousticExecutionConfig {
    pub solver: PhotoacousticSolverConfig,
    pub optical_map: OpticalPropertyMap,
    pub reconstruction: PhotoacousticReconstructionConfig,
    pub wavelength_nm: f64,
}
