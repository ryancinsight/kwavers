// Canonical photoacoustic domain types.

mod config;
mod materials;
mod pressure_series;
mod references;
mod scenario;
mod state;
mod types;
mod validation;

pub use config::{
    IlluminationGeometry, MonteCarloModelConfig, OpticalModel, PhotoacousticAcousticConfig,
    PhotoacousticExecutionConfig, PhotoacousticReconstructionConfig, PhotoacousticSolverConfig,
    ThermoelasticProperties,
};
pub use materials::{PhotoacousticMaterialLibrary, SpectralSample};
pub use pressure_series::PressureFieldSeries;
pub use references::PHOTOACOUSTIC_PRIMARY_REFERENCES;
pub use scenario::PhotoacousticScenario;
pub use state::PhotoacousticState;
pub use types::{
    InitialPressure, PhotoacousticOpticalProperties, PhotoacousticParameters, PhotoacousticResult,
    PhotoacousticSignalSet, PhotoacousticSimulation, PhotoacousticValidationReport, WavelengthBand,
};
pub use validation::{AcceptanceCheck, AcceptanceStatus};
