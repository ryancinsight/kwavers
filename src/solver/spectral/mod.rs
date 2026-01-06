pub mod absorption;
pub mod config;
pub mod data;
pub mod dg; // Hybrid DG module
pub mod operators;
pub mod plugin;
pub mod sources;
pub mod utils;

pub use crate::recorder::{SensorConfig, SensorData, SensorHandler};
pub use crate::solver::spectral::solver::SpectralSolver;
pub use absorption::initialize_absorption_operators;
pub use config::SpectralConfig;
pub use data::initialize_field_arrays;
pub use plugin::SpectralPlugin;
pub use sources::{SourceHandler, SpectralSource};

pub mod solver; // Core solver implementation

pub type KSpaceConfig = SpectralConfig;
pub type KSpaceSolver = SpectralSolver;
pub type KSpaceSource = SpectralSource;
pub type KSpaceSourceHandler = SourceHandler;
pub type KSpaceSourceMode = sources::SourceMode;
