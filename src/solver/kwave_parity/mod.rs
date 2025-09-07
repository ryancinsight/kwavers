//! k-Wave Parity Implementation
//!
//! This module provides exact compatibility with k-Wave MATLAB toolbox,
//! implementing the core algorithms following GRASP architectural principles.
//! 
//! ## Architecture
//! 
//! Modules are organized by concern (<500 lines each, GRASP compliance):
//! - `config`: Configuration types and enums
//! - `solver`: Core solver implementation 
//! - `data`: Internal data structures
//! - `utils`: Helper functions
//! - `operators`: k-space and differential operators
//! - `absorption`: Absorption models
//! - `nonlinearity`: Nonlinear acoustics
//! - `sensors`: Sensor implementations
//! - `sources`: Source implementations
//!
//! # References
//!
//! - Treeby, B. E., & Cox, B. T. (2010). "k-Wave: MATLAB toolbox for the simulation
//!   and reconstruction of photoacoustic wave fields." Journal of Biomedical Optics, 15(2).
//! - Treeby, B. E., Jaros, J., Rendell, A. P., & Cox, B. T. (2012). "Modeling nonlinear
//!   ultrasound propagation in heterogeneous media with power law absorption using a
//!   k-space pseudospectral method." The Journal of the Acoustical Society of America, 131(6).

// GRASP-compliant module organization
pub mod absorption;
pub mod config;
pub mod data;
pub mod nonlinearity;
pub mod operators;
pub mod sensors;
pub mod solver;
pub mod sources;
pub mod utils;

// Public re-exports for clean API
pub use config::{AbsorptionMode, KWaveConfig};
pub use solver::KWaveSolver;

// Internal re-exports for module cohesion
// Note: FieldArrays and KSpaceData available but currently unused