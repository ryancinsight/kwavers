//! Forward solvers
//!
//! This module contains solvers for forward problems (wave propagation,
//! heat diffusion, etc.) that simulate physical phenomena from causes to effects.

pub mod acoustic;
pub mod axisymmetric;
pub mod elastic;
pub mod elastic_wave;
pub mod fdtd;
pub mod helmholtz;
pub mod hybrid;
pub mod imex;
pub mod nonlinear;
pub mod plugin_based;
pub mod poroelastic;
pub mod pstd;
pub mod thermal_diffusion;

pub use axisymmetric::AxisymmetricSolver;
pub use fdtd::FdtdSolver;
pub use helmholtz::born_series::{
    ConvergentBornSolver, ConvergentBornStats, IterativeBornSolver, IterativeBornStats,
    ModifiedBornSolver, ModifiedBornStats,
};
pub use hybrid::HybridSolver;
pub use imex::IMEXIntegrator;
pub use plugin_based::PluginBasedSolver;
pub use pstd::PSTDSolver;
