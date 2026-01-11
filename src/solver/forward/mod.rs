//! Forward solvers
//!
//! This module contains solvers for forward problems (wave propagation,
//! heat diffusion, etc.) that simulate physical phenomena from causes to effects.

pub mod acoustic;
pub mod axisymmetric;
pub mod bem;
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
pub mod sem;
pub mod thermal_diffusion;

pub use axisymmetric::AxisymmetricSolver;
pub use bem::BemSolver;
pub use fdtd::FdtdSolver;
pub use sem::SemSolver;
pub use helmholtz::born_series::{
    ConvergentBornSolver, ConvergentBornStats, IterativeBornSolver, IterativeBornStats,
    ModifiedBornSolver, ModifiedBornStats,
};
pub use hybrid::{BemFemCouplingConfig, BemFemCoupler, BemFemInterface, BemFemSolver, FdtdFemCouplingConfig, FdtdFemCoupler, FdtdFemSolver, HybridSolver, PstdSemCouplingConfig, PstdSemCoupler, PstdSemSolver};
pub use imex::IMEXIntegrator;
pub use plugin_based::PluginBasedSolver;
pub use pstd::PSTDSolver;
