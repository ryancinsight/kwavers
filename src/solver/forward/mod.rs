//! Forward solvers
//!
//! This module contains solvers for forward problems (wave propagation,
//! heat diffusion, etc.) that simulate physical phenomena from causes to effects.

pub mod acoustic;
pub mod bem;
pub mod coupled;
pub mod elastic;
pub mod elastic_wave;
pub mod fdtd;
pub mod helmholtz;
pub mod hybrid;
pub mod imex;
pub mod nonlinear;
pub mod optical;
pub mod plugin_based;
pub mod poroelastic;
pub mod pstd;
pub mod sem;
pub mod thermal;
pub mod thermal_diffusion;

pub use bem::BemSolver;
pub use coupled::{ThermalAcousticConfig, ThermalAcousticCoupler};
pub use fdtd::FdtdSolver;
pub use helmholtz::born_series::{
    ConvergentBornSolver,
    IterativeBornSolver,
    ModifiedBornSolver,
    // TODO: Implement ConvergentBornStats, IterativeBornStats, ModifiedBornStats
    // ConvergentBornStats, IterativeBornStats, ModifiedBornStats,
};
pub use hybrid::{
    BemFemCoupler, BemFemCouplingConfig, BemFemInterface, BemFemSolver, FdtdFemCoupler,
    FdtdFemCouplingConfig, FdtdFemSolver, HybridSolver, PstdSemCoupler, PstdSemCouplingConfig,
    PstdSemSolver,
};
pub use imex::IMEXIntegrator;
pub use plugin_based::PluginBasedSolver;
pub use pstd::PSTDSolver;
pub use sem::SemSolver;
pub use thermal::PennesSolver;
