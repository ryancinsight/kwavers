// src/solver/mod.rs
// Clean module structure focusing only on the plugin-based architecture

// Hierarchical solver module structure
pub mod config;
pub mod factory;
pub mod feature;
pub mod forward;
pub mod progress;

pub mod analytical;
pub mod integration;
pub mod interface;
pub mod inverse;
pub mod multiphysics;
pub mod plugin;
pub mod utilities;
pub mod workspace;

// Re-export field indices from the single source of truth
pub use crate::domain::field::indices::{
    PRESSURE_IDX as P_IDX, STRESS_XX_IDX as SXX_IDX, STRESS_XY_IDX as SXY_IDX,
    STRESS_XZ_IDX as SXZ_IDX, STRESS_YY_IDX as SYY_IDX, STRESS_YZ_IDX as SYZ_IDX,
    STRESS_ZZ_IDX as SZZ_IDX, TOTAL_FIELDS, VX_IDX, VY_IDX, VZ_IDX,
};

// Re-export commonly used types from hierarchical modules
pub use config::{SolverParameters, SolverType};
pub use forward::{FdtdSolver, HybridSolver, IMEXIntegrator, PSTDSolver, PluginBasedSolver};
pub use interface::{Solver, SolverConfig};
pub use inverse::{
    ReconstructionConfig, Reconstructor, TimeReversalConfig, TimeReversalReconstructor,
};
pub use multiphysics::{CouplingStrategy, FieldCoupler, MultiPhysicsSolver};
// pub use utilities::LinearAlgebra;

// Constants module remains at root level for easy access
pub mod constants;

// Progress reporting - use types from interface module
pub use interface::{
    ConsoleProgressReporter, FieldsSummary, ProgressData, ProgressReporter, ProgressUpdate,
};

// Backward-compatible re-exports for commonly used submodules
pub use forward::fdtd;
pub use forward::hybrid;
pub use forward::plugin_based;
pub use forward::pstd;
pub use integration::time_integration;
pub use inverse::reconstruction;
pub use inverse::time_reversal;
pub use utilities::amr;
// pub use utilities::linear_algebra;
pub mod validation;
