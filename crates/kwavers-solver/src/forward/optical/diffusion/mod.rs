//! Optical Diffusion Solver Module
//!
//! Implements the diffusion approximation for optical fluence computation
//! in biological tissues. Moved from physics layer to enforce proper
//! separation of concerns.

mod solver;

pub use solver::{
    DiffusionBoundaryCondition, DiffusionBoundaryConditions, DiffusionSolver, DiffusionSolverConfig,
};
