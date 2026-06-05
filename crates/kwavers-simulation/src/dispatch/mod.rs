//! Solver dispatch modules.
//!
//! Each module handles one solver type's orchestration:
//! setting up the solver, injecting sources, running the simulation,
//! and extracting results into a [`SimulationRunResult`](crate::types::SimulationRunResult).

pub mod bem;
pub mod dg;
pub mod elastic;
pub mod elastic_pstd;
pub mod fdtd;
pub mod helmholtz;
pub mod nonlinear;
pub mod poroelastic;
pub mod pstd;
pub mod rayleigh_sommerfeld;
pub(crate) mod shared;
