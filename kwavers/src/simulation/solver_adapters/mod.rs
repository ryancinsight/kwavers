//! Simulation-owned adapters from numerical cores to `solver::interface::Solver`.
//!
//! The simulation layer owns these adapters because they bind domain grids,
//! sources, and sensors to concrete solver state. Numerical kernels remain in
//! `solver::forward`.

pub mod dg;

pub use dg::DgSimulationSolver;
