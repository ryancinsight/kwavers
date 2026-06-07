//! Numerical optimisation utilities.
//!
//! General-purpose, problem-agnostic optimisers used by inverse solvers (FWI,
//! PINN refinement) and parameter fitting.

pub mod lbfgs;

pub use lbfgs::{minimize, LbfgsConfig, LbfgsMemory, LbfgsResult};
