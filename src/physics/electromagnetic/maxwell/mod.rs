//! Maxwell's equations solvers
//!
//! This module provides numerical solvers for Maxwell's equations,
//! including Finite-Difference Time-Domain (FDTD) methods.

pub mod fdtd;

pub use fdtd::FDTD;
