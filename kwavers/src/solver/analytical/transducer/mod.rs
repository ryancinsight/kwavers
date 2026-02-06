//! Transducer Analytical Solvers
//!
//! This module contains analytical solvers for transducer fields.

pub mod fast_nearfield;

pub use fast_nearfield::{AngularSpectrumFactors, FNMConfig, FastNearfieldSolver};
