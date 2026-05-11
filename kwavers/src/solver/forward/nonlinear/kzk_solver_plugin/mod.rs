//! KZK Equation Solver Plugin
//!
//! Based on Lee & Hamilton (1995): "Time-domain modeling of pulsed finite-amplitude sound beams"

mod frequency_operator;
mod plugin_impl;
mod solver;

pub use frequency_operator::FrequencyOperator;
pub use solver::KzkSolverPlugin;
