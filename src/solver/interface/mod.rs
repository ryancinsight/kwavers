//! Unified Solver Interface
//!
//! This module defines the common interfaces, traits, and configurations
//! for all solver implementations in Kwavers.

pub mod config;
pub mod feature;
pub mod progress;
pub mod solver;

pub use config::SolverConfig;
pub use feature::{FeatureManager, SolverFeature};
pub use progress::{
    ConsoleProgressReporter, FieldsSummary, ProgressData, ProgressReporter, ProgressUpdate,
};
pub use solver::{Solver, SolverStatistics};
