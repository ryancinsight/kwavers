//! Unified Solver Interface
//!
//! This module defines the common interfaces, traits, and configurations
//! for all solver implementations in Kwavers.

// pub mod config; // Consolidated into crate::solver::config
pub use crate::solver::config::SolverConfiguration as SolverConfig;
pub use feature::{FeatureManager, SolverFeature};
pub use progress::{
    ConsoleProgressReporter, FieldsSummary, ProgressData, ProgressReporter, ProgressUpdate,
};
pub use solver::{Solver, SolverStatistics};
