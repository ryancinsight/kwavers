//! Unified Solver Interface
//!
//! This module defines the common interfaces, traits, and configurations
//! for all solver implementations in Kwavers.

// pub mod config; // Consolidated into crate::solver::config
pub mod progress;
pub mod solver;

pub use self::progress::{
    ConsoleProgressReporter, FieldsSummary, ProgressData, ProgressReporter, ProgressUpdate,
};
pub use self::solver::{Solver, SolverStatistics};
pub use crate::solver::config::SolverConfiguration as SolverConfig;
pub use crate::solver::feature::{FeatureManager, SolverFeature};
