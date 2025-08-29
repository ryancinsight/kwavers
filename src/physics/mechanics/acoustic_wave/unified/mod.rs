//! Unified acoustic wave solver architecture - modularized
//!
//! This module provides a unified interface for all acoustic wave models,
//! with proper separation of concerns following GRASP principles.

pub mod config;
pub mod kuznetsov;
pub mod solver;
pub mod westervelt;

pub use config::{AcousticModelType, AcousticSolverConfig};
pub use solver::UnifiedAcousticSolver;

// Re-export model-specific solvers for direct use
pub use kuznetsov::KuznetsovSolver;
pub use westervelt::WesterveltSolver;
