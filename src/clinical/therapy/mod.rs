//! Clinical therapy workflows
//!
//! This module provides application-level therapeutic workflows that combine
//! physics models and solvers for clinical therapy applications.

pub mod lithotripsy;
pub mod swe_3d_workflows;
pub mod therapy_integration;

pub use swe_3d_workflows::*;
pub use therapy_integration::*;
