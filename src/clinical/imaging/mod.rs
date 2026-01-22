//! Clinical imaging workflows
//!
//! This module provides application-level imaging workflows that combine
//! physics models and solvers for clinical imaging applications.

pub mod chromophores;
pub mod doppler;
pub mod phantoms;
pub mod photoacoustic;
pub mod spectroscopy;
pub mod workflows;

pub use workflows::*;
