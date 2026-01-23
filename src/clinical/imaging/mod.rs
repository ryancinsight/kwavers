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

// Functional ultrasound imaging (fUS) with brain GPS neuronavigation
// Based on: Nouhoum et al. (2021) "A functional ultrasound brain GPS for automatic vascular-based neuronavigation"
// DOI: 10.1038/s41598-021-94764-7
pub mod functional_ultrasound;

pub use workflows::*;
