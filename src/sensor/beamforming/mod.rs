//! Beamforming Algorithms for Ultrasound Arrays
//!
//! This module implements state-of-the-art beamforming algorithms for ultrasound
//! imaging and passive acoustic mapping, following established literature and
//! designed for large-scale array processing.
//!
//! # Design Principles
//! - **Literature-Based**: All algorithms follow established papers
//! - **Zero-Copy**: Efficient `ArrayView` usage throughout
//! - **Sparse Operations**: Designed for large arrays with sparse matrices
//! - **Modular Design**: Plugin-compatible architecture
//!
//! # Literature References
//! - Van Veen & Buckley (1988): "Beamforming: A versatile approach to spatial filtering"
//! - Li et al. (2003): "Robust Capon beamforming"
//! - Schmidt (1986): "Multiple emitter location and signal parameter estimation"
//! - Capon (1969): "High-resolution frequency-wavenumber spectrum analysis"
//! - Frost (1972): "An algorithm for linearly constrained adaptive array processing"

mod algorithms;
mod config;
mod covariance;
mod processor;
mod steering;

pub use algorithms::{AlgorithmImplementation, BeamformingAlgorithm};
pub use config::BeamformingConfig;
pub use covariance::{CovarianceEstimator, SpatialSmoothing};
pub use processor::BeamformingProcessor;
pub use steering::{SteeringVector, SteeringVectorMethod};
