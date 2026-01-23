//! Domain-Specific Beamforming Interface for Ultrasound Sensors
//!
//! This module provides the **domain layer interface** for beamforming operations that are
//! tightly coupled to sensor hardware characteristics and array geometry.
//!
//! ## Architectural Role
//!
//! The domain layer focuses on **sensor-specific concerns** while delegating algorithmic
//! complexity to the analysis layer:
//!
//! **Domain Layer Responsibilities (THIS MODULE):**
//! - Sensor geometry and array configuration
//! - Hardware-specific delay calculations
//! - Array-specific optimizations and constraints
//! - Real-time processing interfaces
//!
//! **Analysis Layer Algorithms:** [`crate::analysis::signal_processing::beamforming`]
//! - General-purpose beamforming algorithms (DAS, MVDR, MUSIC, etc.)
//! - Mathematical optimizations and transformations
//! - Advanced signal processing techniques
//! - Neural/ML beamforming methods
//!
//! **Clinical Layer Decision Support:** [`crate::clinical::imaging::workflows::neural`]
//! - Lesion detection and tissue classification
//! - Diagnostic recommendations
//! - Clinical workflow orchestration
//!
//! ## Layer Separation
//!
//! | Concern | Domain Layer | Analysis Layer | Clinical Layer |
//! |---------|-------------|----------------|----------------|
//! | Sensor geometry | ✅ Owns | Uses interface | - |
//! | Beamforming algorithms | Delegates | ✅ Owns | - |
//! | Clinical analysis | - | Feature extraction | ✅ Decision support |
//! | 3D beamforming | Interface | ✅ Algorithms | - |
//! | Neural beamforming | - | ✅ Algorithms | Clinical workflows |
//!
//! ## Migration Guide
//!
//! Code has been migrated to enforce proper layer separation:
//!
//! **For Beamforming Algorithms:** Use `crate::analysis::signal_processing::beamforming`
//! ```rust,ignore
//! // OLD (deprecated):
//! use kwavers::domain::sensor::beamforming::adaptive::MinimumVariance;
//! use kwavers::domain::sensor::beamforming::neural::AIEnhancedBeamformingProcessor;
//! use kwavers::domain::sensor::beamforming::beamforming_3d::BeamformingProcessor3D;
//!
//! // NEW (recommended):
//! use kwavers::analysis::signal_processing::beamforming::adaptive::MinimumVariance;
//! use kwavers::analysis::signal_processing::beamforming::neural::processor::AIEnhancedBeamformingProcessor;
//! use kwavers::analysis::signal_processing::beamforming::three_dimensional::processor::BeamformingProcessor3D;
//! ```
//!
//! **For Clinical Decision Support:** Use `crate::clinical::imaging::workflows::neural`
//! ```rust,ignore
//! // OLD (deprecated):
//! use kwavers::domain::sensor::beamforming::neural::{ClinicalDecisionSupport, DiagnosisAlgorithm};
//!
//! // NEW (recommended):
//! use kwavers::clinical::imaging::workflows::neural::{ClinicalDecisionSupport, DiagnosisAlgorithm};
//! ```
//!
//! **For Sensor Interface (unchanged):** Keep using domain layer
//! ```rust,ignore
//! use kwavers::domain::sensor::beamforming::SensorBeamformer;  // ✅ Still in domain
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::domain::sensor::{GridSensorSet, beamforming::SensorBeamformer};
//! use kwavers::analysis::signal_processing::beamforming::time_domain::DelayAndSum;
//!
//! // 1. Configure sensor array (domain layer)
//! let sensors = GridSensorSet::new(sensor_positions, sampling_rate)?;
//!
//! // 2. Create sensor-specific interface (domain layer)
//! let beamformer = SensorBeamformer::new(&sensors);
//!
//! // 3. Use analysis algorithms
//! let processor = DelayAndSum::new();
//! let image = processor.process(&rf_data, beamformer.geometry(), &grid)?;
//! ```
//!
//! ## References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley-Interscience.
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."

// Domain-specific sensor interface (core functionality)
pub mod sensor_beamformer;
// Configuration types (shared with analysis layer)
mod config;
pub use config::{BeamformingConfig, BeamformingCoreConfig};
// Processor (shared with analysis layer)
mod processor;
pub use processor::BeamformingProcessor;
// GPU shaders (shared with analysis layer)
#[cfg(feature = "gpu")]
pub mod shaders;
// Covariance and steering (shared with analysis layer)
pub mod covariance;
pub mod steering;
pub mod time_domain;
pub use covariance::CovarianceEstimator;
pub use steering::{SteeringVector, SteeringVectorMethod};

pub use sensor_beamformer::{SensorBeamformer, SensorProcessingParams, WindowType};

// NOTE: All beamforming algorithms have been migrated to:
// - Analysis layer: crate::analysis::signal_processing::beamforming
// - Clinical layer: crate::clinical::imaging::workflows::neural
//
// Only the sensor geometry interface (SensorBeamformer) remains in domain layer.
