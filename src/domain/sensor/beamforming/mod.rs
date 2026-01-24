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
//! - Hardware-specific delay calculations (transmit/receive geometry)
//! - Array-specific apodization and constraints
//! - Real-time processing interfaces
//! - Physical sensor array characteristics
//!
//! **Analysis Layer Algorithms:** [`crate::analysis::signal_processing::beamforming`]
//! - General-purpose beamforming algorithms (DAS, MVDR, MUSIC, etc.)
//! - Mathematical optimizations and transformations
//! - Advanced signal processing techniques (adaptive, frequency-domain)
//! - Neural/ML beamforming methods
//! - Receive beamforming signal processing
//!
//! **Clinical Layer Decision Support:** [`crate::clinical::imaging::workflows::neural`]
//! - Lesion detection and tissue classification
//! - Diagnostic recommendations
//! - Clinical workflow orchestration
//!
//! ## Beamforming Concepts
//!
//! **Transmit Beamforming (Domain Layer):**
//! - Hardware control for focusing transmitted energy
//! - Array element excitation timing
//! - Transmit aperture configuration
//! - Physical delay line implementation
//!
//! **Receive Beamforming (Analysis Layer):**
//! - Signal processing of received echoes
//! - Delay-and-sum algorithms
//! - Adaptive weighting (MVDR, Capon)
//! - Image reconstruction
//!
//! **Array Geometry (Domain Layer):**
//! - Sensor positions and spacing
//! - Element characteristics
//! - Coordinate system definitions
//! - Hardware calibration data
//!
//! ## Layer Separation
//!
//! | Concern | Domain Layer | Analysis Layer | Clinical Layer |
//! |---------|-------------|----------------|----------------|
//! | Sensor geometry | ✅ Owns | Uses interface | - |
//! | Transmit beamforming | ✅ Hardware control | - | - |
//! | Receive beamforming | Hardware interface | ✅ Algorithms | - |
//! | Array configuration | ✅ Owns | - | - |
//! | Signal processing | - | ✅ Owns | - |
//! | Clinical analysis | - | Feature extraction | ✅ Decision support |
//! | 3D beamforming | Geometry | ✅ Algorithms | - |
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
//! use kwavers::domain::sensor::beamforming::BeamformingConfig;  // ✅ Configuration
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::domain::sensor::{GridSensorSet, beamforming::SensorBeamformer};
//! use kwavers::analysis::signal_processing::beamforming::time_domain::delay_and_sum;
//!
//! // 1. Configure sensor array (domain layer)
//! let sensors = GridSensorSet::new(sensor_positions, sampling_rate)?;
//!
//! // 2. Create sensor-specific interface (domain layer)
//! let beamformer = SensorBeamformer::new(sensor_array, sampling_rate);
//!
//! // 3. Calculate hardware-specific delays (domain layer)
//! let delays = beamformer.calculate_delays(&image_grid, sound_speed)?;
//!
//! // 4. Use analysis algorithms for receive beamforming (analysis layer)
//! let image = delay_and_sum(&rf_data, &delays)?;
//! ```
//!
//! ## References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley-Interscience.
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//! - Szabo, T. L. (2004). *Diagnostic Ultrasound Imaging: Inside Out*. Academic Press.

// Domain-specific sensor interface (core functionality)
pub mod sensor_beamformer;

// Configuration types (shared with analysis layer)
mod config;
pub use config::{BeamformingConfig, BeamformingCoreConfig};

// GPU shaders (hardware-accelerated implementations)
#[cfg(feature = "gpu")]
pub mod shaders;

pub use sensor_beamformer::{SensorBeamformer, SensorProcessingParams, WindowType};

// Re-exports for backward compatibility (migrated to analysis layer)
pub use crate::analysis::signal_processing::beamforming::domain_processor::BeamformingProcessor;
pub use crate::analysis::signal_processing::beamforming::utils::steering::{
    SteeringVector, SteeringVectorMethod,
};

// Re-export covariance module
pub mod covariance {
    pub use crate::analysis::signal_processing::beamforming::covariance::*;
}

// Re-export time_domain module
pub mod time_domain {
    pub use crate::analysis::signal_processing::beamforming::time_domain::*;
}

// NOTE: All beamforming algorithms have been migrated to:
// - Analysis layer: crate::analysis::signal_processing::beamforming
// - Clinical layer: crate::clinical::imaging::workflows::neural
//
// This domain module now contains only:
// - SensorBeamformer: Hardware-specific array geometry interface
// - BeamformingConfig: Shared configuration types
// - GPU shaders: Hardware-accelerated kernels
// - Re-exports: Backward compatibility for common types
