//! Domain-Specific Beamforming Operations for Ultrasound Sensors
//!
//! This module handles **sensor-specific beamforming operations** that are tightly coupled
//! to hardware characteristics and array geometry. It provides the domain layer interface
//! for beamforming while delegating algorithmic complexity to the analysis layer.
//!
//! ## Architectural Role
//!
//! **Domain Layer Responsibilities:**
//! - Sensor geometry and array configuration
//! - Hardware-specific delay calculations
//! - Array-specific optimizations and constraints
//! - Real-time processing interfaces
//!
//! **Analysis Layer Delegation:**
//! - General-purpose beamforming algorithms ([`crate::analysis::signal_processing::beamforming`])
//! - Mathematical optimizations and transformations
//! - Advanced signal processing techniques
//!
//! ## Design Principles
//!
//! - **Hardware Coupling**: Operations tied to specific sensor geometries and hardware constraints
//! - **Accessor Pattern**: Uses analysis algorithms through well-defined interfaces
//! - **Performance Focus**: Optimized for real-time processing with specific array types
//! - **Zero Abstraction**: Minimal indirection for time-critical operations
//!
//! ## Usage Pattern
//!
//! ```rust,ignore
//! use kwavers::domain::sensor::{GridSensorSet, beamforming::SensorBeamformer};
//!
//! // 1. Configure sensor array (domain layer)
//! let sensors = GridSensorSet::new(sensor_positions, sampling_rate)?;
//!
//! // 2. Create sensor-specific beamformer (domain layer)
//! let beamformer = SensorBeamformer::new(&sensors);
//!
//! // 3. Delegate to analysis algorithms through accessor pattern
//! let delays = beamformer.calculate_delays(&image_grid, sound_speed)?;
//! let weights = beamformer.apply_windowing(delays, WindowType::Hanning)?;
//!
//! // 4. Process with analysis algorithms
//! use kwavers::analysis::signal_processing::beamforming::time_domain::DelayAndSum;
//! let processor = DelayAndSum::new();
//! let image = processor.process(&rf_data, &weights, &image_grid)?;
//! ```
//!
//! ## Layer Separation
//!
//! | Concern | Domain Layer | Analysis Layer |
//! |---------|-------------|----------------|
//! | Sensor geometry | ✅ Owns | Uses interface |
//! | Delay calculations | ✅ Hardware-specific | General algorithms |
//! | Steering vectors | ✅ Array-optimized | Mathematical primitives |
//! | Beamforming algorithms | Delegates | ✅ Owns |
//! | Optimization | Hardware constraints | Mathematical methods |
//!
//! ## References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing*. Wiley-Interscience.
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.

// Domain-specific beamforming interface
pub mod sensor_beamformer;

// Legacy modules (being phased out - use analysis layer instead)
// These remain for backward compatibility but delegate to analysis algorithms
pub mod adaptive;
mod beamforming_3d;
mod config;
pub mod covariance;
pub mod neural;
mod processor;
#[cfg(feature = "gpu")]
mod shaders;
mod steering;
pub mod time_domain;

// Domain-specific interface (recommended for new code)
pub use sensor_beamformer::{SensorBeamformer, SensorProcessingParams, WindowType};

// Legacy re-exports (deprecated - use analysis layer directly)
pub use adaptive::{
    AdaptiveBeamformer, ArrayGeometry, CovarianceTaper, SteeringMatrix,
    SteeringVector as AdaptiveSteeringVector, WeightCalculator, WeightingScheme,
};
pub use beamforming_3d::{
    ApodizationWindow, BeamformingAlgorithm3D, BeamformingConfig3D, BeamformingMetrics,
    BeamformingProcessor3D,
};
pub use config::{BeamformingConfig, BeamformingCoreConfig};
pub use covariance::{
    CovarianceEstimator, CovariancePostProcess, SpatialSmoothing, SpatialSmoothingComplex,
};
pub use processor::BeamformingProcessor;
pub use steering::{SteeringVector, SteeringVectorMethod};
// Note: time_domain submodules (das, delay_reference) have been removed.
// Use analysis::signal_processing::beamforming::time_domain instead.

// Note: NeuralBeamformer and NeuralBeamformingConfig are NOT YET MIGRATED.
// These high-level API types remain in the old monolithic file and will be
// extracted in a future sprint. Use the lower-level primitives directly:
// - NeuralBeamformingNetwork (for network operations)
// - PhysicsConstraints (for physics-informed constraints)
// - UncertaintyEstimator (for uncertainty quantification)

// Neural beamforming with modular architecture
pub use neural::{
    AIBeamformingConfig, AIBeamformingResult, ClinicalAnalysis, ClinicalDecisionSupport,
    ClinicalThresholds, DiagnosisAlgorithm, FeatureConfig, FeatureExtractor, FeatureMap,
    LesionDetection, PerformanceMetrics, RealTimeWorkflow, TissueClassification,
};

// Processor requires PINN feature
#[cfg(feature = "pinn")]
pub use neural::AIEnhancedBeamformingProcessor;
