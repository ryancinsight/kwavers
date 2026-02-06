//! # Analysis Layer Beamforming Algorithms
//!
//! This module provides **general-purpose beamforming algorithms** for array signal
//! processing and ultrasound image formation. It implements the mathematical and
//! signal processing aspects of beamforming while remaining agnostic to specific
//! sensor geometries and hardware characteristics.
//!
//! ## Architectural Separation
//!
//! ### Analysis Layer Responsibilities (THIS MODULE)
//! - **Mathematical Algorithms**: Delay-and-sum, MVDR, MUSIC, adaptive methods
//! - **Signal Processing**: Filtering, windowing, optimization techniques
//! - **Image Formation**: Reconstruction algorithms and pipelines
//! - **Performance Optimization**: SIMD, parallelization, caching strategies
//!
//! ### Domain Layer Responsibilities ([`crate::domain::sensor::beamforming`])
//! - **Hardware Coupling**: Sensor-specific optimizations and constraints
//! - **Geometry Integration**: Array-specific delay calculations and positioning
//! - **Real-time Interfaces**: Hardware-accelerated processing pipelines
//! - **Configuration Management**: Sensor-specific parameter validation
//!
//! ### Accessor Pattern
//! ```rust,ignore
//! // 1. Domain layer handles sensor-specific concerns
//! let sensors = GridSensorSet::new(positions, sampling_rate)?;
//! let beamformer = SensorBeamformer::new(&sensors);
//!
//! // 2. Analysis layer provides algorithms through interfaces
//! let processor = DelayAndSum::new();
//! let image = processor.process(&rf_data, &beamformer.geometry(), &grid)?;
//! ```
//!
//! ## Algorithm Categories
//!
//! ### Time-Domain Beamforming
//! - **Delay-and-Sum (DAS)**: Geometric delay compensation and coherent summation
//! - **Dynamic Focusing**: Real-time focal point adjustment during reception
//! - **Synthetic Aperture**: Virtual array expansion through motion
//!
//! ### Frequency-Domain Beamforming
//! - **Minimum Variance (Capon)**: Adaptive weighting for interference rejection
//! - **MUSIC**: High-resolution source localization via subspace methods
//! - **Robust Capon**: Diagonal loading for improved stability
//!
//! ### Adaptive Beamforming
//! - **Sample Matrix Inversion (SMI)**: Data-dependent weight calculation
//! - **Eigenspace Methods**: Signal subspace exploitation
//! - **Constrained Optimization**: User-defined beam pattern constraints
//!
//! ### Advanced Methods
//! - **Neural Beamforming**: Deep learning-based adaptive processing
//! - **Compressive Sensing**: Sparse reconstruction techniques
//! - **Differentiable Beamforming**: Gradient-based optimization
//!
//! ## Architecture
//!
//! ```text
//! Beamforming Flow:
//!
//! 1. Domain Layer (Sensor Geometry)
//!    ├── Sensor positions
//!    ├── Element spacing
//!    └── Sampling rate
//!
//! 2. Data Acquisition (Sensor Recording)
//!    ├── RF data capture
//!    ├── Time series storage
//!    └── Channel alignment
//!
//! 3. Analysis Layer (THIS MODULE)
//!    ├── Delay calculation
//!    ├── Apodization weights
//!    ├── Signal combination
//!    └── Image formation
//!
//! 4. Output
//!    ├── Beamformed image
//!    ├── Source location
//!    └── Quality metrics
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::DelayAndSum;
//! use kwavers::domain::sensor::GridSensorSet;
//! use ndarray::Array2;
//!
//! // 1. Define sensor geometry (domain layer)
//! let sensor_positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], ...];
//! let sensors = GridSensorSet::new(sensor_positions, 10e6)?; // 10 MHz
//!
//! // 2. Acquire RF data (simulation or hardware)
//! let rf_data: Array2<f64> = sensors.get_recorded_data();
//! // Shape: (n_sensors, n_time_samples)
//!
//! // 3. Create beamformer (analysis layer)
//! let sound_speed = 1540.0; // m/s (soft tissue)
//! let mut beamformer = DelayAndSum::new(&sensors, sound_speed);
//!
//! // 4. Define imaging grid
//! let image_grid = beamformer.create_image_grid(
//!     x_range: (-0.02, 0.02),  // ±20 mm
//!     z_range: (0.01, 0.05),   // 10-50 mm depth
//!     resolution: 0.0001,      // 0.1 mm pixels
//! )?;
//!
//! // 5. Process RF data to form image
//! let beamformed_image = beamformer.process(&rf_data, &image_grid)?;
//! // Shape: (n_x_pixels, n_z_pixels)
//! ```
//!
//! ## Trait Hierarchy
//!
//! ```text
//! Beamformer (trait)
//!   ├── process() -> Image
//!   ├── set_focus()
//!   └── get_psf() -> PointSpreadFunction
//!
//! TimeDomainBeamformer: Beamformer
//!   ├── DelayAndSum
//!   └── SyntheticAperture
//!
//! FrequencyDomainBeamformer: Beamformer
//!   ├── MinimumVariance
//!   ├── MUSIC
//!   └── RobustCapon
//!
//! AdaptiveBeamformer: Beamformer
//!   ├── SampleMatrixInversion
//!   └── EigenspaceMinimumVariance
//! ```
//!
//! ## Performance Considerations
//!
//! - **GPU Acceleration**: Enable with `gpu` feature for real-time processing
//! - **Parallel Processing**: Rayon-based parallelization for CPU
//! - **Cache Optimization**: Memory layouts optimized for cache efficiency
//! - **SIMD**: Vectorized operations where applicable
//!
//! ## Mathematical Foundation
//!
//! ### Delay-and-Sum Beamforming
//!
//! For a sensor array with N elements, the beamformed output at focal point **r** is:
//!
//! ```text
//! y(r, t) = ∑ᵢ₌₁ᴺ wᵢ · xᵢ(t - τᵢ(r))
//! ```
//!
//! where:
//! - `wᵢ` = apodization weight for sensor i
//! - `xᵢ(t)` = received signal at sensor i
//! - `τᵢ(r)` = time delay from focal point r to sensor i
//!
//! ### Minimum Variance (Capon) Beamforming
//!
//! Adaptive weights minimize output power while maintaining unity gain in look direction:
//!
//! ```text
//! w = R⁻¹a / (aᴴR⁻¹a)
//! ```
//!
//! where:
//! - `R` = sample covariance matrix
//! - `a` = steering vector for look direction
//! - `H` = Hermitian (conjugate transpose)
//!
//! ## References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection,
//!   Estimation, and Modulation Theory*. Wiley-Interscience.
//!
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//!   DOI: 10.1109/PROC.1969.7278
//!
//! - Jensen, J. A., et al. (2006). "Synthetic aperture ultrasound imaging."
//!   *Ultrasonics*, 44, e5-e15.
//!   DOI: 10.1016/j.ultras.2006.07.017
//!
//! - Synnevåg, J. F., et al. (2009). "Adaptive beamforming applied to medical
//!   ultrasound imaging." *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 54(8).
//!   DOI: 10.1109/TUFFC.2007.431
//!
//! ## Implementation Status
//!
//! Algorithms are being migrated from `domain::sensor::beamforming` in Phase 2-3:
//!
//! - [x] Time-domain DAS (Delay-and-Sum) ✅
//! - [x] Delay reference policy and utilities ✅
//! - [x] Define `AdaptiveBeamformer` trait ✅
//! - [x] Adaptive beamforming: MinimumVariance (Capon/MVDR) ✅
//! - [x] Subspace methods: MUSIC, ESMV ✅
//! - [ ] Narrowband frequency-domain beamforming
//! - [ ] Migrate neural beamforming (experimental)
//! - [ ] Add GPU implementations
//! - [ ] Comprehensive integration tests
//!
//! ## Status
//!
//! **Current:** 🟢 Phase 2 complete - Infrastructure setup (traits, covariance, utils)
//! **Next:** Phase 3 - Algorithm migration from `domain::sensor::beamforming`
//! **Timeline:** Sprint 4, Phase 3 (Week 4-5 execution)
//!
//! ## Phase 2 Deliverables (Sprint 4, ✅ COMPLETE)
//!
//! - [x] Core trait hierarchy (`traits.rs`) ✅
//! - [x] Covariance matrix estimation (`covariance/`) ✅
//! - [x] Utility functions (`utils/`) ✅
//! - [x] Narrowband module placeholder (`narrowband/`) ✅
//! - [x] Experimental module placeholder (`experimental/`) ✅
//! - [x] Module structure and re-exports ✅

// Core trait hierarchy
pub mod traits;

// Algorithm implementations
pub mod adaptive;
pub mod time_domain;

// Domain processor (shared with domain layer via re-export)
pub mod domain_processor;

// Infrastructure modules (Phase 2 - Sprint 4)
pub mod covariance; // Covariance matrix estimation
pub mod utils; // Steering vectors, windows, interpolation

// Advanced algorithm modules
pub mod experimental;
pub mod neural; // Neural/ML beamforming (PINN, distributed) // Experimental/research-grade algorithms
pub mod slsc; // Short-Lag Spatial Coherence beamforming

// GPU-accelerated implementations (Sprint 214 Session 3)
pub mod gpu; // GPU beamforming (Burn-based + WGSL shaders)

// Future algorithm modules (planned for Phase 3)
pub mod narrowband; // Frequency-domain beamforming (awaiting migration)
pub mod three_dimensional; // 3D beamforming algorithms

// Test utilities (shared accessor layer for test modules)
#[cfg(test)]
pub mod test_utilities;

// Re-exports for convenience
pub use adaptive::{AdaptiveBeamformer, EigenspaceMV, MinimumVariance, MUSIC};
pub use time_domain::{
    alignment_shifts_s, delay_and_sum, relative_delays_s, DelayReference, DEFAULT_DELAY_REFERENCE,
};

// Trait re-exports
pub use traits::{Beamformer, BeamformerConfig, FrequencyDomainBeamformer, TimeDomainBeamformer};

// Utility re-exports
pub use covariance::{
    estimate_forward_backward_covariance, estimate_sample_covariance, is_hermitian, trace,
    validate_covariance_matrix,
};

// GPU beamforming re-exports (conditional on pinn feature)
#[cfg(feature = "pinn")]
pub use gpu::{beamform_cpu, BurnBeamformingConfig, BurnDasBeamformer, InterpolationMethod};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        assert_eq!(DEFAULT_DELAY_REFERENCE, DelayReference::SensorIndex(0));
        let mvdr = MinimumVariance::default();
        assert!(mvdr.diagonal_loading.is_finite());
    }

    #[test]
    fn test_delay_reference_export() {
        // Verify DelayReference is accessible
        let ref_policy = DelayReference::recommended_default();
        assert_eq!(ref_policy, DelayReference::SensorIndex(0));
    }

    #[test]
    fn test_adaptive_beamformer_export() {
        // Verify adaptive beamforming types are accessible
        let mvdr = MinimumVariance::with_diagonal_loading(1e-4);
        assert_eq!(mvdr.diagonal_loading, 1e-4);
    }

    #[test]
    fn test_subspace_export() {
        // Verify subspace methods are accessible
        let music = MUSIC::new(2);
        assert_eq!(music.num_sources, 2);

        let esmv = EigenspaceMV::with_diagonal_loading(3, 1e-4);
        assert_eq!(esmv.num_sources, 3);
        assert_eq!(esmv.diagonal_loading, 1e-4);
    }
}
