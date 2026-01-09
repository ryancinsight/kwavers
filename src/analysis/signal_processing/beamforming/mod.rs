//! # Beamforming Module
//!
//! This module provides beamforming algorithms for array signal processing and
//! ultrasound image formation. Beamforming combines signals from multiple sensors
//! to form directional sensitivity patterns.
//!
//! ## Overview
//!
//! Beamforming is a signal processing technique used in sensor arrays for:
//! - **Spatial Filtering**: Enhance signals from specific directions
//! - **Image Formation**: Convert RF data to B-mode images
//! - **Source Localization**: Identify signal origins
//! - **Interference Rejection**: Suppress noise and artifacts
//!
//! ## Algorithm Categories
//!
//! ### Time-Domain Beamforming
//! - **Delay-and-Sum (DAS)**: Apply geometric delays and sum signals
//! - **Synthetic Aperture**: Reconstruct full aperture from sub-arrays
//!
//! ### Frequency-Domain Beamforming
//! - **Minimum Variance (Capon)**: Adaptive weights minimize output power
//! - **MUSIC**: Multiple signal classification via eigenanalysis
//!
//! ### Adaptive Beamforming
//! - **Sample Matrix Inversion (SMI)**: Data-adaptive weight calculation
//! - **Robust Capon**: Diagonal loading for stability
//! - **ESMV**: Eigenspace-based minimum variance
//!
//! ### Advanced Methods
//! - **Neural Beamforming**: Deep learning-based beamforming
//! - **Compressive Beamforming**: Sparse reconstruction
//!
//! ## Migration from `domain::sensor::beamforming`
//!
//! This module is the new home for beamforming algorithms, previously located in
//! `domain::sensor::beamforming`. The old location violated architectural layering
//! by mixing domain primitives (sensor geometry) with analysis algorithms.
//!
//! ### Migration Timeline
//!
//! - **Week 2 (Current)**: Module structure created, documentation in place
//! - **Week 3-4**: Gradual algorithm migration with backward compatibility
//! - **Week 5+**: Remove deprecated `domain::sensor::beamforming`
//!
//! ### What to Migrate Here
//!
//! ✅ **Should be in `analysis::signal_processing::beamforming`:**
//! - Delay-and-sum algorithms
//! - Adaptive beamforming (Capon, MUSIC)
//! - Neural beamforming
//! - Image formation algorithms
//! - Beamforming processors and pipelines
//!
//! ❌ **Should stay in `domain::sensor`:**
//! - Sensor array geometry
//! - Element positions
//! - Sampling parameters
//! - Sensor data recording
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
//! Algorithms are being migrated from `domain::sensor::beamforming` in Phase 2:
//!
//! - [x] Time-domain DAS (Delay-and-Sum) ✅
//! - [x] Delay reference policy and utilities ✅
//! - [ ] Define `Beamformer` trait (planned)
//! - [ ] Adaptive beamforming: MinimumVariance (Capon)
//! - [ ] Subspace methods: MUSIC, ESMV
//! - [ ] Narrowband frequency-domain beamforming
//! - [ ] Migrate neural beamforming (experimental)
//! - [ ] Add GPU implementations
//! - [ ] Comprehensive integration tests
//!
//! ## Status
//!
//! **Current:** 🟢 Time-domain DAS migration complete (Phase 2 PoC)
//! **Next:** Migrate adaptive beamforming (Capon, MUSIC) from domain layer
//! **Timeline:** Week 3-4 execution

// Algorithm implementations
pub mod time_domain;

// Future modules (planned)
// pub mod traits;           // Trait definitions for Beamformer, etc.
// pub mod adaptive;         // Adaptive beamforming (Capon, MUSIC, ESMV)
// pub mod narrowband;       // Frequency-domain beamforming
// pub mod neural;           // Neural network beamforming (experimental)
// pub mod utils;            // Utility functions

// Re-exports for convenience
pub use time_domain::{
    alignment_shifts_s, delay_and_sum, relative_delays_s, DelayReference, DEFAULT_DELAY_REFERENCE,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_structure() {
        // Verify time_domain module is accessible
        let _ = DEFAULT_DELAY_REFERENCE;
        assert!(true);
    }

    #[test]
    fn test_delay_reference_export() {
        // Verify DelayReference is accessible
        let ref_policy = DelayReference::recommended_default();
        assert_eq!(ref_policy, DelayReference::SensorIndex(0));
    }
}
