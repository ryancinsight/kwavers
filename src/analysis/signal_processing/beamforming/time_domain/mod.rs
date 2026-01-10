//! Time-domain beamforming (broadband / transient processing).
//!
//! # Overview
//!
//! Time-domain beamforming operates directly on time-series sensor data, making it
//! suitable for **broadband** and **transient** signals common in ultrasound imaging,
//! sonar, and passive acoustic monitoring.
//!
//! ## Field Terminology
//!
//! - **Time-domain DAS** is also called **conventional beamforming** or **shift-and-sum**.
//! - For transient localization, the dominant pattern is **SRP-DAS** (Steered Response Power):
//!   evaluate candidate points by steering (TOF alignment) and scoring energy:
//!   ```text
//!   SRP(p) = ∑ₜ |y_p(t)|²
//!   ```
//!   where `y_p(t)` is the steered DAS output for focal point `p`.
//!
//! # Mathematical Foundation
//!
//! ## Delay-and-Sum (DAS)
//!
//! The fundamental operation is:
//!
//! ```text
//! y(t) = Σᵢ₌₁ᴺ wᵢ · xᵢ(t - Δτᵢ)
//! ```
//!
//! where:
//! - `N` = number of sensors
//! - `xᵢ(t)` = received signal at sensor i
//! - `wᵢ` = apodization weight (typically normalized: Σwᵢ = 1)
//! - `Δτᵢ` = relative time delay for sensor i
//! - `y(t)` = beamformed output
//!
//! ## Delay Reference Policy
//!
//! Absolute propagation delays (time-of-flight) are:
//! ```text
//! τᵢ(p) = ||xᵢ - p|| / c
//! ```
//!
//! However, for time-domain processing, we need **relative delays**:
//! ```text
//! Δτᵢ(p) = τᵢ(p) - τᵣₑ𝒻(p)
//! ```
//!
//! The choice of `τᵣₑ𝒻` is a **policy decision** (see [`DelayReference`]):
//! - **SensorIndex(k)**: Use sensor k as reference (deterministic, recommended)
//! - **EarliestArrival**: Use `minᵢ τᵢ(p)` (data-dependent)
//! - **LatestArrival**: Use `maxᵢ τᵢ(p)` (data-dependent)
//!
//! # Architectural Intent (SSOT / Deep Vertical Tree)
//!
//! This module is the **single source of truth** for time-domain beamforming in kwavers.
//! It is placed in the **analysis layer** because:
//!
//! 1. **Domain layer** provides sensor geometry (positions, sampling rate)
//! 2. **Physics layer** provides sound speed
//! 3. **Analysis layer** (THIS MODULE) combines these to compute delays and beamform
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::time_domain (Layer 7)
//!   ↓ imports from
//! domain::sensor (Layer 2) - sensor positions, sampling parameters
//! physics::acoustics (Layer 3) - sound speed
//! math::numerics (Layer 1) - numerical operations
//! core::error (Layer 0) - error types
//! ```
//!
//! ## What Belongs Here vs Domain Layer
//!
//! ✅ **In `analysis::signal_processing::beamforming::time_domain`:**
//! - DAS algorithm implementation
//! - Delay calculation and steering
//! - Apodization weight computation
//! - SRP-DAS localization scoring
//! - Beamforming processors and pipelines
//!
//! ❌ **NOT here (belongs in `domain::sensor`):**
//! - Sensor array geometry (positions, orientations)
//! - Sampling rate and time base
//! - RF data recording and storage
//! - Sensor calibration parameters
//!
//! # Module Layout
//!
//! - [`delay_reference`]: Delay datum policies and TOF→relative-delay utilities
//! - [`das`]: Delay-and-sum (shift-and-sum) building blocks
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::time_domain::{
//!     delay_and_sum, DelayReference, DEFAULT_DELAY_REFERENCE
//! };
//! use kwavers::domain::sensor::GridSensorSet;
//! use ndarray::Array3;
//!
//! // 1. Define sensor geometry (domain layer)
//! let sensor_positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]; // 1mm spacing
//! let fs = 10e6; // 10 MHz sampling
//! let sensors = GridSensorSet::new(sensor_positions, fs)?;
//!
//! // 2. Acquire RF data (simulation or hardware)
//! let rf_data: Array3<f64> = sensors.get_recorded_data();
//! // Shape: (n_sensors, 1, n_time_samples)
//!
//! // 3. Define focal point and compute delays (analysis layer)
//! let focal_point = [0.0, 0.0, 0.02]; // 20mm depth
//! let sound_speed = 1540.0; // m/s (soft tissue)
//! let delays: Vec<f64> = sensors.compute_tof_delays(&focal_point, sound_speed)?;
//!
//! // 4. Apply beamforming
//! let weights = vec![1.0, 1.0]; // Equal weighting (can use apodization)
//! let beamformed = delay_and_sum(
//!     &rf_data,
//!     fs,
//!     &delays,
//!     &weights,
//!     DEFAULT_DELAY_REFERENCE,
//! )?;
//! // Shape: (1, 1, n_time_samples)
//! ```
//!
//! # Performance Considerations
//!
//! ## Current Implementation
//! - **Discretization**: Integer sample shifts (round to nearest sample)
//! - **Complexity**: O(n_elements × n_samples)
//! - **Memory**: O(n_samples) output buffer
//! - **Parallelization**: Single-threaded (CPU)
//!
//! ## Future Optimizations
//! - [ ] Fractional delay filtering (sub-sample accuracy)
//! - [ ] GPU acceleration (`gpu` feature)
//! - [ ] SIMD vectorization
//! - [ ] Parallel processing (Rayon)
//! - [ ] Upsampling + integer shifts for sub-sample precision
//!
//! # Next Steps (Recommended Defaults)
//!
//! The default delay reference for localization should be a **fixed reference sensor**,
//! commonly element 0 (`SensorIndex(0)`). This makes the delay datum explicit and keeps
//! SRP-DAS scoring point-dependent for transient data models.
//!
//! ## Comparison to Frequency-Domain Beamforming
//!
//! | Aspect | Time-Domain DAS | Frequency-Domain |
//! |--------|----------------|------------------|
//! | Signal Type | Broadband, transient | Narrowband, continuous |
//! | Operation | Sample shifts | Phase shifts |
//! | Precision | Integer samples | Sub-sample (phase) |
//! | Speed | O(N×T) | O(N²) for adaptive |
//! | Use Case | Ultrasound imaging, PAM | Radar, sonar arrays |
//!
//! # Literature References
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection,
//!   Estimation, and Modulation Theory*. Wiley-Interscience.
//!   (Chapter 2: Temporal and Spatial Sampling)
//!
//! - Brandstein, M., & Ward, D. (2001). *Microphone Arrays: Signal Processing
//!   Techniques and Applications*. Springer.
//!   (Chapter 5: Beamforming and Chapter 4: Time Delay Estimation)
//!
//! - DiBiase, J. H. (2000). *A High-Accuracy, Low-Latency Technique for Talker
//!   Localization in Reverberant Environments Using Microphone Arrays*.
//!   PhD Thesis, Brown University.
//!   (Develops SRP-PHAT: Steered Response Power with Phase Transform)
//!
//! - Jensen, J. A., et al. (2006). "Synthetic aperture ultrasound imaging."
//!   *Ultrasonics*, 44, e5-e15. DOI: 10.1016/j.ultras.2006.07.017
//!   (Synthetic aperture focusing technique using DAS)
//!
//! # Migration Note
//!
//! This module was migrated from `domain::sensor::beamforming::time_domain` to
//! `analysis::signal_processing::beamforming::time_domain` as part of the
//! architectural purification effort documented in **ADR 003: Signal Processing
//! Migration to Analysis Layer**.
//!
//! ## Why This Move Is Correct
//!
//! 1. **Layering**: Beamforming is an **analysis algorithm**, not a **domain primitive**
//! 2. **Dependencies**: Beamforming depends on domain (sensor geometry), not vice versa
//! 3. **Reusability**: Beamforming should work on data from multiple sources:
//!    - Real sensors (domain)
//!    - Simulated data (simulation layer)
//!    - Clinical workflows (clinical layer)
//! 4. **Literature**: Standard references treat beamforming as signal processing / array processing
//!
//! ## Backward Compatibility
//!
//! The old location `domain::sensor::beamforming::time_domain` will continue to exist
//! with deprecation warnings and re-exports for one minor version cycle. Update your
//! imports to the new location:
//!
//! ```rust,ignore
//! // Old (deprecated):
//! use crate::domain::sensor::beamforming::time_domain::das::delay_and_sum_time_domain_with_reference;
//!
//! // New (correct):
//! use crate::analysis::signal_processing::beamforming::time_domain::delay_and_sum;
//! ```

pub mod das;
pub mod delay_reference;

// Re-exports: keep domain terms discoverable at the `time_domain` level.
pub use das::{delay_and_sum, DEFAULT_DELAY_REFERENCE};
pub use delay_reference::{alignment_shifts_s, relative_delays_s, DelayReference};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify key types are accessible
        let _ = DEFAULT_DELAY_REFERENCE;
        let _ = DelayReference::recommended_default();
    }
}
