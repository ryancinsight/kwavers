//! # Narrowband Beamforming
//!
//! This module provides narrowband (frequency-domain) beamforming algorithms.
//! Narrowband beamformers operate on Fourier-transformed data and are particularly
//! effective for signals with narrow frequency content.
//!
//! # Architectural Intent (SSOT + Analysis Layer)
//!
//! ## Design Principles
//!
//! 1. **Single Source of Truth**: All narrowband beamforming logic consolidated here
//! 2. **Layer Separation**: Analysis layer (not domain primitives)
//! 3. **Frequency-Domain Operations**: FFT-based processing pipeline
//! 4. **Explicit Failure**: No silent fallbacks or error masking
//!
//! ## Migration Target
//!
//! This module will consolidate narrowband algorithms from:
//! - `domain::sensor::beamforming::narrowband/*` (49 files, ~4k LOC)
//! - Inline frequency-domain operations scattered across adaptive beamformers
//!
//! ## SSOT Enforcement (Strict)
//!
//! Once migration is complete:
//! - ❌ **NO narrowband operations** outside this module
//! - ❌ **NO duplicate FFT-based beamforming** in other locations
//! - ❌ **NO silent fallbacks** to time-domain methods
//! - ❌ **NO error masking** via dummy outputs
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::narrowband (Layer 7)
//!   ↓ imports from
//! analysis::signal_processing::beamforming::{traits, covariance, utils} (Layer 7)
//! math::linear_algebra (Layer 1) - matrix operations
//! math::fft (Layer 1) - Fourier transforms
//! core::error (Layer 0) - error types
//! ```
//!
//! # Mathematical Foundation
//!
//! ## Narrowband Assumption
//!
//! The narrowband assumption states that the signal bandwidth Δf is much smaller
//! than the center frequency f₀:
//!
//! ```text
//! Δf << f₀  ⟹  Δf/f₀ < 0.1 (typically)
//! ```
//!
//! Under this assumption, the propagation delay across the array can be approximated
//! as a phase shift rather than a time delay.
//!
//! ## Frequency-Domain Beamforming
//!
//! For narrowband signals at frequency ω, the beamformed output is:
//!
//! ```text
//! Y(ω) = w^H(ω) · X(ω)
//! ```
//!
//! where:
//! - `w(ω)` (N×1) = complex frequency-dependent weights
//! - `X(ω)` (N×1) = sensor data in frequency domain (via FFT)
//! - `H` = Hermitian transpose
//!
//! ## Steering Vector (Narrowband)
//!
//! For a plane wave at angle θ relative to array normal:
//!
//! ```text
//! a(θ, ω) = [1, e^{jωΔ₁}, e^{jωΔ₂}, ..., e^{jωΔₙ₋₁}]^T
//! ```
//!
//! where Δᵢ = (dᵢ sin θ)/c is the time delay for sensor i.
//!
//! # Algorithm Categories
//!
//! ## Data-Independent Beamformers
//!
//! - **Conventional Beamformer**: Delay-and-sum in frequency domain
//! - **Superdirective Beamformer**: Maximizes directivity index
//!
//! ## Adaptive Beamformers
//!
//! - **Minimum Variance (Capon)**: Minimizes output power subject to constraints
//! - **Linearly Constrained Minimum Variance (LCMV)**: Multiple constraints
//! - **Generalized Sidelobe Canceller (GSC)**: Equivalent LCMV formulation
//!
//! ## Subspace Methods
//!
//! - **MUSIC**: Multiple Signal Classification
//! - **ESPRIT**: Estimation of Signal Parameters via Rotational Invariance
//! - **Root-MUSIC**: Polynomial rooting variant
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use kwavers::analysis::signal_processing::beamforming::narrowband;
//! use ndarray::Array2;
//! use num_complex::Complex64;
//!
//! // Time-domain sensor data (N_sensors × N_samples)
//! let rf_data: Array2<f64> = get_sensor_data();
//!
//! // Transform to frequency domain
//! let fft_data: Array2<Complex64> = narrowband::fft_transform(&rf_data)?;
//!
//! // Create beamformer
//! let beamformer = narrowband::ConventionalBeamformer::new(
//!     &sensor_positions,
//!     center_frequency,
//!     sound_speed,
//! )?;
//!
//! // Beamform at specific direction
//! let output = beamformer.beamform_direction(&fft_data, angle)?;
//! ```
//!
//! # Performance Considerations
//!
//! | Operation | Complexity | Memory | Suitable For |
//! |-----------|------------|--------|--------------|
//! | FFT Transform | O(N·M log M) | O(N·M) | All narrowband |
//! | Conventional BF | O(N·K) | O(N) | Real-time |
//! | Adaptive BF | O(N²·K) | O(N²) | High SNR |
//! | MUSIC | O(N³ + N·K) | O(N²) | DOA estimation |
//!
//! where N = sensors, M = time samples, K = scan angles
//!
//! # Applicability Limits
//!
//! Narrowband methods are valid when:
//!
//! 1. **Bandwidth constraint**: Δf/f₀ < 0.1
//! 2. **Array size**: Array aperture < λ_min / 2 (avoid spatial aliasing)
//! 3. **Signal stationarity**: Signal statistics constant over observation time
//!
//! For wideband signals (Δf/f₀ > 0.1), use time-domain or wideband subspace methods.
//!
//! # Literature References
//!
//! ## Foundational Papers
//!
//! - Capon, J. (1969). "High-resolution frequency-wavenumber spectrum analysis."
//!   *Proceedings of the IEEE*, 57(8), 1408-1418.
//!   DOI: 10.1109/PROC.1969.7278
//!
//! - Schmidt, R. O. (1986). "Multiple emitter location and signal parameter estimation."
//!   *IEEE Transactions on Antennas and Propagation*, 34(3), 276-280.
//!   DOI: 10.1109/TAP.1986.1143830
//!
//! ## Advanced Techniques
//!
//! - Roy, R., & Kailath, T. (1989). "ESPRIT—estimation of signal parameters via
//!   rotational invariance techniques." *IEEE Trans. Acoust., Speech, Signal Process.*,
//!   37(7), 984-995. DOI: 10.1109/29.32276
//!
//! - Frost, O. L. (1972). "An algorithm for linearly constrained adaptive array processing."
//!   *Proceedings of the IEEE*, 60(8), 926-935.
//!   DOI: 10.1109/PROC.1972.8817
//!
//! # Implementation Status
//!
//! **Current:** 🟡 Module structure created, awaiting migration
//! **Next:** Phase 3 - Migrate algorithms from `domain::sensor::beamforming::narrowband`
//! **Timeline:** Week 4 (Sprint 4, Phase 3)
//!
//! # Migration Plan
//!
//! ## Phase 3B: Narrowband Algorithm Migration
//!
//! 1. **Conventional Beamformer** (2h)
//!    - Migrate delay-and-sum frequency-domain implementation
//!    - Add FFT utilities
//!
//! 2. **MVDR/Capon** (already done via `adaptive::MinimumVariance`)
//!
//! 3. **LCMV** (3h)
//!    - Linearly constrained minimum variance
//!    - Constraint matrix formulation
//!
//! 4. **MUSIC** (already done via `adaptive::MUSIC`)
//!
//! 5. **Root-MUSIC** (2h)
//!    - Polynomial rooting variant
//!    - Higher resolution than spectral MUSIC
//!
//! 6. **ESPRIT** (3h)
//!    - Rotational invariance technique
//!    - Direct DOA estimates (no grid search)
//!
//! Total estimated effort: **10-12 hours**
//!
//! ## Migration Checklist
//!
//! - [ ] FFT/IFFT utilities
//! - [ ] Conventional beamformer
//! - [ ] LCMV beamformer
//! - [ ] Root-MUSIC
//! - [ ] ESPRIT
//! - [ ] Integration tests with time-domain equivalents
//! - [ ] Benchmarks vs domain::sensor::beamforming::narrowband
//! - [ ] Deprecation notices in old location
//! - [ ] Migration guide

// Future algorithm implementations (Phase 3B)
// pub mod conventional;   // Delay-and-sum in frequency domain
// pub mod lcmv;          // Linearly Constrained Minimum Variance
// pub mod esprit;        // ESPRIT algorithm
// pub mod root_music;    // Root-MUSIC (polynomial rooting)
// pub mod fft_utils;     // FFT/IFFT utilities

#[cfg(test)]
mod tests {
    // Integration tests will be added during migration
}
