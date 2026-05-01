//! # Beamforming Trait Hierarchy
//!
//! This module defines the core trait hierarchy for beamforming algorithms in kwavers.
//! It establishes a clean abstraction boundary between different algorithm categories
//! while enforcing strict architectural principles.
//!
//! # Architectural Intent (SSOT + Layer Separation)
//!
//! ## Design Principles
//!
//! 1. **Single Source of Truth (SSOT)**: All beamforming algorithm interfaces are defined here
//! 2. **Layer Separation**: Traits operate on processed data, not domain primitives
//! 3. **Explicit Failure**: No silent fallbacks, error masking, or dummy outputs
//! 4. **Mathematical Rigor**: Contracts specify invariants and guarantees
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::traits (Layer 7)
//!   ↓ imports from
//! domain::sensor (Layer 2) - array geometry (read-only)
//! math::linear_algebra (Layer 1) - numerical operations
//! core::error (Layer 0) - error types
//! ```
//!
//! # Trait Hierarchy
//!
//! ```text
//! Beamformer (root trait)
//!   ├── TimeDomainBeamformer
//!   │     ├── DelayAndSum
//!   │     └── SyntheticAperture
//!   ├── FrequencyDomainBeamformer
//!   │     ├── MinimumVariance (MVDR/Capon)
//!   │     ├── MUSIC
//!   │     └── RobustCapon
//!   └── AdaptiveBeamformer
//!         ├── SampleMatrixInversion (SMI)
//!         └── EigenspaceMinimumVariance (ESMV)
//! ```
//!
//! # Module layout
//!
//! - [`core`]: root [`Beamformer`] trait — minimal `focus_at_point` contract.
//! - [`time_domain`]: [`TimeDomainBeamformer`] — RF time-series with delay
//!   computation and apodization.
//! - [`frequency_domain`]: [`FrequencyDomainBeamformer`] — FFT-bin steering
//!   vectors and sample covariance.
//! - [`adaptive`]: [`AdaptiveBeamformer`] — data-driven weight optimization
//!   with diagonal loading and pseudospectrum.
//! - [`config`]: [`BeamformerConfig`] — sensor-array initialization seam.
//!
//! # Mathematical Foundation
//!
//! ## General Beamforming Output
//!
//! For a sensor array with N elements, the beamformed output is:
//!
//! ```text
//! y(r, t) = w^H x(t, r)
//! ```
//!
//! where:
//! - `w` (N×1) = complex weight vector
//! - `x(t, r)` (N×1) = sensor data vector (time-aligned or frequency-domain)
//! - `r` = focal point in space
//! - `H` = Hermitian (conjugate transpose)
//!
//! # Error Semantics (Strict)
//!
//! All traits enforce:
//!
//! - ❌ **NO silent fallbacks** - return `Err(...)` on any failure
//! - ❌ **NO error masking** - propagate root cause errors
//! - ❌ **NO dummy outputs** - never return zeros, steering vectors, or placeholders
//! - ❌ **NO undefined behavior** - all inputs validated, all outputs guaranteed
//!
//! # Performance Considerations
//!
//! | Algorithm Category | Complexity | Memory | Parallelizable | GPU-Friendly |
//! |-------------------|------------|--------|----------------|--------------|
//! | Time-Domain DAS   | O(N·M·K)   | O(N·M) | ✅ Yes         | ✅ Yes       |
//! | Frequency-Domain  | O(N²·M)    | O(N²)  | ⚠️ Partial     | ✅ Yes       |
//! | Adaptive          | O(N³·M)    | O(N²)  | ❌ Limited     | ⚠️ Moderate  |
//!
//! where N = sensors, M = time samples, K = focal points
//!
//! # Literature References
//!
//! ## Foundational Texts
//!
//! - Van Trees, H. L. (2002). *Optimum Array Processing: Part IV of Detection,
//!   Estimation, and Modulation Theory*. Wiley-Interscience.
//!   ISBN: 978-0-471-09390-9
//!
//! - Johnson, D. H., & Dudgeon, D. E. (1993). *Array Signal Processing:
//!   Concepts and Techniques*. Prentice Hall.
//!   ISBN: 978-0-13-048513-5
//!
//! ## Medical Ultrasound Applications
//!
//! - Jensen, J. A. (1996). *Field: A Program for Simulating Ultrasound Systems*.
//!   Medical & Biological Engineering & Computing, 34(1), 351-353.
//!
//! - Synnevåg, J. F., et al. (2009). "Adaptive beamforming applied to medical
//!   ultrasound imaging." *IEEE Trans. Ultrason., Ferroelect., Freq. Control*, 56(8).
//!   DOI: 10.1109/TUFFC.2009.1263

mod adaptive;
mod config;
mod core;
mod frequency_domain;
mod time_domain;

#[cfg(test)]
mod tests;

pub use adaptive::AdaptiveBeamformer;
pub use config::BeamformerConfig;
pub use core::Beamformer;
pub use frequency_domain::FrequencyDomainBeamformer;
pub use time_domain::TimeDomainBeamformer;
