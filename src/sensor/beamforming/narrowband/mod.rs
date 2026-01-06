#![deny(missing_docs)]
//! Narrowband (frequency-domain) beamforming primitives and spatial spectra.
//!
//! # Field jargon / scope
//! This module is intended for **narrowband** (single-frequency or small-bandwidth) array
//! processing where steering is naturally expressed via a **complex steering vector**
//!
//! `a_i(p; f0) = exp(-j 2π f0 τ_i(p))`
//!
//! with propagation delay `τ_i(p) = ||x_i - p|| / c`.
//!
//! In this regime, adaptive methods such as **MVDR/Capon** are typically evaluated as a
//! **spatial spectrum**:
//!
//! `P_Capon(p) = 1 / (a(p)^H R^{-1} a(p))`
//!
//! where `R` is the sensor covariance matrix (possibly with **diagonal loading**).
//!
//! # Architectural intent (SSOT / deep vertical tree)
//! - This module hosts narrowband-specific orchestration utilities that compose:
//!   - `crate::sensor::beamforming::steering` (steering vectors)
//!   - `crate::sensor::beamforming::covariance` (covariance estimation / smoothing)
//!   - `crate::sensor::beamforming::adaptive` (MVDR/MUSIC/ESMV implementations)
//! - Localization consumers must not re-implement these numerics; they should call into SSOT.
//!
//! # Snapshot model (recommended for correctness)
//! For narrowband adaptive methods, prefer **complex baseband snapshots** and a **Hermitian**
//! covariance `R = (1/K) ∑ x_k x_kᴴ`. This module exports snapshot extraction helpers under
//! `snapshots` for that purpose.
//!
//! # Recommended defaults
//! - For MVDR/Capon, use **diagonal loading** (a.k.a. ridge regularization) for numerical stability,
//!   especially under low snapshot counts.
//! - Use `SteeringVectorMethod::SphericalWave { source_position: p }` for near-field localization
//!   unless you are explicitly in a far-field plane-wave DOA scenario.

pub mod capon;
pub mod snapshots;
pub mod steering_narrowband;

pub use capon::{
    capon_spatial_spectrum_point, capon_spatial_spectrum_point_complex_baseband,
    CaponSpectrumConfig,
};
pub use snapshots::{
    extract_complex_baseband_snapshots, extract_narrowband_snapshots, BasebandSnapshotConfig,
    SnapshotMethod, SnapshotScenario, SnapshotSelection, StftBinConfig, WindowFunction,
};
pub use steering_narrowband::{NarrowbandSteering, NarrowbandSteeringVector};
