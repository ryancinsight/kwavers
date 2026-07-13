#![deny(missing_docs)]
//! Narrowband snapshot extraction (SSOT) for adaptive array processing.
//!
//! # Literature alignment (MVDR/MUSIC/ESMV)
//! Narrowband adaptive methods are mathematically native to **complex** snapshots `x_k ∈ ℂ^M` and a
//! **Hermitian** covariance:
//!
//! `R = (1/K) ∑ x_k x_kᴴ`.
//!
//! This module provides both:
//! 1. **Windowed snapshots** (preferred): deterministic, scenario-driven auto selection.
//! 2. **Legacy analytic baseband** snapshots: analytic-signal (Hilbert) + downconversion.
//!
//! # No error masking
//! Validation is strict (no silent clamp/ceil that would produce partially filled snapshots).

use eunomia::Complex64;
use kwavers_core::error::KwaversResult;
use leto::{Array2, Array3};

pub mod config;
pub mod legacy;
pub mod windowed;

pub use config::BasebandSnapshotConfig;
pub use legacy::extract_complex_baseband_snapshots;
pub use windowed::{
    extract_stft_bin_snapshots, extract_windowed_snapshots, SnapshotMethod, SnapshotScenario,
    SnapshotSelection, StftBinConfig, WindowFunction,
};

/// Narrowband snapshot extraction API (SSOT).
///
/// This function integrates the preferred windowed snapshot extraction options and provides a
/// scenario-driven auto-selection policy. It is intended to be the single entry point for narrowband
/// snapshot formation in MVDR/MUSIC/ESMV pipelines.
///
/// # Selection
/// - `SnapshotSelection::Explicit(...)`: uses exactly the provided windowed method.
/// - `SnapshotSelection::Auto(...)`: deterministically chooses a robust windowed method.
///
/// # Errors
/// Returns an error if input shapes are invalid or if the resolved method/config violates invariants.
pub fn extract_narrowband_snapshots(
    sensor_data: &Array3<f64>,
    selection: &SnapshotSelection,
) -> KwaversResult<Array2<Complex64>> {
    extract_windowed_snapshots(sensor_data, selection)
}
