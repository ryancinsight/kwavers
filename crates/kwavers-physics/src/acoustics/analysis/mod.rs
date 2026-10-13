//! Field Analysis and Beam Pattern Calculation Module
//!
//! This module provides comprehensive field analysis tools for beam pattern
//! calculation and field analysis functions.
//!
//! # Design Principles
//! - **SOLID**: Single responsibility for field analysis functions
//! - **DRY**: Reusable field analysis implementations
//! - **Zero-Copy**: Uses `ArrayView` and efficient iterators
//! - **KISS**: Clear, well-documented field analysis interfaces

pub mod beam_pattern;
pub mod beam_patterns;
pub mod focus;
pub mod metrics;
pub mod pressure;
mod validation;

pub use beam_pattern::{
    calculate_beam_pattern, calculate_directivity, BeamPatternConfig, FarFieldMethod,
};
pub use beam_patterns::BeamPatterns;
pub use focus::{calculate_beam_width, find_focal_plane, find_focus};
pub use metrics::{calculate_field_metrics, find_peak_pressure, FieldMetrics};
pub use pressure::{calculate_intensity, calculate_mechanical_index, calculate_thermal_index};

/// Shared peak-pressure locator used by both `focus` and `metrics`.
///
/// Scans the entire field for the index with maximum `|p|` and returns
/// the grid-coordinate triple together with the absolute pressure.
/// Does **not** validate domain shape — callers must call
/// `validation::validate_pressure_field_domain` first.
#[inline]
pub(super) fn peak_pressure_unchecked(
    pressure_field: leto::ArrayView3<f64>,
    grid: &kwavers_grid::Grid,
) -> ([f64; 3], f64) {
    let mut max_pressure = 0.0_f64;
    let mut max_index = [0_usize; 3];

    for ix in 0..grid.nx {
        for iy in 0..grid.ny {
            for iz in 0..grid.nz {
                let p = pressure_field[[ix, iy, iz]].abs();
                if p > max_pressure {
                    max_pressure = p;
                    max_index = [ix, iy, iz];
                }
            }
        }
    }

    (
        [
            grid.x(max_index[0]),
            grid.y(max_index[1]),
            grid.z(max_index[2]),
        ],
        max_pressure,
    )
}
