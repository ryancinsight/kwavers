//! Spatial Windowing Module
//!
//! Implements spatial windowing functions for time-reversal reconstruction.

use crate::signal::window::{window_value, WindowType};
use crate::{error::KwaversResult, grid::Grid};
use ndarray::Array3;

/// Tukey window function (tapered cosine window)
#[must_use]
pub fn tukey_window(i: usize, n: usize, alpha: f64) -> f64 {
    if n <= 1 {
        return 1.0;
    }

    let x = i as f64 / (n - 1) as f64;
    window_value(WindowType::Tukey { alpha }, x)
}

/// Apply spatial windowing function to a 3D field
pub fn apply_spatial_window(
    mut field: Array3<f64>,
    grid: &Grid,
    alpha: f64,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

    // Apply Tukey window in each dimension
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let wx = tukey_window(i, nx, alpha);
                let wy = tukey_window(j, ny, alpha);
                let wz = tukey_window(k, nz, alpha);

                field[[i, j, k]] *= wx * wy * wz;
            }
        }
    }

    Ok(field)
}

/// Apply Hann window
#[must_use]
pub fn hann_window(i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    window_value(WindowType::Hann, i as f64 / (n - 1) as f64)
}

/// Apply Hamming window
#[must_use]
pub fn hamming_window(i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    window_value(WindowType::Hamming, i as f64 / (n - 1) as f64)
}

/// Apply Blackman window
#[must_use]
pub fn blackman_window(i: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    window_value(WindowType::Blackman, i as f64 / (n - 1) as f64)
}
