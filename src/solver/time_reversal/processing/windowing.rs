//! Spatial Windowing Module
//!
//! Implements spatial windowing functions for time-reversal reconstruction.

use crate::{error::KwaversResult, grid::Grid};
use ndarray::Array3;

/// Tukey window function (tapered cosine window)
pub fn tukey_window(i: usize, n: usize, alpha: f64) -> f64 {
    if n <= 1 {
        return 1.0;
    }

    let x = i as f64 / (n - 1) as f64;

    if x < alpha / 2.0 {
        // Rising edge - starts at 0 when x=0
        // w(x) = 0.5 * (1 + cos(π * (2x/α - 1)))
        // At x=0: w(0) = 0.5 * (1 + cos(-π)) = 0.5 * (1 - 1) = 0 ✓
        0.5 * (1.0 + (std::f64::consts::PI * (2.0 * x / alpha - 1.0)).cos())
    } else if x > 1.0 - alpha / 2.0 {
        // Falling edge - ends at 0 when x=1
        // w(x) = 0.5 * (1 + cos(π * (2(1-x)/α - 1)))
        // At x=1: w(1) = 0.5 * (1 + cos(-π)) = 0.5 * (1 - 1) = 0 ✓
        0.5 * (1.0 + (std::f64::consts::PI * (2.0 * (1.0 - x) / alpha - 1.0)).cos())
    } else {
        1.0
    }
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
pub fn hann_window(i: usize, n: usize) -> f64 {
    0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos())
}

/// Apply Hamming window
pub fn hamming_window(i: usize, n: usize) -> f64 {
    0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos()
}

/// Apply Blackman window
pub fn blackman_window(i: usize, n: usize) -> f64 {
    let x = i as f64 / (n - 1) as f64;
    0.42 - 0.5 * (2.0 * std::f64::consts::PI * x).cos()
        + 0.08 * (4.0 * std::f64::consts::PI * x).cos()
}
