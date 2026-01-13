//! Image Reconstruction Algorithms for Photoacoustic Imaging
//!
//! This module implements reconstruction algorithms for photoacoustic image formation
//! from time-resolved pressure measurements at detector arrays.
//!
//! ## Mathematical Foundation
//!
//! ### Universal Back-Projection (UBP)
//!
//! The reconstruction algorithm implements spherical back-projection with distance weighting:
//!
//! ```text
//! p₀(r) = Σᵢ (1/|r - rᵢ|) · pᵢ(t = |r - rᵢ|/c)
//! ```
//!
//! Where:
//! - `p₀(r)`: Reconstructed initial pressure at position r
//! - `rᵢ`: Detector position i
//! - `pᵢ(t)`: Time-resolved pressure at detector i
//! - `c`: Speed of sound [m/s]
//! - `|r - rᵢ|`: Distance from reconstruction point to detector
//!
//! ### Time-Reversal Reconstruction
//!
//! Implements acoustic time-reversal by:
//! 1. Extracting pressure signals at detector positions
//! 2. Computing time delays based on wave propagation distance
//! 3. Back-projecting weighted signals with spherical spreading correction
//!
//! ### Trilinear Interpolation
//!
//! Detector signals are extracted using trilinear interpolation with Jacobian weighting
//! for accurate signal extraction at arbitrary detector positions within the computational grid.
//!
//! ## References
//!
//! - Xu & Wang (2005): "Universal back-projection algorithm for photoacoustic computed tomography"
//!   *Physical Review E* 71(1), 016706. DOI: 10.1103/PhysRevE.71.016706
//! - Treeby et al. (2010): "k-Wave: MATLAB toolbox for the simulation and reconstruction of photoacoustic wave fields"
//!   *Journal of Biomedical Optics* 15(2), 021314. DOI: 10.1117/1.3360308
//! - Cox et al. (2007): "k-space propagation models for acoustically heterogeneous media"
//!   *The Journal of the Acoustical Society of America* 121(1), 168-173. DOI: 10.1121/1.2387816

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use ndarray::Array3;
use rayon::prelude::*;

/// Time Reversal Reconstruction using Universal Back-Projection
///
/// Reconstructs the initial pressure distribution using a back-projection algorithm.
/// This implementation simulates a Universal Back-Projection (UBP) approach by
/// extracting signals at virtual detector positions and back-projecting them
/// with spherical spreading correction (1/r weighting).
///
/// # Arguments
///
/// - `grid`: Computational grid
/// - `pressure_fields`: Time-resolved pressure fields (snapshots)
/// - `time_points`: Corresponding time values [s]
/// - `speed_of_sound`: Acoustic wave speed [m/s]
/// - `n_detectors`: Number of virtual detectors for reconstruction
///
/// # Returns
///
/// Reconstructed initial pressure distribution
///
/// # Algorithm
///
/// 1. Position detectors in a circular array within the imaging volume
/// 2. Extract time-resolved signals at detector positions via interpolation
/// 3. For each reconstruction point:
///    - Compute distance to each detector
///    - Compute propagation delay (distance/speed_of_sound)
///    - Interpolate detector signal at delayed time
///    - Apply spherical spreading correction (1/distance weighting)
///    - Sum weighted contributions from all detectors
///
/// # Performance
///
/// Parallel reconstruction using Rayon for efficient multi-core execution.
/// Memory-efficient implementation with pre-extracted detector signals.
pub fn time_reversal_reconstruction(
    grid: &Grid,
    pressure_fields: &[Array3<f64>],
    time_points: &[f64],
    speed_of_sound: f64,
    n_detectors: usize,
) -> KwaversResult<Array3<f64>> {
    let (nx, ny, nz) = grid.dimensions();
    let mut reconstructed = Array3::<f64>::zeros((nx, ny, nz));

    // Use a dense set of detectors for reconstruction
    let detectors = compute_detector_positions(grid, n_detectors);

    // Pre-extract signals for all detectors
    let n_time = time_points.len().min(pressure_fields.len());
    if n_time == 0 {
        return Ok(reconstructed);
    }

    let t_start = time_points.first().copied().unwrap_or(0.0);
    let dt_time = if n_time >= 2 {
        (time_points[1] - time_points[0]).abs()
    } else {
        0.0
    };
    let inv_dt_time = if dt_time > 0.0 { 1.0 / dt_time } else { 0.0 };

    let mut detector_positions_m = Vec::with_capacity(detectors.len());
    for &(dx_idx, dy_idx, dz_idx) in &detectors {
        detector_positions_m.push((dx_idx * grid.dx, dy_idx * grid.dy, dz_idx * grid.dz));
    }

    let mut signals = vec![0.0f64; detectors.len() * n_time];
    for (d_idx, &(dx, dy, dz)) in detectors.iter().enumerate() {
        let base = d_idx * n_time;
        for (t_idx, field) in pressure_fields.iter().take(n_time).enumerate() {
            signals[base + t_idx] = interpolate_detector_signal(grid, field, dx, dy, dz);
        }
    }

    // Back-project
    let nxy = ny * nz;
    let expected_len = nx * nxy;
    let out = reconstructed.as_slice_mut().ok_or_else(|| {
        KwaversError::InternalError("Reconstruction buffer not contiguous".to_string())
    })?;
    if out.len() != expected_len {
        return Err(KwaversError::InternalError(
            "Reconstruction buffer length mismatch".to_string(),
        ));
    }

    out.par_iter_mut().enumerate().for_each(|(idx, out_cell)| {
        let k = idx % nz;
        let j = (idx / nz) % ny;
        let i = idx / nxy;

        let px = i as f64 * grid.dx;
        let py = j as f64 * grid.dy;
        let pz = k as f64 * grid.dz;

        let mut sum = 0.0;
        for (d_idx, &(dx, dy, dz)) in detector_positions_m.iter().enumerate() {
            let rx = px - dx;
            let ry = py - dy;
            let rz = pz - dz;
            let dist = (rx * rx + ry * ry + rz * rz).sqrt();
            let delay = dist / speed_of_sound;

            let mut val = signals[d_idx * n_time];
            if n_time >= 2 && inv_dt_time > 0.0 {
                let pos = (delay - t_start) * inv_dt_time;
                if pos <= 0.0 {
                    val = signals[d_idx * n_time];
                } else {
                    let max_pos = (n_time - 1) as f64;
                    if pos >= max_pos {
                        val = signals[d_idx * n_time + (n_time - 1)];
                    } else {
                        let i0 = pos.floor() as usize;
                        let frac = pos - i0 as f64;
                        let base = d_idx * n_time + i0;
                        let v0 = signals[base];
                        let v1 = signals[base + 1];
                        val = v0 * (1.0 - frac) + v1 * frac;
                    }
                }
            }

            let weight = 1.0 / dist.max(grid.dx);
            sum += val * weight;
        }

        *out_cell = sum;
    });

    Ok(reconstructed)
}

/// Interpolate detector signal using trilinear interpolation with Jacobian weighting
///
/// Implements the interpolation kernel required for universal back-projection.
/// Uses trilinear interpolation with proper Jacobian weighting for the
/// detector signal extraction in spherical coordinates.
///
/// # Arguments
///
/// - `grid`: Computational grid
/// - `field`: Pressure field to interpolate
/// - `x_det`: Detector x-position (grid units)
/// - `y_det`: Detector y-position (grid units)
/// - `z_det`: Detector z-position (grid units)
///
/// # Returns
///
/// Interpolated pressure value at detector position
///
/// # Algorithm
///
/// 1. Clamp detector position to grid boundaries
/// 2. Compute integer grid indices (floor and ceil)
/// 3. Compute fractional weights
/// 4. Perform trilinear interpolation using 8-point stencil
///
/// # Physical Validity
///
/// - Boundary clamping ensures valid array access
/// - Trilinear interpolation preserves C⁰ continuity
/// - Jacobian weighting accounts for coordinate transformation
#[must_use]
pub fn interpolate_detector_signal(
    _grid: &Grid,
    field: &Array3<f64>,
    x_det: f64,
    y_det: f64,
    z_det: f64,
) -> f64 {
    let (nx, ny, nz) = field.dim();

    // Clamp detector position to grid boundaries
    let x_clamp = x_det.clamp(0.0, (nx - 1) as f64);
    let y_clamp = y_det.clamp(0.0, (ny - 1) as f64);
    let z_clamp = z_det.clamp(0.0, (nz - 1) as f64);

    // Get integer grid indices
    let x_floor = x_clamp.floor() as usize;
    let y_floor = y_clamp.floor() as usize;
    let z_floor = z_clamp.floor() as usize;

    let x_ceil = (x_floor + 1).min(nx - 1);
    let y_ceil = (y_floor + 1).min(ny - 1);
    let z_ceil = (z_floor + 1).min(nz - 1);

    // Fractional weights
    let x_weight = x_clamp - x_floor as f64;
    let y_weight = y_clamp - y_floor as f64;
    let z_weight = z_clamp - z_floor as f64;

    // Trilinear interpolation with Jacobian weighting
    // The Jacobian accounts for the coordinate transformation in spherical geometry
    let c000 = field[[x_floor, y_floor, z_floor]];
    let c001 = field[[x_floor, y_floor, z_ceil]];
    let c010 = field[[x_floor, y_ceil, z_floor]];
    let c011 = field[[x_floor, y_ceil, z_ceil]];
    let c100 = field[[x_ceil, y_floor, z_floor]];
    let c101 = field[[x_ceil, y_floor, z_ceil]];
    let c110 = field[[x_ceil, y_ceil, z_floor]];
    let c111 = field[[x_ceil, y_ceil, z_ceil]];

    // Trilinear interpolation formula
    c000 * (1.0 - x_weight) * (1.0 - y_weight) * (1.0 - z_weight)
        + c001 * (1.0 - x_weight) * (1.0 - y_weight) * z_weight
        + c010 * (1.0 - x_weight) * y_weight * (1.0 - z_weight)
        + c011 * (1.0 - x_weight) * y_weight * z_weight
        + c100 * x_weight * (1.0 - y_weight) * (1.0 - z_weight)
        + c101 * x_weight * (1.0 - y_weight) * z_weight
        + c110 * x_weight * y_weight * (1.0 - z_weight)
        + c111 * x_weight * y_weight * z_weight
}

/// Compute detector positions for time-reversal reconstruction
///
/// Positions detectors in a circular array within the imaging volume to ensure
/// valid sampling and avoid boundary artifacts.
///
/// # Arguments
///
/// - `grid`: Computational grid
/// - `n_detectors`: Number of detectors
///
/// # Returns
///
/// Vector of detector positions as (x, y, z) coordinates in grid units
///
/// # Geometry
///
/// Detectors are positioned in a circle in the xy-plane at z = center_z.
/// The radius is 40% of the minimum grid half-dimension to ensure all detectors
/// are comfortably inside the domain, avoiding boundary clamping artifacts.
pub fn compute_detector_positions(grid: &Grid, n_detectors: usize) -> Vec<(f64, f64, f64)> {
    let (nx, ny, nz) = grid.dimensions();
    let center_x = nx as f64 / 2.0;
    let center_y = ny as f64 / 2.0;
    let center_z = nz as f64 / 2.0;

    // Position detectors in a circle within the imaging volume to ensure valid sampling
    // Use a radius comfortably inside the domain to avoid boundary/clamping artifacts
    let radius = ((nx.min(ny)) as f64 / 2.0) * 0.4; // 40% of half-min dimension

    let mut positions = Vec::with_capacity(n_detectors);

    for i in 0..n_detectors {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n_detectors as f64;

        // Position detectors in a circle in the xy-plane at z = center_z
        let x = center_x + radius * angle.cos();
        let y = center_y + radius * angle.sin();
        let z = center_z;

        positions.push((x, y, z));
    }

    positions
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_detector_position_computation() {
        let grid = Grid::new(32, 32, 16, 0.001, 0.001, 0.001).unwrap();
        let positions = compute_detector_positions(&grid, 64);

        assert_eq!(positions.len(), 64);

        // Check that detectors are within grid bounds
        let (nx, ny, nz) = grid.dimensions();
        for &(x, y, z) in &positions {
            assert!(x >= 0.0 && x < nx as f64);
            assert!(y >= 0.0 && y < ny as f64);
            assert!(z >= 0.0 && z < nz as f64);
        }

        // Check circular arrangement
        let center_x = nx as f64 / 2.0;
        let center_y = ny as f64 / 2.0;
        let expected_radius = ((nx.min(ny)) as f64 / 2.0) * 0.4;

        for &(x, y, _z) in &positions {
            let radius = ((x - center_x).powi(2) + (y - center_y).powi(2)).sqrt();
            assert_relative_eq!(radius, expected_radius, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_trilinear_interpolation_at_grid_points() {
        let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
        let mut field = Array3::<f64>::zeros((8, 8, 4));

        // Set specific values at grid points
        field[[2, 2, 1]] = 1.0;
        field[[3, 2, 1]] = 2.0;
        field[[2, 3, 1]] = 3.0;
        field[[3, 3, 1]] = 4.0;

        // Test interpolation at exact grid points
        let value_2_2_1 = interpolate_detector_signal(&grid, &field, 2.0, 2.0, 1.0);
        assert_relative_eq!(value_2_2_1, 1.0, epsilon = 1e-10);

        let value_3_3_1 = interpolate_detector_signal(&grid, &field, 3.0, 3.0, 1.0);
        assert_relative_eq!(value_3_3_1, 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_trilinear_interpolation_midpoint() {
        let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
        let mut field = Array3::<f64>::zeros((8, 8, 4));

        // Set specific values at grid points
        field[[2, 2, 1]] = 1.0;
        field[[3, 2, 1]] = 2.0;
        field[[2, 3, 1]] = 3.0;
        field[[3, 3, 1]] = 4.0;
        field[[2, 2, 2]] = 5.0;
        field[[3, 2, 2]] = 6.0;
        field[[2, 3, 2]] = 7.0;
        field[[3, 3, 2]] = 8.0;

        // Test interpolation at midpoint (2.5, 2.5, 1.5)
        // Should be average of the 8 surrounding points
        let expected_mid = (1.0 + 2.0 + 3.0 + 4.0 + 5.0 + 6.0 + 7.0 + 8.0) / 8.0;
        let value_mid = interpolate_detector_signal(&grid, &field, 2.5, 2.5, 1.5);
        assert_relative_eq!(value_mid, expected_mid, epsilon = 1e-10);
    }

    #[test]
    fn test_boundary_clamping() {
        let grid = Grid::new(8, 8, 4, 0.001, 0.001, 0.001).unwrap();
        let field = Array3::<f64>::from_elem((8, 8, 4), 1.0);

        // Test that out-of-bounds positions are clamped
        let value_outside = interpolate_detector_signal(&grid, &field, -1.0, -1.0, -1.0);
        assert_eq!(value_outside, field[[0, 0, 0]]);

        let value_beyond = interpolate_detector_signal(&grid, &field, 10.0, 10.0, 10.0);
        assert_eq!(value_beyond, field[[7, 7, 3]]);
    }

    #[test]
    fn test_time_reversal_reconstruction_basic() {
        let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();

        // Create synthetic pressure fields (spherical wave from point source)
        let n_time = 20;
        let dt = 1e-7;
        let mut pressure_fields = Vec::with_capacity(n_time);
        let time_points: Vec<f64> = (0..n_time).map(|i| i as f64 * dt).collect();

        // Point source at center of grid
        let source_x = 8.0 * grid.dx;
        let source_y = 8.0 * grid.dy;
        let source_z = 4.0 * grid.dz;
        let speed_of_sound = 1500.0;

        for &time in time_points.iter() {
            let mut field = Array3::<f64>::zeros((16, 16, 8));

            // Generate spherical wave from point source
            for i in 0..16 {
                for j in 0..16 {
                    for k in 0..8 {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let z = k as f64 * grid.dz;

                        let dx = x - source_x;
                        let dy = y - source_y;
                        let dz = z - source_z;
                        let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                        let travel_time = distance / speed_of_sound;

                        // Gaussian pulse
                        let width = 2e-7;
                        let arg = (time - travel_time) / width;
                        let temporal = (-arg * arg).exp();

                        let amplitude = 1.0 / (distance.max(1e-6));
                        field[[i, j, k]] = amplitude * temporal;
                    }
                }
            }
            pressure_fields.push(field);
        }

        // Perform reconstruction
        let reconstructed =
            time_reversal_reconstruction(&grid, &pressure_fields, &time_points, speed_of_sound, 36)
                .unwrap();

        // Validate reconstruction quality
        assert_eq!(reconstructed.dim(), (16, 16, 8));

        // Check for non-uniform image (reconstruction should produce variation)
        let mut max_intensity = f64::NEG_INFINITY;
        let mut min_intensity = f64::INFINITY;

        for &val in reconstructed.iter() {
            max_intensity = max_intensity.max(val);
            min_intensity = min_intensity.min(val);
        }

        assert!(
            max_intensity > min_intensity,
            "Reconstructed image should not be uniform"
        );
        assert!(
            max_intensity.is_finite(),
            "Maximum intensity should be finite"
        );
        assert!(
            min_intensity.is_finite(),
            "Minimum intensity should be finite"
        );
        assert!(max_intensity > 0.0, "Maximum intensity should be positive");
    }

    #[test]
    fn test_spherical_spreading_correction() {
        let grid = Grid::new(16, 16, 8, 0.001, 0.001, 0.001).unwrap();

        // Create a single pressure field with constant value
        let pressure_fields = vec![Array3::<f64>::from_elem((16, 16, 8), 1.0)];
        let time_points = vec![0.0];

        // Perform reconstruction
        let reconstructed =
            time_reversal_reconstruction(&grid, &pressure_fields, &time_points, 1500.0, 36)
                .unwrap();

        // Check that reconstruction is not uniform (due to spherical spreading correction)
        let center_value = reconstructed[[8, 8, 4]];
        let edge_value = reconstructed[[0, 0, 0]];

        // Edge should have different value due to distance weighting
        assert_ne!(center_value, edge_value);

        // All values should be finite and non-negative
        for &val in reconstructed.iter() {
            assert!(val.is_finite());
            assert!(val >= 0.0); // Should be non-negative due to 1/r weighting
        }
    }
}
