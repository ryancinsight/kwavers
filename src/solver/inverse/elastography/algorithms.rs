//! Shared Algorithms for Elastography Inversion
//!
//! Common utility algorithms used across multiple inversion methods, including
//! spatial smoothing, boundary filling, and feature detection.

use crate::domain::grid::Grid;
use crate::physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;
use ndarray::Array3;

/// Apply 3D spatial smoothing to reduce noise in speed estimates
///
/// Uses a 3×3×3 averaging filter to reduce noise while preserving edges.
///
/// # Arguments
///
/// * `speed_field` - Mutable reference to speed field to smooth
///
/// # Algorithm
///
/// Simple box filter: Each voxel is replaced by the average of its 27-neighborhood
/// (3×3×3 cube centered on the voxel).
///
/// # References
///
/// - Gonzalez & Woods (2008). "Digital Image Processing", Chapter 5
pub fn spatial_smoothing(speed_field: &mut Array3<f64>) {
    let (nx, ny, nz) = speed_field.dim();
    let mut smoothed = speed_field.clone();

    // Simple 3x3x3 averaging filter
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let mut sum = 0.0;
                let mut count = 0;

                // Average over 3x3x3 neighborhood
                for di in -1..=1 {
                    for dj in -1..=1 {
                        for dk in -1..=1 {
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;

                            if ii < nx && jj < ny && kk < nz {
                                sum += speed_field[[ii, jj, kk]];
                                count += 1;
                            }
                        }
                    }
                }

                if count > 0 {
                    smoothed[[i, j, k]] = sum / count as f64;
                }
            }
        }
    }

    *speed_field = smoothed;
}

/// Apply volumetric smoothing with edge-preserving characteristics
///
/// More sophisticated than basic spatial smoothing, preserves sharp boundaries
/// between tissue types while reducing noise.
///
/// # Arguments
///
/// * `speed_field` - Mutable reference to speed field to smooth
///
/// # Algorithm
///
/// Weighted averaging that reduces smoothing across large gradients to preserve
/// tissue boundaries.
///
/// # References
///
/// - Tomasi & Manduchi (1998). "Bilateral Filtering for Gray and Color Images"
pub fn volumetric_smoothing(speed_field: &mut Array3<f64>) {
    let (nx, ny, nz) = speed_field.dim();
    let mut smoothed = speed_field.clone();

    // Edge-preserving smoothing with adaptive kernel
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let center = speed_field[[i, j, k]];
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                // Weighted average based on similarity to center value
                for di in -1..=1 {
                    for dj in -1..=1 {
                        for dk in -1..=1 {
                            let ii = (i as i32 + di) as usize;
                            let jj = (j as i32 + dj) as usize;
                            let kk = (k as i32 + dk) as usize;

                            if ii < nx && jj < ny && kk < nz {
                                let neighbor = speed_field[[ii, jj, kk]];
                                // Weight decreases with difference from center
                                let diff = (neighbor - center).abs();
                                let weight = (-diff / 1.0).exp(); // Gaussian-like weighting
                                sum += neighbor * weight;
                                weight_sum += weight;
                            }
                        }
                    }
                }

                if weight_sum > 0.0 {
                    smoothed[[i, j, k]] = sum / weight_sum;
                }
            }
        }
    }

    *speed_field = smoothed;
}

/// Apply directional smoothing based on wave propagation patterns
///
/// Smooths preferentially along coordinate directions, which typically align
/// with wave propagation directions in shear wave elastography.
///
/// # Arguments
///
/// * `speed_field` - Mutable reference to speed field to smooth
///
/// # Algorithm
///
/// Directional weighted average that favors smoothing along coordinate axes
/// (typical wave propagation directions) while preserving cross-directional features.
///
/// # References
///
/// - Perona & Malik (1990). "Scale-space and edge detection using anisotropic diffusion"
pub fn directional_smoothing(speed_field: &mut Array3<f64>) {
    let (nx, ny, nz) = speed_field.dim();
    let mut smoothed = speed_field.clone();

    // Directional smoothing along likely wave propagation paths
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                // Weighted average favoring values along coordinate axes
                let center = speed_field[[i, j, k]];
                let x_dir = (speed_field[[i - 1, j, k]] + speed_field[[i + 1, j, k]]) / 2.0;
                let y_dir = (speed_field[[i, j - 1, k]] + speed_field[[i, j + 1, k]]) / 2.0;
                let z_dir = (speed_field[[i, j, k - 1]] + speed_field[[i, j, k + 1]]) / 2.0;

                // Combine with directional weighting (40% center, 20% each direction)
                smoothed[[i, j, k]] =
                    (center * 0.4 + x_dir * 0.2 + y_dir * 0.2 + z_dir * 0.2).clamp(0.5, 10.0);
            }
        }
    }

    *speed_field = smoothed;
}

/// Fill boundary values with nearest interior values
///
/// Propagates interior values to domain boundaries using nearest-neighbor
/// extrapolation.
///
/// # Arguments
///
/// * `array` - Mutable reference to array with boundaries to fill
///
/// # Algorithm
///
/// Boundary faces are filled by copying adjacent interior slice values:
/// - Faces (6 total): Copy from adjacent interior layer
/// - Edges and corners inherit from filled faces
///
/// # References
///
/// - Numerical Recipes (2007). "Boundary Conditions", Chapter 19
pub fn fill_boundaries(array: &mut Array3<f64>) {
    let (nx, ny, nz) = array.dim();

    // Fill i=0 and i=nx-1 faces
    for k in 0..nz {
        for j in 0..ny {
            array[[0, j, k]] = array[[1, j, k]];
            array[[nx - 1, j, k]] = array[[nx - 2, j, k]];
        }
    }

    // Fill j=0 and j=ny-1 faces
    for k in 0..nz {
        for i in 0..nx {
            array[[i, 0, k]] = array[[i, 1, k]];
            array[[i, ny - 1, k]] = array[[i, ny - 2, k]];
        }
    }

    // Fill k=0 and k=nz-1 faces
    for j in 0..ny {
        for i in 0..nx {
            array[[i, j, 0]] = array[[i, j, 1]];
            array[[i, j, nz - 1]] = array[[i, j, nz - 2]];
        }
    }
}

/// Find multiple push locations in the displacement field
///
/// Identifies local maxima in the displacement field that correspond to
/// acoustic radiation force push locations.
///
/// # Arguments
///
/// * `displacement` - Displacement field to analyze
/// * `grid` - Computational grid for coordinate conversion
///
/// # Returns
///
/// Vector of 3D coordinates [x, y, z] of detected push locations
///
/// # Algorithm
///
/// 1. Compute threshold as 30% of maximum displacement
/// 2. Find local maxima above threshold using 27-neighborhood comparison
/// 3. Convert voxel indices to physical coordinates
///
/// # References
///
/// - Bercoff et al. (2004). "Supersonic shear imaging"
/// - Palmeri et al. (2008). "Quantifying hepatic shear modulus in vivo"
pub fn find_push_locations(displacement: &DisplacementField, grid: &Grid) -> Vec<[f64; 3]> {
    let (nx, ny, nz) = displacement.uz.dim();
    let mut locations = Vec::new();
    let threshold = displacement.uz.iter().cloned().fold(0.0, f64::max) * 0.3; // 30% of max

    // Simple peak finding (could be enhanced with more sophisticated algorithms)
    for i in 1..nx - 1 {
        for j in 1..ny - 1 {
            for k in 1..nz - 1 {
                let val = displacement.uz[[i, j, k]].abs();
                if val > threshold {
                    // Check if it's a local maximum in 27-neighborhood
                    let mut is_local_max = true;
                    'neighbor_loop: for di in -1..=1 {
                        for dj in -1..=1 {
                            for dk in -1..=1 {
                                if di == 0 && dj == 0 && dk == 0 {
                                    continue;
                                }
                                let ii = (i as i32 + di) as usize;
                                let jj = (j as i32 + dj) as usize;
                                let kk = (k as i32 + dk) as usize;

                                if ii < nx
                                    && jj < ny
                                    && kk < nz
                                    && displacement.uz[[ii, jj, kk]].abs() > val
                                {
                                    is_local_max = false;
                                    break 'neighbor_loop;
                                }
                            }
                        }
                    }

                    if is_local_max {
                        locations.push([
                            i as f64 * grid.dx,
                            j as f64 * grid.dy,
                            k as f64 * grid.dz,
                        ]);
                    }
                }
            }
        }
    }

    locations
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_spatial_smoothing() {
        let mut field = Array3::from_elem((10, 10, 10), 3.0);
        field[[5, 5, 5]] = 10.0; // Spike

        spatial_smoothing(&mut field);

        // Center spike should be reduced
        assert!(field[[5, 5, 5]] < 10.0);
        assert!(field[[5, 5, 5]] > 3.0);
    }

    #[test]
    fn test_volumetric_smoothing_preserves_uniform() {
        let mut field = Array3::from_elem((10, 10, 10), 5.0);
        let original = field.clone();

        volumetric_smoothing(&mut field);

        // Uniform field should remain uniform
        for i in 1..9 {
            for j in 1..9 {
                for k in 1..9 {
                    assert!((field[[i, j, k]] - original[[i, j, k]]).abs() < 0.1);
                }
            }
        }
    }

    #[test]
    fn test_directional_smoothing() {
        let mut field = Array3::from_elem((10, 10, 10), 3.0);
        field[[5, 5, 5]] = 8.0; // Local variation

        directional_smoothing(&mut field);

        // Should smooth but clamp to valid range
        assert!(field[[5, 5, 5]] >= 0.5);
        assert!(field[[5, 5, 5]] <= 10.0);
    }

    #[test]
    fn test_fill_boundaries() {
        let mut array = Array3::zeros((10, 10, 10));
        array[[5, 5, 5]] = 42.0;

        fill_boundaries(&mut array);

        // Check that boundaries are filled
        assert_ne!(array[[0, 5, 5]], 0.0);
        assert_ne!(array[[9, 5, 5]], 0.0);
        assert_ne!(array[[5, 0, 5]], 0.0);
        assert_ne!(array[[5, 9, 5]], 0.0);
        assert_ne!(array[[5, 5, 0]], 0.0);
        assert_ne!(array[[5, 5, 9]], 0.0);
    }

    #[test]
    fn test_find_push_locations_empty() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let displacement = DisplacementField::zeros(20, 20, 20);

        let locations = find_push_locations(&displacement, &grid);

        // Zero field should have no peaks
        assert!(locations.is_empty());
    }

    #[test]
    fn test_find_push_locations_single_peak() {
        let grid = Grid::new(20, 20, 20, 0.001, 0.001, 0.001).unwrap();
        let mut displacement = DisplacementField::zeros(20, 20, 20);
        displacement.uz[[10, 10, 10]] = 10.0; // Single peak

        let locations = find_push_locations(&displacement, &grid);

        // Should find one peak
        assert_eq!(locations.len(), 1);
        assert!((locations[0][0] - 10.0 * grid.dx).abs() < 1e-9);
        assert!((locations[0][1] - 10.0 * grid.dy).abs() < 1e-9);
        assert!((locations[0][2] - 10.0 * grid.dz).abs() < 1e-9);
    }

    #[test]
    fn test_smoothing_preserves_positivity() {
        let mut field = Array3::from_elem((10, 10, 10), 2.0);
        field[[5, 5, 5]] = 5.0;

        spatial_smoothing(&mut field);

        // All values should remain positive
        for &val in field.iter() {
            assert!(val > 0.0);
        }
    }
}
