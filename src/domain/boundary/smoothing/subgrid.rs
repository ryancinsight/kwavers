//! Subgrid Averaging for Boundary Smoothing
//!
//! Averages material properties over grid cells that intersect the boundary,
//! weighted by the volume fraction inside the domain. This is the simplest
//! and most commonly used smoothing method.
//!
//! # Algorithm
//!
//! For a grid cell intersected by the boundary:
//! 1. Calculate volume fraction `f` inside the domain (0 ≤ f ≤ 1)
//! 2. Average property: `p_smooth = f * p_inside + (1-f) * p_outside`
//!
//! For acoustic properties (impedance), harmonic averaging is more appropriate:
//! `1/Z_smooth = f/Z_inside + (1-f)/Z_outside`

use crate::core::error::KwaversResult;
use ndarray::{Array3, Zip};

/// Subgrid averaging configuration
#[derive(Debug, Clone)]
pub struct SubgridConfig {
    /// Smoothing kernel size (number of cells)
    pub kernel_size: usize,

    /// Use harmonic averaging (better for impedance)
    pub harmonic_average: bool,

    /// Minimum volume fraction to consider (avoid division by zero)
    pub min_volume_fraction: f64,
}

impl Default for SubgridConfig {
    fn default() -> Self {
        Self {
            kernel_size: 3,
            harmonic_average: false,
            min_volume_fraction: 1e-6,
        }
    }
}

/// Subgrid averaging smoother
#[derive(Debug, Clone)]
pub struct SubgridAveraging {
    config: SubgridConfig,
}

impl SubgridAveraging {
    pub fn new(config: SubgridConfig) -> Self {
        Self { config }
    }

    /// Apply subgrid averaging to smooth boundary
    ///
    /// # Arguments
    ///
    /// * `property` - Material property field
    /// * `geometry` - Volume fraction (1.0 inside, 0.0 outside, 0-1 at boundary)
    pub fn apply(
        &self,
        property: &Array3<f64>,
        geometry: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = property.dim();
        let mut smoothed = property.clone();

        let half_kernel = self.config.kernel_size / 2;

        // Apply smoothing at boundary cells (where geometry is between 0 and 1)
        Zip::indexed(geometry).for_each(|(i, j, k), &vol_frac| {
            // Only smooth cells near boundary (partial volume fraction)
            if vol_frac > self.config.min_volume_fraction
                && vol_frac < (1.0 - self.config.min_volume_fraction)
            {
                // Average over neighboring cells
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                for di in 0..self.config.kernel_size {
                    for dj in 0..self.config.kernel_size {
                        for dk in 0..self.config.kernel_size {
                            let ii = (i as isize + di as isize - half_kernel as isize)
                                .max(0)
                                .min((nx - 1) as isize)
                                as usize;
                            let jj = (j as isize + dj as isize - half_kernel as isize)
                                .max(0)
                                .min((ny - 1) as isize)
                                as usize;
                            let kk = (k as isize + dk as isize - half_kernel as isize)
                                .max(0)
                                .min((nz - 1) as isize)
                                as usize;

                            let geom_weight = geometry[[ii, jj, kk]];
                            let prop_val = property[[ii, jj, kk]];

                            if self.config.harmonic_average {
                                // Harmonic averaging (better for impedance)
                                if prop_val.abs() > 1e-10 {
                                    sum += geom_weight / prop_val;
                                    weight_sum += geom_weight;
                                }
                            } else {
                                // Arithmetic averaging
                                sum += geom_weight * prop_val;
                                weight_sum += geom_weight;
                            }
                        }
                    }
                }

                if weight_sum > self.config.min_volume_fraction {
                    smoothed[[i, j, k]] = if self.config.harmonic_average {
                        weight_sum / sum
                    } else {
                        sum / weight_sum
                    };
                }
            }
        });

        Ok(smoothed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::{s, Array3};

    #[test]
    fn test_subgrid_averaging_no_boundary() {
        let config = SubgridConfig::default();
        let smoother = SubgridAveraging::new(config);

        // Uniform property, all inside (geometry = 1.0)
        let property = Array3::from_elem((10, 10, 10), 1540.0);
        let geometry = Array3::from_elem((10, 10, 10), 1.0);

        let result = smoother.apply(&property, &geometry);
        assert!(result.is_ok());

        let smoothed = result.unwrap();
        // Should be unchanged (no boundary cells)
        assert_relative_eq!(smoothed[[5, 5, 5]], 1540.0, epsilon = 1e-9);
    }

    #[test]
    fn test_subgrid_averaging_at_boundary() {
        let config = SubgridConfig {
            kernel_size: 3,
            harmonic_average: false,
            min_volume_fraction: 1e-6,
        };
        let smoother = SubgridAveraging::new(config);

        // Create a simple boundary scenario
        let mut property = Array3::from_elem((10, 10, 10), 1540.0);
        property.slice_mut(s![5.., .., ..]).fill(3000.0); // Different property outside

        let mut geometry = Array3::from_elem((10, 10, 10), 1.0);
        geometry.slice_mut(s![5.., .., ..]).fill(0.0); // Outside domain
        geometry[[5, 5, 5]] = 0.5; // Boundary cell (50% volume fraction)

        let result = smoother.apply(&property, &geometry);
        assert!(result.is_ok());

        let smoothed = result.unwrap();
        // Boundary cell should be smoothed (averaged value)
        let smoothed_val = smoothed[[5, 5, 5]];
        assert!(
            smoothed_val > 1540.0 && smoothed_val < 3000.0,
            "Boundary cell should be between inside and outside values"
        );
    }

    #[test]
    fn test_harmonic_averaging() {
        let config = SubgridConfig {
            kernel_size: 3,
            harmonic_average: true,
            min_volume_fraction: 1e-6,
        };
        let smoother = SubgridAveraging::new(config);

        let mut property = Array3::from_elem((10, 10, 10), 1000.0);
        property.slice_mut(s![5.., .., ..]).fill(2000.0);

        let mut geometry = Array3::from_elem((10, 10, 10), 1.0);
        geometry.slice_mut(s![5.., .., ..]).fill(0.0);
        geometry[[5, 5, 5]] = 0.5;

        let result = smoother.apply(&property, &geometry);
        assert!(result.is_ok());

        // Harmonic average should be closer to lower value
        let smoothed_val = result.unwrap()[[5, 5, 5]];
        let arithmetic_avg = (1000.0 + 2000.0) / 2.0; // 1500
        assert!(
            smoothed_val < arithmetic_avg,
            "Harmonic average should be less than arithmetic average"
        );
    }
}
