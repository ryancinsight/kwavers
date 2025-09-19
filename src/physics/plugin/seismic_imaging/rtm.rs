//! Reverse Time Migration Implementation
//!
//! RTM algorithm implementation following GRASP principles
//! Reference: Baysal et al. (1983): "Reverse time migration"

use super::parameters::{ImagingCondition, RtmSettings};
use crate::error::KwaversResult;
use crate::grid::Grid;
use ndarray::Array3;

/// Reverse Time Migration processor
/// Follows Single Responsibility Principle - only handles RTM computations
#[derive(Debug)]
pub struct RtmProcessor {
    settings: RtmSettings,
}

impl RtmProcessor {
    /// Create new RTM processor with specified settings
    #[must_use]
    pub fn new(settings: RtmSettings) -> Self {
        Self { settings }
    }

    /// Perform Reverse Time Migration
    /// Based on Baysal et al. (1983): "Reverse time migration"
    pub fn migrate(
        &self,
        source_wavefield: &Array3<f64>,
        receiver_wavefield: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Initialize image with grid dimensions
        let mut image = Array3::zeros(grid.dimensions());

        // Apply imaging condition
        match self.settings.imaging_condition {
            ImagingCondition::ZeroLag => {
                self.apply_zero_lag_correlation(&mut image, source_wavefield, receiver_wavefield);
            }
            ImagingCondition::Normalized => {
                self.apply_normalized_correlation(
                    &mut image,
                    source_wavefield,
                    receiver_wavefield,
                )?;
            }
        }

        // Apply post-processing filters
        if self.settings.apply_laplacian {
            self.apply_laplacian_filter(&mut image, grid);
        }

        Ok(image)
    }

    /// Apply zero-lag cross-correlation imaging condition
    /// I(x) = ∫ S(x,t) * R(x,t) dt
    fn apply_zero_lag_correlation(
        &self,
        image: &mut Array3<f64>,
        source_wavefield: &Array3<f64>,
        receiver_wavefield: &Array3<f64>,
    ) {
        use ndarray::Zip;

        Zip::from(image)
            .and(source_wavefield)
            .and(receiver_wavefield)
            .for_each(|img, &src, &rcv| {
                *img = src * rcv;
            });
    }

    /// Apply normalized cross-correlation imaging condition
    /// I(x) = ∫ S(x,t) * R(x,t) dt / √(∫ S²(x,t) dt * ∫ R²(x,t) dt)
    fn apply_normalized_correlation(
        &self,
        image: &mut Array3<f64>,
        source_wavefield: &Array3<f64>,
        receiver_wavefield: &Array3<f64>,
    ) -> KwaversResult<()> {
        use ndarray::Zip;

        let mut source_energy = Array3::zeros(source_wavefield.dim());
        let mut receiver_energy = Array3::zeros(receiver_wavefield.dim());

        // Compute energy terms
        Zip::from(&mut source_energy)
            .and(source_wavefield)
            .for_each(|energy, &src| *energy = src * src);

        Zip::from(&mut receiver_energy)
            .and(receiver_wavefield)
            .for_each(|energy, &rcv| *energy = rcv * rcv);

        // Apply normalized correlation
        Zip::from(image)
            .and(source_wavefield)
            .and(receiver_wavefield)
            .and(&source_energy)
            .and(&receiver_energy)
            .for_each(|img, &src, &rcv, &src_energy, &rcv_energy| {
                let normalization = (src_energy * rcv_energy).sqrt();
                *img = if normalization > f64::EPSILON {
                    src * rcv / normalization
                } else {
                    0.0
                };
            });

        Ok(())
    }

    /// Apply Laplacian filter for artifact suppression
    /// Removes low-wavenumber artifacts using discrete Laplacian operator
    fn apply_laplacian_filter(&self, image: &mut Array3<f64>, grid: &Grid) {
        let laplacian_weight = 0.1;
        let mut filtered = image.clone();

        for k in 1..grid.nz - 1 {
            for j in 1..grid.ny - 1 {
                for i in 1..grid.nx - 1 {
                    let laplacian = self.compute_laplacian_3d(image, i, j, k, grid);
                    filtered[[i, j, k]] += laplacian_weight * laplacian;
                }
            }
        }

        *image = filtered;
    }

    /// Compute 3D Laplacian at grid point (i,j,k)
    #[must_use]
    fn compute_laplacian_3d(
        &self,
        field: &Array3<f64>,
        i: usize,
        j: usize,
        k: usize,
        grid: &Grid,
    ) -> f64 {
        let d2_dx2 = (field[[i + 1, j, k]] + field[[i - 1, j, k]] - 2.0 * field[[i, j, k]])
            / (grid.dx * grid.dx);
        let d2_dy2 = (field[[i, j + 1, k]] + field[[i, j - 1, k]] - 2.0 * field[[i, j, k]])
            / (grid.dy * grid.dy);
        let d2_dz2 = (field[[i, j, k + 1]] + field[[i, j, k - 1]] - 2.0 * field[[i, j, k]])
            / (grid.dz * grid.dz);

        d2_dx2 + d2_dy2 + d2_dz2
    }
}

impl Default for RtmProcessor {
    fn default() -> Self {
        Self::new(RtmSettings::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Grid;

    #[test]
    fn test_rtm_basic_functionality() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let processor = RtmProcessor::default();

        let source_field = Array3::ones((10, 10, 10));
        let receiver_field = Array3::ones((10, 10, 10));

        let result = processor.migrate(&source_field, &receiver_field, &grid);
        assert!(result.is_ok());

        let image = result.unwrap();
        assert_eq!(image.dim(), (10, 10, 10));

        // For unit fields, zero-lag correlation should produce unit image
        assert!((image[[5, 5, 5]] - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_laplacian_computation() {
        let processor = RtmProcessor::default();
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0).unwrap();

        // Create a field with known Laplacian
        let mut field = Array3::zeros((5, 5, 5));
        field[[2, 2, 2]] = 6.0; // Center point
        field[[1, 2, 2]] = 1.0; // Neighbors
        field[[3, 2, 2]] = 1.0;
        field[[2, 1, 2]] = 1.0;
        field[[2, 3, 2]] = 1.0;
        field[[2, 2, 1]] = 1.0;
        field[[2, 2, 3]] = 1.0;

        let laplacian = processor.compute_laplacian_3d(&field, 2, 2, 2, &grid);

        // Expected: (1+1-2*6) + (1+1-2*6) + (1+1-2*6) = -10 + -10 + -10 = -30
        assert!((laplacian + 30.0).abs() < f64::EPSILON);
    }
}
