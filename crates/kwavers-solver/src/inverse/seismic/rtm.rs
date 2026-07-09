//! Reverse Time Migration Implementation
//!
//! RTM algorithm implementation following GRASP principles
//! Reference: Baysal et al. (1983): "Reverse time migration"

use super::parameters::{ImagingCondition, RtmSettings};
use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use leto::Array3;

/// Reverse Time Migration processor
/// Follows Single Responsibility Principle - only handles RTM computations
#[derive(Debug)]
pub struct RtmProcessor {
    settings: RtmSettings,
}

impl RtmProcessor {
    /// Create new RTM processor with specified settings
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(settings: RtmSettings) -> Self {
        Self { settings }
    }

    /// Compute the RTM image for a single pair of pre-accumulated snapshot fields.
    ///
    /// # Contract
    ///
    /// `source_wavefield` and `receiver_wavefield` must be 3-D arrays of shape
    /// `(nx, ny, nz)` representing the *pre-accumulated* forward and backward
    /// pressure fields respectively.  Each field is typically the result of
    /// summing per-time-step snapshots externally before calling this function,
    /// or the single snapshot of a time-domain forward/backward solver at the
    /// desired correlation lag.
    ///
    /// For full time-domain multi-shot RTM with internally driven propagation,
    /// use [`ReverseTimeMigration::migrate_shot`] from the
    /// `reconstruction::seismic::rtm` module, which performs 4th-order FD
    /// propagation and time-step accumulation internally.
    ///
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// Reference: Baysal et al. (1983), *Geophysics* **48**(11), 1514–1524.
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

    /// Apply zero-lag cross-correlation imaging condition.
    ///
    /// Accumulates `S(x)·R(x)` into the image buffer (`+=`), consistent with
    /// the multi-shot definition `I(x) = Σ_{shots} S(x)·R(x)`.
    ///
    /// For a freshly initialised zero buffer (as produced by [`migrate`]) a
    /// single call is equivalent to assignment; using `+=` makes the operation
    /// compositionally correct if the buffer is pre-loaded or reused.
    ///
    /// [`migrate`]: RtmProcessor::migrate
    fn apply_zero_lag_correlation(
        &self,
        image: &mut Array3<f64>,
        source_wavefield: &Array3<f64>,
        receiver_wavefield: &Array3<f64>,
    ) {
        if let (Some(image), Some(source), Some(receiver)) = (
            image.as_slice_mut(),
            source_wavefield.as_slice(),
            receiver_wavefield.as_slice(),
        ) {
            enumerate_mut_with::<Adaptive, _, _>(image, |index, img| {
                *img += source[index] * receiver[index];
            });
        } else {
            Zip::from(image)
                .and(source_wavefield)
                .and(receiver_wavefield)
                .for_each(|img, &src, &rcv| {
                    *img += src * rcv;
                });
        }
    }

    /// Apply source-illumination-compensated imaging condition.
    ///
    /// # Mathematical specification
    ///
    /// For pre-accumulated snapshot inputs `S(x)` (forward) and `R(x)` (backward):
    ///
    /// ```text
    /// I_norm(x) = S(x)·R(x) / (S²(x) + ε)
    /// ```
    ///
    /// This is the single-snapshot form of the multi-shot accumulation:
    ///
    /// ```text
    /// I_norm(x) = [Σ_t S(x,t)·R(x,t)] / [Σ_t S²(x,t) + ε]
    /// ```
    ///
    /// Source-only denominator (not `√(Σ S² · Σ R²)`) follows Zhang & Sun (2009):
    /// normalising by source illumination preserves reflectivity amplitude while
    /// compensating for uneven illumination.  The `√(Σ S² · Σ R²)` denominator
    /// collapses the image to normalised cross-correlation (cos θ ∈ [−1,1]) and
    /// loses all amplitude information — that is a coherence measure, not an
    /// image.
    ///
    /// Reference: Zhang, Y. & Sun, J. (2009).  Practical issues in reverse time
    /// migration: true amplitude gathers, noise removal and harmonic-source
    /// encoding.  *The Leading Edge* **28**(4), 446–452.
    ///
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn apply_normalized_correlation(
        &self,
        image: &mut Array3<f64>,
        source_wavefield: &Array3<f64>,
        receiver_wavefield: &Array3<f64>,
    ) -> KwaversResult<()> {
        // I_norm(x) = S(x)·R(x) / (Φ(x) + ε)
        if let (Some(image), Some(source), Some(receiver)) = (
            image.as_slice_mut(),
            source_wavefield.as_slice(),
            receiver_wavefield.as_slice(),
        ) {
            enumerate_mut_with::<Adaptive, _, _>(image, |index, img| {
                let src = source[index];
                let phi = src * src;
                *img = if phi > f64::EPSILON {
                    src * receiver[index] / phi
                } else {
                    0.0
                };
            });
        } else {
            Zip::from(image)
                .and(source_wavefield)
                .and(receiver_wavefield)
                .for_each(|img, &src, &rcv| {
                    let phi = src * src;
                    *img = if phi > f64::EPSILON {
                        src * rcv / phi
                    } else {
                        0.0
                    };
                });
        }

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
        let d2_dx2 = 2.0f64.mul_add(
            -field[[i, j, k]],
            field[[i + 1, j, k]] + field[[i - 1, j, k]],
        ) / (grid.dx * grid.dx);
        let d2_dy2 = 2.0f64.mul_add(
            -field[[i, j, k]],
            field[[i, j + 1, k]] + field[[i, j - 1, k]],
        ) / (grid.dy * grid.dy);
        let d2_dz2 = 2.0f64.mul_add(
            -field[[i, j, k]],
            field[[i, j, k + 1]] + field[[i, j, k - 1]],
        ) / (grid.dz * grid.dz);

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
    use kwavers_grid::Grid;
    use leto::Array3;

    /// Zero-lag imaging condition: I(x) = S(x)·R(x).
    ///
    /// For S=1, R=1: I = 1.  Image shape must match grid dimensions.
    #[test]
    fn zero_lag_unit_fields_produces_unit_image() {
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        let processor = RtmProcessor::default(); // default: ImagingCondition::ZeroLag

        let source_field = Array3::ones((10, 10, 10));
        let receiver_field = Array3::ones((10, 10, 10));

        let image = processor
            .migrate(&source_field, &receiver_field, &grid)
            .unwrap();

        assert_eq!(image.dim(), (10, 10, 10));
        // I(x) = 1·1 = 1 everywhere.
        assert!(
            image.iter().all(|&v| (v - 1.0).abs() < f64::EPSILON),
            "centre = {}",
            image[[5, 5, 5]]
        );
    }

    /// Normalized (source-illumination-compensated) imaging condition.
    ///
    /// Formula: I_norm(x) = S(x)·R(x) / (S²(x) + ε)
    ///
    /// Test case:
    ///   S = 3.0, R = 6.0  →  I_norm = 3·6 / 9 = 18/9 = 2.0
    ///
    /// Verifies amplitude is preserved (not collapsed to ±1 as incorrect NCC
    /// would produce) and that the illumination compensation correctly removes
    /// the source amplitude envelope.
    #[test]
    fn normalized_condition_gives_illumination_compensated_reflectivity() {
        let settings = RtmSettings {
            imaging_condition: ImagingCondition::Normalized,
            ..RtmSettings::default()
        };
        let processor = RtmProcessor::new(settings);
        let grid = Grid::new(5, 5, 5, 1e-3, 1e-3, 1e-3).unwrap();

        let source_field = Array3::from_elem((5, 5, 5), 3.0_f64);
        let receiver_field = Array3::from_elem((5, 5, 5), 6.0_f64);

        let image = processor
            .migrate(&source_field, &receiver_field, &grid)
            .unwrap();

        // I_norm = 3*6 / (3*3) = 18/9 = 2.0 exactly (phi = 9 >> f64::EPSILON)
        assert!(
            image.iter().all(|&v| (v - 2.0).abs() < 1e-12),
            "expected 2.0 everywhere, centre = {}",
            image[[2, 2, 2]]
        );
    }

    /// 2nd-order Laplacian at interior point with unit spacing.
    ///
    /// Field: centre = 6, each of 6 face-neighbours = 1.
    /// Expected per axis: (1 + 1 − 2·6) / 1² = −10.
    /// Total: −30.
    #[test]
    fn laplacian_3d_stencil_matches_analytical_value() {
        let processor = RtmProcessor::default();
        let grid = Grid::new(5, 5, 5, 1.0, 1.0, 1.0).unwrap();

        let mut field = Array3::zeros((5, 5, 5));
        field[[2, 2, 2]] = 6.0;
        field[[1, 2, 2]] = 1.0;
        field[[3, 2, 2]] = 1.0;
        field[[2, 1, 2]] = 1.0;
        field[[2, 3, 2]] = 1.0;
        field[[2, 2, 1]] = 1.0;
        field[[2, 2, 3]] = 1.0;

        let lap = processor.compute_laplacian_3d(&field, 2, 2, 2, &grid);
        // (1+1−12) + (1+1−12) + (1+1−12) = −10 − 10 − 10 = −30
        assert!((lap + 30.0).abs() < f64::EPSILON);
    }
}
