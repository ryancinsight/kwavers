//! Inherent impl methods for [`ReverseTimeMigration`]: top-level entry points.
//!
//! Delegates to focused submodules:
//! - [`propagation`]   – forward / backward / wavefield reconstruction
//! - [`wavefield`]     – single finite-difference time step (4th-order FD Laplacian)
//! - [`imaging`]       – apply_imaging_condition (all six k-Wave variants)
//! - [`laplacian`]     – 2nd-order spatial Laplacian and Laplacian filter
//! - [`illumination`]  – source illumination accumulation
//!
//! All triple-nested-for-loop stencils have been replaced with
//! `Zip::par_for_each` slice-view passes (see individual submodules).

mod illumination;
mod imaging;
mod laplacian;
mod propagation;
pub(super) mod tests;
mod wavefield;

use crate::core::error::KwaversResult;
use ndarray::{Array3, Zip};

use super::types::ReverseTimeMigration;

impl ReverseTimeMigration {
    /// Perform RTM for a single shot gather.
    ///
    /// # Algorithm
    /// 1. Forward propagation of source wavefield (stored, decimated).
    /// 2. Time-reversed backward propagation of receiver data.
    /// 3. Apply the configured imaging condition.
    /// 4. Accumulate source illumination.
    ///
    /// Reference: Baysal et al. (1983), "Reverse time migration",
    /// *Geophysics* **48**(11), 1514–1524.
    pub fn migrate_shot(
        &mut self,
        shot_data: &ndarray::Array2<f64>,
        source_position: (usize, usize, usize),
        receiver_positions: &[(usize, usize, usize)],
        grid: &crate::domain::grid::Grid,
    ) -> KwaversResult<()> {
        let n_time_steps = shot_data.shape()[1];

        let source_wavefield = self.forward_propagation(source_position, grid, n_time_steps)?;

        let receiver_wavefield =
            self.backward_propagation(shot_data, receiver_positions, grid, n_time_steps)?;

        self.apply_imaging_condition(&source_wavefield, &receiver_wavefield)?;

        self.update_source_illumination(&source_wavefield)?;

        Ok(())
    }

    /// Apply post-processing to the migrated image.
    ///
    /// Normalises by source illumination (`√illumination`) and optionally
    /// applies a mild Laplacian filter to suppress migration artefacts.
    pub fn post_process_image(&mut self) -> KwaversResult<()> {
        use super::super::constants::RTM_AMPLITUDE_THRESHOLD;

        Zip::from(&mut self.image)
            .and(&self.source_illumination)
            .for_each(|img, &illum| {
                if illum > RTM_AMPLITUDE_THRESHOLD {
                    *img /= illum.sqrt();
                }
            });

        if self.config.base_config.filter != crate::solver::reconstruction::FilterType::None {
            let filtered = self.apply_laplacian_filter_inplace(&self.image.clone())?;
            self.image.assign(&filtered);
        }

        Ok(())
    }

    /// Thin wrapper: calls [`laplacian::apply_laplacian_filter_inplace`].
    pub(super) fn apply_laplacian_filter_inplace(
        &self,
        image: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let laplacian = self.compute_laplacian(image)?;
        Ok(image - &(0.1_f64 * laplacian))
    }
}
