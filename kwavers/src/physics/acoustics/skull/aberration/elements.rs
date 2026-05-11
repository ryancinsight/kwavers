//! Phase correction extraction for planar array elements.

use std::f64::consts::PI;

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array3};

use super::model::AberrationCorrection;

impl AberrationCorrection<'_> {
    /// Compute scalar phase correction for each element of a 2D planar array.
    ///
    /// For element `l` at aperture position `(x_m, y_m)`,
    /// `phi_corr,l = -sum_z [k_skull(i,j,z) - k_water] dz`, where `(i,j)` is
    /// the nearest grid cell to the element position.
    /// # Errors
    /// - Returns [`KwaversError::DimensionMismatch`] if the precondition for mismatched array or grid dimensions is violated.
    ///
    pub fn compute_element_corrections(
        &self,
        frequency: f64,
        element_x_m: &[f64],
        element_y_m: &[f64],
    ) -> KwaversResult<Array1<f64>> {
        if element_x_m.len() != element_y_m.len() {
            return Err(KwaversError::DimensionMismatch(format!(
                "element_x_m has length {}, element_y_m has length {}",
                element_x_m.len(),
                element_y_m.len()
            )));
        }

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let k_water = 2.0 * PI * frequency / self.c_water;
        let dz = self.grid.dz;
        let mut corrections = Array1::zeros(element_x_m.len());

        for (elem_idx, (&xm, &ym)) in element_x_m.iter().zip(element_y_m.iter()).enumerate() {
            let i = ((xm / self.grid.dx).round() as isize).clamp(0, nx as isize - 1) as usize;
            let j = ((ym / self.grid.dy).round() as isize).clamp(0, ny as isize - 1) as usize;

            let mut total_phase = 0.0_f64;
            for k in 0..nz {
                let c_local = self.skull.sound_speed[[i, j, k]];
                if c_local > 0.0 {
                    let k_local = 2.0 * PI * frequency / c_local;
                    total_phase += (k_local - k_water) * dz;
                }
            }
            corrections[elem_idx] = -total_phase;
        }

        Ok(corrections)
    }

    /// Compute element phase corrections from a precomputed phase map.
    ///
    /// The correction for element `l` is `-phases[gi, gj, Nz-1]`, where
    /// `(gi,gj)` is the nearest grid cell to the element position.
    /// # Panics
    /// - Panics if an internal precondition is violated.
    ///
    #[must_use]
    pub fn element_corrections_from_map(
        &self,
        phases: &Array3<f64>,
        element_x_m: &[f64],
        element_y_m: &[f64],
    ) -> Array1<f64> {
        debug_assert_eq!(element_x_m.len(), element_y_m.len());
        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let mut corr = Array1::zeros(element_x_m.len().min(element_y_m.len()));
        for (idx, (&xm, &ym)) in element_x_m.iter().zip(element_y_m.iter()).enumerate() {
            let i = ((xm / self.grid.dx).round() as isize).clamp(0, nx as isize - 1) as usize;
            let j = ((ym / self.grid.dy).round() as isize).clamp(0, ny as isize - 1) as usize;
            corr[idx] = -phases[[i, j, nz - 1]];
        }
        corr
    }
}
