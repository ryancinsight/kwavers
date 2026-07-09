//! WENO5 (fifth-order) shock limiting implementation.

use super::types::WENOLimiter;
use kwavers_core::error::KwaversResult;
use leto::Array3;

impl WENOLimiter {
    /// WENO5 limiting implementation
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn weno5_limit(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();

        // Process each direction
        for direction in 0..3 {
            match direction {
                0 => self.weno5_limit_x(field, shock_indicator, nx, ny, nz)?,
                1 => self.weno5_limit_y(field, shock_indicator, nx, ny, nz)?,
                2 => self.weno5_limit_z(field, shock_indicator, nx, ny, nz)?,
                _ => continue,
            }
        }

        Ok(())
    }

    /// WENO5 limiting in x-direction
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn weno5_limit_x(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                i >= 2 && i < nx - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();

        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i.saturating_sub(2), j, k]],
                field[[i.saturating_sub(1), j, k]],
                field[[i, j, k]],
                field[[i.min(nx - 1).saturating_add(1), j, k]],
                field[[(i + 2).min(nx - 1), j, k]],
            ];

            field[[i, j, k]] = self.compute_weno5_value(&v);
        }

        Ok(())
    }

    /// WENO5 limiting in y-direction
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn weno5_limit_y(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                j >= 2 && j < ny - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();

        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i, j.saturating_sub(2), k]],
                field[[i, j.saturating_sub(1), k]],
                field[[i, j, k]],
                field[[i, j.min(ny - 1).saturating_add(1), k]],
                field[[i, (j + 2).min(ny - 1), k]],
            ];

            field[[i, j, k]] = self.compute_weno5_value(&v);
        }

        Ok(())
    }

    /// WENO5 limiting in z-direction
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn weno5_limit_z(
        &self,
        field: &mut Array3<f64>,
        shock_indicator: &Array3<f64>,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<()> {
        // Collect indices where shock indicator exceeds threshold
        let indices: Vec<(usize, usize, usize)> = (0..nx)
            .flat_map(|i| (0..ny).flat_map(move |j| (0..nz).map(move |k| (i, j, k))))
            .filter(|&(i, j, k)| {
                k >= 2 && k < nz - 2 && shock_indicator[[i, j, k]] > self.shock_threshold
            })
            .collect();

        // Process each index
        for (i, j, k) in indices {
            // Extract stencil values
            let v = [
                field[[i, j, k.saturating_sub(2)]],
                field[[i, j, k.saturating_sub(1)]],
                field[[i, j, k]],
                field[[i, j, k.min(nz - 1).saturating_add(1)]],
                field[[i, j, (k + 2).min(nz - 1)]],
            ];

            field[[i, j, k]] = self.compute_weno5_value(&v);
        }

        Ok(())
    }

    /// Compute WENO5 reconstruction value from stencil
    pub(super) fn compute_weno5_value(&self, v: &[f64; 5]) -> f64 {
        // Three stencils for WENO5
        let p0 = 11.0f64.mul_add(v[2], 2.0f64.mul_add(v[0], -(7.0 * v[1]))) / 6.0;
        let p1 = 2.0f64.mul_add(v[3], 5.0f64.mul_add(v[2], -v[1])) / 6.0;
        let p2 = (2.0f64.mul_add(v[2], 5.0 * v[3]) - v[4]) / 6.0;

        // Smoothness indicators
        let beta0 = (13.0_f64 / 12.0).mul_add(
            (2.0f64.mul_add(-v[1], v[0]) + v[2]).powi(2),
            0.25 * 3.0f64.mul_add(v[2], 4.0f64.mul_add(-v[1], v[0])).powi(2),
        );
        let beta1 = (13.0_f64 / 12.0).mul_add(
            (2.0f64.mul_add(-v[2], v[1]) + v[3]).powi(2),
            0.25 * (v[1] - v[3]).powi(2),
        );
        let beta2 = (13.0_f64 / 12.0).mul_add(
            (2.0f64.mul_add(-v[3], v[2]) + v[4]).powi(2),
            0.25 * (3.0f64.mul_add(v[2], -(4.0 * v[3])) + v[4]).powi(2),
        );

        // Optimal weights
        let d0 = 0.1;
        let d1 = 0.6;
        let d2 = 0.3;

        // WENO weights
        let alpha0 = d0 / (self.epsilon + beta0).powi(2);
        let alpha1 = d1 / (self.epsilon + beta1).powi(2);
        let alpha2 = d2 / (self.epsilon + beta2).powi(2);
        let sum_alpha = alpha0 + alpha1 + alpha2;

        let w0 = alpha0 / sum_alpha;
        let w1 = alpha1 / sum_alpha;
        let w2 = alpha2 / sum_alpha;

        // WENO5 reconstruction
        w0 * p0 + w1 * p1 + w2 * p2
    }
}
