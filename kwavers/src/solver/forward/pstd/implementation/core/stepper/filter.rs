//! Anti-aliasing spectral filter for `PSTDSolver`.

use super::super::orchestrator::PSTDSolver;
use crate::core::error::KwaversResult;
use crate::math::fft::Complex64;
use ndarray::Zip;

impl PSTDSolver {
    /// Apply anti-aliasing filter to field variables.
    ///
    /// Removes high-frequency spatial components that can cause instability
    /// or aliasing when using PSTD with nonlinearities.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn apply_anti_aliasing_filter(&mut self) -> KwaversResult<()> {
        if let Some(filter) = &self.filter {
            // filter has shape (nx, ny, nz_c) — truncated to the r2c half-spectrum
            // in construction. All k-space buffers (p_k, ux_k, …) are (nx, ny, nz_c).
            // R2C forward: real field (nx,ny,nz) → half-spectrum (nx,ny,nz_c).
            // C2R inverse: half-spectrum → real field (nx,ny,nz); scratch = nz_c buffer.

            // Apply filter to pressure using p_k as transform buffer, ux_k as scratch.
            self.fft.forward_r2c_into(&self.fields.p, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .par_for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_c2r_into(&self.p_k, &mut self.fields.p, &mut self.ux_k);

            // Apply filter to split density components.
            self.fft.forward_r2c_into(&self.rhox, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .par_for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_c2r_into(&self.p_k, &mut self.rhox, &mut self.uy_k);

            self.fft.forward_r2c_into(&self.rhoy, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .par_for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_c2r_into(&self.p_k, &mut self.rhoy, &mut self.uy_k);

            self.fft.forward_r2c_into(&self.rhoz, &mut self.p_k);
            Zip::from(&mut self.p_k)
                .and(filter)
                .par_for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_c2r_into(&self.p_k, &mut self.rhoz, &mut self.uy_k);

            // Apply filter to Ux using ux_k as transform buffer, p_k as scratch.
            self.fft.forward_r2c_into(&self.fields.ux, &mut self.ux_k);
            Zip::from(&mut self.ux_k)
                .and(filter)
                .par_for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_c2r_into(&self.ux_k, &mut self.fields.ux, &mut self.p_k);

            // Apply filter to Uy using uy_k as transform buffer, p_k as scratch.
            self.fft.forward_r2c_into(&self.fields.uy, &mut self.uy_k);
            Zip::from(&mut self.uy_k)
                .and(filter)
                .par_for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_c2r_into(&self.uy_k, &mut self.fields.uy, &mut self.p_k);

            // Apply filter to Uz using uz_k as transform buffer, p_k as scratch.
            self.fft.forward_r2c_into(&self.fields.uz, &mut self.uz_k);
            Zip::from(&mut self.uz_k)
                .and(filter)
                .par_for_each(|val, &f| *val *= Complex64::new(f, 0.0));
            self.fft
                .inverse_c2r_into(&self.uz_k, &mut self.fields.uz, &mut self.p_k);
        }
        Ok(())
    }
}
