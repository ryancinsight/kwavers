//! Anti-aliasing spectral filter for `PSTDSolver`.

use super::super::orchestrator::PSTDSolver;
use super::ops::multiply_complex_by_real_field;
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::Fft3dInOutExt;

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
            // C2R inverse: half-spectrum → real field (nx,ny,nz), consuming the half-spectrum.

            // Apply filter to pressure using p_k as transform buffer.
            self.fft.forward_r2c_into(&self.fields.p, &mut self.p_k);
            multiply_complex_by_real_field(&mut self.p_k, filter);
            self.fft.inverse_c2r_into(&mut self.p_k, &mut self.fields.p);

            // Apply filter to split density components.
            self.fft.forward_r2c_into(&self.rhox, &mut self.p_k);
            multiply_complex_by_real_field(&mut self.p_k, filter);
            self.fft.inverse_c2r_into(&mut self.p_k, &mut self.rhox);

            self.fft.forward_r2c_into(&self.rhoy, &mut self.p_k);
            multiply_complex_by_real_field(&mut self.p_k, filter);
            self.fft.inverse_c2r_into(&mut self.p_k, &mut self.rhoy);

            self.fft.forward_r2c_into(&self.rhoz, &mut self.p_k);
            multiply_complex_by_real_field(&mut self.p_k, filter);
            self.fft.inverse_c2r_into(&mut self.p_k, &mut self.rhoz);

            // Apply filter to Ux using ux_k as transform buffer.
            self.fft.forward_r2c_into(&self.fields.ux, &mut self.ux_k);
            multiply_complex_by_real_field(&mut self.ux_k, filter);
            self.fft
                .inverse_c2r_into(&mut self.ux_k, &mut self.fields.ux);

            // Apply filter to Uy using ux_k as transform buffer.
            self.fft.forward_r2c_into(&self.fields.uy, &mut self.ux_k);
            multiply_complex_by_real_field(&mut self.ux_k, filter);
            self.fft
                .inverse_c2r_into(&mut self.ux_k, &mut self.fields.uy);

            // Apply filter to Uz using ux_k as transform buffer.
            self.fft.forward_r2c_into(&self.fields.uz, &mut self.ux_k);
            multiply_complex_by_real_field(&mut self.ux_k, filter);
            self.fft
                .inverse_c2r_into(&mut self.ux_k, &mut self.fields.uz);
        }
        Ok(())
    }
}
