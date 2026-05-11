//! IVP staggered leapfrog velocity initialisation for `PSTDSolver`.

use super::PSTDSolver;
use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{s, Zip};

impl PSTDSolver {
    /// Initialize velocity fields at t = −dt/2 for exact IVP staggered leapfrog start.
    ///
    /// Matches k-Wave convention (kspaceFirstOrder2D.m line 920):
    ///   u_α(x, dt/2) = dt/(2·ρ₀(x)) · [∂p₀/∂α]_κ
    ///
    /// where [∂p₀/∂α]_κ = IFFT(dd_α_shift_pos · sinc(c_ref·|k|·dt/2) · FFT(p₀)) is the
    /// k-space-corrected staggered derivative. The density factor dt/(2·ρ₀(x)) is applied
    /// pointwise in real space to preserve spatial heterogeneity; using a scalar mean density
    /// instead would produce an O(δρ/ρ̄) amplitude error in heterogeneous media.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(super) fn initialize_ivp_velocity(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;
        // nz_c: half-spectrum z-length; p_k/grad_k have shape (nx, ny, nz_c).
        let nz_c = self.p_k.dim().2;

        if !dt.is_finite() || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "dt must be finite and positive, got {dt}"
            )));
        }

        // Precompute sinc(c_ref·|k|·dt/2) correction into div_u scratch.
        // Density scaling is applied pointwise after the IFFT (see below).
        Zip::from(&mut self.div_u)
            .and(&self.source_kappa)
            .par_for_each(|sinc_val, &kap| {
                let theta = kap.clamp(-1.0, 1.0).acos();
                *sinc_val = if theta < 1e-30 {
                    1.0
                } else {
                    theta.sin() / theta
                };
            });

        let half_dt = dt / 2.0;

        // R2C forward: real pressure (nx,ny,nz) → half-spectrum (nx,ny,nz_c).
        self.fft.forward_r2c_into(&self.fields.p, &mut self.p_k);

        // X-axis: grad_k[i,j,k] = ddx[i] · sinc[i,j,k] · p_k[i,j,k]
        // sinc is sliced to (nx,ny,nz_c) to match the half-spectrum k-space shape.
        {
            let ddx = self.ddx_k_shift_pos.view();
            let sin_s = self.div_u.slice(s![.., .., ..nz_c]);
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .par_for_each(|(i, _j, _k), gk, &ss, &p| {
                    *gk = ddx[i] * ss * p;
                });
        }
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.fields.ux, &mut self.ux_k);
        Zip::from(&mut self.fields.ux)
            .and(&self.materials.rho0)
            .par_for_each(|u, &rho| {
                *u *= half_dt / rho;
            });

        if has_y {
            let ddy = self.ddy_k_shift_pos.view();
            let sin_s = self.div_u.slice(s![.., .., ..nz_c]);
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .par_for_each(|(_i, j, _k), gk, &ss, &p| {
                    *gk = ddy[j] * ss * p;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.fields.uy, &mut self.uy_k);
            Zip::from(&mut self.fields.uy)
                .and(&self.materials.rho0)
                .par_for_each(|u, &rho| {
                    *u *= half_dt / rho;
                });
        } else {
            self.fields.uy.fill(0.0);
        }

        if has_z {
            // ddz has length nz_c (truncated in construction); k_idx ∈ [0, nz_c).
            let ddz = self.ddz_k_shift_pos.view();
            let sin_s = self.div_u.slice(s![.., .., ..nz_c]);
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .par_for_each(|(_i, _j, k_idx), gk, &ss, &p| {
                    *gk = ddz[k_idx] * ss * p;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.fields.uz, &mut self.uz_k);
            Zip::from(&mut self.fields.uz)
                .and(&self.materials.rho0)
                .par_for_each(|u, &rho| {
                    *u *= half_dt / rho;
                });
        } else {
            self.fields.uz.fill(0.0);
        }

        Ok(())
    }
}
