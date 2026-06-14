//! IVP staggered leapfrog velocity initialisation for `PSTDSolver`.

use super::PSTDSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::Fft3dInOutExt;
use ndarray::Zip;

impl PSTDSolver {
    /// Seed the full PSTD state for a zero-initial-velocity initial-value problem
    /// from an externally-supplied initial pressure already stored in
    /// `self.fields.p`.
    ///
    /// PSTD's state variables are the partial densities `(ρx, ρy, ρz)` with
    /// pressure *derived* via the equation of state `p = c²·(ρx+ρy+ρz)`. Setting
    /// only `fields.p` is therefore insufficient — the densities must be seeded so
    /// the EOS reproduces `p₀` and each directional density can evolve from its own
    /// divergence. The total perturbation `ρ = p₀/c²` is split equally among the
    /// **active** spatial dimensions (matching the construction-time IVP path):
    /// 3-D → `/3`, 2-D → `/2`, 1-D → `/1`. The half-step velocity is then seeded by
    /// [`Self::initialize_ivp_velocity`].
    ///
    /// # Errors
    /// Propagates [`Self::initialize_ivp_velocity`] failures.
    pub(crate) fn seed_ivp_from_pressure(&mut self) -> KwaversResult<()> {
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;
        let divisor = (1 + has_y as usize + has_z as usize) as f64;
        Zip::from(&mut self.rhox)
            .and(&mut self.rhoy)
            .and(&mut self.rhoz)
            .and(&self.fields.p)
            .and(&self.materials.c0)
            .par_for_each(|rx, ry, rz, &p, &c| {
                let share = if c > 1e-6 { p / (c * c) / divisor } else { 0.0 };
                *rx = share;
                *ry = if has_y { share } else { 0.0 };
                *rz = if has_z { share } else { 0.0 };
            });
        self.initialize_ivp_velocity()
    }

    /// Initialize velocity fields at t = −dt/2 for exact IVP staggered leapfrog start.
    ///
    /// Matches k-Wave convention (kspaceFirstOrder2D.m line 920):
    ///   u_α(x, dt/2) = dt/(2·ρ₀(x)) · [∂p₀/∂α]_κ
    ///
    /// where [∂p₀/∂α]_κ = IFFT(dd_α_shift_pos · sinc(c_ref·|k|·dt/2) · FFT(p₀)) is the
    /// k-space-corrected staggered derivative. The density factor dt/(2·ρ₀(x)) is applied
    /// pointwise in real space to preserve spatial heterogeneity; using a scalar mean density
    /// instead would produce an O(δρ/ρ̄) amplitude error in heterogeneous media.
    ///
    /// ## Implementation note (Opt-10)
    ///
    /// `source_kappa` stores `cos(c_ref·|k|·dt/2)` pre-truncated to the r2c half-spectrum
    /// shape `(nx, ny, nz_c)`.  The sinc factor `sinc(c_ref·|k|·dt/2)` is recovered via
    /// `sinc(θ) = sin(θ)/θ` with `θ = arccos(source_kappa[i,j,k])`, computed inline in the
    /// k-space Zip rather than precomputed into the full-shape `div_u` scratch.  This avoids
    /// the N-element write to `div_u` used before Opt-10 and eliminates a factor-of-2
    /// mismatch in element count (`div_u` has nz elements vs the half-spectrum nz_c).
    ///
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(super) fn initialize_ivp_velocity(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        if !dt.is_finite() || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "dt must be finite and positive, got {dt}"
            )));
        }

        let half_dt = dt / 2.0;

        // R2C forward: real pressure (nx,ny,nz) → half-spectrum (nx,ny,nz_c).
        self.fft.forward_r2c_into(&self.fields.p, &mut self.p_k);

        // X-axis: grad_k[i,j,k] = ddx[i] · sinc(arccos(source_kappa[i,j,k])) · p_k[i,j,k]
        //
        // sinc is computed inline from source_kappa (pre-truncated to nz_c):
        //   source_kappa = cos(c_ref·|k|·dt/2) ∈ [−1,1]
        //   θ = arccos(source_kappa)   (= c_ref·|k|·dt/2)
        //   sinc(θ) = sin(θ)/θ   (or 1.0 for θ < ε)
        //
        // This replaces the old precompute-into-div_u pass, which allocated a full
        // (nx,ny,nz) scratch for a value used only on (nx,ny,nz_c) — Opt-10.
        {
            let ddx = self.ddx_k_shift_pos.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.source_kappa.view())
                .and(p_k)
                .par_for_each(|(i, _j, _k), gk, &kap, &p| {
                    let theta = kap.clamp(-1.0, 1.0).acos();
                    let sinc_kap = if theta < 1e-30 {
                        1.0
                    } else {
                        theta.sin() / theta
                    };
                    *gk = ddx[i] * sinc_kap * p;
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
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.source_kappa.view())
                .and(p_k)
                .par_for_each(|(_i, j, _k), gk, &kap, &p| {
                    let theta = kap.clamp(-1.0, 1.0).acos();
                    let sinc_kap = if theta < 1e-30 {
                        1.0
                    } else {
                        theta.sin() / theta
                    };
                    *gk = ddy[j] * sinc_kap * p;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.fields.uy, &mut self.ux_k);
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
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(self.source_kappa.view())
                .and(p_k)
                .par_for_each(|(_i, _j, k_idx), gk, &kap, &p| {
                    let theta = kap.clamp(-1.0, 1.0).acos();
                    let sinc_kap = if theta < 1e-30 {
                        1.0
                    } else {
                        theta.sin() / theta
                    };
                    *gk = ddz[k_idx] * sinc_kap * p;
                });
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.fields.uz, &mut self.ux_k);
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
