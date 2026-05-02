//! Per-axis fractional Laplacian absorption correction for `PSTDSolver`.
//!
//! ## Theorem: Per-Axis Fractional Laplacian Absorption
//! For each axis α ∈ {x, y, z} (Treeby & Cox 2010, Eq. 21):
//! ```text
//!   ρ_α += Δt · τ · IFFT(nabla1 · FFT(∂u_α/∂α))
//!        − Δt · η · IFFT(nabla2 · FFT(∂u_α/∂α))
//! ```
//! where `nabla1 = |k|^(y−2)` and `nabla2 = |k|^(y−1)`.

use crate::core::error::{KwaversError, KwaversResult, ValidationError};
use crate::math::fft::Complex64;
use crate::physics::acoustics::mechanics::absorption::AbsorptionMode;
use crate::solver::pstd::PSTDSolver;
use ndarray::Zip;

impl PSTDSolver {
    /// Apply per-axis power-law absorption correction to split density components.
    ///
    /// Uses scratch fields already in solver state to avoid allocations.
    /// Must be called AFTER `update_density()` and BEFORE `apply_pml_to_density()`.
    ///
    /// ## References
    /// - Treeby & Cox (2010) Eqs. 19–21.
    /// - k-Wave MATLAB: `kspaceFirstOrder3D.m`, density absorption block.
    pub(crate) fn apply_absorption(&mut self, dt: f64) -> KwaversResult<()> {
        match self.config.absorption_mode {
            AbsorptionMode::Lossless => return Ok(()),
            AbsorptionMode::Stokes | AbsorptionMode::PowerLaw { .. } => {}
            AbsorptionMode::MultiRelaxation { .. } | AbsorptionMode::Causal { .. } => {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: "Relaxation absorption modes are not supported by spectral solver"
                            .to_string(),
                    },
                ));
            }
        }

        let Some(ref abs) = self.absorption else {
            return Ok(());
        };

        // ── X-AXIS ────────────────────────────────────────────────────────────────
        self.fft.forward_into(&self.dpx, &mut self.grad_k);
        {
            let n1 = abs.nabla1.view();
            Zip::from(&mut self.ux_k)
                .and(&self.grad_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.ux_k, &mut self.div_u, &mut self.uy_k);
        {
            let n2 = abs.nabla2.view();
            Zip::from(&mut self.ux_k)
                .and(&self.grad_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.ux_k, &mut self.dpx, &mut self.uy_k);
        {
            let tau = abs.tau.view();
            let eta = abs.eta.view();
            Zip::from(&mut self.rhox)
                .and(&tau)
                .and(&eta)
                .and(&self.div_u)
                .and(&self.dpx)
                .for_each(|rho, &t, &e, &l1, &l2| {
                    *rho += dt * (t * l1 - e * l2);
                });
        }

        // ── Y-AXIS ────────────────────────────────────────────────────────────────
        self.fft.forward_into(&self.dpy, &mut self.grad_k);
        {
            let n1 = abs.nabla1.view();
            Zip::from(&mut self.uy_k)
                .and(&self.grad_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uy_k, &mut self.div_u, &mut self.ux_k);
        {
            let n2 = abs.nabla2.view();
            Zip::from(&mut self.uy_k)
                .and(&self.grad_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uy_k, &mut self.dpy, &mut self.ux_k);
        {
            let tau = abs.tau.view();
            let eta = abs.eta.view();
            Zip::from(&mut self.rhoy)
                .and(&tau)
                .and(&eta)
                .and(&self.div_u)
                .and(&self.dpy)
                .for_each(|rho, &t, &e, &l1, &l2| {
                    *rho += dt * (t * l1 - e * l2);
                });
        }

        // ── Z-AXIS ────────────────────────────────────────────────────────────────
        self.fft.forward_into(&self.dpz, &mut self.grad_k);
        {
            let n1 = abs.nabla1.view();
            Zip::from(&mut self.uz_k)
                .and(&self.grad_k)
                .and(&n1)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uz_k, &mut self.div_u, &mut self.ux_k);
        {
            let n2 = abs.nabla2.view();
            Zip::from(&mut self.uz_k)
                .and(&self.grad_k)
                .and(&n2)
                .for_each(|out, &hat, &n| {
                    *out = hat * Complex64::new(n, 0.0);
                });
        }
        self.fft
            .inverse_into(&self.uz_k, &mut self.dpz, &mut self.ux_k);
        {
            let tau = abs.tau.view();
            let eta = abs.eta.view();
            Zip::from(&mut self.rhoz)
                .and(&tau)
                .and(&eta)
                .and(&self.div_u)
                .and(&self.dpz)
                .for_each(|rho, &t, &e, &l1, &l2| {
                    *rho += dt * (t * l1 - e * l2);
                });
        }

        Ok(())
    }
}
