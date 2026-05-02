use crate::core::error::KwaversResult;
use crate::math::fft::shift_operators::{generate_kappa, generate_shift_1d, generate_source_kappa};
use crate::math::fft::{get_fft_for_grid, Complex64, ProcessorFft3d};
use crate::solver::forward::acoustic_ivp::spectral_velocity_scale_from_source_kappa;
use ndarray::{Array1, Array3, Zip};
use std::f64::consts::PI;
use std::sync::Arc;

/// Pre-computed operators and scratch buffers for k-space corrected FDTD.
///
/// Constructed once per simulation from the grid dimensions, sound speed
/// reference, and time step. All scratch arrays are pre-allocated to avoid
/// per-step heap allocation in the hot time loop.
pub struct KSpaceFdtdOperators {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    c_ref: f64,
    /// Shared FFT plan (cached; not duplicated if PSTD is also running).
    fft: Arc<ProcessorFft3d>,
    /// Temporal correction factor κ[i,j,k] = sinc(0.5·c_ref·dt·|k|).
    /// Public so that tests can compare with PSTD's kappa.
    pub kappa: Array3<f64>,
    // 1-D staggered shift operators — pressure→velocity (positive half-shift)
    pub ddx_k_shift_pos: Array1<Complex64>,
    pub ddy_k_shift_pos: Array1<Complex64>,
    pub ddz_k_shift_pos: Array1<Complex64>,
    // 1-D staggered shift operators — velocity→pressure (negative half-shift)
    pub ddx_k_shift_neg: Array1<Complex64>,
    pub ddy_k_shift_neg: Array1<Complex64>,
    pub ddz_k_shift_neg: Array1<Complex64>,
    // ---- scratch arrays (pre-allocated, reused each step) ----
    /// FFT of input field (shared across gradient/divergence operations)
    field_k: Array3<Complex64>,
    /// k-space gradient buffers (one per axis)
    grad_x_k: Array3<Complex64>,
    grad_y_k: Array3<Complex64>,
    grad_z_k: Array3<Complex64>,
    /// IFFT scratch (required by `Fft3d::inverse_into`)
    scratch_k: Array3<Complex64>,
    // ---- real-space output buffers ----
    /// x-component gradient (filled by `compute_grad_pos` / `compute_grad_neg`)
    pub grad_x: Array3<f64>,
    /// y-component gradient
    pub grad_y: Array3<f64>,
    /// z-component gradient
    pub grad_z: Array3<f64>,
    /// Scalar divergence = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z (filled by `compute_divergence_neg`)
    pub divergence: Array3<f64>,
}

impl std::fmt::Debug for KSpaceFdtdOperators {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KSpaceFdtdOperators")
            .field("nx", &self.nx)
            .field("ny", &self.ny)
            .field("nz", &self.nz)
            .finish()
    }
}

impl KSpaceFdtdOperators {
    /// Construct operators from grid parameters.
    ///
    /// Calls [`crate::math::fft::shift_operators::generate_shift_1d`] and
    /// [`crate::math::fft::shift_operators::generate_kappa`] — the same shared
    /// utilities used by the PSTD orchestrator.
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        c_ref: f64,
        dt: f64,
    ) -> Self {
        let fft = get_fft_for_grid(nx, ny, nz);

        let dk_x = 2.0 * PI / (nx as f64 * dx);
        let dk_y = 2.0 * PI / (ny as f64 * dy);
        let dk_z = 2.0 * PI / (nz as f64 * dz);

        let (ddx_k_shift_pos, ddx_k_shift_neg) = generate_shift_1d(nx, dk_x, dx);
        let (ddy_k_shift_pos, ddy_k_shift_neg) = generate_shift_1d(ny, dk_y, dy);
        let (ddz_k_shift_pos, ddz_k_shift_neg) = generate_shift_1d(nz, dk_z, dz);

        let kappa = generate_kappa(nx, ny, nz, dx, dy, dz, c_ref, dt);

        let shape = (nx, ny, nz);

        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            c_ref,
            fft,
            kappa,
            ddx_k_shift_pos,
            ddy_k_shift_pos,
            ddz_k_shift_pos,
            ddx_k_shift_neg,
            ddy_k_shift_neg,
            ddz_k_shift_neg,
            field_k: Array3::zeros(shape),
            grad_x_k: Array3::zeros(shape),
            grad_y_k: Array3::zeros(shape),
            grad_z_k: Array3::zeros(shape),
            scratch_k: Array3::zeros(shape),
            grad_x: Array3::zeros(shape),
            grad_y: Array3::zeros(shape),
            grad_z: Array3::zeros(shape),
            divergence: Array3::zeros(shape),
        }
    }

    /// Initialize the exact staggered-grid velocity state for a given initial pressure.
    ///
    /// When the medium is homogeneous and no explicit initial velocity is
    /// supplied, the compatible leapfrog start is obtained by applying the
    /// k-space pressure→velocity operator at `t = -Δt/2`.
    pub fn initialize_ivp_velocity(
        &mut self,
        p0: &Array3<f64>,
        dt: f64,
        rho0_ref: f64,
        ux: &mut Array3<f64>,
        uy: &mut Array3<f64>,
        uz: &mut Array3<f64>,
    ) -> KwaversResult<()> {
        let source_kappa = generate_source_kappa(
            self.nx, self.ny, self.nz, self.dx, self.dy, self.dz, self.c_ref, dt,
        );
        let sin_scale = spectral_velocity_scale_from_source_kappa(&source_kappa, dt, rho0_ref)?;

        self.fft.forward_into(p0, &mut self.field_k);

        {
            let ddx = self.ddx_k_shift_pos.view();
            let sin_s = sin_scale.view();
            let p_k = self.field_k.view();
            Zip::indexed(self.grad_x_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(i, _j, _k), gx, &ss, &p| {
                    *gx = ddx[i] * ss * p;
                });
        }
        self.fft
            .inverse_into(&self.grad_x_k, ux, &mut self.scratch_k);

        {
            let ddy = self.ddy_k_shift_pos.view();
            let sin_s = sin_scale.view();
            let p_k = self.field_k.view();
            Zip::indexed(self.grad_y_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(_i, j, _k), gy, &ss, &p| {
                    *gy = ddy[j] * ss * p;
                });
        }
        self.fft
            .inverse_into(&self.grad_y_k, uy, &mut self.scratch_k);

        {
            let ddz = self.ddz_k_shift_pos.view();
            let sin_s = sin_scale.view();
            let p_k = self.field_k.view();
            Zip::indexed(self.grad_z_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(_i, _j, k_idx), gz, &ss, &p| {
                    *gz = ddz[k_idx] * ss * p;
                });
        }
        self.fft
            .inverse_into(&self.grad_z_k, uz, &mut self.scratch_k);

        Ok(())
    }

    /// Compute spectral gradients of `field` in all three directions.
    ///
    /// Uses the **positive** staggered shift operators (pressure→velocity path):
    /// ```text
    ///   grad_x = Re[ IFFT( ddx_k_shift_pos · κ · FFT(field) ) ]
    ///   grad_y = Re[ IFFT( ddy_k_shift_pos · κ · FFT(field) ) ]
    ///   grad_z = Re[ IFFT( ddz_k_shift_pos · κ · FFT(field) ) ]
    /// ```
    ///
    /// Results stored in `self.grad_x`, `self.grad_y`, `self.grad_z`.
    pub fn compute_grad_pos(&mut self, field: &Array3<f64>) {
        self.fft.forward_into(field, &mut self.field_k);

        {
            let ddx = self.ddx_k_shift_pos.view();
            let ddy = self.ddy_k_shift_pos.view();
            let ddz = self.ddz_k_shift_pos.view();
            Zip::indexed(self.grad_x_k.view_mut())
                .and(self.grad_y_k.view_mut())
                .and(self.grad_z_k.view_mut())
                .and(self.field_k.view())
                .and(self.kappa.view())
                .for_each(|(i, j, k), gx, gy, gz, &fk, &kap| {
                    let e = Complex64::new(kap, 0.0) * fk;
                    *gx = ddx[i] * e;
                    *gy = ddy[j] * e;
                    *gz = ddz[k] * e;
                });
        }

        self.fft
            .inverse_into(&self.grad_x_k, &mut self.grad_x, &mut self.scratch_k);
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.grad_y, &mut self.scratch_k);
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.grad_z, &mut self.scratch_k);
    }

    /// Compute spectral velocity divergence.
    ///
    /// Uses the **negative** staggered shift operators (velocity→pressure path):
    /// ```text
    ///   divergence = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
    /// ```
    ///
    /// Result accumulated into `self.divergence`.
    pub fn compute_divergence_neg(&mut self, ux: &Array3<f64>, uy: &Array3<f64>, uz: &Array3<f64>) {
        self.divergence.fill(0.0);

        // ∂ux/∂x
        self.fft.forward_into(ux, &mut self.field_k);
        {
            let ddx = self.ddx_k_shift_neg.view();
            Zip::indexed(self.grad_x_k.view_mut())
                .and(self.field_k.view())
                .and(self.kappa.view())
                .for_each(|(i, _j, _k), gx, &fk, &kap| {
                    *gx = ddx[i] * Complex64::new(kap, 0.0) * fk;
                });
        }
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.grad_x, &mut self.scratch_k);
        self.divergence += &self.grad_x;

        // ∂uy/∂y
        self.fft.forward_into(uy, &mut self.field_k);
        {
            let ddy = self.ddy_k_shift_neg.view();
            Zip::indexed(self.grad_y_k.view_mut())
                .and(self.field_k.view())
                .and(self.kappa.view())
                .for_each(|(_i, j, _k), gy, &fk, &kap| {
                    *gy = ddy[j] * Complex64::new(kap, 0.0) * fk;
                });
        }
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.grad_y, &mut self.scratch_k);
        self.divergence += &self.grad_y;

        // ∂uz/∂z
        self.fft.forward_into(uz, &mut self.field_k);
        {
            let ddz = self.ddz_k_shift_neg.view();
            Zip::indexed(self.grad_z_k.view_mut())
                .and(self.field_k.view())
                .and(self.kappa.view())
                .for_each(|(_i, _j, k), gz, &fk, &kap| {
                    *gz = ddz[k] * Complex64::new(kap, 0.0) * fk;
                });
        }
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.grad_z, &mut self.scratch_k);
        self.divergence += &self.grad_z;
    }
}
