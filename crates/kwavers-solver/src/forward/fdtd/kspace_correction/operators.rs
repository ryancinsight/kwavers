use crate::forward::acoustic_ivp::spectral_velocity_scale_from_source_kappa;
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::shift_operators::{
    generate_kappa, generate_shift_1d, generate_source_kappa,
};
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d, Fft3dInOutExt};
use leto::Array3 as LetoArray3;
use moirai_parallel::{enumerate_mut_with, Adaptive};
use leto::{
    Array1,
    Array3,
};
use std::sync::Arc;

#[derive(Clone, Copy)]
enum SpectralAxis {
    X,
    Y,
    Z,
}

impl SpectralAxis {
    fn index(self, linear_index: usize, ny: usize, nz: usize) -> usize {
        match self {
            Self::X => linear_index / (ny * nz),
            Self::Y => (linear_index / nz) % ny,
            Self::Z => linear_index % nz,
        }
    }

    fn indexed(self, i: usize, j: usize, k: usize) -> usize {
        match self {
            Self::X => i,
            Self::Y => j,
            Self::Z => k,
        }
    }
}

fn apply_shifted_spectral_gradient(
    output: &mut LetoArray3<Complex64>,
    field_k: &LetoArray3<Complex64>,
    kappa: &Array3<f64>,
    shift: &Array1<Complex64>,
    axis: SpectralAxis,
) {
    assert_eq!(
        output.shape(),
        field_k.shape(),
        "invariant: FDTD spectral output shape matches transformed field"
    );
    assert_eq!(
        output.shape(),
        kappa.shape(),
        "invariant: FDTD spectral kappa shape matches transformed field"
    );
    let [nx, ny, nz] = output.shape();

    if let (Some(output_values), Some(field_values), Some(kappa_values), Some(shift_values)) = (
        output.as_slice_mut(),
        field_k.as_slice(),
        kappa.as_slice_memory_order(),
        shift.as_slice_memory_order(),
    ) {
        enumerate_mut_with::<Adaptive, _, _>(output_values, |linear_index, value| {
            let shift_index = axis.index(linear_index, ny, nz);
            *value = shift_values[shift_index]
                * (field_values[linear_index] * kappa_values[linear_index]);
        });
    } else {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let shift_index = axis.indexed(i, j, k);
                    output[[i, j, k]] =
                        shift[shift_index] * (field_k[[i, j, k]] * kappa[[i, j, k]]);
                }
            }
        }
    }
}

fn ndarray_real_field(field: LetoArray3<f64>) -> Array3<f64> {
    let [nx, ny, nz] = field.shape();
    Array3::from_shape_vec((nx, ny, nz), field.into_vec())
        .expect("Leto real field length must match FDTD field shape")
}

fn add_assign_ndarray(dst: &mut Array3<f64>, src: &Array3<f64>) {
    assert_eq!(
        dst.shape(),
        src.shape(),
        "invariant: FDTD accumulation field shapes must match"
    );
    if let (Some(dst_values), Some(src_values)) =
        (dst.as_slice_memory_order_mut(), src.as_slice_memory_order())
    {
        enumerate_mut_with::<Adaptive, _, _>(dst_values, |index, value| {
            *value += src_values[index];
        });
    } else {
        let (nx, ny, nz) = dst.dim();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    dst[[i, j, k]] += src[[i, j, k]];
                }
            }
        }
    }
}

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
    fft: Arc<Fft3d>,
    /// Temporal correction factor `κ[i,j,k] = sinc(0.5·c_ref·dt·|k|)`.
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
    field_k: LetoArray3<Complex64>,
    /// k-space gradient buffers (one per axis)
    grad_x_k: LetoArray3<Complex64>,
    grad_y_k: LetoArray3<Complex64>,
    grad_z_k: LetoArray3<Complex64>,
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
    /// Calls [`kwavers_math::fft::shift_operators::generate_shift_1d`] and
    /// [`kwavers_math::fft::shift_operators::generate_kappa`] — the same shared
    /// utilities used by the PSTD orchestrator.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
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

        let dk_x = TWO_PI / (nx as f64 * dx);
        let dk_y = TWO_PI / (ny as f64 * dy);
        let dk_z = TWO_PI / (nz as f64 * dz);

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
            field_k: LetoArray3::zeros([nx, ny, nz]),
            grad_x_k: LetoArray3::zeros([nx, ny, nz]),
            grad_y_k: LetoArray3::zeros([nx, ny, nz]),
            grad_z_k: LetoArray3::zeros([nx, ny, nz]),
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn initialize_ivp_velocity(
        &mut self,
        p0: &LetoArray3<f64>,
        dt: f64,
        rho0_ref: f64,
        ux: &mut LetoArray3<f64>,
        uy: &mut LetoArray3<f64>,
        uz: &mut LetoArray3<f64>,
    ) -> KwaversResult<()> {
        let source_kappa = generate_source_kappa(
            self.nx, self.ny, self.nz, self.dx, self.dy, self.dz, self.c_ref, dt,
        );
        let sin_scale = spectral_velocity_scale_from_source_kappa(&source_kappa, dt, rho0_ref)?;

        self.field_k = self.fft.forward(p0);

        {
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        self.grad_x_k[[i, j, k]] = self.ddx_k_shift_pos[i]
                            * sin_scale[[i, j, k]]
                            * self.field_k[[i, j, k]];
                    }
                }
            }
        }
        ux.assign(&self.fft.inverse(&self.grad_x_k));

        {
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        self.grad_y_k[[i, j, k]] = self.ddy_k_shift_pos[j]
                            * sin_scale[[i, j, k]]
                            * self.field_k[[i, j, k]];
                    }
                }
            }
        }
        uy.assign(&self.fft.inverse(&self.grad_y_k));

        {
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        self.grad_z_k[[i, j, k]] = self.ddz_k_shift_pos[k]
                            * sin_scale[[i, j, k]]
                            * self.field_k[[i, j, k]];
                    }
                }
            }
        }
        uz.assign(&self.fft.inverse(&self.grad_z_k));

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
    pub fn compute_grad_pos(&mut self, field: &LetoArray3<f64>) {
        self.field_k = self.fft.forward(field);

        apply_shifted_spectral_gradient(
            &mut self.grad_x_k,
            &self.field_k,
            &self.kappa,
            &self.ddx_k_shift_pos,
            SpectralAxis::X,
        );
        apply_shifted_spectral_gradient(
            &mut self.grad_y_k,
            &self.field_k,
            &self.kappa,
            &self.ddy_k_shift_pos,
            SpectralAxis::Y,
        );
        apply_shifted_spectral_gradient(
            &mut self.grad_z_k,
            &self.field_k,
            &self.kappa,
            &self.ddz_k_shift_pos,
            SpectralAxis::Z,
        );

        self.grad_x = ndarray_real_field(self.fft.inverse(&self.grad_x_k));
        self.grad_y = ndarray_real_field(self.fft.inverse(&self.grad_y_k));
        self.grad_z = ndarray_real_field(self.fft.inverse(&self.grad_z_k));
    }

    /// Compute spectral velocity divergence.
    ///
    /// Uses the **negative** staggered shift operators (velocity→pressure path):
    /// ```text
    ///   divergence = ∂ux/∂x + ∂uy/∂y + ∂uz/∂z
    /// ```
    ///
    /// Result accumulated into `self.divergence`.
    pub fn compute_divergence_neg(
        &mut self,
        ux: &LetoArray3<f64>,
        uy: &LetoArray3<f64>,
        uz: &LetoArray3<f64>,
    ) {
        self.divergence.fill(0.0);

        // ∂ux/∂x
        self.field_k = self.fft.forward(ux);
        apply_shifted_spectral_gradient(
            &mut self.grad_x_k,
            &self.field_k,
            &self.kappa,
            &self.ddx_k_shift_neg,
            SpectralAxis::X,
        );
        self.grad_x = ndarray_real_field(self.fft.inverse(&self.grad_x_k));
        add_assign_ndarray(&mut self.divergence, &self.grad_x);

        // ∂uy/∂y
        self.field_k = self.fft.forward(uy);
        apply_shifted_spectral_gradient(
            &mut self.grad_y_k,
            &self.field_k,
            &self.kappa,
            &self.ddy_k_shift_neg,
            SpectralAxis::Y,
        );
        self.grad_y = ndarray_real_field(self.fft.inverse(&self.grad_y_k));
        add_assign_ndarray(&mut self.divergence, &self.grad_y);

        // ∂uz/∂z
        self.field_k = self.fft.forward(uz);
        apply_shifted_spectral_gradient(
            &mut self.grad_z_k,
            &self.field_k,
            &self.kappa,
            &self.ddz_k_shift_neg,
            SpectralAxis::Z,
        );
        self.grad_z = ndarray_real_field(self.fft.inverse(&self.grad_z_k));
        add_assign_ndarray(&mut self.divergence, &self.grad_z);
    }
}
