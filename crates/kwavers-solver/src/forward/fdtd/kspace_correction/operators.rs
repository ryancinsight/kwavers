use crate::forward::acoustic_ivp::spectral_velocity_scale_from_source_kappa;
use crate::forward::lanes::{axis_index, for_each_z_lane, LaneAxis};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::shift_operators::{
    generate_kappa, generate_shift_1d, generate_source_kappa,
};
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3d, Fft3dInOutExt};
use leto::{Array1, Array3};
use std::sync::Arc;

fn apply_shifted_spectral_gradient(
    output: &mut Array3<Complex64>,
    field_k: &Array3<Complex64>,
    kappa: &Array3<f64>,
    shift: &Array1<Complex64>,
    axis: LaneAxis,
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
        kappa.as_slice(),
        shift.as_slice(),
    ) {
        let element_bytes = 2 * size_of::<Complex64>() + size_of::<f64>();
        for_each_z_lane(
            output_values,
            [ny, nz],
            element_bytes,
            |start, i, j, lane| {
                let inputs = field_values[start..start + nz]
                    .iter()
                    .zip(&kappa_values[start..start + nz]);
                match axis {
                    LaneAxis::X | LaneAxis::Y => {
                        let shift = shift_values[axis_index(axis, i, j, 0)];
                        for (value, (&field, &kappa)) in lane.iter_mut().zip(inputs) {
                            *value = shift * (field * kappa);
                        }
                    }
                    LaneAxis::Z => {
                        for ((value, (&field, &kappa)), &shift) in
                            lane.iter_mut().zip(inputs).zip(&shift_values[..nz])
                        {
                            *value = shift * (field * kappa);
                        }
                    }
                }
            },
        );
    } else {
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let shift_index = axis_index(axis, i, j, k);
                    output[[i, j, k]] =
                        shift[shift_index] * (field_k[[i, j, k]] * kappa[[i, j, k]]);
                }
            }
        }
    }
}

/// In-place accumulation: `dst += src` for Leto arrays.
///
/// Replaces the old `add_assign_ndarray` that bridged ndarray→Leto.
fn add_assign(dst: &mut Array3<f64>, src: &Array3<f64>) {
    assert_eq!(
        dst.shape(),
        src.shape(),
        "invariant: FDTD accumulation field shapes must match"
    );
    let [_nx, ny, nz] = src.shape();
    if let (Some(dst_values), Some(src_values)) = (dst.as_slice_mut(), src.as_slice()) {
        for_each_z_lane(
            dst_values,
            [ny, nz],
            2 * size_of::<f64>(),
            |start, _, _, lane| {
                for (value, &source) in lane.iter_mut().zip(&src_values[start..start + nz]) {
                    *value += source;
                }
            },
        );
    } else {
        let [nx, ny, nz] = dst.shape();
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
    /// `kappa` over the half spectrum `(nx, ny, nz/2+1)` the step transforms
    /// into: `kappa` depends on `|k|` alone, so its first `nz/2+1` z-values are
    /// the complete set a real field's half spectrum needs.
    kappa_half: Array3<f64>,
    // ---- scratch arrays (pre-allocated, reused each step) ----
    /// Half spectrum of the input field (shared across gradient/divergence operations)
    field_k: Array3<Complex64>,
    /// k-space gradient buffers (one per axis)
    grad_x_k: Array3<Complex64>,
    grad_y_k: Array3<Complex64>,
    grad_z_k: Array3<Complex64>,
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
            .field("dx", &self.dx)
            .field("dy", &self.dy)
            .field("dz", &self.dz)
            .field("c_ref", &self.c_ref)
            .field("fft", &"<fft-plan>")
            .field("kappa", &self.kappa.shape())
            .field("kappa_half", &self.kappa_half.shape())
            .field("ddx_k_shift_pos", &self.ddx_k_shift_pos.len())
            .field("ddy_k_shift_pos", &self.ddy_k_shift_pos.len())
            .field("ddz_k_shift_pos", &self.ddz_k_shift_pos.len())
            .field("ddx_k_shift_neg", &self.ddx_k_shift_neg.len())
            .field("ddy_k_shift_neg", &self.ddy_k_shift_neg.len())
            .field("ddz_k_shift_neg", &self.ddz_k_shift_neg.len())
            .field("field_k", &self.field_k.shape())
            .field("grad_x_k", &self.grad_x_k.shape())
            .field("grad_y_k", &self.grad_y_k.shape())
            .field("grad_z_k", &self.grad_z_k.shape())
            .field("grad_x", &self.grad_x.shape())
            .field("grad_y", &self.grad_y.shape())
            .field("grad_z", &self.grad_z.shape())
            .field("divergence", &self.divergence.shape())
            .finish()
    }
}

impl KSpaceFdtdOperators {
    /// Construct operators from grid parameters.
    ///
    /// Calls [`kwavers_math::fft::shift_operators::generate_shift_1d`] and
    /// [`kwavers_math::fft::shift_operators::generate_kappa`] — the same shared
    /// utilities used by the PSTD orchestrator.
    ///
    /// # Panics
    ///
    /// Unreachable: the half-spectrum depth `nz/2+1` never exceeds `nz`, so
    /// slicing `kappa` to it cannot fail.
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
        // Real fields transform through the half-spectrum pair; the z shifts stay
        // full length (the initial-value path indexes all of them) and the step
        // reads their first `nz/2+1` entries, which rfftfreq order makes exact.
        let nz_half = nz / 2 + 1;
        let kappa_half = kappa
            .slice_with(&s![.., .., ..nz_half])
            .expect("invariant: nz/2+1 never exceeds nz")
            .to_contiguous();

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
            kappa_half,
            ddx_k_shift_pos,
            ddy_k_shift_pos,
            ddz_k_shift_pos,
            ddx_k_shift_neg,
            ddy_k_shift_neg,
            ddz_k_shift_neg,
            field_k: Array3::zeros([nx, ny, nz_half]),
            grad_x_k: Array3::zeros([nx, ny, nz_half]),
            grad_y_k: Array3::zeros([nx, ny, nz_half]),
            grad_z_k: Array3::zeros([nx, ny, nz_half]),
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
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
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

        let field_k = self.fft.forward(p0);
        let mut gradient_k = Array3::<Complex64>::zeros([self.nx, self.ny, self.nz]);

        {
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        gradient_k[[i, j, k]] =
                            self.ddx_k_shift_pos[i] * sin_scale[[i, j, k]] * field_k[[i, j, k]];
                    }
                }
            }
        }
        ux.assign(&self.fft.inverse(&gradient_k));

        {
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        gradient_k[[i, j, k]] =
                            self.ddy_k_shift_pos[j] * sin_scale[[i, j, k]] * field_k[[i, j, k]];
                    }
                }
            }
        }
        uy.assign(&self.fft.inverse(&gradient_k));

        {
            for i in 0..self.nx {
                for j in 0..self.ny {
                    for k in 0..self.nz {
                        gradient_k[[i, j, k]] =
                            self.ddz_k_shift_pos[k] * sin_scale[[i, j, k]] * field_k[[i, j, k]];
                    }
                }
            }
        }
        uz.assign(&self.fft.inverse(&gradient_k));

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
        self.fft.forward_r2c_into(field, &mut self.field_k);

        apply_shifted_spectral_gradient(
            &mut self.grad_x_k,
            &self.field_k,
            &self.kappa_half,
            &self.ddx_k_shift_pos,
            LaneAxis::X,
        );
        apply_shifted_spectral_gradient(
            &mut self.grad_y_k,
            &self.field_k,
            &self.kappa_half,
            &self.ddy_k_shift_pos,
            LaneAxis::Y,
        );
        apply_shifted_spectral_gradient(
            &mut self.grad_z_k,
            &self.field_k,
            &self.kappa_half,
            &self.ddz_k_shift_pos,
            LaneAxis::Z,
        );

        self.fft
            .inverse_c2r_into(&mut self.grad_x_k, &mut self.grad_x);
        self.fft
            .inverse_c2r_into(&mut self.grad_y_k, &mut self.grad_y);
        self.fft
            .inverse_c2r_into(&mut self.grad_z_k, &mut self.grad_z);
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
        self.fft.forward_r2c_into(ux, &mut self.field_k);
        apply_shifted_spectral_gradient(
            &mut self.grad_x_k,
            &self.field_k,
            &self.kappa_half,
            &self.ddx_k_shift_neg,
            LaneAxis::X,
        );
        self.fft
            .inverse_c2r_into(&mut self.grad_x_k, &mut self.grad_x);
        add_assign(&mut self.divergence, &self.grad_x);

        // ∂uy/∂y
        self.fft.forward_r2c_into(uy, &mut self.field_k);
        apply_shifted_spectral_gradient(
            &mut self.grad_y_k,
            &self.field_k,
            &self.kappa_half,
            &self.ddy_k_shift_neg,
            LaneAxis::Y,
        );
        self.fft
            .inverse_c2r_into(&mut self.grad_y_k, &mut self.grad_y);
        add_assign(&mut self.divergence, &self.grad_y);

        // ∂uz/∂z
        self.fft.forward_r2c_into(uz, &mut self.field_k);
        apply_shifted_spectral_gradient(
            &mut self.grad_z_k,
            &self.field_k,
            &self.kappa_half,
            &self.ddz_k_shift_neg,
            LaneAxis::Z,
        );
        self.fft
            .inverse_c2r_into(&mut self.grad_z_k, &mut self.grad_z);
        add_assign(&mut self.divergence, &self.grad_z);
    }
}
