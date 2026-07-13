//! IVP staggered leapfrog velocity initialisation for `PSTDSolver`.

use super::PSTDSolver;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{Complex64, Fft3dInOutExt};
use leto::Array1;
use leto::Array3 as LetoArray3;
use moirai_parallel::{enumerate_mut_with, for_each_chunk_triple_mut_enumerated_with, Adaptive};

const DENSE_IVP_CHUNK: usize = 4096;

struct DensitySeedFields<'a> {
    rhox: &'a mut LetoArray3<f64>,
    rhoy: &'a mut LetoArray3<f64>,
    rhoz: &'a mut LetoArray3<f64>,
    pressure: &'a LetoArray3<f64>,
    c0: &'a LetoArray3<f64>,
}

struct DensitySeedConfig {
    divisor: f64,
    has_y: bool,
    has_z: bool,
}

#[derive(Clone, Copy)]
enum GradientAxis {
    X,
    Y,
    Z,
}

impl GradientAxis {
    fn factor_index(self, i: usize, j: usize, k: usize) -> usize {
        match self {
            Self::X => i,
            Self::Y => j,
            Self::Z => k,
        }
    }
}

fn dense_indices(index: usize, ny: usize, nz: usize) -> (usize, usize, usize) {
    let plane = ny * nz;
    let i = index / plane;
    let rem = index % plane;
    let j = rem / nz;
    let k = rem % nz;
    (i, j, k)
}

fn sinc_from_source_kappa(kap: f64) -> f64 {
    let theta = kap.clamp(-1.0, 1.0).acos();
    if theta < 1e-30 {
        1.0
    } else {
        theta.sin() / theta
    }
}

fn seed_density_components_from_pressure(fields: DensitySeedFields<'_>, config: DensitySeedConfig) {
    let DensitySeedFields {
        rhox,
        rhoy,
        rhoz,
        pressure,
        c0,
    } = fields;
    assert_eq!(
        rhox.shape(),
        rhoy.shape(),
        "invariant: PSTD IVP rhox and rhoy shapes match"
    );
    assert_eq!(
        rhox.shape(),
        rhoz.shape(),
        "invariant: PSTD IVP rhox and rhoz shapes match"
    );
    assert_eq!(
        rhox.shape(),
        pressure.shape(),
        "invariant: PSTD IVP density shape matches pressure shape"
    );
    assert_eq!(
        rhox.shape(),
        c0.shape(),
        "invariant: PSTD IVP density shape matches sound-speed shape"
    );

    let used_dense_path = match (
        rhox.as_slice_mut(),
        rhoy.as_slice_mut(),
        rhoz.as_slice_mut(),
        pressure.as_slice(),
        c0.as_slice(),
    ) {
        (Some(rx_values), Some(ry_values), Some(rz_values), Some(p_values), Some(c_values)) => {
            for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
                rx_values,
                ry_values,
                rz_values,
                DENSE_IVP_CHUNK,
                |chunk_index, rx_chunk, ry_chunk, rz_chunk| {
                    let start = chunk_index * DENSE_IVP_CHUNK;
                    for (offset, rx) in rx_chunk.iter_mut().enumerate() {
                        let index = start + offset;
                        let c = c_values[index];
                        let share = if c > 1e-6 {
                            p_values[index] / (c * c) / config.divisor
                        } else {
                            0.0
                        };
                        *rx = share;
                        ry_chunk[offset] = if config.has_y { share } else { 0.0 };
                        rz_chunk[offset] = if config.has_z { share } else { 0.0 };
                    }
                },
            );
            true
        }
        _ => false,
    };

    if !used_dense_path {
        let [nx, ny, nz] = rhox.shape();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let c = c0[[i, j, k]];
                    let share = if c > 1e-6 {
                        pressure[[i, j, k]] / (c * c) / config.divisor
                    } else {
                        0.0
                    };
                    rhox[[i, j, k]] = share;
                    rhoy[[i, j, k]] = if config.has_y { share } else { 0.0 };
                    rhoz[[i, j, k]] = if config.has_z { share } else { 0.0 };
                }
            }
        }
    }
}

fn write_spectral_gradient_axis(
    grad_k: &mut LetoArray3<Complex64>,
    source_kappa: &LetoArray3<f64>,
    p_k: &LetoArray3<Complex64>,
    shift: &Array1<Complex64>,
    axis: GradientAxis,
) {
    assert_eq!(
        grad_k.shape(),
        source_kappa.shape(),
        "invariant: PSTD IVP gradient shape matches source kappa shape"
    );
    assert_eq!(
        grad_k.shape(),
        p_k.shape(),
        "invariant: PSTD IVP gradient shape matches pressure spectrum shape"
    );

    let [nx, ny, nz] = grad_k.shape();
    let expected_len = match axis {
        GradientAxis::X => nx,
        GradientAxis::Y => ny,
        GradientAxis::Z => nz,
    };
    assert_eq!(
        (shift.len()),
        expected_len,
        "invariant: PSTD IVP shift length matches selected gradient axis"
    );

    let used_dense_path = match (
        grad_k.as_slice_mut(),
        source_kappa.as_slice(),
        p_k.as_slice(),
    ) {
        (Some(grad_values), Some(kappa_values), Some(p_values)) => {
            enumerate_mut_with::<Adaptive, _, _>(grad_values, |index, grad| {
                let (i, j, k) = dense_indices(index, ny, nz);
                let shift_factor = shift[axis.factor_index(i, j, k)];
                *grad =
                    shift_factor * sinc_from_source_kappa(kappa_values[index]) * p_values[index];
            });
            true
        }
        _ => false,
    };

    if !used_dense_path {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let shift_factor = shift[axis.factor_index(i, j, k)];
                    grad_k[[i, j, k]] = shift_factor
                        * sinc_from_source_kappa(source_kappa[[i, j, k]])
                        * p_k[[i, j, k]];
                }
            }
        }
    }
}

fn scale_velocity_by_density(velocity: &mut LetoArray3<f64>, rho0: &LetoArray3<f64>, half_dt: f64) {
    assert_eq!(
        velocity.shape(),
        rho0.shape(),
        "invariant: PSTD IVP velocity shape matches density shape"
    );

    let used_dense_path = match (velocity.as_slice_mut(), rho0.as_slice()) {
        (Some(velocity_values), Some(rho_values)) => {
            enumerate_mut_with::<Adaptive, _, _>(velocity_values, |index, velocity| {
                *velocity *= half_dt / rho_values[index];
            });
            true
        }
        _ => false,
    };

    if !used_dense_path {
        let [nx, ny, nz] = velocity.shape();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    velocity[[i, j, k]] *= half_dt / rho0[[i, j, k]];
                }
            }
        }
    }
}

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
        seed_density_components_from_pressure(
            DensitySeedFields {
                rhox: &mut self.rhox,
                rhoy: &mut self.rhoy,
                rhoz: &mut self.rhoz,
                pressure: &self.fields.p,
                c0: &self.materials.c0,
            },
            DensitySeedConfig {
                divisor,
                has_y,
                has_z,
            },
        );
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
    /// - Returns [`crate::KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
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
            write_spectral_gradient_axis(
                &mut self.grad_k,
                &self.source_kappa,
                &self.p_k,
                &self.ddx_k_shift_pos,
                GradientAxis::X,
            );
        }
        self.fft
            .inverse_c2r_into(&self.grad_k, &mut self.fields.ux, &mut self.ux_k);
        scale_velocity_by_density(&mut self.fields.ux, &self.materials.rho0, half_dt);

        if has_y {
            write_spectral_gradient_axis(
                &mut self.grad_k,
                &self.source_kappa,
                &self.p_k,
                &self.ddy_k_shift_pos,
                GradientAxis::Y,
            );
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.fields.uy, &mut self.ux_k);
            scale_velocity_by_density(&mut self.fields.uy, &self.materials.rho0, half_dt);
        } else {
            self.fields.uy.fill(0.0);
        }

        if has_z {
            // ddz has length nz_c (truncated in construction); k_idx ∈ [0, nz_c).
            write_spectral_gradient_axis(
                &mut self.grad_k,
                &self.source_kappa,
                &self.p_k,
                &self.ddz_k_shift_pos,
                GradientAxis::Z,
            );
            self.fft
                .inverse_c2r_into(&self.grad_k, &mut self.fields.uz, &mut self.ux_k);
            scale_velocity_by_density(&mut self.fields.uz, &self.materials.rho0, half_dt);
        } else {
            self.fields.uz.fill(0.0);
        }

        Ok(())
    }
}
