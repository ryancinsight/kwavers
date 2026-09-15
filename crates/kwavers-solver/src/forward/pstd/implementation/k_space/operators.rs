//! PSTD k-Space Operators Implementation

use super::grid::PSTDKSGrid;
use kwavers_core::error::KwaversResult;
use kwavers_math::fft::{Complex64, Fft3d, Fft3dInOutExt, Shape3D};
use leto::Array3;
use leto::{Array1, Array3 as LetoArray3};
use moirai_parallel::{enumerate_mut_with, Adaptive};

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
}

fn apply_helmholtz_multiplier(field: &mut Array3<Complex64>, k_mag: &Array3<f64>, k0_sq: f64) {
    assert_eq!(
        field.shape(),
        k_mag.shape(),
        "invariant: PSTD Helmholtz field shape matches k-space magnitude grid"
    );

    if let (Some(field_values), Some(k_values)) = (field.as_slice_mut(), k_mag.as_slice()) {
        enumerate_mut_with::<Adaptive, _, _>(field_values, |index, value| {
            let k = k_values[index];
            *value *= k.mul_add(-k, k0_sq);
        });
    } else {
        let [nx, ny, nz] = field.shape();
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let k_value = k_mag[[i, j, k]];
                    field[[i, j, k]] *= k_value.mul_add(-k_value, k0_sq);
                }
            }
        }
    }
}

fn apply_spectral_axis_multiplier(
    field: &mut Array3<Complex64>,
    axis_values: &Array1<f64>,
    axis: SpectralAxis,
) {
    let [nx, ny, nz] = field.shape();
    let expected_axis_len = match axis {
        SpectralAxis::X => nx,
        SpectralAxis::Y => ny,
        SpectralAxis::Z => nz,
    };
    assert_eq!(
        axis_values.shape()[0],
        expected_axis_len,
        "invariant: PSTD spectral axis vector length matches selected field dimension"
    );

    if let (Some(field_values), Some(axis_slice)) = (field.as_slice_mut(), axis_values.as_slice()) {
        enumerate_mut_with::<Adaptive, _, _>(field_values, |linear_index, value| {
            let axis_index = axis.index(linear_index, ny, nz);
            *value *= Complex64::new(0.0, axis_slice[axis_index]);
        });
    } else {
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let axis_index = match axis {
                        SpectralAxis::X => i,
                        SpectralAxis::Y => j,
                        SpectralAxis::Z => k,
                    };
                    field[[i, j, k]] *= Complex64::new(0.0, axis_values[axis_index]);
                }
            }
        }
    }
}

fn leto_complex_field(field: &Array3<Complex64>) -> LetoArray3<Complex64> {
    let [nx, ny, nz] = field.shape();
    LetoArray3::from_shape_vec([nx, ny, nz], field.iter().copied().collect())
        .expect("PSTD complex field length must match Leto field shape")
}

fn leto_real_field(field: LetoArray3<f64>) -> Array3<f64> {
    let [nx, ny, nz] = field.shape();
    Array3::from_shape_vec([nx, ny, nz], field.into_vec())
        .expect("Leto real field length must match PSTD field shape")
}

fn clone_complex_field(field: LetoArray3<Complex64>) -> Array3<Complex64> {
    field.to_contiguous()
}

/// k-Space operators for PSTD spectral computations
#[derive(Debug, Clone)]
pub struct PSTDKSOperators {
    pub k_grid: PSTDKSGrid,
    pub fft_processor: std::sync::Arc<Fft3d>,
    /// State of the exact second-order leapfrog, built on the first FullKSpace
    /// step (its coefficient needs `c_ref·Δt`).
    leapfrog: Option<KSpaceLeapfrog>,
}

/// Buffers of the exact second-order k-space leapfrog
/// `pⁿ⁺¹ = 2cos(c·|k|·Δt)·pⁿ − pⁿ⁻¹ + sⁿ`.
///
/// Pressure is real and `|k|` is even in every axis, so the coefficient acts
/// on the half spectrum `(nx, ny, nz/2+1)` exactly: the Hermitian completion
/// of `coefficient · half` is `coefficient ·` the completion of `half`. The
/// step therefore runs on the half-spectrum pair, and every buffer here is
/// reused across steps — a step allocates nothing.
#[derive(Debug, Clone)]
pub(crate) struct KSpaceLeapfrog {
    /// `2·cos(c_ref·|k|·Δt)` over the half spectrum.
    pub(crate) coefficient: Array3<f64>,
    /// Half spectrum of pⁿ; the inverse transform consumes it every step.
    pub(crate) spectrum: Array3<Complex64>,
    /// pⁿ⁻¹, absent before the first step, which takes the zero-velocity
    /// initial-value half coefficient instead.
    pub(crate) previous: Option<Array3<f64>>,
    /// pⁿ⁺¹ under construction; it rotates with the pressure field.
    pub(crate) next: Array3<f64>,
}

impl PSTDKSOperators {
    /// New.
    ///
    ///
    /// # Panics
    ///
    /// Panics when a k-space grid dimension is zero: the FFT plan shape validates at
    /// construction, and a zero extent has no transform.
    #[must_use]
    pub fn new(k_grid: PSTDKSGrid) -> Self {
        let (nx, ny, nz) = k_grid.dimensions();
        Self {
            k_grid,
            fft_processor: std::sync::Arc::new(Fft3d::new(
                Shape3D::new(nx, ny, nz).expect("invariant: k-space grid dimensions are non-zero"),
            )),
            leapfrog: None,
        }
    }

    /// The transform plan beside the leapfrog state, which is built on first use
    /// from `c_ref·Δt`; later calls return the cached state unchanged.
    pub(crate) fn leapfrog(&mut self, c_ref: f64, dt: f64) -> (&Fft3d, &mut KSpaceLeapfrog) {
        let Self {
            k_grid,
            fft_processor,
            leapfrog,
        } = self;
        let state = leapfrog.get_or_insert_with(|| {
            let [nx, ny, nz] = k_grid.k_mag.shape();
            let depth = nz / 2 + 1;
            let cdt = c_ref * dt;
            let mut coefficient = Array3::zeros([nx, ny, depth]);
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..depth {
                        coefficient[[i, j, k]] = 2.0 * (cdt * k_grid.k_mag[[i, j, k]]).cos();
                    }
                }
            }
            KSpaceLeapfrog {
                coefficient,
                spectrum: Array3::zeros([nx, ny, depth]),
                previous: None,
                next: Array3::zeros([nx, ny, nz]),
            }
        });
        (fft_processor, state)
    }

    /// Apply Helmholtz operator: (∇² + k₀²)p in wavenumber domain
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn apply_helmholtz(
        &self,
        field: &LetoArray3<f64>,
        wavenumber: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let k0_sq = wavenumber.powi(2);

        apply_helmholtz_multiplier(&mut k_field, &self.k_grid.k_mag, k0_sq);

        self.inverse_fft_3d(&k_field)
    }
    /// Forward fft 3d.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn forward_fft_3d(&self, input: &LetoArray3<f64>) -> KwaversResult<Array3<Complex64>> {
        let output = self.fft_processor.forward(input);
        Ok(clone_complex_field(output))
    }
    /// Inverse fft 3d.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn inverse_fft_3d(&self, input: &Array3<Complex64>) -> KwaversResult<Array3<f64>> {
        let output = self.fft_processor.inverse(&leto_complex_field(input));
        Ok(leto_real_field(output))
    }

    // ── Spectral gradient operators ──────────────────────────────────────────

    /// Spectral x-derivative: `IFFT(i·kx · FFT(field))`.
    ///
    /// Used to convert a velocity-source mask into its pressure-equivalent
    /// contribution for the FullKSpace pressure-only wave equation.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn spectral_grad_x(&self, field: &LetoArray3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        apply_spectral_axis_multiplier(&mut k_field, &self.k_grid.kx, SpectralAxis::X);
        self.inverse_fft_3d(&k_field)
    }

    /// Spectral y-derivative: `IFFT(i·ky · FFT(field))`.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn spectral_grad_y(&self, field: &LetoArray3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        apply_spectral_axis_multiplier(&mut k_field, &self.k_grid.ky, SpectralAxis::Y);
        self.inverse_fft_3d(&k_field)
    }

    /// Spectral z-derivative: `IFFT(i·kz · FFT(field))`.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn spectral_grad_z(&self, field: &LetoArray3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        apply_spectral_axis_multiplier(&mut k_field, &self.k_grid.kz, SpectralAxis::Z);
        self.inverse_fft_3d(&k_field)
    }
}
