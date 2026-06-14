//! PSTD k-Space Operators Implementation

use super::grid::PSTDKSGrid;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_math::fft::{Complex64, Fft3d, Fft3dInOutExt, Shape3D};
use ndarray::{Array3, Zip};

/// k-Space operators for PSTD spectral computations
#[derive(Debug, Clone)]
pub struct PSTDKSOperators {
    pub k_grid: PSTDKSGrid,
    pub fft_processor: std::sync::Arc<Fft3d>,
    /// Previous-step pressure pⁿ⁻¹ for the exact second-order k-space leapfrog
    /// `pⁿ⁺¹ = 2cos(c·|k|·Δt)·pⁿ − pⁿ⁻¹`. `None` before the first FullKSpace step;
    /// the first step uses the zero-velocity IVP half-coefficient instead.
    pub p_prev: Option<Array3<f64>>,
    /// Exact homogeneous wave-propagation coefficient `2·cos(c_ref·|k|·Δt)` over the
    /// full spectrum, lazily built on the first FullKSpace step (needs `c_ref·Δt`).
    pub wave_coeff: Option<Array3<f64>>,
}

impl PSTDKSOperators {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(k_grid: PSTDKSGrid) -> Self {
        let (nx, ny, nz) = k_grid.dimensions();
        Self {
            k_grid,
            fft_processor: std::sync::Arc::new(Fft3d::new(Shape3D { nx, ny, nz })),
            p_prev: None,
            wave_coeff: None,
        }
    }

    /// Build (or return) the exact second-order propagation coefficient
    /// `2·cos(c_ref·|k|·Δt)` for every spectral bin. Computed once and cached in
    /// `wave_coeff`. Each bin depends only on its own `|k|` → race-free.
    pub fn ensure_wave_coeff(&mut self, c_ref: f64, dt: f64) -> &Array3<f64> {
        if self.wave_coeff.is_none() {
            let cdt = c_ref * dt;
            let coeff = self.k_grid.k_mag.mapv(|k| 2.0 * (cdt * k).cos());
            self.wave_coeff = Some(coeff);
        }
        self.wave_coeff
            .as_ref()
            .expect("invariant: wave_coeff populated immediately above")
    }

    /// Apply Helmholtz operator: (∇² + k₀²)p in wavenumber domain
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn apply_helmholtz(
        &self,
        field: &Array3<f64>,
        wavenumber: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let k0_sq = wavenumber.powi(2);

        Zip::from(&mut k_field)
            .and(&self.k_grid.k_mag)
            .par_for_each(|val, &k_mag| {
                *val *= k_mag.mul_add(-k_mag, k0_sq); // real-scalar multiply: 2 mults vs complex×complex (4+2)
            });

        self.inverse_fft_3d(&k_field)
    }
    /// Forward fft 3d.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn forward_fft_3d(&self, input: &Array3<f64>) -> KwaversResult<Array3<Complex64>> {
        let output = self.fft_processor.forward(input);
        Ok(output)
    }
    /// Inverse fft 3d.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn inverse_fft_3d(&self, input: &Array3<Complex64>) -> KwaversResult<Array3<f64>> {
        let output = self.fft_processor.inverse(input);
        Ok(output)
    }

    // ── Spectral gradient operators ──────────────────────────────────────────

    /// Spectral x-derivative: `IFFT(i·kx · FFT(field))`.
    ///
    /// Used to convert a velocity-source mask into its pressure-equivalent
    /// contribution for the FullKSpace pressure-only wave equation.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    /// - Returns [`KwaversError::InternalError`] if `kx` is not contiguous in memory.
    ///
    pub fn spectral_grad_x(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let kx_s = self
            .k_grid
            .kx
            .as_slice()
            .ok_or_else(|| KwaversError::InternalError("kx must be contiguous".into()))?;
        Zip::indexed(k_field.view_mut())
            .par_for_each(|(i, _, _), v| *v *= Complex64::new(0.0, kx_s[i]));
        self.inverse_fft_3d(&k_field)
    }

    /// Spectral y-derivative: `IFFT(i·ky · FFT(field))`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    /// - Returns [`KwaversError::InternalError`] if `ky` is not contiguous in memory.
    ///
    pub fn spectral_grad_y(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let ky_s = self
            .k_grid
            .ky
            .as_slice()
            .ok_or_else(|| KwaversError::InternalError("ky must be contiguous".into()))?;
        Zip::indexed(k_field.view_mut())
            .par_for_each(|(_, j, _), v| *v *= Complex64::new(0.0, ky_s[j]));
        self.inverse_fft_3d(&k_field)
    }

    /// Spectral z-derivative: `IFFT(i·kz · FFT(field))`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    /// - Returns [`KwaversError::InternalError`] if `kz` is not contiguous in memory.
    ///
    pub fn spectral_grad_z(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let kz_s = self
            .k_grid
            .kz
            .as_slice()
            .ok_or_else(|| KwaversError::InternalError("kz must be contiguous".into()))?;
        Zip::indexed(k_field.view_mut())
            .par_for_each(|(_, _, k), v| *v *= Complex64::new(0.0, kz_s[k]));
        self.inverse_fft_3d(&k_field)
    }
}
