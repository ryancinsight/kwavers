//! PSTD k-Space Operators Implementation

use super::grid::PSTDKSGrid;
use crate::core::error::KwaversResult;
use crate::math::fft::{Complex64, ProcessorFft3d, Shape3D};
use ndarray::{Array3, Zip};

/// k-Space operators for PSTD spectral computations
#[derive(Debug, Clone)]
pub struct PSTDKSOperators {
    pub k_grid: PSTDKSGrid,
    pub fft_processor: std::sync::Arc<ProcessorFft3d>,
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
            fft_processor: std::sync::Arc::new(ProcessorFft3d::new(Shape3D { nx, ny, nz })),
        }
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
    ///
    /// # Panics
    /// - Panics if `kx must be contiguous`.
    ///
    pub fn spectral_grad_x(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let kx_s = self.k_grid.kx.as_slice().expect("kx must be contiguous");
        Zip::indexed(k_field.view_mut())
            .par_for_each(|(i, _, _), v| *v *= Complex64::new(0.0, kx_s[i]));
        self.inverse_fft_3d(&k_field)
    }

    /// Spectral y-derivative: `IFFT(i·ky · FFT(field))`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `ky must be contiguous`.
    ///
    pub fn spectral_grad_y(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let ky_s = self.k_grid.ky.as_slice().expect("ky must be contiguous");
        Zip::indexed(k_field.view_mut())
            .par_for_each(|(_, j, _), v| *v *= Complex64::new(0.0, ky_s[j]));
        self.inverse_fft_3d(&k_field)
    }

    /// Spectral z-derivative: `IFFT(i·kz · FFT(field))`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if `kz must be contiguous`.
    ///
    pub fn spectral_grad_z(&self, field: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let kz_s = self.k_grid.kz.as_slice().expect("kz must be contiguous");
        Zip::indexed(k_field.view_mut())
            .par_for_each(|(_, _, k), v| *v *= Complex64::new(0.0, kz_s[k]));
        self.inverse_fft_3d(&k_field)
    }
}
