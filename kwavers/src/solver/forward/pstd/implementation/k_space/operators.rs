//! PSTD k-Space Operators Implementation

use super::grid::PSTDKSGrid;
use crate::core::error::KwaversResult;
use crate::math::fft::{Complex64, ProcessorFft3d};
use ndarray::{Array3, Zip};

/// k-Space operators for PSTD spectral computations
#[derive(Debug, Clone)]
pub struct PSTDKSOperators {
    pub k_grid: PSTDKSGrid,
    pub fft_processor: std::sync::Arc<ProcessorFft3d>,
}

impl PSTDKSOperators {
    pub fn new(k_grid: PSTDKSGrid) -> Self {
        let (nx, ny, nz) = k_grid.dimensions();
        Self {
            k_grid,
            fft_processor: std::sync::Arc::new(ProcessorFft3d::new(nx, ny, nz)),
        }
    }

    /// Apply Helmholtz operator: (∇² + k₀²)p in wavenumber domain
    pub fn apply_helmholtz(
        &self,
        field: &Array3<f64>,
        wavenumber: f64,
    ) -> KwaversResult<Array3<f64>> {
        let mut k_field = self.forward_fft_3d(field)?;
        let k0_sq = wavenumber.powi(2);

        Zip::from(&mut k_field)
            .and(&self.k_grid.k_mag)
            .for_each(|val, &k_mag| {
                let laplacian_k = -k_mag.powi(2);
                *val *= Complex64::new(laplacian_k + k0_sq, 0.0);
            });

        self.inverse_fft_3d(&k_field)
    }

    pub fn forward_fft_3d(&self, input: &Array3<f64>) -> KwaversResult<Array3<Complex64>> {
        let output = self.fft_processor.forward(input);
        Ok(output)
    }

    pub fn inverse_fft_3d(&self, input: &Array3<Complex64>) -> KwaversResult<Array3<f64>> {
        let output = self.fft_processor.inverse(input);
        Ok(output)
    }
}
