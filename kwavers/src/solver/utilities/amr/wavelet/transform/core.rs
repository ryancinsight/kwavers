//! WaveletTransform struct and public API.

use crate::core::error::KwaversResult;
use ndarray::Array3;

use super::super::types::WaveletBasis;

/// Wavelet transform for AMR
#[derive(Debug)]
pub struct WaveletTransform {
    pub(super) basis: WaveletBasis,
    pub(super) levels: usize,
}

impl WaveletTransform {
    /// Create a new wavelet transform
    #[must_use]
    pub fn new(basis: WaveletBasis, levels: usize) -> Self {
        Self { basis, levels }
    }

    /// Forward wavelet transform
    pub fn forward(&self, data: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut result = data.clone();

        match self.basis {
            WaveletBasis::Haar => self.haar_forward(&mut result)?,
            WaveletBasis::Daubechies(n) => self.daubechies_forward(&mut result, n)?,
            WaveletBasis::CDF(p, q) => self.cdf_forward(&mut result, p, q)?,
        }

        Ok(result)
    }

    /// Inverse wavelet transform
    pub fn inverse(&self, coeffs: &Array3<f64>) -> KwaversResult<Array3<f64>> {
        let mut result = coeffs.clone();

        match self.basis {
            WaveletBasis::Haar => self.haar_inverse(&mut result)?,
            WaveletBasis::Daubechies(n) => self.daubechies_inverse(&mut result, n)?,
            WaveletBasis::CDF(p, q) => self.cdf_inverse(&mut result, p, q)?,
        }

        Ok(result)
    }

    /// Threshold wavelet coefficients for compression
    pub fn threshold(&self, coeffs: &mut Array3<f64>, threshold: f64) {
        coeffs.mapv_inplace(|c| if c.abs() < threshold { 0.0 } else { c });
    }

    /// Compute compression ratio
    #[must_use]
    pub fn compression_ratio(&self, coeffs: &Array3<f64>, threshold: f64) -> f64 {
        let total = coeffs.len();
        let nonzero = coeffs.iter().filter(|&&c| c.abs() >= threshold).count();

        total as f64 / nonzero.max(1) as f64
    }
}
