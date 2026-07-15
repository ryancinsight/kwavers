//! WaveletTransform struct and public API.

use crate::workspace::inplace_ops::apply_inplace;
use kwavers_core::error::KwaversResult;
use leto::Array3;

use super::super::types::WaveletBasis;

/// Wavelet transform for AMR
#[derive(Debug)]
pub struct WaveletTransform {
    pub(super) basis: WaveletBasis,
    pub(super) levels: usize,
}

impl WaveletTransform {
    /// Create a new wavelet transform
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(basis: WaveletBasis, levels: usize) -> Self {
        Self { basis, levels }
    }

    /// Forward wavelet transform
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
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
        apply_inplace(coeffs, |c| if c.abs() < threshold { 0.0 } else { c });
    }

    /// Compute compression ratio
    #[must_use]
    pub fn compression_ratio(&self, coeffs: &Array3<f64>, threshold: f64) -> f64 {
        let total = coeffs.len();
        let nonzero = coeffs.iter().filter(|&&c| c.abs() >= threshold).count();

        total as f64 / nonzero.max(1) as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_zeroes_only_coefficients_below_threshold() {
        let wavelet = WaveletTransform::new(WaveletBasis::Haar, 1);
        let mut coeffs = Array3::from_shape_vec([2, 2, 1], vec![-0.2, -0.05, 0.0, 0.3])
            .expect("invariant: shape matches coefficient count");

        wavelet.threshold(&mut coeffs, 0.1);

        assert_eq!(coeffs[[0, 0, 0]], -0.2);
        assert_eq!(coeffs[[0, 1, 0]], 0.0);
        assert_eq!(coeffs[[1, 0, 0]], 0.0);
        assert_eq!(coeffs[[1, 1, 0]], 0.3);
    }
}
