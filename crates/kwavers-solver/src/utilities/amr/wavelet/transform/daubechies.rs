//! Daubechies wavelet transform implementation.
//!
//! Implements Daubechies-N wavelet using filter coefficients.
//! The Daubechies family provides compact support and maximum
//! number of vanishing moments for given support length.
//!
//! References:
//! - Daubechies (1992): "Ten Lectures on Wavelets"
//! - Mallat (2008): "A Wavelet Tour of Signal Processing"

use kwavers_core::error::KwaversResult;
use leto::Array3;

use super::core::WaveletTransform;

impl WaveletTransform {
    /// Daubechies wavelet forward transform (1D)
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn daubechies_forward(
        &self,
        data: &mut Array3<f64>,
        order: usize,
    ) -> KwaversResult<()> {
        // Get Daubechies filter coefficients
        let h = Self::daubechies_coefficients(order);
        let g = Self::wavelet_highpass_from_lowpass(&h);

        // Apply 1D transform along each dimension
        let (nx, ny, nz) = data.dim();

        // Transform along x-axis
        for j in 0..ny {
            for k in 0..nz {
                let mut row: Vec<f64> = (0..nx).map(|i| data[[i, j, k]]).collect();
                Self::apply_wavelet_1d(&mut row, &h, &g);
                for (i, val) in row.iter().enumerate() {
                    data[[i, j, k]] = *val;
                }
            }
        }

        // Transform along y-axis
        for i in 0..nx {
            for k in 0..nz {
                let mut col: Vec<f64> = (0..ny).map(|j| data[[i, j, k]]).collect();
                Self::apply_wavelet_1d(&mut col, &h, &g);
                for (j, val) in col.iter().enumerate() {
                    data[[i, j, k]] = *val;
                }
            }
        }

        // Transform along z-axis
        for i in 0..nx {
            for j in 0..ny {
                let mut depth: Vec<f64> = (0..nz).map(|k| data[[i, j, k]]).collect();
                Self::apply_wavelet_1d(&mut depth, &h, &g);
                for (k, val) in depth.iter().enumerate() {
                    data[[i, j, k]] = *val;
                }
            }
        }

        Ok(())
    }

    /// Daubechies wavelet inverse transform
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn daubechies_inverse(
        &self,
        coeffs: &mut Array3<f64>,
        order: usize,
    ) -> KwaversResult<()> {
        // Get reconstruction filters (time-reversed analysis filters)
        let h = Self::daubechies_coefficients(order);
        let g = Self::wavelet_highpass_from_lowpass(&h);

        let (nx, ny, nz) = coeffs.dim();

        // Inverse transform along z-axis (reverse order from forward)
        for i in 0..nx {
            for j in 0..ny {
                let mut depth: Vec<f64> = (0..nz).map(|k| coeffs[[i, j, k]]).collect();
                Self::apply_inverse_wavelet_1d(&mut depth, &h, &g);
                for (k, val) in depth.iter().enumerate() {
                    coeffs[[i, j, k]] = *val;
                }
            }
        }

        // Inverse transform along y-axis
        for i in 0..nx {
            for k in 0..nz {
                let mut col: Vec<f64> = (0..ny).map(|j| coeffs[[i, j, k]]).collect();
                Self::apply_inverse_wavelet_1d(&mut col, &h, &g);
                for (j, val) in col.iter().enumerate() {
                    coeffs[[i, j, k]] = *val;
                }
            }
        }

        // Inverse transform along x-axis
        for j in 0..ny {
            for k in 0..nz {
                let mut row: Vec<f64> = (0..nx).map(|i| coeffs[[i, j, k]]).collect();
                Self::apply_inverse_wavelet_1d(&mut row, &h, &g);
                for (i, val) in row.iter().enumerate() {
                    coeffs[[i, j, k]] = *val;
                }
            }
        }

        Ok(())
    }

    /// Get Daubechies-N filter coefficients.
    /// Returns normalized lowpass filter coefficients.
    pub(super) fn daubechies_coefficients(order: usize) -> Vec<f64> {
        use std::f64::consts::FRAC_1_SQRT_2;

        match order {
            2 => vec![
                0.6830127018922193,
                1.1830127018922193,
                0.3169872981077807,
                -0.1830127018922193,
            ],
            4 => vec![
                0.482962913144534,
                0.836516303737808,
                0.224143868042013,
                -0.129409522551260,
            ],
            _ => {
                // Default to Haar for unsupported orders
                vec![FRAC_1_SQRT_2, FRAC_1_SQRT_2]
            }
        }
    }

    /// Generate highpass filter from lowpass using quadrature mirror relationship.
    pub(super) fn wavelet_highpass_from_lowpass(h: &[f64]) -> Vec<f64> {
        let n = h.len();
        let mut g = vec![0.0; n];
        for i in 0..n {
            g[i] = if i % 2 == 0 {
                h[n - 1 - i]
            } else {
                -h[n - 1 - i]
            };
        }
        g
    }
}
