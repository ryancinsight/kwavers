//! CDF (Cohen-Daubechies-Feauveau) biorthogonal wavelet transform and filter bank.
//!
//! CDF wavelets are symmetric and biorthogonal, commonly used in image
//! compression (e.g., JPEG2000 uses CDF 9/7).
//!
//! References:
//! - Cohen, Daubechies & Feauveau (1992): "Biorthogonal bases of compactly supported wavelets"

use kwavers_core::error::KwaversResult;
use leto::Array3;

use super::core::WaveletTransform;

impl WaveletTransform {
    /// CDF biorthogonal wavelet forward transform.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn cdf_forward(
        &self,
        data: &mut Array3<f64>,
        p: usize,
        q: usize,
    ) -> KwaversResult<()> {
        // Get CDF filter coefficients
        let (h_analysis, g_analysis) = Self::cdf_coefficients(p, q);

        let [nx, ny, nz] = data.shape();

        // Apply 3D separable transform
        for j in 0..ny {
            for k in 0..nz {
                let mut row: Vec<f64> = (0..nx).map(|i| data[[i, j, k]]).collect();
                Self::apply_wavelet_1d(&mut row, &h_analysis, &g_analysis);
                for (i, val) in row.iter().enumerate() {
                    data[[i, j, k]] = *val;
                }
            }
        }

        for i in 0..nx {
            for k in 0..nz {
                let mut col: Vec<f64> = (0..ny).map(|j| data[[i, j, k]]).collect();
                Self::apply_wavelet_1d(&mut col, &h_analysis, &g_analysis);
                for (j, val) in col.iter().enumerate() {
                    data[[i, j, k]] = *val;
                }
            }
        }

        for i in 0..nx {
            for j in 0..ny {
                let mut depth: Vec<f64> = (0..nz).map(|k| data[[i, j, k]]).collect();
                Self::apply_wavelet_1d(&mut depth, &h_analysis, &g_analysis);
                for (k, val) in depth.iter().enumerate() {
                    data[[i, j, k]] = *val;
                }
            }
        }

        Ok(())
    }

    /// CDF wavelet inverse transform.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn cdf_inverse(
        &self,
        coeffs: &mut Array3<f64>,
        p: usize,
        q: usize,
    ) -> KwaversResult<()> {
        // Get CDF synthesis filters
        let (h_synthesis, g_synthesis) = Self::cdf_synthesis_coefficients(p, q);

        let [nx, ny, nz] = coeffs.shape();

        // Inverse transform (reverse order)
        for i in 0..nx {
            for j in 0..ny {
                let mut depth: Vec<f64> = (0..nz).map(|k| coeffs[[i, j, k]]).collect();
                Self::apply_inverse_wavelet_1d(&mut depth, &h_synthesis, &g_synthesis);
                for (k, val) in depth.iter().enumerate() {
                    coeffs[[i, j, k]] = *val;
                }
            }
        }

        for i in 0..nx {
            for k in 0..nz {
                let mut col: Vec<f64> = (0..ny).map(|j| coeffs[[i, j, k]]).collect();
                Self::apply_inverse_wavelet_1d(&mut col, &h_synthesis, &g_synthesis);
                for (j, val) in col.iter().enumerate() {
                    coeffs[[i, j, k]] = *val;
                }
            }
        }

        for j in 0..ny {
            for k in 0..nz {
                let mut row: Vec<f64> = (0..nx).map(|i| coeffs[[i, j, k]]).collect();
                Self::apply_inverse_wavelet_1d(&mut row, &h_synthesis, &g_synthesis);
                for (i, val) in row.iter().enumerate() {
                    coeffs[[i, j, k]] = *val;
                }
            }
        }

        Ok(())
    }

    /// Get CDF p/q analysis filter coefficients.
    pub(super) fn cdf_coefficients(p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
        // CDF 5/3 (used in lossless JPEG2000)
        if p == 5 && q == 3 {
            let h = vec![-0.125, 0.25, 0.75, 0.25, -0.125];
            let g = vec![-0.5, 1.0, -0.5];
            return (h, g);
        }

        // CDF 9/7 (used in lossy JPEG2000)
        if p == 9 && q == 7 {
            let h = vec![
                0.026748757411,
                -0.016864118443,
                -0.078223266529,
                0.266864118443,
                0.602949018236,
                0.266864118443,
                -0.078223266529,
                -0.016864118443,
                0.026748757411,
            ];
            let g = vec![
                0.045635881557,
                -0.028771763114,
                -0.295635881557,
                0.557543526229,
                -0.295635881557,
                -0.028771763114,
                0.045635881557,
            ];
            return (h, g);
        }

        // Default to simple coefficients
        (vec![0.5, 1.0, 0.5], vec![-0.5, 1.0, -0.5])
    }

    /// Get CDF synthesis filter coefficients (for reconstruction).
    /// For biorthogonal wavelets, synthesis filters are time-reversed analysis filters.
    pub(super) fn cdf_synthesis_coefficients(p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
        let (h, g) = Self::cdf_coefficients(p, q);
        let h_syn: Vec<f64> = h.iter().rev().copied().collect();
        let g_syn: Vec<f64> = g.iter().rev().copied().collect();
        (h_syn, g_syn)
    }

    /// Apply 1D wavelet transform using filter bank.
    pub(super) fn apply_wavelet_1d(signal: &mut [f64], h: &[f64], g: &[f64]) {
        let n = signal.len();
        if n < 2 {
            return;
        }

        let mut approx = vec![0.0; n / 2];
        let mut detail = vec![0.0; n / 2];

        // Convolve with lowpass and highpass filters, then downsample
        for i in 0..n / 2 {
            let mut sum_h = 0.0;
            let mut sum_g = 0.0;

            for (k, &h_k) in h.iter().enumerate() {
                let idx = (2 * i + k) % n;
                sum_h += h_k * signal[idx];
            }

            for (k, &g_k) in g.iter().enumerate() {
                let idx = (2 * i + k) % n;
                sum_g += g_k * signal[idx];
            }

            approx[i] = sum_h;
            detail[i] = sum_g;
        }

        // Store results: approximation in first half, details in second half
        signal[0..n / 2].copy_from_slice(&approx);
        signal[n / 2..].copy_from_slice(&detail);
    }

    /// Apply 1D inverse wavelet transform.
    pub(super) fn apply_inverse_wavelet_1d(coeffs: &mut [f64], h: &[f64], g: &[f64]) {
        let n = coeffs.len();
        if n < 2 {
            return;
        }

        let half = n / 2;
        let mut reconstructed = vec![0.0; n];

        // Upsample and convolve
        for (i, recon_val) in reconstructed.iter_mut().enumerate().take(n) {
            let mut sum = 0.0;

            // Contribution from approximation coefficients
            for (k, &h_k) in h.iter().enumerate() {
                let idx = if i >= k { (i - k) / 2 } else { (n + i - k) / 2 } % half;
                if (i + (h.len()) - k).is_multiple_of(2) {
                    sum += h_k * coeffs[idx];
                }
            }

            // Contribution from detail coefficients
            for (k, &g_k) in g.iter().enumerate() {
                let idx = if i >= k { (i - k) / 2 } else { (n + i - k) / 2 } % half;
                if (i + (g.len()) - k).is_multiple_of(2) {
                    sum += g_k * coeffs[half + idx];
                }
            }

            *recon_val = sum;
        }

        coeffs.copy_from_slice(&reconstructed);
    }
}
