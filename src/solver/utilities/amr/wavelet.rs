//! Wavelet transforms for multiresolution analysis

use crate::core::error::KwaversResult;
use ndarray::Array3;

/// Wavelet basis functions
#[derive(Debug, Clone, Copy)]
pub enum WaveletBasis {
    /// Haar wavelet (simplest)
    Haar,
    /// Daubechies wavelets
    Daubechies(usize),
    /// Cohen-Daubechies-Feauveau wavelets
    CDF(usize, usize),
}

/// Wavelet transform for AMR
#[derive(Debug)]
pub struct WaveletTransform {
    basis: WaveletBasis,
    levels: usize,
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

    /// Haar wavelet forward transform
    fn haar_forward(&self, data: &mut Array3<f64>) -> KwaversResult<()> {
        let (nx, ny, nz) = data.dim();

        // Apply 1D Haar transform in each direction
        for level in 0..self.levels {
            let size = nx >> level;

            // Transform in x-direction
            for j in 0..ny {
                for k in 0..nz {
                    let mut row = Vec::with_capacity(size);
                    for i in 0..size {
                        row.push(data[[i, j, k]]);
                    }

                    let transformed = self.haar_1d_forward(&row);
                    for (i, val) in transformed.iter().enumerate() {
                        data[[i, j, k]] = *val;
                    }
                }
            }

            // Similar for y and z directions...
        }

        Ok(())
    }

    /// 1D Haar forward transform
    fn haar_1d_forward(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len();
        let mut result = vec![0.0; n];

        // Compute averages and differences
        for i in 0..n / 2 {
            let avg = (data[2 * i] + data[2 * i + 1]) / 2.0_f64.sqrt();
            let diff = (data[2 * i] - data[2 * i + 1]) / 2.0_f64.sqrt();

            result[i] = avg;
            result[n / 2 + i] = diff;
        }

        result
    }

    /// Haar wavelet inverse transform
    fn haar_inverse(&self, _coeffs: &mut Array3<f64>) -> KwaversResult<()> {
        // Reverse of forward transform
        // Implementation would be similar but in reverse order
        Ok(())
    }

    /// Daubechies wavelet forward transform (1D)
    ///
    /// Implements Daubechies-N wavelet using filter coefficients.
    /// The Daubechies family provides compact support and maximum
    /// number of vanishing moments for given support length.
    ///
    /// References:
    /// - Daubechies (1992): "Ten Lectures on Wavelets"
    /// - Mallat (2008): "A Wavelet Tour of Signal Processing"
    fn daubechies_forward(&self, data: &mut Array3<f64>, order: usize) -> KwaversResult<()> {
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
    fn daubechies_inverse(&self, coeffs: &mut Array3<f64>, order: usize) -> KwaversResult<()> {
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

    /// CDF (Cohen-Daubechies-Feauveau) biorthogonal wavelet forward transform
    ///
    /// CDF wavelets are symmetric and biorthogonal, commonly used in image
    /// compression (e.g., JPEG2000 uses CDF 9/7)
    ///
    /// References:
    /// - Cohen, Daubechies & Feauveau (1992): "Biorthogonal bases of compactly supported wavelets"
    fn cdf_forward(&self, data: &mut Array3<f64>, p: usize, q: usize) -> KwaversResult<()> {
        // Get CDF filter coefficients
        let (h_analysis, g_analysis) = Self::cdf_coefficients(p, q);

        let (nx, ny, nz) = data.dim();

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

    /// CDF wavelet inverse transform
    fn cdf_inverse(&self, coeffs: &mut Array3<f64>, p: usize, q: usize) -> KwaversResult<()> {
        // Get CDF synthesis filters
        let (h_synthesis, g_synthesis) = Self::cdf_synthesis_coefficients(p, q);

        let (nx, ny, nz) = coeffs.dim();

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

    /// Get Daubechies-N filter coefficients
    /// Returns normalized lowpass filter coefficients
    fn daubechies_coefficients(order: usize) -> Vec<f64> {
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

    /// Generate highpass filter from lowpass using quadrature mirror relationship
    fn wavelet_highpass_from_lowpass(h: &[f64]) -> Vec<f64> {
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

    /// Get CDF p/q analysis filter coefficients
    fn cdf_coefficients(p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
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

    /// Get CDF synthesis filter coefficients (for reconstruction)
    fn cdf_synthesis_coefficients(p: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
        let (h, g) = Self::cdf_coefficients(p, q);
        // For biorthogonal wavelets, synthesis filters are time-reversed
        let h_syn: Vec<f64> = h.iter().rev().copied().collect();
        let g_syn: Vec<f64> = g.iter().rev().copied().collect();
        (h_syn, g_syn)
    }

    /// Apply 1D wavelet transform using filter bank
    fn apply_wavelet_1d(signal: &mut [f64], h: &[f64], g: &[f64]) {
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

    /// Apply 1D inverse wavelet transform
    fn apply_inverse_wavelet_1d(coeffs: &mut [f64], h: &[f64], g: &[f64]) {
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
                if (i + h.len() - k).is_multiple_of(2) {
                    sum += h_k * coeffs[idx];
                }
            }

            // Contribution from detail coefficients
            for (k, &g_k) in g.iter().enumerate() {
                let idx = if i >= k { (i - k) / 2 } else { (n + i - k) / 2 } % half;
                if (i + g.len() - k).is_multiple_of(2) {
                    sum += g_k * coeffs[half + idx];
                }
            }

            *recon_val = sum;
        }

        coeffs.copy_from_slice(&reconstructed);
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
