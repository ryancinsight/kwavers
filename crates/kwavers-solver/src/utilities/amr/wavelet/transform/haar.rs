//! Haar wavelet transform implementation.

use kwavers_core::error::KwaversResult;
use leto::Array3;

use super::core::WaveletTransform;

impl WaveletTransform {
    /// Haar wavelet forward transform
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn haar_forward(&self, data: &mut Array3<f64>) -> KwaversResult<()> {
        let [nx, ny, nz] = data.shape();

        // Apply a separable transform to the low-pass subvolume at each level.
        for level in 0..self.levels {
            let x_size = nx >> level;
            let y_size = ny >> level;
            let z_size = nz >> level;

            if x_size >= 2 {
                for j in 0..y_size {
                    for k in 0..z_size {
                        let row: Vec<f64> = (0..x_size).map(|i| data[[i, j, k]]).collect();
                        let mut transformed = vec![0.0; x_size];
                        Self::haar_1d_forward(&row, &mut transformed);
                        for (i, value) in transformed.into_iter().enumerate() {
                            data[[i, j, k]] = value;
                        }
                    }
                }
            }

            if y_size >= 2 {
                for i in 0..x_size {
                    for k in 0..z_size {
                        let column: Vec<f64> = (0..y_size).map(|j| data[[i, j, k]]).collect();
                        let mut transformed = vec![0.0; y_size];
                        Self::haar_1d_forward(&column, &mut transformed);
                        for (j, value) in transformed.into_iter().enumerate() {
                            data[[i, j, k]] = value;
                        }
                    }
                }
            }

            if z_size >= 2 {
                for i in 0..x_size {
                    for j in 0..y_size {
                        let depth: Vec<f64> = (0..z_size).map(|k| data[[i, j, k]]).collect();
                        let mut transformed = vec![0.0; z_size];
                        Self::haar_1d_forward(&depth, &mut transformed);
                        for (k, value) in transformed.into_iter().enumerate() {
                            data[[i, j, k]] = value;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// 1D Haar forward transform
    pub(super) fn haar_1d_forward(data: &[f64], result: &mut [f64]) {
        let n = data.len();

        // Compute averages and differences
        for i in 0..n / 2 {
            let avg = (data[2 * i] + data[2 * i + 1]) / 2.0_f64.sqrt();
            let diff = (data[2 * i] - data[2 * i + 1]) / 2.0_f64.sqrt();

            result[i] = avg;
            result[n / 2 + i] = diff;
        }

        if !n.is_multiple_of(2) {
            result[n - 1] = data[n - 1];
        }
    }

    /// Haar wavelet inverse transform
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn haar_inverse(&self, coeffs: &mut Array3<f64>) -> KwaversResult<()> {
        let [nx, ny, nz] = coeffs.shape();
        for level in (0..self.levels).rev() {
            let x_size = nx >> level;
            let y_size = ny >> level;
            let z_size = nz >> level;

            if z_size >= 2 {
                for i in 0..x_size {
                    for j in 0..y_size {
                        let mut depth: Vec<f64> = (0..z_size).map(|k| coeffs[[i, j, k]]).collect();
                        Self::haar_1d_inverse(&mut depth);
                        for (k, value) in depth.into_iter().enumerate() {
                            coeffs[[i, j, k]] = value;
                        }
                    }
                }
            }

            if y_size >= 2 {
                for i in 0..x_size {
                    for k in 0..z_size {
                        let mut column: Vec<f64> = (0..y_size).map(|j| coeffs[[i, j, k]]).collect();
                        Self::haar_1d_inverse(&mut column);
                        for (j, value) in column.into_iter().enumerate() {
                            coeffs[[i, j, k]] = value;
                        }
                    }
                }
            }

            if x_size >= 2 {
                for j in 0..y_size {
                    for k in 0..z_size {
                        let mut row: Vec<f64> = (0..x_size).map(|i| coeffs[[i, j, k]]).collect();
                        Self::haar_1d_inverse(&mut row);
                        for (i, value) in row.into_iter().enumerate() {
                            coeffs[[i, j, k]] = value;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn haar_1d_inverse(coeffs: &mut [f64]) {
        let n = coeffs.len();
        let half = n / 2;
        let mut reconstructed = vec![0.0; n];
        let inv_sqrt_two = 2.0_f64.sqrt().recip();
        for i in 0..half {
            let average = coeffs[i];
            let difference = coeffs[half + i];
            reconstructed[2 * i] = (average + difference) * inv_sqrt_two;
            reconstructed[2 * i + 1] = (average - difference) * inv_sqrt_two;
        }
        if !n.is_multiple_of(2) {
            reconstructed[n - 1] = coeffs[n - 1];
        }
        coeffs.copy_from_slice(&reconstructed);
    }
}
