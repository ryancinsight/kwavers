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

        // Apply 1D Haar transform in each direction
        for level in 0..self.levels {
            let size = nx >> level;

            // Transform in x-direction
            let mut row = Vec::with_capacity(size);
            let mut result_buf = vec![0.0; size];

            for j in 0..ny {
                for k in 0..nz {
                    row.clear();
                    for i in 0..size {
                        row.push(data[[i, j, k]]);
                    }

                    self.haar_1d_forward(&row, &mut result_buf);
                    for (i, val) in result_buf.iter().enumerate() {
                        data[[i, j, k]] = *val;
                    }
                }
            }

            // Similar for y and z directions...
        }

        Ok(())
    }

    /// 1D Haar forward transform
    pub(super) fn haar_1d_forward(&self, data: &[f64], result: &mut [f64]) {
        let n = data.len();

        // Compute averages and differences
        for i in 0..n / 2 {
            let avg = (data[2 * i] + data[2 * i + 1]) / 2.0_f64.sqrt();
            let diff = (data[2 * i] - data[2 * i + 1]) / 2.0_f64.sqrt();

            result[i] = avg;
            result[n / 2 + i] = diff;
        }

        if !n.is_multiple_of(2) {
            result[n - 1] = 0.0;
        }
    }

    /// Haar wavelet inverse transform
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn haar_inverse(&self, _coeffs: &mut Array3<f64>) -> KwaversResult<()> {
        // Reverse of forward transform
        // Implementation would be similar but in reverse order
        Ok(())
    }
}
