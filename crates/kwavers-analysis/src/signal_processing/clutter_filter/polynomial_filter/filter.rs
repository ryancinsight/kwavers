//! `PolynomialFilter`: polynomial regression clutter filter implementation.

use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array2, SliceArg};

use super::config::PolynomialFilterConfig;

/// Polynomial regression clutter filter.
///
/// Removes slow-moving tissue clutter by fitting and subtracting polynomial
/// trends from temporal signals.
#[derive(Debug)]
pub struct PolynomialFilter {
    config: PolynomialFilterConfig,
}

impl PolynomialFilter {
    /// Create a new polynomial filter with given configuration.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: PolynomialFilterConfig) -> KwaversResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Apply polynomial regression filter to slow-time data.
    ///
    /// # Arguments
    ///
    /// * `slow_time_data` - Input data with shape (n_pixels, n_frames)
    ///
    /// # Returns
    ///
    /// Filtered data with same shape, where polynomial trend has been removed.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Number of frames is ≤ polynomial order
    /// - Numerical issues in polynomial fitting (singular matrix)
    ///
    /// # Algorithm
    ///
    /// For each pixel (spatial location):
    /// 1. Extract temporal signal x(t)
    /// 2. Construct Vandermonde matrix V for polynomial basis
    /// 3. Solve normal equations (VᵀV)a = Vᵀx for coefficients
    /// 4. Compute polynomial fit p(t) = Σᵢ aᵢtⁱ
    /// 5. Subtract: y(t) = x(t) - p(t)
    ///
    /// # Performance
    ///
    /// - Time complexity: O(n_pixels × n_frames × order²)
    /// - Space complexity: O(n_frames × order)
    pub fn filter(&self, slow_time_data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let [n_pixels, n_frames] = slow_time_data.shape();

        if n_frames <= self.config.polynomial_order {
            return Err(KwaversError::InvalidInput(format!(
                "Number of frames ({}) must be > polynomial order ({})",
                n_frames, self.config.polynomial_order
            )));
        }

        let mut time: Array1<f64> = Array1::from_iter((0..n_frames).map(|t| t as f64));

        if self.config.normalize_time {
            let max_time = (n_frames - 1) as f64;
            for t in time.iter_mut() {
                *t /= max_time;
            }
        }

        // Vandermonde matrix: V[t, i] = t^i for i = 0..=polynomial_order.
        let mut vandermonde = Array2::<f64>::zeros((n_frames, self.config.polynomial_order + 1));
        for (t_idx, &t) in time.iter().enumerate() {
            for power in 0..=self.config.polynomial_order {
                vandermonde[[t_idx, power]] = t.powi(power as i32);
            }
        }

        // Precompute (VᵀV)⁻¹Vᵀ — same for all pixels.
        let k = self.config.polynomial_order + 1;
        let vt = vandermonde
            .transpose([1, 0])
            .expect("2D transpose axes valid")
            .to_contiguous();
        let mut vtv = Array2::<f64>::zeros((k, k));
        leto_ops::matmul(&vt.view(), &vandermonde.view(), &mut vtv.view_mut())
            .expect("VᵀV matmul shapes conform");
        let vtv_inv = self.pseudo_inverse(&vtv)?;
        let mut projection = Array2::<f64>::zeros((k, n_frames));
        leto_ops::matmul(&vtv_inv.view(), &vt.view(), &mut projection.view_mut())
            .expect("projection matmul shapes conform");

        let mut filtered_data = Array2::<f64>::zeros((n_pixels, n_frames));

        for pixel_idx in 0..n_pixels {
            let signal = slow_time_data
                .slice_with::<1>(&[SliceArg::Index(pixel_idx as isize), SliceArg::All])
                .expect("pixel row slice valid")
                .to_contiguous();
            let mut coefficients = Array1::<f64>::zeros(k);
            leto_ops::matvec(
                &projection.view(),
                &signal.view(),
                &mut coefficients.view_mut(),
            )
            .expect("coefficients matvec shapes conform");
            let mut polynomial_fit = Array1::<f64>::zeros(n_frames);
            leto_ops::matvec(
                &vandermonde.view(),
                &coefficients.view(),
                &mut polynomial_fit.view_mut(),
            )
            .expect("polynomial-fit matvec shapes conform");

            for (t, (&original, &fit)) in signal.iter().zip(polynomial_fit.iter()).enumerate() {
                filtered_data[[pixel_idx, t]] = original - fit;
            }
        }

        Ok(filtered_data)
    }

    /// Compute inverse of a square matrix using Gauss-Jordan elimination with partial pivoting.
    ///
    /// Sufficient for typical polynomial orders (1–10); O(n³) per call.
    ///
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the matrix is not square.
    /// - Returns [`KwaversError::Numerical`] if the matrix is singular (pivot < 1e-12).
    fn pseudo_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let [n, m] = matrix.shape();

        if n != m {
            return Err(KwaversError::InvalidInput(
                "Matrix must be square for simple inversion".to_owned(),
            ));
        }

        let mut augmented = Array2::<f64>::zeros((n, 2 * n));

        // Build augmented matrix [A | I].
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = matrix[[i, j]];
                if i == j {
                    augmented[[i, j + n]] = 1.0;
                }
            }
        }

        // Gauss-Jordan elimination with partial pivoting.
        for col in 0..n {
            let mut max_row = col;
            let mut max_val = augmented[[col, col]].abs();

            for row in (col + 1)..n {
                if augmented[[row, col]].abs() > max_val {
                    max_row = row;
                    max_val = augmented[[row, col]].abs();
                }
            }

            if max_val < 1e-12 {
                return Err(KwaversError::Numerical(
                    kwavers_core::error::NumericalError::SolverFailed {
                        method: "Matrix inversion".to_owned(),
                        reason: "Matrix is singular or nearly singular".to_owned(),
                    },
                ));
            }

            if max_row != col {
                for j in 0..(2 * n) {
                    let temp = augmented[[col, j]];
                    augmented[[col, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            for row in 0..n {
                if row != col {
                    let factor = augmented[[row, col]] / augmented[[col, col]];
                    for j in 0..(2 * n) {
                        augmented[[row, j]] -= factor * augmented[[col, j]];
                    }
                }
            }

            let pivot = augmented[[col, col]];
            for j in 0..(2 * n) {
                augmented[[col, j]] /= pivot;
            }
        }

        let mut inverse = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, j + n]];
            }
        }

        Ok(inverse)
    }
}