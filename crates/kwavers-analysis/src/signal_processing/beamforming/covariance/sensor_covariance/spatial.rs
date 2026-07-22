use eunomia::Complex64;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array2, SliceArg};

/// Spatial smoothing for coherent source decorrelation (real-valued)
#[derive(Debug, Clone)]
pub struct SpatialSmoothing {
    pub subarray_size: usize,
}

impl SpatialSmoothing {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(subarray_size: usize) -> Self {
        Self { subarray_size }
    }
    /// Apply.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn apply(&self, covariance: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = covariance.shape()[0];

        if self.subarray_size >= n {
            return Ok(covariance.clone());
        }

        let num_subarrays = n - self.subarray_size + 1;
        let mut smoothed = Array2::zeros((self.subarray_size, self.subarray_size));

        for start_idx in 0..num_subarrays {
            let end_idx = start_idx + self.subarray_size;
            let sub_cov = covariance
                .slice_with::<2>(&[
                    SliceArg::Range {
                        start: Some(start_idx as isize),
                        end: Some(end_idx as isize),
                        step: 1,
                    },
                    SliceArg::Range {
                        start: Some(start_idx as isize),
                        end: Some(end_idx as isize),
                        step: 1,
                    },
                ])
                .expect("subarray slice within bounds");
            for i in 0..self.subarray_size {
                for j in 0..self.subarray_size {
                    smoothed[[i, j]] += sub_cov[[i, j]];
                }
            }
        }

        for x in smoothed.iter_mut() {
            *x /= num_subarrays as f64;
        }
        Ok(smoothed)
    }
}

/// Spatial smoothing for coherent source decorrelation (complex-valued).
#[derive(Debug, Clone)]
pub struct SpatialSmoothingComplex {
    pub subarray_size: usize,
}

impl SpatialSmoothingComplex {
    /// New.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(subarray_size: usize) -> Self {
        Self { subarray_size }
    }

    /// Apply spatial smoothing: `R_smooth = (1/L) ∑ R[start..start+p, start..start+p]`.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn apply(&self, covariance: &Array2<Complex64>) -> KwaversResult<Array2<Complex64>> {
        let n = covariance.shape()[0];
        if covariance.shape()[1] != n {
            return Err(KwaversError::InvalidInput(
                "SpatialSmoothingComplex::apply: covariance must be square".to_owned(),
            ));
        }

        if self.subarray_size >= n {
            return Ok(covariance.clone());
        }
        if self.subarray_size == 0 {
            return Err(KwaversError::InvalidInput(
                "SpatialSmoothingComplex::apply: subarray_size must be >= 1".to_owned(),
            ));
        }

        let num_subarrays = n - self.subarray_size + 1;
        let mut smoothed = Array2::<Complex64>::from_elem(
            (self.subarray_size, self.subarray_size),
            Complex64::default(),
        );

        for start_idx in 0..num_subarrays {
            let end_idx = start_idx + self.subarray_size;
            let sub_cov = covariance
                .slice_with::<2>(&[
                    SliceArg::Range {
                        start: Some(start_idx as isize),
                        end: Some(end_idx as isize),
                        step: 1,
                    },
                    SliceArg::Range {
                        start: Some(start_idx as isize),
                        end: Some(end_idx as isize),
                        step: 1,
                    },
                ])
                .expect("subarray slice within bounds");
            for i in 0..self.subarray_size {
                for j in 0..self.subarray_size {
                    smoothed[[i, j]] += sub_cov[[i, j]];
                }
            }
        }

        let inv = 1.0 / (num_subarrays as f64);
        for v in smoothed.iter_mut() {
            *v *= inv;
        }
        Ok(smoothed)
    }
}
