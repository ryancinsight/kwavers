//! Covariance estimation for beamforming
//!
//! ## Mathematical Foundation
//! **Sample Covariance**: `R = (1/N) Σ x_n x_nᴴ`
//! **Forward-Backward Averaging**: Reduces correlation matrix bias for finite samples
//! **Spatial Smoothing**: Decorrelates coherent sources using subarray averaging
//! **Shrinkage**: Improves conditioning by mixing the sample covariance with a structured target
//!
//! ## Narrowband / complex-baseband note (MVDR/MUSIC correctness)
//! Narrowband adaptive methods (MVDR/Capon, MUSIC, ESMV) are naturally expressed on **complex**
//! snapshots `x_n ∈ ℂ^M` with a **Hermitian** covariance `R = E[x xᴴ]`.

use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array2;
use num_complex::Complex64;

mod post_process;
mod shrinkage;
mod spatial;
#[cfg(test)]
mod tests;

pub use post_process::CovariancePostProcess;
pub use spatial::{SpatialSmoothing, SpatialSmoothingComplex};

/// Covariance matrix estimator with multiple methods
#[derive(Debug, Clone)]
pub struct CovarianceEstimator {
    /// Use forward-backward averaging for improved estimation
    pub forward_backward_averaging: bool,
    /// Number of snapshots for averaging
    pub num_snapshots: usize,
    /// Post-processing strategy (shrinkage / spatial smoothing).
    pub post_process: CovariancePostProcess,
}

impl Default for CovarianceEstimator {
    fn default() -> Self {
        Self {
            forward_backward_averaging: true,
            num_snapshots: 1,
            post_process: CovariancePostProcess::default(),
        }
    }
}

impl CovarianceEstimator {
    /// Create new covariance estimator
    #[must_use]
    pub fn new(forward_backward_averaging: bool, num_snapshots: usize) -> Self {
        Self {
            forward_backward_averaging,
            num_snapshots,
            post_process: CovariancePostProcess::default(),
        }
    }

    /// Create a new covariance estimator with an explicit post-processing policy.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn with_post_process(
        forward_backward_averaging: bool,
        num_snapshots: usize,
        post_process: CovariancePostProcess,
    ) -> Self {
        Self {
            forward_backward_averaging,
            num_snapshots,
            post_process,
        }
    }

    /// Estimate sample covariance matrix (real-valued): `R = (1/N) Σ x_n x_nᵀ`.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn estimate(&self, data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let (num_sensors, num_snapshots) = data.dim();

        if num_sensors == 0 || num_snapshots == 0 {
            return Err(KwaversError::InvalidInput(
                "CovarianceEstimator::estimate: data must have shape (num_sensors>0, num_snapshots>0)".to_owned(),
            ));
        }

        let mut covariance = Array2::zeros((num_sensors, num_sensors));

        for snapshot in 0..num_snapshots {
            let x = data.column(snapshot);
            for i in 0..num_sensors {
                for j in 0..num_sensors {
                    covariance[[i, j]] += x[i] * x[j];
                }
            }
        }

        covariance.par_mapv_inplace(|x| x / num_snapshots as f64);

        if self.forward_backward_averaging {
            covariance = self.apply_forward_backward_averaging(&covariance);
        }

        covariance = self.apply_post_process_real(&covariance)?;
        Ok(covariance)
    }

    /// Estimate sample covariance matrix (complex-valued): `R = (1/N) Σ x_n x_nᴴ`.
    ///
    /// Intended for narrowband adaptive processing (MVDR/Capon, MUSIC, ESMV) on complex
    /// baseband snapshots.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn estimate_complex(&self, data: &Array2<Complex64>) -> KwaversResult<Array2<Complex64>> {
        let (num_sensors, num_snapshots) = data.dim();

        if num_sensors == 0 || num_snapshots == 0 {
            return Err(KwaversError::InvalidInput(
                "CovarianceEstimator::estimate_complex: data must have shape (num_sensors>0, num_snapshots>0)".to_owned(),
            ));
        }

        let mut covariance = Array2::<Complex64>::zeros((num_sensors, num_sensors));

        for snapshot in 0..num_snapshots {
            let x = data.column(snapshot);
            for i in 0..num_sensors {
                for j in 0..num_sensors {
                    covariance[(i, j)] += x[i] * x[j].conj();
                }
            }
        }

        let inv_n = 1.0 / (num_snapshots as f64);
        covariance.par_mapv_inplace(|v| v * inv_n);

        if self.forward_backward_averaging {
            covariance = self.apply_forward_backward_averaging_complex(&covariance);
        }

        covariance = self.apply_post_process_complex(&covariance)?;
        Ok(covariance)
    }

    /// Apply forward-backward averaging (real-valued): `R_fb = 0.5 * (R + J R J)`.
    #[must_use]
    pub fn apply_forward_backward_averaging(&self, covariance: &Array2<f64>) -> Array2<f64> {
        let n = covariance.nrows();
        let mut fb_covariance = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let forward = covariance[[i, j]];
                let backward = covariance[[n - 1 - i, n - 1 - j]];
                fb_covariance[[i, j]] = 0.5 * (forward + backward);
            }
        }
        fb_covariance
    }

    /// Apply forward-backward averaging (complex-valued): `R_fb = 0.5 * (R + J R* J)`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn apply_forward_backward_averaging_complex(
        &self,
        covariance: &Array2<Complex64>,
    ) -> Array2<Complex64> {
        let n = covariance.nrows();
        let mut fb = Array2::<Complex64>::zeros((n, n));

        for i in 0..n {
            for j in 0..n {
                let forward = covariance[(i, j)];
                let backward = covariance[(n - 1 - i, n - 1 - j)].conj();
                fb[(i, j)] = (forward + backward) * 0.5;
            }
        }
        fb
    }

    fn apply_post_process_real(&self, covariance: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        self.post_process.validate()?;
        let m = covariance.nrows();
        if m == 0 || covariance.ncols() != m {
            return Err(KwaversError::InvalidInput(
                "CovarianceEstimator::apply_post_process_real: covariance must be square and non-empty".to_owned(),
            ));
        }

        match &self.post_process {
            CovariancePostProcess::None => Ok(covariance.clone()),
            CovariancePostProcess::ShrinkageToIdentity { alpha } => {
                Ok(shrinkage::shrinkage_to_identity_real(covariance, *alpha))
            }
            CovariancePostProcess::SpatialSmoothing { subarray_size } => {
                SpatialSmoothing::new(*subarray_size).apply(covariance)
            }
            CovariancePostProcess::ShrinkageThenSpatialSmoothing {
                alpha,
                subarray_size,
            } => {
                let shrunk = shrinkage::shrinkage_to_identity_real(covariance, *alpha);
                SpatialSmoothing::new(*subarray_size).apply(&shrunk)
            }
        }
    }

    fn apply_post_process_complex(
        &self,
        covariance: &Array2<Complex64>,
    ) -> KwaversResult<Array2<Complex64>> {
        self.post_process.validate()?;
        let m = covariance.nrows();
        if m == 0 || covariance.ncols() != m {
            return Err(KwaversError::InvalidInput(
                "CovarianceEstimator::apply_post_process_complex: covariance must be square and non-empty".to_owned(),
            ));
        }

        match &self.post_process {
            CovariancePostProcess::None => Ok(covariance.clone()),
            CovariancePostProcess::ShrinkageToIdentity { alpha } => {
                Ok(shrinkage::shrinkage_to_identity_complex(covariance, *alpha))
            }
            CovariancePostProcess::SpatialSmoothing { subarray_size } => {
                SpatialSmoothingComplex::new(*subarray_size).apply(covariance)
            }
            CovariancePostProcess::ShrinkageThenSpatialSmoothing {
                alpha,
                subarray_size,
            } => {
                let shrunk = shrinkage::shrinkage_to_identity_complex(covariance, *alpha);
                SpatialSmoothingComplex::new(*subarray_size).apply(&shrunk)
            }
        }
    }

    /// Estimate covariance matrix with spatial smoothing for coherent sources (real-valued only).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn estimate_with_spatial_smoothing(
        &self,
        data: &Array2<f64>,
        subarray_size: usize,
    ) -> KwaversResult<Array2<f64>> {
        let base_covariance = self.estimate(data)?;
        SpatialSmoothing::new(subarray_size).apply(&base_covariance)
    }

    /// Estimate covariance matrix with spatial smoothing for coherent sources (complex-valued).
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn estimate_complex_with_spatial_smoothing(
        &self,
        data: &Array2<Complex64>,
        subarray_size: usize,
    ) -> KwaversResult<Array2<Complex64>> {
        let base_covariance = self.estimate_complex(data)?;
        SpatialSmoothingComplex::new(subarray_size).apply(&base_covariance)
    }
}
