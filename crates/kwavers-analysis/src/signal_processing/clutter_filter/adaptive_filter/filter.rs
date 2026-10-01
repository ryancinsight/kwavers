use super::types::{AdaptiveFilterConfig, CbrEstimationMethod, SubspaceSeparationMethod};
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::{Array1, Array2};

/// Adaptive clutter filter using eigendecomposition.
///
/// ## Algorithm
///
/// 1. Construct temporal covariance matrix from slow-time data
/// 2. Eigendecompose to separate clutter and blood subspaces
/// 3. Apply adaptive thresholding based on CBR estimation
/// 4. Reconstruct signal with clutter eigenmodes removed
///
/// ## Memory
///
/// The per-pixel working buffers (covariance matrix, sorted eigenvalues,
/// sort index, filtered signal) are allocated once per [`AdaptiveFilter::filter`]
/// call and reused across pixels, instead of five allocations per pixel. The
/// remaining per-pixel allocations live inside the eigensolver itself.
///
/// ## References
///
/// - Yu & Lovstakken (2010) — eigen-based clutter filter design
/// - Bjaerum et al. (2002) — clutter filter design for color flow imaging
#[derive(Debug)]
pub struct AdaptiveFilter {
    pub(super) config: AdaptiveFilterConfig,
    pub(super) cbr_history: Vec<f64>,
}

/// Per-call scratch buffers reused across pixels.
#[derive(Debug)]
struct PixelScratch {
    covariance: Array2<f64>,
    sorted_eigenvalues: Array1<f64>,
    indices: Vec<usize>,
    filtered: Array1<f64>,
}

impl PixelScratch {
    fn new(n_frames: usize) -> Self {
        Self {
            covariance: Array2::zeros((n_frames, n_frames)),
            sorted_eigenvalues: Array1::zeros(n_frames),
            indices: (0..n_frames).collect(),
            filtered: Array1::zeros(n_frames),
        }
    }
}

impl AdaptiveFilter {
    /// Create a new adaptive filter with the given configuration.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn new(config: AdaptiveFilterConfig) -> KwaversResult<Self> {
        if config.noise_floor_threshold <= 0.0 || config.noise_floor_threshold >= 1.0 {
            return Err(KwaversError::InvalidInput(
                "noise_floor_threshold must be in range (0, 1)".to_owned(),
            ));
        }
        if config.temporal_smoothing && config.smoothing_window < 1 {
            return Err(KwaversError::InvalidInput(
                "smoothing_window must be >= 1 when temporal_smoothing is enabled".to_owned(),
            ));
        }
        match config.separation_method {
            SubspaceSeparationMethod::FixedRank { clutter_rank } => {
                if clutter_rank == 0 {
                    return Err(KwaversError::InvalidInput(
                        "clutter_rank must be > 0".to_owned(),
                    ));
                }
            }
            SubspaceSeparationMethod::AdaptiveThreshold { decay_factor } => {
                if decay_factor <= 0.0 || decay_factor >= 1.0 {
                    return Err(KwaversError::InvalidInput(
                        "decay_factor must be in range (0, 1)".to_owned(),
                    ));
                }
            }
            SubspaceSeparationMethod::CbrBased { target_cbr_db } => {
                if target_cbr_db <= 0.0 {
                    return Err(KwaversError::InvalidInput(
                        "target_cbr_db must be > 0".to_owned(),
                    ));
                }
            }
        }
        Ok(Self {
            config,
            cbr_history: Vec::new(),
        })
    }

    /// Apply adaptive clutter filter to slow-time data.
    ///
    /// `slow_time_data` shape: `(n_pixels, n_frames)`.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    ///
    /// # Panics
    ///
    /// Panics if a caller-supplied shape or an internal analysis state violates
    /// the precondition required by this operation.
    pub fn filter(&mut self, slow_time_data: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let [n_pixels, n_frames] = slow_time_data.shape();

        if n_frames < 3 {
            return Err(KwaversError::InvalidInput(
                "Need at least 3 frames for adaptive filtering".to_owned(),
            ));
        }

        if let SubspaceSeparationMethod::FixedRank { clutter_rank } = self.config.separation_method
        {
            if clutter_rank >= n_frames {
                return Err(KwaversError::InvalidInput(format!(
                    "clutter_rank ({}) must be < n_frames ({})",
                    clutter_rank, n_frames
                )));
            }
        }

        let mut filtered_data = Array2::<f64>::zeros((n_pixels, n_frames));
        let mut scratch = PixelScratch::new(n_frames);

        for pixel_idx in 0..n_pixels {
            let signal = slow_time_data
                .index_axis::<1>(0, pixel_idx)
                .expect("row index within pixel bounds");
            self.filter_single_pixel(&signal, &mut scratch)?;
            for (t, &value) in scratch.filtered.iter().enumerate() {
                filtered_data[[pixel_idx, t]] = value;
            }
        }

        Ok(filtered_data)
    }

    fn filter_single_pixel(
        &mut self,
        signal: &leto::ArrayView1<f64>,
        scratch: &mut PixelScratch,
    ) -> KwaversResult<()> {
        let n_frames = signal.shape()[0];

        // Construct the temporal covariance R[i,j] = E[x(t+i) x(t+j)]. R
        // depends only on the lag |i-j|, so each lag's inner product is
        // evaluated once — in the same ascending-t order the previous
        // all-pairs loop used — and written to both triangles. The stored
        // values are bitwise identical to computing every (i, j) pair.
        let mut cov = scratch.covariance.view_mut();
        for lag in 0..n_frames {
            let count = n_frames - lag;
            let mut sum = 0.0;
            for t in 0..count {
                sum += signal[t] * signal[t + lag];
            }
            let value = sum / count as f64;
            for i in 0..count {
                cov[[i, i + lag]] = value;
                cov[[i + lag, i]] = value;
            }
        }

        // The temporal covariance is symmetric by construction (R[i,j] = R[j,i]),
        // so use the symmetric Jacobi eigensolver, which guarantees real
        // eigenvalues and orthonormal eigenvectors. The general complex QR
        // (`eig`) is unnecessary here and orders of magnitude slower on matrices
        // this size (>60 s per 120×120 vs milliseconds), which per-pixel across
        // an image made the adaptive filter unusable.
        let decomposition = leto_ops::symmetric_eigen_jacobi(&scratch.covariance.view())?;
        let eigenvalues = &decomposition.eigenvalues;
        let eigenvectors = &decomposition.eigenvectors;

        // Sort descending.
        let indices = &mut scratch.indices;
        for (i, index) in indices.iter_mut().enumerate() {
            *index = i;
        }
        indices.sort_by(|&i, &j| eigenvalues[j].total_cmp(&eigenvalues[i]));
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            scratch.sorted_eigenvalues[new_idx] = eigenvalues[old_idx];
        }

        let clutter_rank = self.determine_clutter_rank(&scratch.sorted_eigenvalues)?;
        let cbr = self.estimate_cbr(&scratch.sorted_eigenvalues, clutter_rank);
        self.cbr_history.push(cbr);

        // Project onto blood subspace: filtered = x − Σᵢ <x,eᵢ> eᵢ. Only the
        // first `clutter_rank` sorted eigenvector columns are consumed, so the
        // full sorted matrix is never materialized; column `indices[i]` holds
        // the same values the materialized matrix had at column `i`.
        let filtered = &mut scratch.filtered;
        for (dst, &src) in filtered.iter_mut().zip(signal.iter()) {
            *dst = src;
        }
        for i in 0..clutter_rank.min(n_frames) {
            let column = indices[i];
            let mut projection_coef = 0.0;
            for (idx, &x_val) in signal.iter().enumerate() {
                projection_coef += x_val * eigenvectors[[idx, column]];
            }
            for (idx, filtered_val) in filtered.iter_mut().enumerate() {
                *filtered_val -= projection_coef * eigenvectors[[idx, column]];
            }
        }

        Ok(())
    }

    fn determine_clutter_rank(&self, eigenvalues: &Array1<f64>) -> KwaversResult<usize> {
        let n_eigenvalues = eigenvalues.len();
        if n_eigenvalues == 0 {
            return Ok(0);
        }

        let max_eigenvalue = eigenvalues[0];
        let noise_threshold = max_eigenvalue * self.config.noise_floor_threshold;

        match self.config.separation_method {
            SubspaceSeparationMethod::FixedRank { clutter_rank } => {
                Ok(clutter_rank.min(n_eigenvalues))
            }
            SubspaceSeparationMethod::AdaptiveThreshold { decay_factor } => {
                let threshold = decay_factor * max_eigenvalue;
                for (i, &eigenvalue) in eigenvalues.iter().enumerate() {
                    if eigenvalue < threshold || eigenvalue < noise_threshold {
                        return Ok(i.max(1));
                    }
                }
                Ok((n_eigenvalues / 2).max(1))
            }
            SubspaceSeparationMethod::CbrBased { target_cbr_db } => {
                let target_cbr_linear = 10.0_f64.powf(target_cbr_db / 10.0);
                for rank in 1..n_eigenvalues {
                    let cbr = self.estimate_cbr(eigenvalues, rank);
                    if cbr <= target_cbr_linear {
                        return Ok(rank);
                    }
                }
                Ok((n_eigenvalues / 2).max(1))
            }
        }
    }

    fn estimate_cbr(&self, eigenvalues: &Array1<f64>, clutter_rank: usize) -> f64 {
        let n_eigenvalues = eigenvalues.len();
        if clutter_rank >= n_eigenvalues {
            return f64::INFINITY;
        }

        match self.config.cbr_estimation {
            CbrEstimationMethod::EigenvalueSum => {
                let clutter_power: f64 = eigenvalues.iter().take(clutter_rank).sum();
                let blood_power: f64 = eigenvalues.iter().skip(clutter_rank).sum();
                if blood_power > 0.0 {
                    clutter_power / blood_power
                } else {
                    f64::INFINITY
                }
            }
            CbrEstimationMethod::PowerRatio => {
                let clutter_mean: f64 =
                    eigenvalues.iter().take(clutter_rank).sum::<f64>() / clutter_rank.max(1) as f64;
                let blood_count = n_eigenvalues - clutter_rank;
                let blood_mean: f64 =
                    eigenvalues.iter().skip(clutter_rank).sum::<f64>() / blood_count.max(1) as f64;
                if blood_mean > 0.0 {
                    clutter_mean / blood_mean
                } else {
                    f64::INFINITY
                }
            }
            CbrEstimationMethod::MaximumLikelihood => {
                let clutter_prod: f64 = eigenvalues
                    .iter()
                    .take(clutter_rank)
                    .map(|&x| x.max(1e-12))
                    .product();
                let blood_prod: f64 = eigenvalues
                    .iter()
                    .skip(clutter_rank)
                    .map(|&x| x.max(1e-12))
                    .product();
                let clutter_geomean = clutter_prod.powf(1.0 / clutter_rank.max(1) as f64);
                let blood_count = n_eigenvalues - clutter_rank;
                let blood_geomean = blood_prod.powf(1.0 / blood_count.max(1) as f64);
                if blood_geomean > 0.0 {
                    clutter_geomean / blood_geomean
                } else {
                    f64::INFINITY
                }
            }
        }
    }

    /// Get the history of CBR estimates.
    #[must_use]
    pub fn cbr_history(&self) -> &[f64] {
        &self.cbr_history
    }

    /// Get the current estimated CBR in dB.
    #[must_use]
    pub fn current_cbr_db(&self) -> Option<f64> {
        self.cbr_history.last().map(|&cbr| 10.0 * cbr.log10())
    }

    /// Clear CBR history.
    pub fn clear_history(&mut self) {
        self.cbr_history.clear();
    }
}
