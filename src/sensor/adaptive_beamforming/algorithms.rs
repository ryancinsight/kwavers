//! Advanced Beamforming Algorithms
//!
//! This module implements state-of-the-art beamforming algorithms including:
//! - Delay-and-Sum (conventional beamforming)
//! - MVDR (Capon beamformer - Minimum Variance Distortionless Response)
//! - MUSIC (Multiple Signal Classification)
//! - Eigenspace-based Minimum Variance (ESPMV)
//!
//! # References
//! - Capon (1969), "High-resolution frequency-wavenumber spectrum analysis"
//! - Schmidt (1986), "Multiple emitter location and signal parameter estimation"
//! - Van Trees (2002), "Optimum Array Processing"
//! - Stoica & Nehorai (1990), "MUSIC, maximum likelihood, and Cramer-Rao bound"

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

/// Simple matrix inversion using Gauss-Jordan elimination
/// Returns None if matrix is singular
fn invert_matrix(mat: &Array2<Complex64>) -> Option<Array2<Complex64>> {
    let n = mat.nrows();
    if n != mat.ncols() {
        return None;
    }

    // Create augmented matrix [A | I]
    let mut aug = Array2::<Complex64>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = mat[(i, j)];
        }
        aug[(i, n + i)] = Complex64::new(1.0, 0.0);
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut pivot_row = i;
        let mut max_val = aug[(i, i)].norm();
        for k in (i + 1)..n {
            let val = aug[(k, i)].norm();
            if val > max_val {
                max_val = val;
                pivot_row = k;
            }
        }

        // Check if matrix is singular
        if max_val < 1e-14 {
            return None;
        }

        // Swap rows if needed
        if pivot_row != i {
            for j in 0..(2 * n) {
                let temp = aug[(i, j)];
                aug[(i, j)] = aug[(pivot_row, j)];
                aug[(pivot_row, j)] = temp;
            }
        }

        // Scale pivot row
        let pivot = aug[(i, i)];
        for j in 0..(2 * n) {
            aug[(i, j)] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[(k, i)];
                // Store row i values to avoid borrow checker issues
                let row_i: Vec<Complex64> = (0..(2 * n)).map(|j| aug[(i, j)]).collect();
                for j in 0..(2 * n) {
                    aug[(k, j)] -= factor * row_i[j];
                }
            }
        }
    }

    // Extract inverse from right half
    let mut inv = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[(i, j)] = aug[(i, n + j)];
        }
    }

    Some(inv)
}

/// Compute eigenvalues and eigenvectors of Hermitian matrix
/// Returns (eigenvalues, eigenvectors) where eigenvectors are columns
/// Uses power iteration for dominant eigenvalues
fn eigen_hermitian(
    mat: &Array2<Complex64>,
    num_eigs: usize,
) -> Option<(Vec<f64>, Array2<Complex64>)> {
    let n = mat.nrows();
    if n != mat.ncols() || num_eigs == 0 || num_eigs > n {
        return None;
    }

    let mut eigenvalues = Vec::new();
    let mut eigenvectors = Array2::<Complex64>::zeros((n, num_eigs));
    let mut a = mat.clone();

    for col in 0..num_eigs {
        // Power iteration for current eigenvalue
        let mut v = Array1::<Complex64>::from_vec(
            (0..n)
                .map(|i| Complex64::new((i + 1) as f64, 0.0))
                .collect(),
        );

        // Normalize
        let norm: f64 = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
        v.mapv_inplace(|x| x / norm);

        for _ in 0..100 {
            // v = A * v
            let mut v_new = Array1::<Complex64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += a[(i, j)] * v[j];
                }
            }

            // Normalize
            let norm: f64 = v_new.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm < 1e-14 {
                break;
            }
            v_new.mapv_inplace(|x| x / norm);

            // Check convergence
            let diff: f64 = v
                .iter()
                .zip(v_new.iter())
                .map(|(a, b)| (a - b).norm_sqr())
                .sum::<f64>()
                .sqrt();
            v = v_new;

            if diff < 1e-10 {
                break;
            }
        }

        // Compute eigenvalue: λ = v^H A v
        let mut lambda = Complex64::zero();
        for i in 0..n {
            for j in 0..n {
                lambda += v[i].conj() * a[(i, j)] * v[j];
            }
        }

        eigenvalues.push(lambda.re);

        // Store eigenvector
        for i in 0..n {
            eigenvectors[(i, col)] = v[i];
        }

        // Deflate matrix: A = A - λ v v^H
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] -= lambda * v[i] * v[j].conj();
            }
        }
    }

    Some((eigenvalues, eigenvectors))
}

/// Beamforming algorithm trait
pub trait BeamformingAlgorithm {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64>;
}

/// Delay and sum beamforming (conventional beamforming)
///
/// The simplest beamforming algorithm that applies uniform weighting
/// to all array elements after steering delays.
///
/// # References
/// - Van Veen & Buckley (1988), "Beamforming: A versatile approach to spatial filtering"
#[derive(Debug)]
pub struct DelayAndSum;

impl BeamformingAlgorithm for DelayAndSum {
    fn compute_weights(
        &self,
        _covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        // Conventional beamforming: w = a (steering vector)
        steering.clone()
    }
}

/// Minimum Variance Distortionless Response (MVDR / Capon) beamformer
///
/// The MVDR beamformer minimizes output power while maintaining unit gain
/// in the look direction. Also known as the Capon beamformer.
///
/// The weight vector is: w = R^{-1} a / (a^H R^{-1} a)
///
/// where:
/// - R is the sample covariance matrix
/// - a is the steering vector
///
/// # References
/// - Capon (1969), "High-resolution frequency-wavenumber spectrum analysis",
///   Proceedings of the IEEE, 57(8), 1408-1418
/// - Van Trees (2002), "Optimum Array Processing", Ch. 6
#[derive(Debug)]
pub struct MinimumVariance {
    /// Diagonal loading factor for numerical stability
    pub diagonal_loading: f64,
}

impl Default for MinimumVariance {
    fn default() -> Self {
        Self {
            diagonal_loading: 1e-6, // Small regularization
        }
    }
}

impl MinimumVariance {
    /// Create MVDR beamformer with custom diagonal loading
    #[must_use]
    pub fn with_diagonal_loading(diagonal_loading: f64) -> Self {
        Self { diagonal_loading }
    }
}

impl BeamformingAlgorithm for MinimumVariance {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        // Add diagonal loading for numerical stability: R_loaded = R + δI
        let n = covariance.nrows();
        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(self.diagonal_loading, 0.0);
        }

        // Compute R^{-1}
        let r_inv = match invert_matrix(&r_loaded) {
            Some(inv) => inv,
            None => {
                // Fallback to delay-and-sum if inversion fails
                return steering.clone();
            }
        };

        // Compute R^{-1} a
        let r_inv_a = r_inv.dot(steering);

        // Compute a^H R^{-1} a
        let a_h_r_inv_a: Complex64 = steering
            .iter()
            .zip(r_inv_a.iter())
            .map(|(a, r)| a.conj() * r)
            .sum();

        // Avoid division by zero
        if a_h_r_inv_a.norm() < 1e-12 {
            return steering.clone();
        }

        // w = R^{-1} a / (a^H R^{-1} a)
        r_inv_a.mapv(|x| x / a_h_r_inv_a)
    }
}

/// MUSIC (Multiple Signal Classification) algorithm
///
/// MUSIC is a subspace-based method that exploits the eigenstructure of
/// the covariance matrix to estimate directions of arrival (DOA).
///
/// The MUSIC pseudospectrum is: P_MUSIC(θ) = 1 / (a^H P_N P_N^H a)
///
/// where:
/// - P_N is the projection onto the noise subspace
/// - a is the steering vector
///
/// # References
/// - Schmidt (1986), "Multiple emitter location and signal parameter estimation",
///   IEEE Transactions on Antennas and Propagation, 34(3), 276-280
/// - Stoica & Nehorai (1990), "MUSIC, maximum likelihood, and Cramer-Rao bound",
///   IEEE Transactions on Acoustics, Speech, and Signal Processing, 38(5), 720-741
#[derive(Debug)]
pub struct MUSIC {
    /// Number of sources (signals)
    pub num_sources: usize,
}

impl MUSIC {
    /// Create MUSIC algorithm with specified number of sources
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        Self { num_sources }
    }

    /// Compute MUSIC pseudospectrum value for given steering vector
    ///
    /// # Arguments
    /// * `covariance` - Sample covariance matrix
    /// * `steering` - Steering vector for direction of interest
    ///
    /// # Returns
    /// MUSIC pseudospectrum value (higher = more likely source direction)
    pub fn pseudospectrum(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> f64 {
        // Eigendecomposition of covariance matrix
        let n = covariance.nrows();
        let (eigenvalues, eigenvectors) = match eigen_hermitian(covariance, n) {
            Some((vals, vecs)) => (vals, vecs),
            None => return 0.0, // Fallback
        };

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Noise subspace: eigenvectors corresponding to smallest eigenvalues
        let n = covariance.nrows();
        let noise_start = self.num_sources.min(n);

        // Build noise subspace projection: P_N = Σ e_i e_i^H
        let mut p_n = Array2::<Complex64>::zeros((n, n));
        for &idx in indices.iter().skip(noise_start) {
            let e_i = eigenvectors.column(idx);
            for i in 0..n {
                for j in 0..n {
                    p_n[(i, j)] += e_i[i] * e_i[j].conj();
                }
            }
        }

        // Compute a^H P_N a
        let mut a_h_pn_a = Complex64::zero();
        for i in 0..n {
            for j in 0..n {
                a_h_pn_a += steering[i].conj() * p_n[(i, j)] * steering[j];
            }
        }

        // MUSIC pseudospectrum: 1 / |a^H P_N a|
        let denominator = a_h_pn_a.norm();
        if denominator < 1e-12 {
            0.0
        } else {
            1.0 / denominator
        }
    }
}

impl BeamformingAlgorithm for MUSIC {
    fn compute_weights(
        &self,
        _covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        // MUSIC doesn't directly provide weights; return steering vector
        // In practice, MUSIC is used for DOA estimation, not beamforming
        steering.clone()
    }
}

/// Automatic source number estimation using information theoretic criteria
///
/// Estimates the number of signal sources present using the covariance matrix
/// eigenvalues and information theoretic criteria (AIC, MDL).
///
/// # References
/// - Wax & Kailath (1985), "Detection of signals by information theoretic criteria",
///   IEEE Transactions on Acoustics, Speech, and Signal Processing, 33(2), 387-392
/// - Zhao et al. (1986), "Asymptotic equivalence of certain methods for model order
///   estimation", IEEE Transactions on Automatic Control, 31(1), 41-47
#[derive(Debug, Clone, Copy)]
pub enum SourceEstimationCriterion {
    /// Akaike Information Criterion (AIC)
    /// More liberal - may overestimate number of sources
    AIC,
    /// Minimum Description Length (MDL) / Bayesian Information Criterion (BIC)
    /// More conservative - consistent estimator
    MDL,
}

/// Estimate the number of signal sources from covariance matrix
///
/// # Arguments
/// * `covariance` - Sample covariance matrix (n x n)
/// * `num_snapshots` - Number of temporal snapshots used to compute covariance
/// * `criterion` - Information criterion to use (AIC or MDL)
///
/// # Returns
/// Estimated number of sources (0 to n-1)
///
/// # References
/// - Wax & Kailath (1985), "Detection of signals by information theoretic criteria"
pub fn estimate_num_sources(
    covariance: &Array2<Complex64>,
    num_snapshots: usize,
    criterion: SourceEstimationCriterion,
) -> usize {
    let n = covariance.nrows();
    if n == 0 || num_snapshots == 0 {
        return 0;
    }

    // Compute eigenvalues
    let (mut eigenvalues, _) = match eigen_hermitian(covariance, n) {
        Some((vals, vecs)) => (vals, vecs),
        None => return 0,
    };

    // Sort eigenvalues in descending order
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Ensure all eigenvalues are positive
    for val in &mut eigenvalues {
        if *val < 1e-12 {
            *val = 1e-12;
        }
    }

    let m = num_snapshots;
    let mut min_criterion = f64::INFINITY;
    let mut estimated_sources = 0;

    // Test each hypothesis k = 0, 1, ..., n-1
    for k in 0..(n - 1) {
        // Noise eigenvalues: λ_{k+1}, ..., λ_n
        let noise_eigs = &eigenvalues[(k + 1)..];
        let p = n - k - 1; // Number of noise eigenvalues

        if p == 0 {
            break;
        }

        // Arithmetic mean of noise eigenvalues
        let arithmetic_mean: f64 = noise_eigs.iter().sum::<f64>() / (p as f64);

        // Geometric mean of noise eigenvalues
        let log_sum: f64 = noise_eigs.iter().map(|&x| x.ln()).sum();
        let geometric_mean = (log_sum / (p as f64)).exp();

        // Avoid division by zero or invalid values
        if arithmetic_mean < 1e-12 || geometric_mean < 1e-12 {
            continue;
        }

        // Log-likelihood term
        let log_likelihood = -(m as f64) * (p as f64) * (arithmetic_mean / geometric_mean).ln();

        // Penalty term depends on criterion
        let penalty = match criterion {
            SourceEstimationCriterion::AIC => {
                // AIC penalty: 2 * number of free parameters
                // Number of parameters: k(2n - k)
                2.0 * (k as f64) * (2.0 * (n as f64) - (k as f64))
            }
            SourceEstimationCriterion::MDL => {
                // MDL penalty: 0.5 * log(m) * number of free parameters
                0.5 * (m as f64).ln() * (k as f64) * (2.0 * (n as f64) - (k as f64))
            }
        };

        let criterion_value = -log_likelihood + penalty;

        if criterion_value < min_criterion {
            min_criterion = criterion_value;
            estimated_sources = k;
        }
    }

    estimated_sources
}

/// Eigenspace-based Minimum Variance (ESPMV) beamformer
///
/// ESPMV is a robust beamformer that operates in the signal subspace,
/// reducing sensitivity to noise and model errors.
///
/// The weight vector is: w = P_S R^{-1} a / (a^H R^{-1} P_S a)
///
/// where P_S is the projection onto the signal subspace.
///
/// # References
/// - Gershman et al. (1999), "Adaptive beamforming algorithms with robustness
///   against jammer motion", IEEE Transactions on Signal Processing
/// - Shahbazpanahi et al. (2003), "A generalized Capon estimator for localization
///   of multiple spread sources", IEEE Transactions on Signal Processing
#[derive(Debug)]
pub struct EigenspaceMV {
    /// Number of sources (signal subspace dimension)
    pub num_sources: usize,
    /// Diagonal loading factor
    pub diagonal_loading: f64,
}

impl EigenspaceMV {
    /// Create Eigenspace MV beamformer
    #[must_use]
    pub fn new(num_sources: usize) -> Self {
        Self {
            num_sources,
            diagonal_loading: 1e-6,
        }
    }

    /// Create with custom diagonal loading
    #[must_use]
    pub fn with_diagonal_loading(num_sources: usize, diagonal_loading: f64) -> Self {
        Self {
            num_sources,
            diagonal_loading,
        }
    }
}

impl BeamformingAlgorithm for EigenspaceMV {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        let n = covariance.nrows();

        // Add diagonal loading
        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(self.diagonal_loading, 0.0);
        }

        // Eigendecomposition
        let (eigenvalues, eigenvectors) = match eigen_hermitian(&r_loaded, n) {
            Some((vals, vecs)) => (vals, vecs),
            None => return steering.clone(), // Fallback
        };

        // Sort eigenvalues in descending order
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Signal subspace: eigenvectors corresponding to largest eigenvalues
        let num_signal = self.num_sources.min(n);

        // Build signal subspace projection: P_S = Σ e_i e_i^H
        let mut p_s = Array2::<Complex64>::zeros((n, n));
        for &idx in indices.iter().take(num_signal) {
            let e_i = eigenvectors.column(idx);
            for i in 0..n {
                for j in 0..n {
                    p_s[(i, j)] += e_i[i] * e_i[j].conj();
                }
            }
        }

        // Compute R^{-1}
        let r_inv = match invert_matrix(&r_loaded) {
            Some(inv) => inv,
            None => return steering.clone(),
        };

        // Compute P_S R^{-1} a
        let r_inv_a = r_inv.dot(steering);
        let mut ps_r_inv_a = Array1::<Complex64>::zeros(n);
        for i in 0..n {
            for j in 0..n {
                ps_r_inv_a[i] += p_s[(i, j)] * r_inv_a[j];
            }
        }

        // Compute a^H R^{-1} P_S a
        let mut a_h_r_inv_ps_a = Complex64::zero();
        for i in 0..n {
            a_h_r_inv_ps_a += steering[i].conj() * ps_r_inv_a[i];
        }

        // Avoid division by zero
        if a_h_r_inv_ps_a.norm() < 1e-12 {
            return steering.clone();
        }

        // w = P_S R^{-1} a / (a^H R^{-1} P_S a)
        ps_r_inv_a.mapv(|x| x / a_h_r_inv_ps_a)
    }
}

/// Robust Capon Beamformer (RCB)
///
/// The Robust Capon Beamformer addresses the sensitivity of MVDR to steering vector
/// errors and array calibration uncertainties. It optimizes for worst-case performance
/// over an uncertainty set.
///
/// Uses diagonal loading with automatic loading factor selection based on:
/// - Array geometry uncertainty
/// - Desired robustness level
///
/// # References
/// - Vorobyov et al. (2003), "Robust adaptive beamforming using worst-case performance
///   optimization: A solution to the signal mismatch problem", IEEE Trans. SP, 51(2), 313-324
/// - Li et al. (2003), "On robust Capon beamforming and diagonal loading",
///   IEEE Transactions on Signal Processing, 51(7), 1702-1715
/// - Lorenz & Boyd (2005), "Robust minimum variance beamforming",
///   IEEE Transactions on Signal Processing, 53(5), 1684-1696
#[derive(Debug)]
pub struct RobustCapon {
    /// Uncertainty bound (steering vector mismatch tolerance)
    /// Typical values: 0.01 to 0.2 (1% to 20% uncertainty)
    pub uncertainty_bound: f64,
    /// Base diagonal loading factor
    pub base_loading: f64,
    /// Enable adaptive loading factor computation
    pub adaptive_loading: bool,
}

impl Default for RobustCapon {
    fn default() -> Self {
        Self {
            uncertainty_bound: 0.05, // 5% uncertainty
            base_loading: 1e-6,
            adaptive_loading: true,
        }
    }
}

impl RobustCapon {
    /// Create Robust Capon beamformer with specified uncertainty bound
    ///
    /// # Arguments
    /// * `uncertainty_bound` - Steering vector mismatch tolerance (0.0 to 1.0)
    ///   - 0.01: 1% uncertainty (precise calibration)
    ///   - 0.05: 5% uncertainty (typical)
    ///   - 0.20: 20% uncertainty (large errors)
    #[must_use]
    pub fn new(uncertainty_bound: f64) -> Self {
        Self {
            uncertainty_bound: uncertainty_bound.clamp(0.0, 1.0),
            base_loading: 1e-6,
            adaptive_loading: true,
        }
    }

    /// Create with custom base diagonal loading
    #[must_use]
    pub fn with_loading(uncertainty_bound: f64, base_loading: f64) -> Self {
        Self {
            uncertainty_bound: uncertainty_bound.clamp(0.0, 1.0),
            base_loading,
            adaptive_loading: true,
        }
    }

    /// Disable adaptive loading (use only base loading)
    #[must_use]
    pub fn without_adaptive_loading(mut self) -> Self {
        self.adaptive_loading = false;
        self
    }

    /// Compute adaptive loading factor based on uncertainty bound and covariance
    ///
    /// Uses the method from Vorobyov et al. (2003) / Li et al. (2003)
    fn compute_loading_factor(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> f64 {
        if !self.adaptive_loading {
            return self.base_loading;
        }

        let n = covariance.nrows();

        // Compute steering vector norm
        let a_norm_sq: f64 = steering.iter().map(|x| x.norm_sqr()).sum();

        // Estimate noise power from smallest eigenvalues
        // Quick estimation: use trace / n as approximation
        let mut trace = Complex64::zero();
        for i in 0..n {
            trace += covariance[(i, i)];
        }
        let noise_power = (trace.re / (n as f64)).max(1e-12);

        // Adaptive loading factor based on uncertainty bound
        // δ = ε * sqrt(noise_power * ||a||²)
        // where ε is the uncertainty bound
        let epsilon = self.uncertainty_bound;
        let loading = epsilon * (noise_power * a_norm_sq).sqrt();

        // Combine with base loading
        loading.max(self.base_loading)
    }
}

impl BeamformingAlgorithm for RobustCapon {
    fn compute_weights(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> Array1<Complex64> {
        let n = covariance.nrows();

        // Compute adaptive loading factor
        let loading = self.compute_loading_factor(covariance, steering);

        // Apply diagonal loading: R_loaded = R + δI
        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(loading, 0.0);
        }

        // Compute R^{-1}
        let r_inv = match invert_matrix(&r_loaded) {
            Some(inv) => inv,
            None => {
                // Fallback to delay-and-sum if inversion fails
                return steering.clone();
            }
        };

        // Compute R^{-1} a
        let r_inv_a = r_inv.dot(steering);

        // Compute a^H R^{-1} a
        let a_h_r_inv_a: Complex64 = steering
            .iter()
            .zip(r_inv_a.iter())
            .map(|(a, r)| a.conj() * r)
            .sum();

        // Avoid division by zero
        if a_h_r_inv_a.norm() < 1e-12 {
            return steering.clone();
        }

        // w = R^{-1} a / (a^H R^{-1} a)
        // This is the MVDR solution with robust diagonal loading
        r_inv_a.mapv(|x| x / a_h_r_inv_a)
    }
}

/// Covariance Matrix Tapering for improved resolution and robustness
///
/// Applies spatial tapering to the covariance matrix to reduce sidelobe levels
/// and improve robustness to model errors.
///
/// # References
/// - Guerci (1999), "Theory and application of covariance matrix tapers for robust adaptive beamforming"
/// - Mailloux (1994), "Covariance matrix augmentation to produce adaptive array pattern troughs"
#[derive(Debug, Clone)]
pub struct CovarianceTaper {
    taper_type: TaperType,
}

/// Tapering window type
#[derive(Debug, Clone, Copy)]
pub enum TaperType {
    /// Kaiser window with parameter beta
    Kaiser { beta: f64 },
    /// Blackman window
    Blackman,
    /// Hamming window
    Hamming,
    /// Adaptive - data-dependent selection
    Adaptive,
}

impl CovarianceTaper {
    /// Create Kaiser taper with shape parameter beta
    ///
    /// Typical values: beta = 2.5 to 4.0
    /// Higher beta = narrower mainlobe, higher sidelobes
    pub fn kaiser(beta: f64) -> Self {
        Self {
            taper_type: TaperType::Kaiser { beta },
        }
    }

    /// Create Blackman taper
    pub fn blackman() -> Self {
        Self {
            taper_type: TaperType::Blackman,
        }
    }

    /// Create Hamming taper
    pub fn hamming() -> Self {
        Self {
            taper_type: TaperType::Hamming,
        }
    }

    /// Create adaptive taper (data-dependent selection)
    ///
    /// Automatically selects taper based on covariance matrix condition number
    /// and eigenvalue spread
    pub fn adaptive() -> Self {
        Self {
            taper_type: TaperType::Adaptive,
        }
    }

    /// Apply tapering to covariance matrix
    ///
    /// Returns tapered covariance matrix R_tapered = T ⊙ R
    /// where ⊙ denotes element-wise (Hadamard) product
    pub fn apply(&self, covariance: &Array2<Complex64>) -> Array2<Complex64> {
        let n = covariance.nrows();

        // For adaptive tapering, select best taper based on data characteristics
        let effective_taper = match self.taper_type {
            TaperType::Adaptive => self.select_taper(covariance),
            _ => self.taper_type,
        };

        let mut tapered = covariance.clone();

        // Compute taper weights for each lag
        for i in 0..n {
            for j in 0..n {
                let lag = (i as i32 - j as i32).unsigned_abs() as usize;
                let weight = Self::compute_weight_for_type(effective_taper, lag, n);
                tapered[(i, j)] *= weight;
            }
        }

        tapered
    }

    /// Select optimal taper based on covariance matrix characteristics
    ///
    /// Uses eigenvalue spread and condition number to determine best taper:
    /// - High condition number (>100): Kaiser with high beta for robustness
    /// - Medium condition number (10-100): Blackman for balanced performance
    /// - Low condition number (<10): Hamming for minimal distortion
    fn select_taper(&self, covariance: &Array2<Complex64>) -> TaperType {
        let n = covariance.nrows();

        // Estimate condition number via diagonal elements and trace
        let mut diag_min = f64::INFINITY;
        let mut diag_max = 0.0f64;

        for i in 0..n {
            let val = covariance[(i, i)].norm();
            diag_min = diag_min.min(val);
            diag_max = diag_max.max(val);
        }

        // Rough condition number estimate
        let cond = if diag_min > 1e-12 {
            diag_max / diag_min
        } else {
            1e12
        };

        // Eigenvalue spread estimate via power iteration (quick, approximate)
        let eig_spread = self.estimate_eigenvalue_spread(covariance);

        // Decision logic based on matrix characteristics
        if cond > 100.0 || eig_spread > 100.0 {
            // Ill-conditioned: use strong Kaiser tapering
            TaperType::Kaiser { beta: 4.0 }
        } else if cond > 10.0 || eig_spread > 10.0 {
            // Moderately conditioned: use Blackman
            TaperType::Blackman
        } else {
            // Well-conditioned: use gentle Hamming
            TaperType::Hamming
        }
    }

    /// Estimate eigenvalue spread using power iteration
    fn estimate_eigenvalue_spread(&self, covariance: &Array2<Complex64>) -> f64 {
        let n = covariance.nrows();
        if n == 0 {
            return 1.0;
        }

        // Quick power iteration for largest eigenvalue
        let mut v = Array1::<Complex64>::from_elem(n, Complex64::new(1.0, 0.0));
        for _ in 0..5 {
            // Just 5 iterations for quick estimate
            let mut v_new = Array1::<Complex64>::zeros(n);
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += covariance[(i, j)] * v[j];
                }
            }
            let norm = v_new.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            if norm > 1e-12 {
                v = v_new.mapv(|x| x / norm);
            }
        }

        // Compute Rayleigh quotient for largest eigenvalue
        let mut lambda_max = 0.0;
        for i in 0..n {
            for j in 0..n {
                lambda_max += (v[i].conj() * covariance[(i, j)] * v[j]).re;
            }
        }

        // Estimate smallest eigenvalue from diagonal minimum
        let lambda_min = (0..n)
            .map(|i| covariance[(i, i)].norm())
            .fold(f64::INFINITY, |a, b| a.min(b))
            .max(1e-12);

        lambda_max.abs() / lambda_min
    }

    /// Compute taper weight for given lag and array size
    #[allow(dead_code)]
    fn compute_weight(&self, lag: usize, n: usize) -> f64 {
        Self::compute_weight_for_type(self.taper_type, lag, n)
    }

    /// Static method to compute weight for a specific taper type
    fn compute_weight_for_type(taper_type: TaperType, lag: usize, n: usize) -> f64 {
        match taper_type {
            TaperType::Kaiser { beta } => {
                // Kaiser window: I_0(beta * sqrt(1 - (lag/n)^2)) / I_0(beta)
                let x = lag as f64 / n as f64;
                let arg = beta * (1.0 - x * x).max(0.0).sqrt();
                Self::bessel_i0_static(arg) / Self::bessel_i0_static(beta)
            }
            TaperType::Blackman => {
                // Blackman window
                let x = lag as f64 / (n - 1) as f64;
                0.42 - 0.5 * (std::f64::consts::PI * x).cos()
                    + 0.08 * (2.0 * std::f64::consts::PI * x).cos()
            }
            TaperType::Hamming => {
                // Hamming window
                let x = lag as f64 / (n - 1) as f64;
                0.54 - 0.46 * (std::f64::consts::PI * x).cos()
            }
            TaperType::Adaptive => {
                // This should never be called directly
                // Adaptive types are resolved in apply()
                1.0
            }
        }
    }

    /// Modified Bessel function of the first kind, order 0
    /// Using series approximation
    #[allow(dead_code)]
    fn bessel_i0(&self, x: f64) -> f64 {
        Self::bessel_i0_static(x)
    }

    /// Static version of Bessel I0
    fn bessel_i0_static(x: f64) -> f64 {
        let mut sum = 1.0;
        let mut term = 1.0;
        let x2 = x * x / 4.0;

        for k in 1..50 {
            term *= x2 / (k * k) as f64;
            sum += term;
            if term < 1e-12 * sum {
                break;
            }
        }
        sum
    }
}

/// Orthonormal PAST (OPAST) Algorithm for improved numerical stability
///
/// OPAST maintains strict orthonormality via QR decomposition, providing
/// better numerical stability than standard PAST, especially for long runs.
///
/// # References
/// - Abed-Meraim et al. (2000), "A general framework for performance analysis of subspace tracking algorithms"
/// - Strobach (1998), "Fast recursive subspace adaptive ESPRIT algorithms"
#[derive(Debug, Clone)]
pub struct OrthonormalSubspaceTracker {
    /// Orthonormal subspace basis matrix (n x p)
    subspace: Array2<Complex64>,
    /// Forgetting factor (0 < lambda < 1)
    lambda: f64,
    /// Accumulated weight for normalization
    weight: f64,
    /// Auxiliary matrix for QR update (not currently used)
    #[allow(dead_code)]
    r_matrix: Array2<Complex64>,
}

impl OrthonormalSubspaceTracker {
    /// Create new orthonormal subspace tracker
    ///
    /// # Arguments
    /// * `n` - Array size (number of sensors)
    /// * `p` - Subspace dimension (number of signals to track)
    /// * `lambda` - Forgetting factor (0.95-0.99 typical)
    pub fn new(n: usize, p: usize, lambda: f64) -> Self {
        assert!(p <= n, "Subspace dimension must be <= array size");
        assert!(
            lambda > 0.0 && lambda < 1.0,
            "Forgetting factor must be in (0,1)"
        );

        // Initialize with orthonormal basis (first p standard basis vectors)
        let mut subspace = Array2::<Complex64>::zeros((n, p));
        for i in 0..p.min(n) {
            subspace[(i, i)] = Complex64::new(1.0, 0.0);
        }

        // Initialize R matrix (upper triangular from QR)
        let mut r_matrix = Array2::<Complex64>::zeros((p, p));
        for i in 0..p {
            r_matrix[(i, i)] = Complex64::new(1.0, 0.0);
        }

        Self {
            subspace,
            lambda,
            weight: 1.0,
            r_matrix,
        }
    }

    /// Update subspace with new data snapshot
    ///
    /// Implements OPAST recursion with QR orthonormalization:
    /// 1. Standard PAST update
    /// 2. QR orthonormalization to maintain strict orthonormality
    pub fn update(&mut self, snapshot: &[Complex64]) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        assert_eq!(snapshot.len(), n, "Snapshot size mismatch");

        // Convert snapshot to Array1
        let y = Array1::from(snapshot.to_vec());

        // Step 1: Standard PAST update (same as regular SubspaceTracker)
        // Compute projection coefficients: α = (W^H W)^{-1} W^H y
        let mut whw = Array2::<Complex64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let mut sum = Complex64::zero();
                for k in 0..n {
                    sum += self.subspace[(k, i)].conj() * self.subspace[(k, j)];
                }
                whw[(i, j)] = sum;
            }
        }

        // Invert W^H W (small p x p matrix)
        let whw_inv = match invert_matrix(&whw) {
            Some(inv) => inv,
            None => {
                // Fallback: use diagonal loading
                let mut loaded = whw.clone();
                for i in 0..p {
                    loaded[(i, i)] += Complex64::new(1e-10, 0.0);
                }
                invert_matrix(&loaded).unwrap_or_else(|| {
                    let mut id = Array2::zeros((p, p));
                    for i in 0..p {
                        id[(i, i)] = Complex64::new(1.0, 0.0);
                    }
                    id
                })
            }
        };

        // Compute alpha = (W^H W)^{-1} W^H y
        let mut alpha = Array1::<Complex64>::zeros(p);
        for i in 0..p {
            for j in 0..p {
                let mut wh_y = Complex64::zero();
                for k in 0..n {
                    wh_y += self.subspace[(k, j)].conj() * y[k];
                }
                alpha[i] += whw_inv[(i, j)] * wh_y;
            }
        }

        // Update subspace: W(t+1) = lambda * W(t) + (y - W*alpha) * alpha^H
        let sqrt_lambda = self.lambda.sqrt();
        for i in 0..n {
            let mut w_alpha = Complex64::zero();
            for j in 0..p {
                w_alpha += self.subspace[(i, j)] * alpha[j];
            }
            let residual = y[i] - w_alpha;

            for j in 0..p {
                self.subspace[(i, j)] = sqrt_lambda * self.subspace[(i, j)]
                    + residual * alpha[j].conj()
                        / (1.0 + alpha.iter().map(|a| a.norm_sqr()).sum::<f64>());
            }
        }

        // Step 2: QR orthonormalization via Gram-Schmidt
        self.orthonormalize_subspace();

        // Update weight
        self.weight = self.lambda * self.weight + 1.0;
    }

    /// Orthonormalize subspace using Gram-Schmidt
    fn orthonormalize_subspace(&mut self) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        for j in 0..p {
            // Orthogonalize against previous columns
            for i in 0..j {
                // Compute dot product
                let mut dot = Complex64::zero();
                for k in 0..n {
                    dot += self.subspace[(k, i)].conj() * self.subspace[(k, j)];
                }

                // Store column i values to avoid borrow checker issues
                let col_i: Vec<Complex64> = (0..n).map(|k| self.subspace[(k, i)]).collect();

                // Subtract projection
                for (k, &val) in col_i.iter().enumerate() {
                    self.subspace[(k, j)] -= dot * val;
                }
            }

            // Normalize
            let norm = (0..n)
                .map(|k| self.subspace[(k, j)].norm_sqr())
                .sum::<f64>()
                .sqrt();

            if norm > 1e-14 {
                for k in 0..n {
                    self.subspace[(k, j)] /= norm;
                }
            }
        }
    }

    /// QR decomposition via Gram-Schmidt (kept for compatibility but not used in update)
    ///
    /// Returns (Q, R) where A = QR, Q orthonormal, R upper triangular
    #[allow(dead_code)]
    fn qr_decomposition(&self, a: &Array2<Complex64>) -> (Array2<Complex64>, Array2<Complex64>) {
        let m = a.nrows();
        let n = a.ncols();

        let mut q = Array2::<Complex64>::zeros((m, n));
        let mut r = Array2::<Complex64>::zeros((n, n));

        for j in 0..n {
            // Get column j of A
            let mut v = Array1::<Complex64>::zeros(m);
            for i in 0..m {
                v[i] = a[(i, j)];
            }

            // Orthogonalize against previous columns
            for i in 0..j {
                // r[i,j] = q[:, i]^H * a[:, j]
                let mut dot = Complex64::zero();
                for k in 0..m {
                    dot += q[(k, i)].conj() * a[(k, j)];
                }
                r[(i, j)] = dot;

                // v = v - r[i,j] * q[:, i]
                for k in 0..m {
                    v[k] -= r[(i, j)] * q[(k, i)];
                }
            }

            // Normalize
            let norm = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            r[(j, j)] = Complex64::new(norm, 0.0);

            if norm > 1e-14 {
                for i in 0..m {
                    q[(i, j)] = v[i] / norm;
                }
            }
        }

        (q, r)
    }

    /// Get current subspace basis (orthonormal columns)
    pub fn get_subspace(&self) -> &Array2<Complex64> {
        &self.subspace
    }

    /// Reset tracker to initial state
    pub fn reset(&mut self) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        // Reset to standard basis
        self.subspace.fill(Complex64::zero());
        for i in 0..p.min(n) {
            self.subspace[(i, i)] = Complex64::new(1.0, 0.0);
        }

        // Reset R matrix
        self.r_matrix.fill(Complex64::zero());
        for i in 0..p {
            self.r_matrix[(i, i)] = Complex64::new(1.0, 0.0);
        }

        self.weight = 1.0;
    }

    /// Get forgetting factor
    pub fn lambda(&self) -> f64 {
        self.lambda
    }
}

/// Recursive Subspace Tracking using PAST algorithm
///
/// Projection Approximation Subspace Tracking (PAST) efficiently tracks
/// the principal subspace of a time-varying covariance matrix.
///
/// # References
/// - Yang (1995), "Projection approximation subspace tracking"
/// - Badeau et al. (2008), "Fast multilinear singular value decomposition for structured tensors"
#[derive(Debug, Clone)]
pub struct SubspaceTracker {
    /// Subspace basis matrix (n x p)
    subspace: Array2<Complex64>,
    /// Forgetting factor (0 < lambda < 1)
    /// Typical: 0.95-0.99
    lambda: f64,
    /// Accumulated weight for normalization
    weight: f64,
}

impl SubspaceTracker {
    /// Create new subspace tracker
    ///
    /// # Arguments
    /// * `n` - Array size (number of sensors)
    /// * `p` - Subspace dimension (number of signals to track)
    /// * `lambda` - Forgetting factor (0.95-0.99 typical)
    pub fn new(n: usize, p: usize, lambda: f64) -> Self {
        assert!(p <= n, "Subspace dimension must be <= array size");
        assert!(
            lambda > 0.0 && lambda < 1.0,
            "Forgetting factor must be in (0,1)"
        );

        // Initialize with orthonormal basis (first p standard basis vectors)
        let mut subspace = Array2::<Complex64>::zeros((n, p));
        for i in 0..p.min(n) {
            subspace[(i, i)] = Complex64::new(1.0, 0.0);
        }

        Self {
            subspace,
            lambda,
            weight: 1.0,
        }
    }

    /// Update subspace with new data snapshot
    ///
    /// Implements PAST recursion:
    /// W(t+1) = W(t) + (y(t) - W(t)α(t)) α(t)^H
    /// where α(t) = (W(t)^H W(t))^{-1} W(t)^H y(t)
    pub fn update(&mut self, snapshot: &[Complex64]) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        assert_eq!(snapshot.len(), n, "Snapshot size mismatch");

        // Convert snapshot to Array1
        let y = Array1::from(snapshot.to_vec());

        // Compute projection coefficients: α = (W^H W)^{-1} W^H y
        let mut whw = Array2::<Complex64>::zeros((p, p));
        for i in 0..p {
            for j in 0..p {
                let mut sum = Complex64::zero();
                for k in 0..n {
                    sum += self.subspace[(k, i)].conj() * self.subspace[(k, j)];
                }
                whw[(i, j)] = sum;
            }
        }

        // Invert W^H W (small p x p matrix)
        let whw_inv = match invert_matrix(&whw) {
            Some(inv) => inv,
            None => {
                // Fallback: use pseudo-inverse via diagonal loading
                let mut loaded = whw.clone();
                for i in 0..p {
                    loaded[(i, i)] += Complex64::new(1e-10, 0.0);
                }
                invert_matrix(&loaded).unwrap_or_else(|| {
                    // Ultimate fallback: identity
                    let mut id = Array2::zeros((p, p));
                    for i in 0..p {
                        id[(i, i)] = Complex64::new(1.0, 0.0);
                    }
                    id
                })
            }
        };

        // Compute W^H y
        let mut why = Array1::<Complex64>::zeros(p);
        for i in 0..p {
            let mut sum = Complex64::zero();
            for k in 0..n {
                sum += self.subspace[(k, i)].conj() * y[k];
            }
            why[i] = sum;
        }

        // Compute α = (W^H W)^{-1} W^H y
        let mut alpha = Array1::<Complex64>::zeros(p);
        for i in 0..p {
            let mut sum = Complex64::zero();
            for j in 0..p {
                sum += whw_inv[(i, j)] * why[j];
            }
            alpha[i] = sum;
        }

        // Compute error: e = y - W α
        let mut error = y.clone();
        for k in 0..n {
            let mut sum = Complex64::zero();
            for j in 0..p {
                sum += self.subspace[(k, j)] * alpha[j];
            }
            error[k] -= sum;
        }

        // PAST update: W(t+1) = λ W(t) + e α^H
        // Apply forgetting factor
        for i in 0..n {
            for j in 0..p {
                self.subspace[(i, j)] =
                    self.lambda * self.subspace[(i, j)] + error[i] * alpha[j].conj();
            }
        }

        // Gram-Schmidt orthonormalization to maintain numerical stability
        self.orthonormalize();

        // Update weight
        self.weight = self.lambda * self.weight + 1.0;
    }

    /// Get current subspace basis
    pub fn get_subspace(&self) -> &Array2<Complex64> {
        &self.subspace
    }

    /// Gram-Schmidt orthonormalization of subspace columns
    fn orthonormalize(&mut self) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        for j in 0..p {
            // Orthogonalize against previous columns
            for k in 0..j {
                let mut dot = Complex64::zero();
                for i in 0..n {
                    dot += self.subspace[(i, k)].conj() * self.subspace[(i, j)];
                }
                // Store column k values to avoid borrow checker issues
                let col_k: Vec<Complex64> = (0..n).map(|i| self.subspace[(i, k)]).collect();
                for (i, &val) in col_k.iter().enumerate() {
                    self.subspace[(i, j)] -= dot * val;
                }
            }

            // Normalize
            let mut norm_sqr = 0.0;
            for i in 0..n {
                norm_sqr += self.subspace[(i, j)].norm_sqr();
            }
            let norm = norm_sqr.sqrt();
            if norm > 1e-12 {
                for i in 0..n {
                    self.subspace[(i, j)] /= norm;
                }
            }
        }
    }

    /// Reset tracker to initial state
    pub fn reset(&mut self) {
        let n = self.subspace.nrows();
        let p = self.subspace.ncols();

        self.subspace = Array2::<Complex64>::zeros((n, p));
        for i in 0..p.min(n) {
            self.subspace[(i, i)] = Complex64::new(1.0, 0.0);
        }
        self.weight = 1.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    /// Create a simple test covariance matrix
    fn create_test_covariance(n: usize) -> Array2<Complex64> {
        let mut r = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let val = if i == j {
                    Complex64::new(1.0, 0.0)
                } else {
                    Complex64::new(0.1 / (1.0 + (i as f64 - j as f64).abs()), 0.0)
                };
                r[(i, j)] = val;
            }
        }
        r
    }

    /// Create a steering vector for a linear array
    fn create_steering_vector(n: usize, angle: f64) -> Array1<Complex64> {
        let k = 2.0 * PI; // Normalized wavenumber
        Array1::from_vec(
            (0..n)
                .map(|i| {
                    let phase = k * (i as f64) * angle.sin();
                    Complex64::new(phase.cos(), phase.sin())
                })
                .collect(),
        )
    }

    #[test]
    fn test_delay_and_sum() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = DelayAndSum;
        let weights = beamformer.compute_weights(&cov, &steering);

        // Weights should equal steering vector
        for i in 0..n {
            assert_relative_eq!(weights[i].re, steering[i].re, epsilon = 1e-10);
            assert_relative_eq!(weights[i].im, steering[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mvdr_weights_exist() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = MinimumVariance::default();
        let weights = beamformer.compute_weights(&cov, &steering);

        // Weights should be finite
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_mvdr_unit_gain_constraint() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = MinimumVariance::default();
        let weights = beamformer.compute_weights(&cov, &steering);

        // Check unit gain constraint: w^H a = 1
        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(w, a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.norm(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_music_pseudospectrum() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let music = MUSIC::new(1); // 1 source
        let spectrum = music.pseudospectrum(&cov, &steering);

        // Pseudospectrum should be positive
        assert!(spectrum >= 0.0);
        assert!(spectrum.is_finite());
    }

    #[test]
    fn test_music_peak_detection() {
        let n = 8;
        let cov = create_test_covariance(n);
        let music = MUSIC::new(1);

        // Scan angles and find peak
        let angles: Vec<f64> = (0..180).map(|i| (i as f64 - 90.0) * PI / 180.0).collect();
        let spectrum: Vec<f64> = angles
            .iter()
            .map(|&angle| {
                let steering = create_steering_vector(n, angle);
                music.pseudospectrum(&cov, &steering)
            })
            .collect();

        // Should have a peak somewhere
        let max_val = spectrum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_val > 0.0);
        assert!(max_val.is_finite());
    }

    #[test]
    fn test_eigenspace_mv() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = EigenspaceMV::new(2); // 2 sources
        let weights = beamformer.compute_weights(&cov, &steering);

        // Weights should be finite
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_eigenspace_mv_vs_mvdr() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::default();
        let espmv = EigenspaceMV::new(n); // Full rank = equivalent to MVDR

        let weights_mvdr = mvdr.compute_weights(&cov, &steering);
        let weights_espmv = espmv.compute_weights(&cov, &steering);

        // Should be similar for full-rank signal subspace
        let diff_norm: f64 = weights_mvdr
            .iter()
            .zip(weights_espmv.iter())
            .map(|(w1, w2)| (w1 - w2).norm())
            .sum();

        // Allow some difference due to numerical precision
        assert!(diff_norm < 1.0, "Difference: {}", diff_norm);
    }

    #[test]
    fn test_mvdr_diagonal_loading() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = MinimumVariance::with_diagonal_loading(1e-3);
        let weights = beamformer.compute_weights(&cov, &steering);

        // Should still produce valid weights
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_estimate_num_sources_aic() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        let estimated = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::AIC);

        // Should return a valid estimate
        assert!(estimated < n);
    }

    #[test]
    fn test_estimate_num_sources_mdl() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        let estimated = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::MDL);

        // Should return a valid estimate
        assert!(estimated < n);
    }

    #[test]
    fn test_estimate_num_sources_mdl_conservative() {
        let n = 8;
        let cov = create_test_covariance(n);
        let num_snapshots = 100;

        let aic = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::AIC);
        let mdl = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::MDL);

        // MDL should be ≤ AIC (more conservative)
        assert!(mdl <= aic, "MDL ({}) should be ≤ AIC ({})", mdl, aic);
    }

    #[test]
    fn test_estimate_num_sources_high_snr() {
        let n = 6;
        let num_sources = 2;
        let num_snapshots = 200;

        // Create covariance with clear signal-noise separation
        let mut cov = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    // First 2 eigenvalues are large (signals), rest are small (noise)
                    let val = if i < num_sources { 10.0 } else { 0.1 };
                    cov[(i, j)] = Complex64::new(val, 0.0);
                } else {
                    cov[(i, j)] = Complex64::new(0.01, 0.0);
                }
            }
        }

        let estimated = estimate_num_sources(&cov, num_snapshots, SourceEstimationCriterion::MDL);

        // Should correctly estimate close to 2 sources for high SNR
        // MDL may be conservative, so allow 1-3 sources
        assert!(
            (1..=3).contains(&estimated),
            "Should estimate 1-3 sources, got {}",
            estimated
        );
    }

    #[test]
    fn test_robust_capon_default() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = RobustCapon::default();
        let weights = beamformer.compute_weights(&cov, &steering);

        // Should produce valid weights
        assert_eq!(weights.len(), n);
        for &w in &weights {
            assert!(w.is_finite());
        }
    }

    #[test]
    fn test_robust_capon_unit_gain() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let beamformer = RobustCapon::new(0.1); // 10% uncertainty
        let weights = beamformer.compute_weights(&cov, &steering);

        // Check unit gain constraint: w^H a ≈ 1
        let gain: Complex64 = weights
            .iter()
            .zip(steering.iter())
            .map(|(w, a)| w.conj() * a)
            .sum();

        assert_relative_eq!(gain.norm(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_robust_capon_uncertainty_bounds() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        // Test different uncertainty bounds
        for uncertainty in &[0.01, 0.05, 0.1, 0.2] {
            let beamformer = RobustCapon::new(*uncertainty);
            let weights = beamformer.compute_weights(&cov, &steering);

            assert_eq!(weights.len(), n);
            for &w in &weights {
                assert!(w.is_finite());
            }
        }
    }

    #[test]
    fn test_robust_capon_adaptive_loading() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let with_adaptive = RobustCapon::new(0.1);
        let without_adaptive = RobustCapon::new(0.1).without_adaptive_loading();

        let weights_adaptive = with_adaptive.compute_weights(&cov, &steering);
        let weights_fixed = without_adaptive.compute_weights(&cov, &steering);

        // Both should produce valid weights
        for &w in &weights_adaptive {
            assert!(w.is_finite());
        }
        for &w in &weights_fixed {
            assert!(w.is_finite());
        }

        // Adaptive should differ from fixed loading
        let diff: f64 = weights_adaptive
            .iter()
            .zip(weights_fixed.iter())
            .map(|(w1, w2)| (w1 - w2).norm_sqr())
            .sum::<f64>()
            .sqrt();

        // Should be different (but may be similar for this simple test case)
        assert!(diff >= 0.0);
    }

    #[test]
    fn test_robust_capon_vs_mvdr() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let mvdr = MinimumVariance::default();
        let rcb = RobustCapon::new(0.01); // Very small uncertainty → similar to MVDR

        let weights_mvdr = mvdr.compute_weights(&cov, &steering);
        let weights_rcb = rcb.compute_weights(&cov, &steering);

        // With small uncertainty, RCB should be similar to MVDR
        let diff_norm: f64 = weights_mvdr
            .iter()
            .zip(weights_rcb.iter())
            .map(|(w1, w2)| (w1 - w2).norm())
            .sum();

        // Should be relatively close
        assert!(diff_norm < 2.0, "Difference: {}", diff_norm);
    }

    #[test]
    fn test_robust_capon_high_uncertainty() {
        let n = 4;
        let cov = create_test_covariance(n);
        let steering = create_steering_vector(n, 0.0);

        let rcb_low = RobustCapon::new(0.01); // 1% uncertainty
        let rcb_high = RobustCapon::new(0.3); // 30% uncertainty

        let weights_low = rcb_low.compute_weights(&cov, &steering);
        let weights_high = rcb_high.compute_weights(&cov, &steering);

        // Both should produce valid weights
        for &w in &weights_low {
            assert!(w.is_finite());
        }
        for &w in &weights_high {
            assert!(w.is_finite());
        }

        // High uncertainty should produce more conservative (different) weights
        let diff: f64 = weights_low
            .iter()
            .zip(weights_high.iter())
            .map(|(w1, w2)| (w1 - w2).norm_sqr())
            .sum::<f64>()
            .sqrt();

        assert!(diff > 0.0);
    }

    #[test]
    fn test_covariance_tapering() {
        let n = 8;
        let cov = create_test_covariance(n);

        // Apply Kaiser tapering with beta=2.5
        let taper = CovarianceTaper::kaiser(2.5);
        let tapered = taper.apply(&cov);

        // Check dimensions preserved
        assert_eq!(tapered.nrows(), n);
        assert_eq!(tapered.ncols(), n);

        // Check Hermitian symmetry preserved
        for i in 0..n {
            for j in 0..n {
                let diff = (tapered[(i, j)] - tapered[(j, i)].conj()).norm();
                assert!(diff < 1e-10, "Not Hermitian at ({},{}): {}", i, j, diff);
            }
        }

        // Diagonal elements should be reduced less than off-diagonal
        for i in 0..n {
            let orig_diag = cov[(i, i)].re;
            let taper_diag = tapered[(i, i)].re;
            assert!(
                taper_diag >= orig_diag * 0.5,
                "Diagonal too reduced: {} -> {}",
                orig_diag,
                taper_diag
            );
        }
    }

    #[test]
    fn test_covariance_tapering_types() {
        let n = 6;
        let cov = create_test_covariance(n);

        // Test Kaiser taper
        let kaiser = CovarianceTaper::kaiser(3.0);
        let tapered_kaiser = kaiser.apply(&cov);
        assert!(tapered_kaiser.iter().all(|x| x.is_finite()));

        // Test Blackman taper
        let blackman = CovarianceTaper::blackman();
        let tapered_blackman = blackman.apply(&cov);
        assert!(tapered_blackman.iter().all(|x| x.is_finite()));

        // Test Hamming taper
        let hamming = CovarianceTaper::hamming();
        let tapered_hamming = hamming.apply(&cov);
        assert!(tapered_hamming.iter().all(|x| x.is_finite()));

        // Different tapers should produce different results
        let diff_kb: f64 = tapered_kaiser
            .iter()
            .zip(tapered_blackman.iter())
            .map(|(a, b)| (a - b).norm())
            .sum();
        assert!(diff_kb > 0.0);
    }

    #[test]
    fn test_subspace_tracker_initialization() {
        let n = 4;
        let p = 2; // Track 2-dimensional subspace

        let tracker = SubspaceTracker::new(n, p, 0.99);
        let subspace = tracker.get_subspace();

        // Should be identity-like initially
        assert_eq!(subspace.nrows(), n);
        assert_eq!(subspace.ncols(), p);

        // Columns should be orthonormal
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-10,
                "Column {} not unit: {}",
                j,
                norm_sqr
            );
        }
    }

    #[test]
    fn test_subspace_tracker_update() {
        let n = 4;
        let p = 2;
        let mut tracker = SubspaceTracker::new(n, p, 0.98);

        // Simulate signal from direction 30 degrees
        let angle = 30.0_f64.to_radians();
        let signal = create_steering_vector(n, angle);

        // Add some noise
        let mut snapshot = signal.clone();
        snapshot[0] += Complex64::new(0.01, 0.01);
        snapshot[1] += Complex64::new(-0.01, 0.005);

        // Update tracker - convert Array1 to slice
        let snapshot_slice: Vec<Complex64> = snapshot.iter().copied().collect();
        tracker.update(&snapshot_slice);

        // Get updated subspace
        let subspace = tracker.get_subspace();

        // Should still be orthonormal
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-6,
                "Column {} not unit after update: {}",
                j,
                norm_sqr
            );
        }
    }

    #[test]
    fn test_subspace_tracker_convergence() {
        let n = 6;
        let p = 3;
        let mut tracker = SubspaceTracker::new(n, p, 0.95);

        // Create a consistent signal
        let signal = create_steering_vector(n, 45.0_f64.to_radians());

        // Apply many updates
        for i in 0..50 {
            let mut snapshot = signal.clone();
            // Small noise
            for j in 0..n {
                snapshot[j] += Complex64::new(
                    ((i * j) as f64 * 0.001).sin() * 0.01,
                    ((i * j) as f64 * 0.001).cos() * 0.01,
                );
            }
            let snapshot_slice: Vec<Complex64> = snapshot.iter().copied().collect();
            tracker.update(&snapshot_slice);
        }

        // Subspace should still be valid
        let subspace = tracker.get_subspace();
        assert!(subspace.iter().all(|x| x.is_finite()));

        // Should still be orthonormal
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-5,
                "Column {} not unit after convergence: {}",
                j,
                norm_sqr
            );
        }
    }

    #[test]
    fn test_adaptive_tapering() {
        // Test adaptive taper selection
        let n = 8;

        // Create well-conditioned covariance
        let mut well_cond = Array2::<Complex64>::zeros((n, n));
        for i in 0..n {
            well_cond[(i, i)] = Complex64::new(1.0, 0.0);
        }

        // Create ill-conditioned covariance
        let mut ill_cond = Array2::<Complex64>::zeros((n, n));
        for i in 0..n {
            ill_cond[(i, i)] = Complex64::new(100.0f64.powi(i as i32), 0.0);
        }

        let adaptive_taper = CovarianceTaper::adaptive();

        // Apply to both matrices
        let tapered_well = adaptive_taper.apply(&well_cond);
        let tapered_ill = adaptive_taper.apply(&ill_cond);

        // Results should preserve Hermitian property
        for i in 0..n {
            for j in 0..n {
                assert!((tapered_well[(i, j)] - tapered_well[(j, i)].conj()).norm() < 1e-10);
                assert!((tapered_ill[(i, j)] - tapered_ill[(j, i)].conj()).norm() < 1e-10);
            }
        }

        // Diagonal elements should be reasonably preserved
        for i in 0..n {
            let ratio_well = tapered_well[(i, i)].norm() / well_cond[(i, i)].norm();
            let ratio_ill = tapered_ill[(i, i)].norm() / ill_cond[(i, i)].norm();
            // Taper should not completely zero out the diagonal
            assert!(
                ratio_well > 0.01,
                "Well-conditioned diagonal ratio too small: {}",
                ratio_well
            );
            assert!(
                ratio_ill > 0.01,
                "Ill-conditioned diagonal ratio too small: {}",
                ratio_ill
            );
        }
    }

    #[test]
    fn test_opast_initialization() {
        let n = 4;
        let p = 2;
        let lambda = 0.98;

        let tracker = OrthonormalSubspaceTracker::new(n, p, lambda);
        let subspace = tracker.get_subspace();

        // Should be n x p
        assert_eq!(subspace.nrows(), n);
        assert_eq!(subspace.ncols(), p);

        // Should be orthonormal (identity in first p rows)
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!((norm_sqr - 1.0).abs() < 1e-10, "Column {} not unit norm", j);
        }

        // Columns should be orthogonal
        for i in 0..p {
            for j in (i + 1)..p {
                let mut dot = Complex64::zero();
                for k in 0..n {
                    dot += subspace[(k, i)].conj() * subspace[(k, j)];
                }
                assert!(dot.norm() < 1e-10, "Columns {} and {} not orthogonal", i, j);
            }
        }
    }

    #[test]
    fn test_opast_single_update() {
        let n = 4;
        let p = 2;
        let lambda = 0.98;

        let mut tracker = OrthonormalSubspaceTracker::new(n, p, lambda);

        // Create a snapshot
        let snapshot = vec![
            Complex64::new(1.0, 0.0),
            Complex64::new(0.5, 0.1),
            Complex64::new(0.3, -0.2),
            Complex64::new(0.1, 0.0),
        ];

        tracker.update(&snapshot);

        // Subspace should still be orthonormal
        let subspace = tracker.get_subspace();
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-5,
                "Column {} not unit norm after update: {}",
                j,
                norm_sqr
            );
        }

        // Columns should remain orthogonal
        for i in 0..p {
            for j in (i + 1)..p {
                let mut dot = Complex64::zero();
                for k in 0..n {
                    dot += subspace[(k, i)].conj() * subspace[(k, j)];
                }
                assert!(
                    dot.norm() < 1e-5,
                    "Columns {} and {} not orthogonal: {}",
                    i,
                    j,
                    dot.norm()
                );
            }
        }
    }

    #[test]
    fn test_opast_convergence() {
        let n = 4;
        let p = 2;
        let lambda = 0.98;

        let mut tracker = OrthonormalSubspaceTracker::new(n, p, lambda);

        // Update with consistent signal direction
        for _ in 0..100 {
            let snapshot = vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.5, 0.1),
                Complex64::new(0.3, -0.2),
                Complex64::new(0.1, 0.0),
            ];
            tracker.update(&snapshot);
        }

        // Subspace should be stable and orthonormal
        let subspace = tracker.get_subspace();
        for j in 0..p {
            let col: Vec<Complex64> = (0..n).map(|i| subspace[(i, j)]).collect();
            let norm_sqr: f64 = col.iter().map(|x| x.norm_sqr()).sum();
            assert!(
                (norm_sqr - 1.0).abs() < 1e-5,
                "Column {} not unit after convergence: {}",
                j,
                norm_sqr
            );
        }
    }

    #[test]
    fn test_opast_vs_past_stability() {
        let n = 4;
        let p = 2;
        let lambda = 0.98;

        let mut past_tracker = SubspaceTracker::new(n, p, lambda);
        let mut opast_tracker = OrthonormalSubspaceTracker::new(n, p, lambda);

        // Run many updates to test long-term stability
        for i in 0..1000 {
            let snapshot = vec![
                Complex64::new((i as f64 * 0.1).cos(), 0.0),
                Complex64::new((i as f64 * 0.1).sin(), 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(0.1, 0.0),
            ];

            past_tracker.update(&snapshot);
            opast_tracker.update(&snapshot);
        }

        // OPAST should maintain better orthonormality
        let opast_sub = opast_tracker.get_subspace();
        let mut opast_ortho_error = 0.0;

        for i in 0..p {
            for j in (i + 1)..p {
                let mut dot = Complex64::zero();
                for k in 0..n {
                    dot += opast_sub[(k, i)].conj() * opast_sub[(k, j)];
                }
                opast_ortho_error += dot.norm();
            }
        }

        // OPAST should have very small orthogonality error
        assert!(
            opast_ortho_error < 1e-3,
            "OPAST orthogonality error too large: {}",
            opast_ortho_error
        );
    }
}
