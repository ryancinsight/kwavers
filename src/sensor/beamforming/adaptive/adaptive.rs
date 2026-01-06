//! Adaptive beamforming algorithms
//!
//! # Strict SSOT policy
//! This file contains a legacy adaptive implementation that historically duplicated numerics
//! (matrix inversion / eigendecomposition) and used silent fallbacks.
//!
//! Under **strict SSOT**, the canonical adaptive beamforming implementations live under
//! `crate::sensor::beamforming::adaptive::algorithms` and SSOT numerics live under
//! `crate::utils::linear_algebra`.
//!
//! Therefore, this legacy implementation is **feature-gated** and must not be compiled unless
//! `--features legacy_algorithms` is explicitly enabled.

#![cfg(feature = "legacy_algorithms")]

use ndarray::{Array1, Array2};
use num_complex::Complex64;
use num_traits::Zero;

use super::conventional::BeamformingAlgorithm;
use super::matrix_utils::invert_matrix;

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
/// ## Usage
///
/// ```rust
/// use kwavers::sensor::adaptive_beamforming::MinimumVariance;
///
/// // Create MVDR beamformer with default diagonal loading (1e-6)
/// let beamformer = MinimumVariance::default();
///
/// // Or specify custom diagonal loading for numerical stability
/// let custom_beamformer = MinimumVariance::with_diagonal_loading(1e-4);
///
/// // In practice, you would then use this with a covariance matrix and steering vector:
/// // let weights = beamformer.compute_weights(&covariance, &steering);
/// ```
///
/// ## Performance
///
/// - **Time Complexity**: O(N³) due to matrix inversion
/// - **Space Complexity**: O(N²) for covariance matrix storage
/// - **Suitable for**: Arrays with N ≤ 32 elements in real-time applications
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

    /// Compute condition number of the covariance matrix
    ///
    /// The condition number κ(R) = λ_max / λ_min indicates numerical stability.
    /// High condition numbers (>1000) suggest potential singularity issues.
    ///
    /// # Returns
    /// Condition number of the covariance matrix
    pub fn covariance_condition_number(covariance: &Array2<Complex64>) -> f64 {
        use super::matrix_utils::eigen_hermitian;

        let eigenvalues = match eigen_hermitian(covariance, covariance.nrows()) {
            Some((vals, _)) => vals,
            None => return f64::INFINITY, // Matrix is singular
        };

        // Find min and max eigenvalues (all should be real and positive for covariance)
        let mut min_eigen = f64::INFINITY;
        let mut max_eigen = 0.0f64;

        for &eigen in &eigenvalues {
            if eigen > 0.0 {
                min_eigen = min_eigen.min(eigen);
                max_eigen = max_eigen.max(eigen);
            }
        }

        if min_eigen > 0.0 && max_eigen > 0.0 {
            max_eigen / min_eigen
        } else {
            f64::INFINITY // Singular or invalid matrix
        }
    }

    /// Check if covariance matrix is well-conditioned for MVDR
    ///
    /// # Returns
    /// (is_well_conditioned, condition_number, recommended_loading)
    pub fn check_covariance_condition(covariance: &Array2<Complex64>) -> (bool, f64, f64) {
        let condition_number = Self::covariance_condition_number(covariance);

        // Thresholds for well-conditioned matrices
        let is_well_conditioned = condition_number < 1000.0;

        // Recommended diagonal loading based on condition number
        let recommended_loading = if condition_number > 100.0 {
            // High condition number - use larger loading
            (condition_number / 1000.0).sqrt() * 1e-4
        } else {
            // Normal condition number - use default loading
            1e-6
        };

        (is_well_conditioned, condition_number, recommended_loading)
    }

    /// Compute MVDR weights with condition monitoring
    ///
    /// Returns weights along with condition information for diagnostics
    pub fn compute_weights_with_monitoring(
        &self,
        covariance: &Array2<Complex64>,
        steering: &Array1<Complex64>,
    ) -> (Array1<Complex64>, MVDRConditionInfo) {
        let weights = self.compute_weights(covariance, steering);

        // Compute condition information
        let (is_well_conditioned, condition_number, recommended_loading) =
            Self::check_covariance_condition(covariance);

        let info = MVDRConditionInfo {
            condition_number,
            is_well_conditioned,
            recommended_loading,
            actual_loading: self.diagonal_loading,
            matrix_rank: Self::estimate_matrix_rank(covariance),
            matrix_size: covariance.nrows(),
        };

        (weights, info)
    }

    /// Estimate the rank of the covariance matrix
    fn estimate_matrix_rank(covariance: &Array2<Complex64>) -> usize {
        use super::matrix_utils::eigen_hermitian;

        let eigenvalues = match eigen_hermitian(covariance, covariance.nrows()) {
            Some((vals, _)) => vals,
            None => return 0,
        };

        // Count eigenvalues above numerical threshold
        let threshold = eigenvalues.iter().cloned().fold(0.0, f64::max) * 1e-12;
        eigenvalues.iter().filter(|&&e| e > threshold).count()
    }
}

/// Condition information for MVDR diagnostics
#[derive(Debug, Clone)]
pub struct MVDRConditionInfo {
    /// Condition number of covariance matrix
    pub condition_number: f64,
    /// Whether matrix is well-conditioned for MVDR
    pub is_well_conditioned: bool,
    /// Recommended diagonal loading factor
    pub recommended_loading: f64,
    /// Actual diagonal loading used
    pub actual_loading: f64,
    /// Estimated rank of covariance matrix
    pub matrix_rank: usize,
    /// Matrix size (N x N)
    pub matrix_size: usize,
}

impl BeamformingAlgorithm for MinimumVariance {
    fn compute_weights(
        &self,
        covariance: &ndarray::Array2<Complex64>,
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
        covariance: &ndarray::Array2<Complex64>,
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
        covariance: &ndarray::Array2<Complex64>,
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

/// Linearly Constrained Minimum Variance (LCMV) beamformer
///
/// LCMV generalizes MVDR by allowing multiple linear constraints beyond
/// the single unit-gain constraint. This enables null steering, derivative
/// constraints, and other advanced beamforming capabilities.
///
/// The optimization problem is:
/// minimize w^H R w subject to C^H w = f
///
/// where C is the constraint matrix and f is the response vector.
///
/// # Usage
///
/// ```rust
/// use kwavers::sensor::adaptive_beamforming::adaptive::LCMV;
/// use ndarray::{Array1, Array2};
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// // Create LCMV beamformer
/// let mut lcmv = LCMV::new();
///
/// // Example parameters
/// let n = 8; // 8-element array
/// let look_angle = PI / 6.0; // 30 degrees
/// let interference_angle = -PI / 4.0; // -45 degrees
///
/// // Create steering vectors (1D arrays)
/// let steering = Array1::<Complex64>::from_elem(n, Complex64::new(1.0, 0.0));
/// let interference_steering = Array1::<Complex64>::from_elem(n, Complex64::new(0.5, 0.5));
///
/// // Add unit gain constraint in look direction
/// lcmv.add_constraint(&steering, Complex64::new(1.0, 0.0));
///
/// // Add null constraint in interference direction
/// lcmv.add_constraint(&interference_steering, Complex64::new(0.0, 0.0));
///
/// // Create sample covariance matrix
/// let covariance = Array2::<Complex64>::eye(n);
///
/// // Compute weights
/// let weights = lcmv.compute_weights(&covariance);
/// assert_eq!(weights.len(), n);
/// ```
///
/// # References
/// - Frost, O. L. (1972), "An algorithm for linearly constrained adaptive array processing",
///   Proceedings of the IEEE, 60(8), 926-935
/// - Van Trees (2002), "Optimum Array Processing", Ch. 6
#[derive(Debug)]
pub struct LCMV {
    /// Constraint matrix (each column is a constraint vector)
    constraint_matrix: Array2<Complex64>,
    /// Response vector (desired response for each constraint)
    response_vector: Array1<Complex64>,
    /// Diagonal loading factor
    diagonal_loading: f64,
}

impl Default for LCMV {
    fn default() -> Self {
        Self {
            constraint_matrix: Array2::zeros((0, 0)),
            response_vector: Array1::zeros(0),
            diagonal_loading: 1e-6,
        }
    }
}

impl LCMV {
    /// Create a new LCMV beamformer
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create LCMV with custom diagonal loading
    #[must_use]
    pub fn with_diagonal_loading(diagonal_loading: f64) -> Self {
        Self {
            constraint_matrix: Array2::zeros((0, 0)),
            response_vector: Array1::zeros(0),
            diagonal_loading,
        }
    }

    /// Add a linear constraint
    ///
    /// # Arguments
    /// * `constraint_vector` - The constraint vector (e.g., steering vector)
    /// * `desired_response` - Desired response for this constraint
    pub fn add_constraint(
        &mut self,
        constraint_vector: &Array1<Complex64>,
        desired_response: Complex64,
    ) {
        let n = constraint_vector.len();

        // Initialize constraint matrix if empty
        if self.constraint_matrix.is_empty() {
            self.constraint_matrix = Array2::zeros((n, 0));
            self.response_vector = Array1::zeros(0);
        }

        // Add new constraint column
        let mut new_matrix = Array2::zeros((n, self.constraint_matrix.ncols() + 1));
        let mut new_response = Array1::zeros(self.response_vector.len() + 1);

        // Copy existing constraints
        for i in 0..n {
            for j in 0..self.constraint_matrix.ncols() {
                new_matrix[(i, j)] = self.constraint_matrix[(i, j)];
            }
        }
        new_response
            .slice_mut(ndarray::s![..self.response_vector.len()])
            .assign(&self.response_vector);

        // Add new constraint
        for i in 0..n {
            new_matrix[(i, self.constraint_matrix.ncols())] = constraint_vector[i];
        }
        new_response[self.response_vector.len()] = desired_response;

        self.constraint_matrix = new_matrix;
        self.response_vector = new_response;
    }

    /// Clear all constraints
    pub fn clear_constraints(&mut self) {
        self.constraint_matrix = Array2::zeros((0, 0));
        self.response_vector = Array1::zeros(0);
    }

    /// Get number of constraints
    #[must_use]
    pub fn num_constraints(&self) -> usize {
        self.constraint_matrix.ncols()
    }

    /// Compute LCMV weights for the given covariance matrix
    ///
    /// # Arguments
    /// * `covariance` - Sample covariance matrix
    ///
    /// # Returns
    /// LCMV weight vector
    pub fn compute_weights(&self, covariance: &Array2<Complex64>) -> Array1<Complex64> {
        if self.constraint_matrix.is_empty() {
            // No constraints - return zero vector
            return Array1::zeros(covariance.nrows());
        }

        let n = covariance.nrows();
        let _m = self.num_constraints();

        // Add diagonal loading
        let mut r_loaded = covariance.clone();
        for i in 0..n {
            r_loaded[(i, i)] += Complex64::new(self.diagonal_loading, 0.0);
        }

        // Compute R^{-1}
        let r_inv = match invert_matrix(&r_loaded) {
            Some(inv) => inv,
            None => {
                // Fallback: return uniform weights
                return Array1::from_elem(n, Complex64::new(1.0 / (n as f64).sqrt(), 0.0));
            }
        };

        // Compute C^H R^{-1} C (constraint matrix)
        let c_h: Array2<Complex64> = self.constraint_matrix.t().mapv(|x| x.conj());
        let r_inv_c: Array2<Complex64> = r_inv.dot(&self.constraint_matrix);
        let c_h_r_inv_c: Array2<Complex64> = c_h.dot(&r_inv_c);

        // Invert the constraint matrix
        let c_h_r_inv_c_inv = match invert_matrix(&c_h_r_inv_c) {
            Some(inv) => inv,
            None => {
                // Constraints are not linearly independent
                return Array1::from_elem(n, Complex64::new(1.0 / (n as f64).sqrt(), 0.0));
            }
        };

        // Compute weights: w = R^{-1} C (C^H R^{-1} C)^{-1} f
        let temp = c_h_r_inv_c_inv.dot(&self.response_vector);
        r_inv_c.dot(&temp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;
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
}
