//! Beamforming Algorithms - Mathematical Theorems and Implementations
//!
//! ## Fundamental Theorems
//!
//! ### Delay-and-Sum Beamforming
//! **Theorem**: wᵢ = δ(t - τᵢ), where τᵢ is propagation delay
//! **Foundation**: Time-domain alignment of signals from array elements (Van Veen & Buckley 1988)
//! **Mathematical Basis**: Coherent summation maximizes SNR for plane waves from look direction
//!
//! ### Minimum Variance Distortionless Response (MVDR/Capon)
//! **Theorem**: w = (R⁻¹a)/(aᴴR⁻¹a), where R is covariance matrix, a is steering vector
//! **Foundation**: Optimizes array weights to minimize output power subject to unity gain constraint (Capon 1969)
//! **Mathematical Basis**: Solution to constrained optimization: min wᴴRw subject to wᴴa = 1
//!
//! ### Multiple Signal Classification (MUSIC)
//! **Theorem**: Pseudospectrum from noise subspace eigenvalues
//! **Foundation**: Signals and noise occupy different subspaces (Schmidt 1986)
//! **Mathematical Basis**: Eigendecomposition separates signal and noise subspaces
//!
//! ### Linearly Constrained Minimum Variance (LCMV)
//! **Theorem**: w = R⁻¹C(CᴴR⁻¹C)⁻¹f, where C is constraint matrix, f is response vector
//! **Foundation**: Generalizes MVDR with arbitrary linear constraints (Frost 1972)
//! **Mathematical Basis**: Quadratic programming with linear equality constraints
//!
//! ## Literature References
//! - Van Veen, B.D. & Buckley, K.M. (1988): "Beamforming: A versatile approach to spatial filtering"
//! - Capon, J. (1969): "High-resolution frequency-wavenumber spectrum analysis"
//! - Schmidt, R.O. (1986): "Multiple emitter location and signal parameter estimation"
//! - Frost, O.L. (1972): "An algorithm for linearly constrained adaptive array processing"

use crate::error::KwaversResult;
use ndarray::{Array1, Array2};

/// Beamforming algorithm types with literature-based implementations
#[derive(Debug, Clone)]
pub enum BeamformingAlgorithm {
    /// Delay-and-Sum (conventional beamforming)
    DelaySum,
    /// Minimum Variance Distortionless Response (MVDR/Capon)
    MVDR {
        diagonal_loading: f64,
        spatial_smoothing: bool,
    },
    /// `MUltiple` `SIgnal` Classification
    MUSIC {
        signal_subspace_dimension: usize,
        spatial_smoothing: bool,
    },
    /// Capon Beamforming with Regularization
    CaponRegularized {
        diagonal_loading: f64,
        uncertainty_set_size: f64,
    },
    /// Linearly Constrained Minimum Variance (LCMV)
    LCMV {
        constraint_matrix: Array2<f64>,
        response_vector: Array1<f64>,
    },
    /// Generalized Sidelobe Canceller (GSC)
    GSC {
        main_beam_weight: f64,
        adaptation_step_size: f64,
    },
    /// Compressive Beamforming
    Compressive {
        sparsity_parameter: f64,
        dictionary_size: usize,
    },
}

/// MVDR Beamforming Implementation
/// **Theorem**: w = (R⁻¹a)/(aᴴR⁻¹a), where R is covariance matrix, a is steering vector
/// **Foundation**: Optimizes array weights to minimize output power subject to unity gain constraint (Capon 1969)
/// **Mathematical Basis**: Solution to constrained optimization: min wᴴRw subject to wᴴa = 1
#[derive(Debug)]
pub struct MVDRBeamformer {
    pub diagonal_loading: f64,
    pub spatial_smoothing: bool,
}

impl MVDRBeamformer {
    /// Create new MVDR beamformer
    #[must_use]
    pub fn new(diagonal_loading: f64, spatial_smoothing: bool) -> Self {
        Self {
            diagonal_loading,
            spatial_smoothing,
        }
    }

    /// Compute MVDR weights: w = (R⁻¹a)/(aᴴR⁻¹a)
    /// where R is the covariance matrix and a is the steering vector
    pub fn compute_weights(
        &self,
        covariance_matrix: &Array2<f64>,
        steering_vector: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let n = covariance_matrix.nrows();

        // Apply diagonal loading for regularization: R_regularized = R + εI
        let mut r_regularized = covariance_matrix.clone();
        for i in 0..n {
            r_regularized[[i, i]] += self.diagonal_loading;
        }

        // Compute R⁻¹
        let r_inv = self.matrix_inverse(&r_regularized)?;

        // Compute R⁻¹a
        let r_inv_a = r_inv.dot(steering_vector);

        // Compute aᴴR⁻¹a (denominator)
        let denominator = steering_vector.dot(&r_inv_a);

        // Ensure denominator is not too small (numerical stability)
        if denominator.abs() < 1e-12 {
            return Err(crate::error::KwaversError::Numerical(
                crate::error::NumericalError::InvalidOperation(
                    format!("MVDR denominator too small ({:.2e}) - steering vector may be orthogonal to signal subspace", denominator)
                )
            ));
        }

        // Compute weights: w = R⁻¹a / (aᴴR⁻¹a)
        let weights = r_inv_a.mapv(|x| x / denominator);

        Ok(weights)
    }

    /// Apply spatial smoothing to covariance matrix for coherent sources
    #[must_use]
    pub fn apply_spatial_smoothing(
        &self,
        covariance: &Array2<f64>,
        subarray_size: usize,
    ) -> Array2<f64> {
        if !self.spatial_smoothing || subarray_size >= covariance.nrows() {
            return covariance.clone();
        }

        let n = covariance.nrows();
        let mut smoothed = Array2::zeros((n - subarray_size + 1, n - subarray_size + 1));

        // Forward spatial smoothing
        for i in 0..(n - subarray_size + 1) {
            for j in 0..(n - subarray_size + 1) {
                let mut sum = 0.0;
                for k in 0..subarray_size {
                    for l in 0..subarray_size {
                        sum += covariance[[i + k, j + l]];
                    }
                }
                smoothed[[i, j]] = sum / (subarray_size * subarray_size) as f64;
            }
        }

        smoothed
    }

    /// Matrix inversion using Cholesky decomposition for numerical stability
    fn matrix_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        // Use Cholesky decomposition for positive definite matrices (covariance matrices are PSD)
        // This is more numerically stable than general matrix inversion
        self.cholesky_inverse(matrix)
    }

    /// Cholesky-based matrix inversion for positive semi-definite matrices
    fn cholesky_inverse(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = matrix.nrows();

        // Cholesky decomposition: A = L*L^T
        let l = self.cholesky_decomposition(matrix)?;

        // Solve L*L^T * X = I for X = A⁻¹
        let mut inverse = Array2::eye(n);

        // Forward substitution: L * Y = I
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    inverse[[i, j]] /= l[[i, i]];
                } else if i < j {
                    inverse[[i, j]] = 0.0; // Upper triangle
                } else {
                    // i > j, lower triangle
                    let mut sum = 0.0;
                    for k in j..i {
                        sum += l[[i, k]] * inverse[[k, j]];
                    }
                    inverse[[i, j]] = (inverse[[i, j]] - sum) / l[[i, i]];
                }
            }
        }

        // Backward substitution: L^T * X = Y
        for i in (0..n).rev() {
            for j in 0..n {
                if i == j {
                    inverse[[i, j]] /= l[[i, i]];
                } else if i > j {
                    inverse[[i, j]] = 0.0; // Lower triangle
                } else {
                    // i < j, upper triangle
                    let mut sum = 0.0;
                    for k in (i + 1)..=j {
                        sum += l[[k, i]] * inverse[[k, j]];
                    }
                    inverse[[i, j]] = (inverse[[i, j]] - sum) / l[[i, i]];
                }
            }
        }

        Ok(inverse)
    }

    /// Cholesky decomposition: A = L*L^T with regularization for numerical stability
    fn cholesky_decomposition(&self, matrix: &Array2<f64>) -> KwaversResult<Array2<f64>> {
        let n = matrix.nrows();
        let mut l = Array2::zeros((n, n));

        // Add small regularization to handle near-singular matrices
        let regularization = 1e-12;

        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;

                // Sum of L[i][k] * L[j][k] for k = 0 to j-1
                if j == i {
                    for k in 0..j {
                        sum += l[[j, k]] * l[[j, k]];
                    }
                    let diag = matrix[[j, j]] - sum + regularization; // Add regularization
                    if diag <= 0.0 {
                        return Err(crate::error::KwaversError::Numerical(
                            crate::error::NumericalError::SingularMatrix {
                                operation: "Cholesky decomposition".to_string(),
                                condition_number: 0.0, // Would need to compute this properly
                            },
                        ));
                    }
                    l[[j, j]] = diag.sqrt();
                } else {
                    for k in 0..j {
                        sum += l[[i, k]] * l[[j, k]];
                    }
                    l[[i, j]] = (matrix[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        Ok(l)
    }
}

/// Trait for algorithm implementations
pub trait AlgorithmImplementation {
    fn process(&self, data: &Array2<f64>) -> Array1<f64>;
}
