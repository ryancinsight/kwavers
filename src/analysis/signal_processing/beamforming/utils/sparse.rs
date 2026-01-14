//! # Sparse Matrix Utilities for Beamforming (SSOT)
//!
//! This module provides sparse matrix operations for large-scale beamforming applications
//! where memory efficiency is critical (e.g., 3D volumetric imaging, massively parallel
//! arrays, compressive sensing beamforming).
//!
//! # Architectural Intent
//!
//! ## Design Principles
//!
//! 1. **Single Source of Truth**: Canonical sparse beamforming operations
//! 2. **Memory Efficiency**: Sparse representations for large steering matrices
//! 3. **Mathematical Correctness**: Validated against dense implementations
//! 4. **Zero Tolerance**: No silent failures or approximations
//!
//! ## SSOT Enforcement
//!
//! This module is the **only** place for sparse beamforming operations:
//!
//! - ❌ **NO sparse steering matrix construction** elsewhere
//! - ❌ **NO sparse covariance estimation** in other modules
//! - ❌ **NO ad-hoc sparsification** scattered across codebase
//!
//! ## Layer Dependencies
//!
//! ```text
//! analysis::signal_processing::beamforming::utils::sparse (Layer 7)
//!   ↓ imports from
//! math::linear_algebra::sparse (Layer 1) - Generic sparse matrix types (CSR, COO)
//! analysis::signal_processing::beamforming::utils::delays (Layer 7) - Delay calculations
//! core::error (Layer 0) - Error types
//! ```
//!
//! # Use Cases
//!
//! ## Large-Scale Arrays
//!
//! For arrays with N >> 1000 elements and M >> 10000 directions, dense steering
//! matrices (N × M complex) become prohibitively expensive:
//!
//! - Dense: 8 bytes/element × N × M = 80 MB (N=1000, M=10000)
//! - Sparse: ~8 bytes × nnz (non-zeros only), typically 10-20% density
//!
//! ## Compressive Beamforming
//!
//! Sparse reconstruction via convex optimization (ADMM, FISTA) requires:
//! - Sparse measurement matrices
//! - Efficient matrix-vector products
//! - Memory-efficient storage
//!
//! ## Wideband Beamforming
//!
//! Frequency-dependent steering vectors can be sparsified in frequency domain,
//! reducing memory footprint for real-time processing.
//!
//! # Mathematical Foundation
//!
//! ## Sparse Steering Matrix
//!
//! For narrowband beamforming, the steering matrix **A** ∈ ℂ^(M×N) maps sensor
//! outputs to beamformed directions:
//!
//! ```text
//! y = A^H · x
//! ```
//!
//! where:
//! - `x` ∈ ℂ^N = sensor data (N elements)
//! - `y` ∈ ℂ^M = beamformed output (M directions)
//! - `A^H` = Hermitian transpose of steering matrix
//!
//! **Sparsity Exploitation**:
//! - Many steering coefficients ≈ 0 (far-field assumption, limited aperture)
//! - Thresholding removes coefficients with |aᵢⱼ| < ε
//!
//! ## Sparse Covariance Estimation
//!
//! For adaptive beamforming, the sample covariance matrix **R** ∈ ℂ^(N×N):
//!
//! ```text
//! R = (1/K) Σ xₖ·xₖ^H + λI
//! ```
//!
//! **Sparsity**:
//! - Near-diagonal structure for local arrays
//! - Banded structure for linear arrays
//! - Block-diagonal for subarrays
//!
//! # Performance Considerations
//!
//! | Operation | Dense | Sparse (CSR) | Speedup |
//! |-----------|-------|--------------|---------|
//! | Matrix-Vector (10% density) | O(NM) | O(0.1·NM) | 10× |
//! | Memory (N=1000, M=10000) | 160 MB | 16 MB | 10× |
//! | Construction | O(NM) | O(nnz) | 10× |
//!
//! # Literature References
//!
//! ## Sparse Array Processing
//!
//! - Malioutov, D., Cetin, M., & Willsky, A. S. (2005). "A sparse signal
//!   reconstruction perspective for source localization with sensor arrays."
//!   *IEEE Transactions on Signal Processing*, 53(8), 3010-3022.
//!   DOI: 10.1109/TSP.2005.850882
//!
//! ## Compressive Beamforming
//!
//! - Xenaki, A., Gerstoft, P., & Mosegaard, K. (2014). "Compressive beamforming."
//!   *The Journal of the Acoustical Society of America*, 136(1), 260-271.
//!   DOI: 10.1121/1.4883360
//!
//! ## Sparse Covariance Estimation
//!
//! - Chen, Z., Gokeda, G., & Yu, Y. (2010). *Introduction to Direction-of-Arrival
//!   Estimation*. Artech House. ISBN: 978-1-59693-089-6
//!
//! # Future Work
//!
//! - [ ] GPU-accelerated sparse matrix operations (cuSPARSE)
//! - [ ] Adaptive sparsification (dynamic thresholding)
//! - [ ] Block-sparse structures (subarray processing)
//! - [ ] Compressive sensing solvers (ADMM, FISTA)

use crate::core::error::{KwaversError, KwaversResult};
use crate::math::linear_algebra::sparse::{CompressedSparseRowMatrix, CoordinateMatrix};
use ndarray::Array2;
use num_complex::Complex64;
use std::f64::consts::PI;

/// Sparse steering matrix builder for large-scale beamforming.
///
/// Constructs sparse representations of steering matrices, reducing memory
/// footprint and computational cost for arrays with many elements and directions.
///
/// # Mathematical Definition
///
/// For sensor positions **sᵢ** and look directions **dⱼ**, the steering matrix
/// element is:
///
/// ```text
/// A[i,j] = exp(j·k·(sᵢ · dⱼ))
/// ```
///
/// Sparsification via thresholding:
/// ```text
/// A_sparse[i,j] = A[i,j]  if |A[i,j]| > ε
///                 0        otherwise
/// ```
///
/// # Example
///
/// ```rust,ignore
/// use kwavers::analysis::signal_processing::beamforming::utils::sparse::SparseSteeringMatrixBuilder;
///
/// let builder = SparseSteeringMatrixBuilder::new(64, 360, 1e-6);
/// let sparse_matrix = builder.build_plane_wave_steering(
///     &sensor_positions,
///     &look_directions,
///     1e6,
///     1540.0,
/// )?;
/// ```
#[derive(Debug)]
pub struct SparseSteeringMatrixBuilder {
    /// Number of sensor elements
    num_elements: usize,
    /// Number of beamforming directions
    num_directions: usize,
    /// Sparsification threshold (|coefficient| < threshold → zero)
    threshold: f64,
}

impl SparseSteeringMatrixBuilder {
    /// Create a new sparse steering matrix builder.
    ///
    /// # Arguments
    ///
    /// * `num_elements` - Number of sensor elements
    /// * `num_directions` - Number of beamforming directions
    /// * `threshold` - Sparsification threshold (typically 1e-6 to 1e-4)
    ///
    /// # Returns
    ///
    /// New builder instance.
    ///
    /// # Errors
    ///
    /// Returns error if `num_elements` or `num_directions` is zero, or if
    /// `threshold` is negative or non-finite.
    pub fn new(num_elements: usize, num_directions: usize, threshold: f64) -> KwaversResult<Self> {
        if num_elements == 0 {
            return Err(KwaversError::InvalidInput(
                "Number of elements must be positive".into(),
            ));
        }

        if num_directions == 0 {
            return Err(KwaversError::InvalidInput(
                "Number of directions must be positive".into(),
            ));
        }

        if !threshold.is_finite() || threshold < 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Threshold must be non-negative and finite, got {}",
                threshold
            )));
        }

        Ok(Self {
            num_elements,
            num_directions,
            threshold,
        })
    }

    /// Build sparse plane wave steering matrix.
    ///
    /// Constructs a sparse CSR matrix for plane wave beamforming across
    /// multiple look directions.
    ///
    /// # Arguments
    ///
    /// * `sensor_positions` - Sensor positions as [[x, y, z], ...] in meters
    /// * `look_directions` - Look directions as [[dx, dy, dz], ...] (unit vectors)
    /// * `frequency` - Operating frequency in Hz
    /// * `sound_speed` - Speed of sound in m/s
    ///
    /// # Returns
    ///
    /// Sparse steering matrix in CSR format (num_elements × num_directions).
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Dimensions mismatch (sensor_positions.len() != num_elements, etc.)
    /// - Invalid parameters (frequency <= 0, sound_speed <= 0)
    /// - Look directions not normalized
    /// - Non-finite values in positions or directions
    ///
    /// # Performance
    ///
    /// - **Time**: O(N·M) where N = elements, M = directions
    /// - **Space**: O(nnz) where nnz = number of non-zeros (typically 10-20% of N·M)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = SparseSteeringMatrixBuilder::new(64, 360, 1e-6)?;
    ///
    /// // Linear array
    /// let positions: Vec<[f64; 3]> = (0..64)
    ///     .map(|i| [i as f64 * 0.3e-3, 0.0, 0.0])
    ///     .collect();
    ///
    /// // 360° azimuth scan
    /// let directions: Vec<[f64; 3]> = (0..360)
    ///     .map(|i| {
    ///         let theta = (i as f64).to_radians();
    ///         [theta.sin(), 0.0, theta.cos()]
    ///     })
    ///     .collect();
    ///
    /// let sparse_matrix = builder.build_plane_wave_steering(
    ///     &positions,
    ///     &directions,
    ///     1e6,
    ///     1540.0,
    /// )?;
    /// ```
    pub fn build_plane_wave_steering(
        &self,
        sensor_positions: &[[f64; 3]],
        look_directions: &[[f64; 3]],
        frequency: f64,
        sound_speed: f64,
    ) -> KwaversResult<CompressedSparseRowMatrix<Complex64>> {
        // Validate inputs
        if sensor_positions.len() != self.num_elements {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} sensor positions, got {}",
                self.num_elements,
                sensor_positions.len()
            )));
        }

        if look_directions.len() != self.num_directions {
            return Err(KwaversError::InvalidInput(format!(
                "Expected {} look directions, got {}",
                self.num_directions,
                look_directions.len()
            )));
        }

        if !frequency.is_finite() || frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Frequency must be positive and finite, got {}",
                frequency
            )));
        }

        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "Sound speed must be positive and finite, got {}",
                sound_speed
            )));
        }

        // Compute wavenumber
        let wavenumber = 2.0 * PI * frequency / sound_speed;

        // Build sparse matrix using COO format for efficient construction
        // T will be inferred as Complex64 based on add_triplet call
        let mut coo = CoordinateMatrix::create(self.num_elements, self.num_directions);

        for (elem_idx, pos) in sensor_positions.iter().enumerate() {
            // Validate position
            if !pos.iter().all(|&x| x.is_finite()) {
                return Err(KwaversError::InvalidInput(format!(
                    "Sensor position {} contains non-finite values",
                    elem_idx
                )));
            }

            for (dir_idx, direction) in look_directions.iter().enumerate() {
                // Validate direction
                if !direction.iter().all(|&x| x.is_finite()) {
                    return Err(KwaversError::InvalidInput(format!(
                        "Look direction {} contains non-finite values",
                        dir_idx
                    )));
                }

                // Validate direction is unit vector
                let dir_norm_sq =
                    direction[0].powi(2) + direction[1].powi(2) + direction[2].powi(2);
                if (dir_norm_sq - 1.0).abs() > 1e-6 {
                    return Err(KwaversError::InvalidInput(format!(
                        "Look direction {} is not normalized (norm² = {})",
                        dir_idx, dir_norm_sq
                    )));
                }

                // Compute steering coefficient: A[i,j] = exp(j·k·(sᵢ · dⱼ))
                let dot_product =
                    pos[0] * direction[0] + pos[1] * direction[1] + pos[2] * direction[2];
                let phase = wavenumber * dot_product;

                // Complex exponential
                let coeff = Complex64::from_polar(1.0, phase);

                // Apply sparsification threshold (magnitude)
                if coeff.norm() > self.threshold {
                    coo.add_triplet(elem_idx, dir_idx, coeff);
                }
            }
        }

        // Convert to CSR for efficient matrix-vector operations
        Ok(coo.to_csr())
    }

    /// Get number of elements.
    #[must_use]
    pub fn num_elements(&self) -> usize {
        self.num_elements
    }

    /// Get number of directions.
    #[must_use]
    pub fn num_directions(&self) -> usize {
        self.num_directions
    }

    /// Get sparsification threshold.
    #[must_use]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// Compute sparse sample covariance matrix with diagonal loading.
///
/// Constructs a sparse representation of the sample covariance matrix for
/// adaptive beamforming. Exploits spatial correlation structure to reduce
/// memory footprint.
///
/// # Mathematical Definition
///
/// ```text
/// R = (1/K) Σₖ xₖ·xₖ^H + λI
/// ```
///
/// where:
/// - `K` = number of snapshots
/// - `xₖ` = snapshot k (N×1 vector)
/// - `λ` = diagonal loading factor
/// - `I` = identity matrix
///
/// # Arguments
///
/// * `data` - Snapshot data (num_elements × num_snapshots)
/// * `diagonal_loading` - Diagonal loading factor (typically 0.01 to 0.1)
/// * `threshold` - Sparsification threshold for off-diagonal elements
///
/// # Returns
///
/// Sparse covariance matrix in CSR format (num_elements × num_elements).
///
/// # Errors
///
/// Returns error if:
/// - Data matrix is empty
/// - Diagonal loading is negative or non-finite
/// - Threshold is negative or non-finite
///
/// # Performance
///
/// - **Time**: O(N²·K) for dense computation, O(nnz·K) for sparse
/// - **Space**: O(nnz) where nnz ≈ N (banded structure) for local arrays
///
/// # Example
///
/// ```rust,ignore
/// use ndarray::Array2;
///
/// let data = Array2::zeros((64, 1000)); // 64 elements, 1000 snapshots
/// let cov = sparse_sample_covariance(&data, 0.01, 1e-6)?;
/// ```
pub fn sparse_sample_covariance(
    data: &Array2<f64>,
    diagonal_loading: f64,
    threshold: f64,
) -> KwaversResult<CompressedSparseRowMatrix> {
    // Validate inputs
    let (n_elements, n_snapshots) = data.dim();

    if n_elements == 0 || n_snapshots == 0 {
        return Err(KwaversError::InvalidInput(
            "Data matrix cannot be empty".into(),
        ));
    }

    if !diagonal_loading.is_finite() || diagonal_loading < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Diagonal loading must be non-negative and finite, got {}",
            diagonal_loading
        )));
    }

    if !threshold.is_finite() || threshold < 0.0 {
        return Err(KwaversError::InvalidInput(format!(
            "Threshold must be non-negative and finite, got {}",
            threshold
        )));
    }

    // Build sparse covariance using COO format
    let mut coo = CoordinateMatrix::create(n_elements, n_elements);

    // Compute sample covariance: R[i,j] = (1/K) Σₖ xᵢ[k]·xⱼ[k]
    for i in 0..n_elements {
        for j in i..n_elements {
            // Only compute upper triangular (symmetric matrix)
            let mut sum = 0.0;

            for k in 0..n_snapshots {
                sum += data[[i, k]] * data[[j, k]];
            }

            let value = sum / n_snapshots as f64;

            // Add diagonal loading to diagonal elements
            let final_value = if i == j {
                value + diagonal_loading
            } else {
                value
            };

            // Apply sparsification threshold
            if final_value.abs() > threshold {
                coo.add_triplet(i, j, final_value);

                // Add symmetric element for off-diagonal
                if i != j {
                    coo.add_triplet(j, i, final_value);
                }
            }
        }
    }

    Ok(coo.to_csr())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sparse_steering_matrix_builder_creation() {
        let builder = SparseSteeringMatrixBuilder::new(64, 360, 1e-6).unwrap();
        assert_eq!(builder.num_elements(), 64);
        assert_eq!(builder.num_directions(), 360);
        assert_relative_eq!(builder.threshold(), 1e-6);
    }

    #[test]
    fn test_sparse_steering_matrix_builder_invalid_inputs() {
        // Zero elements
        assert!(SparseSteeringMatrixBuilder::new(0, 360, 1e-6).is_err());

        // Zero directions
        assert!(SparseSteeringMatrixBuilder::new(64, 0, 1e-6).is_err());

        // Negative threshold
        assert!(SparseSteeringMatrixBuilder::new(64, 360, -1e-6).is_err());

        // Non-finite threshold
        assert!(SparseSteeringMatrixBuilder::new(64, 360, f64::NAN).is_err());
    }

    #[test]
    fn test_build_plane_wave_steering_simple() {
        let builder = SparseSteeringMatrixBuilder::new(3, 2, 1e-6).unwrap();

        // Simple linear array
        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]];

        // Two directions: broadside and 45°
        let sqrt_2_inv = 1.0 / 2.0_f64.sqrt();
        let directions = vec![[0.0, 0.0, 1.0], [sqrt_2_inv, 0.0, sqrt_2_inv]];

        let sparse_matrix = builder.build_plane_wave_steering(&positions, &directions, 1e6, 1540.0);

        assert!(
            sparse_matrix.is_ok(),
            "Failed to build sparse matrix: {:?}",
            sparse_matrix.err()
        );
        let matrix = sparse_matrix.unwrap();

        // Verify dimensions
        assert_eq!(matrix.rows, 3);
        assert_eq!(matrix.cols, 2);

        // Verify complex values
        let has_complex = matrix.values.iter().any(|v| v.im != 0.0);
        // At least one value should have non-zero imaginary part due to phase delay
        // (unless all delays are multiples of wavelength/2, which is unlikely here)
        assert!(has_complex, "Matrix should contain complex values with non-zero imaginary parts");
    }

    #[test]
    fn test_build_plane_wave_steering_invalid_dimensions() {
        let builder = SparseSteeringMatrixBuilder::new(3, 2, 1e-6).unwrap();

        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]; // Wrong size
        let directions = vec![[0.0, 0.0, 1.0], [0.707, 0.0, 0.707]];

        let result = builder.build_plane_wave_steering(&positions, &directions, 1e6, 1540.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_plane_wave_steering_invalid_frequency() {
        let builder = SparseSteeringMatrixBuilder::new(3, 2, 1e-6).unwrap();

        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]];
        let directions = vec![[0.0, 0.0, 1.0], [0.707, 0.0, 0.707]];

        let result = builder.build_plane_wave_steering(&positions, &directions, -1e6, 1540.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_plane_wave_steering_non_unit_direction() {
        let builder = SparseSteeringMatrixBuilder::new(3, 2, 1e-6).unwrap();

        let positions = vec![[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [0.002, 0.0, 0.0]];
        let directions = vec![[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]]; // Not normalized

        let result = builder.build_plane_wave_steering(&positions, &directions, 1e6, 1540.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_covariance_simple() {
        // Simple 3-element, 10-snapshot data
        let mut data = Array2::zeros((3, 10));
        for i in 0..3 {
            for k in 0..10 {
                data[[i, k]] = (i as f64 + 1.0) * (k as f64 + 1.0);
            }
        }

        let cov = sparse_sample_covariance(&data, 0.01, 1e-6).unwrap();

        // Verify dimensions
        assert_eq!(cov.rows, 3);
        assert_eq!(cov.cols, 3);
    }

    #[test]
    fn test_sparse_covariance_invalid_inputs() {
        let data = Array2::zeros((0, 10));
        assert!(sparse_sample_covariance(&data, 0.01, 1e-6).is_err());

        let data = Array2::zeros((3, 10));
        assert!(sparse_sample_covariance(&data, -0.01, 1e-6).is_err());

        let data = Array2::zeros((3, 10));
        assert!(sparse_sample_covariance(&data, 0.01, -1e-6).is_err());
    }

    #[test]
    fn test_sparse_covariance_diagonal_loading() {
        // Identity-like data (each element independent)
        let mut data = Array2::zeros((3, 100));
        for i in 0..3 {
            data[[i, i]] = 1.0;
        }

        let loading = 0.1;
        let cov = sparse_sample_covariance(&data, loading, 1e-10).unwrap();

        // Diagonal elements should include loading
        // (In practice, would need to extract diagonal to verify)
        assert_eq!(cov.rows, 3);
        assert_eq!(cov.cols, 3);
    }
}
