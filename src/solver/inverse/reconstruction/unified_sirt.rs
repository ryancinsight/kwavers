//! Unified SIRT/ART/OSEM Reconstruction Interface
//!
//! This module provides a unified, trait-based interface for iterative reconstruction
//! algorithms (SIRT, ART, OSEM) with pluggable regularization and stopping criteria.
//!
//! **Algorithm Overview**:
//!
//! All algorithms solve the linear inverse problem: minimize ||Ax - b||²
//! where A is system matrix, x is image, b is sensor data
//!
//! 1. **SIRT** (Simultaneous Iterative Reconstruction Technique):
//!    - Updates all voxels simultaneously using relaxation
//!    - x^(k+1) = x^(k) + λ·(D_R · A^T · (b - A·x^(k)))
//!    - Stable, slow convergence
//!
//! 2. **ART** (Algebraic Reconstruction Technique):
//!    - Updates voxels sequentially, row by row
//!    - x^(k+1) = x^(k) + (λ/(||a_i||²)) · a_i^T · (b_i - a_i·x^(k))
//!    - Faster convergence, cyclic pattern
//!
//! 3. **OSEM** (Ordered Subset Expectation Maximization):
//!    - Uses ordered subsets for faster convergence
//!    - Processes P subsets per iteration
//!    - Combines EM stability with faster practical convergence
//!
//! **References**:
//! - Natterer, F., & Wübbeling, F. (2001). *Mathematical Methods in Image Reconstruction*
//! - Kaczmarz, S. (1937). "Angenäherte Auflösung von Systemen linearer Gleichungen"
//! - Hudson, H. M., & Larkin, R. S. (1994). "Accelerated image reconstruction using ordered subsets"

use crate::core::error::KwaversResult;
use crate::math::inverse_problems::{ModelRegularizer3D, RegularizationConfig};
use ndarray::{Array1, Array2, Array3};
use std::fmt;

/// Configuration for SIRT-based reconstruction
#[derive(Debug, Clone)]
pub struct SirtConfig {
    /// Algorithm to use (SIRT, ART, OSEM)
    pub algorithm: SirtAlgorithm,
    /// Number of iterations
    pub max_iterations: usize,
    /// Relaxation parameter λ (0 < λ ≤ 1, typical: 0.5)
    pub relaxation_factor: f64,
    /// Regularization configuration
    pub regularization: RegularizationConfig,
    /// Convergence tolerance (relative residual)
    pub tolerance: f64,
    /// Minimum relative change to continue iterations
    pub min_relative_change: f64,
    /// Enable convergence monitoring/logging
    pub verbose: bool,
}

impl Default for SirtConfig {
    fn default() -> Self {
        Self {
            algorithm: SirtAlgorithm::Sirt,
            max_iterations: 100,
            relaxation_factor: 0.5,
            regularization: RegularizationConfig::default(),
            tolerance: 1e-6,
            min_relative_change: 1e-8,
            verbose: false,
        }
    }
}

impl SirtConfig {
    /// Create configuration for SIRT
    pub fn with_sirt(mut self) -> Self {
        self.algorithm = SirtAlgorithm::Sirt;
        self
    }

    /// Create configuration for ART
    pub fn with_art(mut self) -> Self {
        self.algorithm = SirtAlgorithm::Art;
        self
    }

    /// Create configuration for OSEM with number of subsets
    pub fn with_osem(mut self, num_subsets: usize) -> Self {
        self.algorithm = SirtAlgorithm::Osem { num_subsets };
        self
    }

    /// Set iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.max_iterations = iterations;
        self
    }

    /// Set relaxation factor
    pub fn with_relaxation(mut self, factor: f64) -> Self {
        self.relaxation_factor = factor.max(0.001).min(1.0);
        self
    }

    /// Set regularization
    pub fn with_regularization(mut self, reg: RegularizationConfig) -> Self {
        self.regularization = reg;
        self
    }

    /// Enable verbose logging
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }
}

/// SIRT algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SirtAlgorithm {
    /// Simultaneous Iterative Reconstruction Technique (stable, slow)
    Sirt,
    /// Algebraic Reconstruction Technique (faster, cyclic)
    Art,
    /// Ordered Subset EM (practical fast convergence)
    Osem { num_subsets: usize },
}

impl fmt::Display for SirtAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Sirt => write!(f, "SIRT"),
            Self::Art => write!(f, "ART"),
            Self::Osem { num_subsets } => write!(f, "OSEM (subsets={})", num_subsets),
        }
    }
}

/// Result from SIRT reconstruction
#[derive(Debug, Clone)]
pub struct SirtResult {
    /// Reconstructed image
    pub image: Array3<f64>,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual norm ||Ax - b||
    pub final_residual: f64,
    /// Residual history (for convergence analysis)
    pub residual_history: Vec<f64>,
    /// Whether convergence criteria were met
    pub converged: bool,
    /// Computation time (seconds)
    pub computation_time: f64,
}

/// Unified SIRT Reconstructor
///
/// Provides trait-based interface for SIRT, ART, and OSEM algorithms
/// with integrated regularization and convergence monitoring.
#[derive(Debug)]
pub struct SirtReconstructor {
    config: SirtConfig,
}

impl SirtReconstructor {
    /// Create new SIRT reconstructor
    pub fn new(config: SirtConfig) -> Self {
        Self { config }
    }

    /// Reconstruct image from sensor data
    ///
    /// # Arguments
    /// * `system_matrix` - Forward problem matrix A (m × n)
    /// * `sensor_data` - Measured data b (m)
    /// * `grid_size` - 3D grid dimensions (nx, ny, nz)
    pub fn reconstruct(
        &self,
        system_matrix: &Array2<f64>,
        sensor_data: &Array1<f64>,
        grid_size: (usize, usize, usize),
    ) -> KwaversResult<SirtResult> {
        let start_time = std::time::Instant::now();

        let (m, n) = system_matrix.dim();
        assert_eq!(n, grid_size.0 * grid_size.1 * grid_size.2);
        assert_eq!(m, sensor_data.len());

        // Initialize solution
        let mut x = Array1::zeros(n);
        let mut residual_history = Vec::new();

        // Precompute row/column norms for efficiency
        let row_norms = self.compute_row_norms(system_matrix);
        let col_norms = self.compute_col_norms(system_matrix);

        // Regularizer for iterative updates
        let regularizer = ModelRegularizer3D::new(self.config.regularization);

        // Main iteration loop
        for iteration in 0..self.config.max_iterations {
            match self.config.algorithm {
                SirtAlgorithm::Sirt => {
                    x = self.sirt_iteration(
                        system_matrix,
                        &x,
                        sensor_data,
                        &row_norms,
                        &col_norms,
                    )?;
                }
                SirtAlgorithm::Art => {
                    self.art_iteration(system_matrix, &mut x, sensor_data, &row_norms)?;
                }
                SirtAlgorithm::Osem { num_subsets } => {
                    self.osem_iteration(
                        system_matrix,
                        &mut x,
                        sensor_data,
                        num_subsets,
                        &row_norms,
                    )?;
                }
            }

            // Apply regularization
            if self.config.regularization.is_active() {
                let mut image_3d = self.reshape_to_3d(&x, grid_size);
                let model_3d = Array3::zeros(grid_size);
                regularizer.apply_to_gradient(&mut image_3d, &model_3d);
                x = self.reshape_to_1d(&image_3d);
            }

            // Compute residual for convergence check
            let residual = system_matrix.dot(&x) - sensor_data;
            let residual_norm = residual.iter().map(|r| r * r).sum::<f64>().sqrt();
            residual_history.push(residual_norm);

            if self.config.verbose {
                eprintln!(
                    "Iteration {}: residual = {:.6e}",
                    iteration + 1,
                    residual_norm
                );
            }

            // Check convergence
            if iteration > 0 {
                let rel_change = (residual_history[iteration] - residual_history[iteration - 1])
                    .abs()
                    / (residual_history[iteration - 1] + 1e-12);

                if rel_change < self.config.min_relative_change
                    || residual_norm < self.config.tolerance
                {
                    if self.config.verbose {
                        eprintln!("Converged after {} iterations", iteration + 1);
                    }
                    return Ok(SirtResult {
                        image: self.reshape_to_3d(&x, grid_size),
                        iterations: iteration + 1,
                        final_residual: residual_norm,
                        residual_history,
                        converged: true,
                        computation_time: start_time.elapsed().as_secs_f64(),
                    });
                }
            }
        }

        // Max iterations reached
        let final_residual = residual_history.last().copied().unwrap_or(0.0);
        Ok(SirtResult {
            image: self.reshape_to_3d(&x, grid_size),
            iterations: self.config.max_iterations,
            final_residual,
            residual_history,
            converged: false,
            computation_time: start_time.elapsed().as_secs_f64(),
        })
    }

    // ==================== Algorithm Implementations ====================

    /// SIRT iteration: simultaneous update with relaxation
    /// x^(k+1) = x^(k) + λ·(D_R · A^T · (b - A·x^(k)))
    fn sirt_iteration(
        &self,
        a: &Array2<f64>,
        x: &Array1<f64>,
        b: &Array1<f64>,
        _row_norms: &Array1<f64>,
        col_norms: &Array1<f64>,
    ) -> KwaversResult<Array1<f64>> {
        let mut x_new = x.clone();

        // Residual: r = b - A·x
        let residual = b - &a.dot(x);

        // Backprojection: A^T · r
        let backproj = a.t().dot(&residual);

        // Diagonal scaling
        for (j, &col_norm) in col_norms.iter().enumerate() {
            if col_norm > 1e-12 {
                x_new[j] += self.config.relaxation_factor * backproj[j] / col_norm;
            }
        }

        Ok(x_new)
    }

    /// ART iteration: sequential row-by-row update
    /// x^(k+1) = x^(k) + (λ/(||a_i||²)) · a_i^T · (b_i - a_i·x^(k))
    fn art_iteration(
        &self,
        a: &Array2<f64>,
        x: &mut Array1<f64>,
        b: &Array1<f64>,
        row_norms: &Array1<f64>,
    ) -> KwaversResult<()> {
        let (m, _n) = a.dim();

        for i in 0..m {
            let row = a.row(i);
            let row_norm_sq = row_norms[i] * row_norms[i];

            if row_norm_sq > 1e-12 {
                let inner_prod = row.dot(x);
                let residual = b[i] - inner_prod;

                let update = self.config.relaxation_factor * residual / row_norm_sq;

                for (j, &a_ij) in row.iter().enumerate() {
                    x[j] += update * a_ij;
                }
            }
        }

        Ok(())
    }

    /// OSEM iteration: ordered subsets EM
    fn osem_iteration(
        &self,
        a: &Array2<f64>,
        x: &mut Array1<f64>,
        b: &Array1<f64>,
        num_subsets: usize,
        row_norms: &Array1<f64>,
    ) -> KwaversResult<()> {
        let (m, _n) = a.dim();
        let subset_size = m.div_ceil(num_subsets);

        for subset_idx in 0..num_subsets {
            let start_row = subset_idx * subset_size;
            let end_row = ((subset_idx + 1) * subset_size).min(m);

            // Process this subset
            for i in start_row..end_row {
                let row = a.row(i);
                let row_norm_sq = row_norms[i] * row_norms[i];

                if row_norm_sq > 1e-12 {
                    let inner_prod = row.dot(x);
                    let residual = b[i] - inner_prod;
                    let update = self.config.relaxation_factor * residual / row_norm_sq;

                    for (j, &a_ij) in row.iter().enumerate() {
                        x[j] += update * a_ij;
                    }
                }
            }
        }

        Ok(())
    }

    // ==================== Helper Functions ====================

    /// Compute squared norm of each row
    fn compute_row_norms(&self, a: &Array2<f64>) -> Array1<f64> {
        let (m, _n) = a.dim();
        let mut norms = Array1::zeros(m);

        for i in 0..m {
            let row = a.row(i);
            norms[i] = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        }

        norms
    }

    /// Compute squared norm of each column
    fn compute_col_norms(&self, a: &Array2<f64>) -> Array1<f64> {
        let (_m, n) = a.dim();
        let mut norms = Array1::zeros(n);

        for j in 0..n {
            let col = a.column(j);
            norms[j] = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        }

        norms
    }

    /// Reshape 1D vector to 3D array
    fn reshape_to_3d(&self, x: &Array1<f64>, grid_size: (usize, usize, usize)) -> Array3<f64> {
        let mut img = Array3::zeros(grid_size);
        for (idx, &val) in x.iter().enumerate() {
            let (i, j, k) = self.linear_to_3d(idx, grid_size);
            img[[i, j, k]] = val;
        }
        img
    }

    /// Reshape 3D array to 1D vector
    fn reshape_to_1d(&self, img: &Array3<f64>) -> Array1<f64> {
        let (nx, ny, nz) = img.dim();
        let mut x = Array1::zeros(nx * ny * nz);
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let idx = self.to_linear_index(i, j, k, (nx, ny, nz));
                    x[idx] = img[[i, j, k]];
                }
            }
        }
        x
    }

    /// Convert linear index to 3D coordinates
    fn linear_to_3d(&self, idx: usize, grid_size: (usize, usize, usize)) -> (usize, usize, usize) {
        let (_nx, ny, _nz) = grid_size;
        let i = idx / (ny * grid_size.2);
        let j = (idx % (ny * grid_size.2)) / grid_size.2;
        let k = idx % grid_size.2;
        (i, j, k)
    }

    /// Convert 3D coordinates to linear index
    fn to_linear_index(
        &self,
        i: usize,
        j: usize,
        k: usize,
        grid_size: (usize, usize, usize),
    ) -> usize {
        i * grid_size.1 * grid_size.2 + j * grid_size.2 + k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sirt_config_default() {
        let cfg = SirtConfig::default();
        assert_eq!(cfg.algorithm, SirtAlgorithm::Sirt);
        assert_eq!(cfg.max_iterations, 100);
        assert!(cfg.relaxation_factor > 0.0 && cfg.relaxation_factor <= 1.0);
    }

    #[test]
    fn test_sirt_config_builder() {
        let cfg = SirtConfig::default()
            .with_sirt()
            .with_iterations(50)
            .with_relaxation(0.3);

        assert_eq!(cfg.algorithm, SirtAlgorithm::Sirt);
        assert_eq!(cfg.max_iterations, 50);
        assert!((cfg.relaxation_factor - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_osem_config() {
        let cfg = SirtConfig::default().with_osem(4);
        assert_eq!(cfg.algorithm, SirtAlgorithm::Osem { num_subsets: 4 });
    }

    #[test]
    fn test_art_config() {
        let cfg = SirtConfig::default().with_art();
        assert_eq!(cfg.algorithm, SirtAlgorithm::Art);
    }

    #[test]
    fn test_algorithm_display() {
        assert_eq!(format!("{}", SirtAlgorithm::Sirt), "SIRT");
        assert_eq!(format!("{}", SirtAlgorithm::Art), "ART");
        assert_eq!(
            format!("{}", SirtAlgorithm::Osem { num_subsets: 4 }),
            "OSEM (subsets=4)"
        );
    }

    #[test]
    fn test_reconstructor_creation() {
        let cfg = SirtConfig::default();
        let _reconstructor = SirtReconstructor::new(cfg);
    }

    #[test]
    fn test_relaxation_factor_clamping() {
        let cfg1 = SirtConfig::default().with_relaxation(2.0);
        assert!(cfg1.relaxation_factor <= 1.0);

        let cfg2 = SirtConfig::default().with_relaxation(-0.5);
        assert!(cfg2.relaxation_factor >= 0.001);
    }

    #[test]
    fn test_sirt_result_structure() {
        let result = SirtResult {
            image: Array3::zeros((10, 10, 10)),
            iterations: 42,
            final_residual: 0.001,
            residual_history: vec![1.0, 0.9, 0.8, 0.7],
            converged: true,
            computation_time: 1.23,
        };

        assert_eq!(result.iterations, 42);
        assert!(result.converged);
        assert!(result.computation_time > 0.0);
    }
}
