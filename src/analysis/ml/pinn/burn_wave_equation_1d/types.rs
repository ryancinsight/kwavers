//! Type definitions for Burn-based 1D Wave Equation PINN
//!
//! This module provides domain types for training metrics, results, and
//! statistical analysis of physics-informed neural network training.
//!
//! ## Training Metrics
//!
//! The `BurnTrainingMetrics` structure tracks the evolution of loss components
//! during training, enabling convergence analysis and hyperparameter tuning.
//!
//! ## Loss Components
//!
//! Physics-informed training optimizes a composite loss function:
//! - **Total loss**: Weighted combination of all components
//! - **Data loss**: MSE between predictions and training data
//! - **PDE loss**: MSE of wave equation residual
//! - **Boundary loss**: MSE of boundary condition violations
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks: A deep learning framework
//!   for solving forward and inverse problems involving nonlinear partial differential equations"
//!   Journal of Computational Physics, 378:686-707. DOI: 10.1016/j.jcp.2018.10.045
//!
//! ## Examples
//!
//! ```rust,ignore
//! use kwavers::analysis::ml::pinn::burn_wave_equation_1d::types::*;
//!
//! // Create empty metrics
//! let mut metrics = BurnTrainingMetrics::with_capacity(1000);
//!
//! // Record training progress
//! metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
//!
//! // Analyze convergence
//! if let Some(final_loss) = metrics.final_total_loss() {
//!     println!("Final loss: {}", final_loss);
//! }
//!
//! // Check convergence rate
//! if metrics.is_converged(1e-6) {
//!     println!("Training converged!");
//! }
//! ```

use std::time::Duration;

/// Training metrics for monitoring PINN convergence
///
/// This structure records the evolution of loss components during physics-informed
/// training, enabling:
/// - Convergence analysis
/// - Hyperparameter tuning
/// - Training diagnostics
/// - Performance profiling
///
/// ## Loss History
///
/// Each vector tracks the corresponding loss component at each epoch:
/// - `total_loss`: L_total = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc
/// - `data_loss`: MSE between predictions and training data
/// - `pde_loss`: MSE of PDE residual (physics constraint violation)
/// - `bc_loss`: MSE of boundary condition violations
///
/// ## Convergence Criteria
///
/// Training is considered converged when:
/// 1. Total loss decreases monotonically (or with small fluctuations)
/// 2. All loss components stabilize below acceptable thresholds
/// 3. Relative loss change falls below tolerance (e.g., 1e-6)
///
/// ## Performance Metrics
///
/// The structure also tracks:
/// - `training_time_secs`: Total wall-clock training time
/// - `epochs_completed`: Number of epochs completed (may differ from requested if early stopping)
///
/// # Examples
///
/// ```rust,ignore
/// // Initialize metrics for 1000 epochs
/// let mut metrics = BurnTrainingMetrics::with_capacity(1000);
///
/// // Training loop
/// for epoch in 0..1000 {
///     // ... compute losses ...
///     metrics.record_epoch(total, data, pde, bc);
///
///     if metrics.is_converged(1e-6) {
///         println!("Converged at epoch {}", epoch);
///         break;
///     }
/// }
///
/// metrics.training_time_secs = start_time.elapsed().as_secs_f64();
/// ```
#[derive(Debug, Clone)]
pub struct BurnTrainingMetrics {
    /// Total loss history (L_total = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc)
    ///
    /// The composite loss function that is actually optimized during training.
    /// This is the weighted sum of data, PDE, and boundary losses.
    pub total_loss: Vec<f64>,

    /// Data loss history (MSE between predictions and training data)
    ///
    /// Measures how well the neural network fits the observed training data.
    /// Lower values indicate better data fitting.
    pub data_loss: Vec<f64>,

    /// PDE residual loss history (MSE of wave equation residual)
    ///
    /// Measures how well the neural network satisfies the wave equation
    /// ∂²u/∂t² = c²∂²u/∂x² at collocation points throughout the domain.
    /// Lower values indicate better physics constraint satisfaction.
    pub pde_loss: Vec<f64>,

    /// Boundary condition loss history (MSE of BC violations)
    ///
    /// Measures how well the neural network satisfies boundary conditions
    /// at domain boundaries (e.g., u=0 at x=±L).
    /// Lower values indicate better boundary constraint satisfaction.
    pub bc_loss: Vec<f64>,

    /// Total training time in seconds
    ///
    /// Wall-clock time from training start to completion.
    /// Useful for performance profiling and resource estimation.
    pub training_time_secs: f64,

    /// Number of epochs completed
    ///
    /// May differ from requested epochs if early stopping is triggered.
    /// Always equals `total_loss.len()` after training.
    pub epochs_completed: usize,
}

impl BurnTrainingMetrics {
    /// Create a new metrics structure with pre-allocated capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected number of epochs (for efficient memory allocation)
    ///
    /// # Returns
    ///
    /// Empty metrics structure with pre-allocated vectors
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let metrics = BurnTrainingMetrics::new();
    /// assert_eq!(metrics.epochs_completed, 0);
    /// assert!(metrics.total_loss.is_empty());
    /// ```
    pub fn new() -> Self {
        Self {
            total_loss: Vec::new(),
            data_loss: Vec::new(),
            pde_loss: Vec::new(),
            bc_loss: Vec::new(),
            training_time_secs: 0.0,
            epochs_completed: 0,
        }
    }

    /// Create new training metrics with pre-allocated capacity
    ///
    /// Pre-allocates vectors to avoid reallocation during training.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected number of epochs (for pre-allocation)
    ///
    /// # Returns
    ///
    /// New metrics instance with pre-allocated capacity
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let metrics = BurnTrainingMetrics::with_capacity(1000);
    /// assert_eq!(metrics.epochs_completed, 0);
    /// assert!(metrics.total_loss.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            total_loss: Vec::with_capacity(capacity),
            data_loss: Vec::with_capacity(capacity),
            pde_loss: Vec::with_capacity(capacity),
            bc_loss: Vec::with_capacity(capacity),
            training_time_secs: 0.0,
            epochs_completed: 0,
        }
    }

    /// Record loss values for a single epoch
    ///
    /// # Arguments
    ///
    /// * `total` - Total loss value
    /// * `data` - Data loss value
    /// * `pde` - PDE residual loss value
    /// * `bc` - Boundary condition loss value
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut metrics = BurnTrainingMetrics::with_capacity(100);
    /// metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    /// assert_eq!(metrics.epochs_completed, 1);
    /// ```
    pub fn record_epoch(&mut self, total: f64, data: f64, pde: f64, bc: f64) {
        self.total_loss.push(total);
        self.data_loss.push(data);
        self.pde_loss.push(pde);
        self.bc_loss.push(bc);
        self.epochs_completed += 1;
    }

    /// Get the final total loss value
    ///
    /// # Returns
    ///
    /// - `Some(loss)` if training has occurred
    /// - `None` if no epochs have been recorded
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut metrics = BurnTrainingMetrics::with_capacity(100);
    /// assert!(metrics.final_total_loss().is_none());
    ///
    /// metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    /// assert_eq!(metrics.final_total_loss(), Some(0.1));
    /// ```
    pub fn final_total_loss(&self) -> Option<f64> {
        self.total_loss.last().copied()
    }

    /// Get the final data loss value
    pub fn final_data_loss(&self) -> Option<f64> {
        self.data_loss.last().copied()
    }

    /// Get the final PDE loss value
    pub fn final_pde_loss(&self) -> Option<f64> {
        self.pde_loss.last().copied()
    }

    /// Get the final boundary condition loss value
    pub fn final_bc_loss(&self) -> Option<f64> {
        self.bc_loss.last().copied()
    }

    /// Check if training has converged based on relative loss change
    ///
    /// Convergence is detected when the relative change in total loss
    /// between consecutive epochs falls below the specified tolerance.
    ///
    /// # Arguments
    ///
    /// * `tolerance` - Relative tolerance for convergence (e.g., 1e-6)
    ///
    /// # Returns
    ///
    /// - `true` if converged (relative change < tolerance)
    /// - `false` otherwise or if insufficient history
    ///
    /// # Formula
    ///
    /// Relative change = |L[n] - L[n-1]| / |L[n-1]|
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut metrics = BurnTrainingMetrics::with_capacity(100);
    /// metrics.record_epoch(1.0, 0.5, 0.3, 0.2);
    /// metrics.record_epoch(0.999999, 0.5, 0.3, 0.2);
    /// assert!(metrics.is_converged(1e-5));
    /// ```
    pub fn is_converged(&self, tolerance: f64) -> bool {
        if self.total_loss.len() < 2 {
            return false;
        }

        let current = self.total_loss[self.total_loss.len() - 1];
        let previous = self.total_loss[self.total_loss.len() - 2];

        // Avoid division by zero
        if previous.abs() < 1e-15 {
            return current.abs() < tolerance;
        }

        let relative_change = (current - previous).abs() / previous.abs();
        relative_change < tolerance
    }

    /// Compute average loss over last N epochs
    ///
    /// Useful for assessing recent training stability.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of recent epochs to average
    ///
    /// # Returns
    ///
    /// - `Some(average)` if at least `n` epochs have been recorded
    /// - `None` otherwise
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut metrics = BurnTrainingMetrics::with_capacity(100);
    /// metrics.record_epoch(0.3, 0.1, 0.1, 0.1);
    /// metrics.record_epoch(0.2, 0.1, 0.05, 0.05);
    /// metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    ///
    /// let avg = metrics.average_loss_last_n(3).unwrap();
    /// assert!((avg - 0.2).abs() < 1e-6);
    /// ```
    pub fn average_loss_last_n(&self, n: usize) -> Option<f64> {
        if self.total_loss.len() < n {
            return None;
        }

        let start = self.total_loss.len() - n;
        let sum: f64 = self.total_loss[start..].iter().sum();
        Some(sum / n as f64)
    }

    /// Get training throughput (epochs per second)
    ///
    /// # Returns
    ///
    /// - `Some(throughput)` if training time is positive
    /// - `None` if no training has occurred
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut metrics = BurnTrainingMetrics::with_capacity(100);
    /// for _ in 0..100 {
    ///     metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    /// }
    /// metrics.training_time_secs = 10.0;
    ///
    /// let throughput = metrics.throughput().unwrap();
    /// assert_eq!(throughput, 10.0); // 100 epochs / 10 seconds
    /// ```
    pub fn throughput(&self) -> Option<f64> {
        if self.training_time_secs > 0.0 && self.epochs_completed > 0 {
            Some(self.epochs_completed as f64 / self.training_time_secs)
        } else {
            None
        }
    }

    /// Get training duration as Duration
    ///
    /// # Returns
    ///
    /// Duration representing training time
    pub fn training_duration(&self) -> Duration {
        Duration::from_secs_f64(self.training_time_secs)
    }

    /// Check if any loss component contains NaN or Inf
    ///
    /// This indicates numerical instability or divergence during training.
    ///
    /// # Returns
    ///
    /// - `true` if any loss contains NaN or Inf
    /// - `false` if all losses are finite
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut metrics = BurnTrainingMetrics::with_capacity(100);
    /// metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
    /// assert!(!metrics.has_numerical_issues());
    ///
    /// metrics.record_epoch(f64::NAN, 0.05, 0.03, 0.02);
    /// assert!(metrics.has_numerical_issues());
    /// ```
    pub fn has_numerical_issues(&self) -> bool {
        self.total_loss.iter().any(|&x| !x.is_finite())
            || self.data_loss.iter().any(|&x| !x.is_finite())
            || self.pde_loss.iter().any(|&x| !x.is_finite())
            || self.bc_loss.iter().any(|&x| !x.is_finite())
    }

    /// Compute loss reduction percentage from first to last epoch
    ///
    /// # Returns
    ///
    /// - `Some(percentage)` if at least 2 epochs recorded
    /// - `None` otherwise
    ///
    /// # Formula
    ///
    /// Reduction = (1 - L_final / L_initial) × 100%
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let mut metrics = BurnTrainingMetrics::with_capacity(100);
    /// metrics.record_epoch(1.0, 0.5, 0.3, 0.2);
    /// metrics.record_epoch(0.5, 0.25, 0.15, 0.1);
    ///
    /// let reduction = metrics.loss_reduction_percent().unwrap();
    /// assert_eq!(reduction, 50.0);
    /// ```
    pub fn loss_reduction_percent(&self) -> Option<f64> {
        if self.total_loss.len() < 2 {
            return None;
        }

        let initial = self.total_loss[0];
        let final_loss = self.total_loss[self.total_loss.len() - 1];

        if initial.abs() < 1e-15 {
            return None;
        }

        Some((1.0 - final_loss / initial) * 100.0)
    }
}

impl Default for BurnTrainingMetrics {
    /// Create empty metrics with default capacity
    fn default() -> Self {
        Self::with_capacity(1000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_metrics() {
        let metrics = BurnTrainingMetrics::with_capacity(100);
        assert_eq!(metrics.epochs_completed, 0);
        assert!(metrics.total_loss.is_empty());
        assert_eq!(metrics.training_time_secs, 0.0);
    }

    #[test]
    fn test_record_epoch() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);
        metrics.record_epoch(0.1, 0.05, 0.03, 0.02);

        assert_eq!(metrics.epochs_completed, 1);
        assert_eq!(metrics.total_loss.len(), 1);
        assert_eq!(metrics.total_loss[0], 0.1);
        assert_eq!(metrics.data_loss[0], 0.05);
        assert_eq!(metrics.pde_loss[0], 0.03);
        assert_eq!(metrics.bc_loss[0], 0.02);
    }

    #[test]
    fn test_final_losses() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);
        assert!(metrics.final_total_loss().is_none());

        metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
        assert_eq!(metrics.final_total_loss(), Some(0.1));
        assert_eq!(metrics.final_data_loss(), Some(0.05));
        assert_eq!(metrics.final_pde_loss(), Some(0.03));
        assert_eq!(metrics.final_bc_loss(), Some(0.02));
    }

    #[test]
    fn test_convergence_detection() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);

        // Not enough history
        assert!(!metrics.is_converged(1e-6));

        // Large change - not converged
        metrics.record_epoch(1.0, 0.5, 0.3, 0.2);
        metrics.record_epoch(0.5, 0.25, 0.15, 0.1);
        assert!(!metrics.is_converged(1e-6));

        // Small change - converged
        metrics.record_epoch(0.5000001, 0.25, 0.15, 0.1);
        assert!(metrics.is_converged(1e-5));
    }

    #[test]
    fn test_average_loss_last_n() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);

        // Not enough history
        assert!(metrics.average_loss_last_n(3).is_none());

        metrics.record_epoch(0.3, 0.1, 0.1, 0.1);
        metrics.record_epoch(0.2, 0.1, 0.05, 0.05);
        metrics.record_epoch(0.1, 0.05, 0.03, 0.02);

        let avg = metrics.average_loss_last_n(3).unwrap();
        assert!((avg - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_throughput() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);

        // No training yet
        assert!(metrics.throughput().is_none());

        for _ in 0..100 {
            metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
        }
        metrics.training_time_secs = 10.0;

        let throughput = metrics.throughput().unwrap();
        assert_eq!(throughput, 10.0);
    }

    #[test]
    fn test_numerical_issues_detection() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);
        metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
        assert!(!metrics.has_numerical_issues());

        metrics.record_epoch(f64::NAN, 0.05, 0.03, 0.02);
        assert!(metrics.has_numerical_issues());
    }

    #[test]
    fn test_numerical_issues_infinity() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);
        metrics.record_epoch(0.1, 0.05, 0.03, 0.02);
        assert!(!metrics.has_numerical_issues());

        metrics.record_epoch(f64::INFINITY, 0.05, 0.03, 0.02);
        assert!(metrics.has_numerical_issues());
    }

    #[test]
    fn test_loss_reduction_percent() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);

        // Not enough history
        assert!(metrics.loss_reduction_percent().is_none());

        metrics.record_epoch(1.0, 0.5, 0.3, 0.2);
        metrics.record_epoch(0.5, 0.25, 0.15, 0.1);

        let reduction = metrics.loss_reduction_percent().unwrap();
        assert_eq!(reduction, 50.0);
    }

    #[test]
    fn test_training_duration() {
        let mut metrics = BurnTrainingMetrics::with_capacity(10);
        metrics.training_time_secs = 12.5;

        let duration = metrics.training_duration();
        assert_eq!(duration.as_secs_f64(), 12.5);
    }
}
