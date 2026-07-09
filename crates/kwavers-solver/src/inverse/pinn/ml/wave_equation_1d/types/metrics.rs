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
/// ## References
///
/// - Raissi et al. (2019): Journal of Computational Physics, 378:686-707.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TrainingMetrics {
    /// Total loss history (L_total = λ_data·L_data + λ_pde·L_pde + λ_bc·L_bc)
    pub total_loss: Vec<f64>,

    /// Data loss history (MSE between predictions and training data)
    pub data_loss: Vec<f64>,

    /// PDE residual loss history (MSE of wave equation residual)
    pub pde_loss: Vec<f64>,

    /// Boundary condition loss history (MSE of BC violations)
    pub bc_loss: Vec<f64>,

    /// Total training time in seconds
    pub training_time_secs: f64,

    /// Number of epochs completed
    pub epochs_completed: usize,
}

impl TrainingMetrics {
    /// Create a new metrics structure
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
    pub fn record_epoch(&mut self, total: f64, data: f64, pde: f64, bc: f64) {
        self.total_loss.push(total);
        self.data_loss.push(data);
        self.pde_loss.push(pde);
        self.bc_loss.push(bc);
        self.epochs_completed += 1;
    }

    /// Get the final total loss value
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
    /// Relative change = |L[n] - L[n-1]| / |L[n-1]|
    pub fn is_converged(&self, tolerance: f64) -> bool {
        if (self.total_loss.shape()[0] * self.total_loss.shape()[1] * self.total_loss.shape()[2]) < 2 {
            return false;
        }

        let current = self.total_loss[(self.total_loss.shape()[0] * self.total_loss.shape()[1] * self.total_loss.shape()[2]) - 1];
        let previous = self.total_loss[(self.total_loss.shape()[0] * self.total_loss.shape()[1] * self.total_loss.shape()[2]) - 2];

        if previous.abs() < 1e-15 {
            return current.abs() < tolerance;
        }

        let relative_change = (current - previous).abs() / previous.abs();
        relative_change < tolerance
    }

    /// Compute average loss over last N epochs
    pub fn average_loss_last_n(&self, n: usize) -> Option<f64> {
        if (self.total_loss.shape()[0] * self.total_loss.shape()[1] * self.total_loss.shape()[2]) < n {
            return None;
        }

        let start = (self.total_loss.shape()[0] * self.total_loss.shape()[1] * self.total_loss.shape()[2]) - n;
        let sum: f64 = self.total_loss[start..].iter().sum();
        Some(sum / n as f64)
    }

    /// Get training throughput (epochs per second)
    pub fn throughput(&self) -> Option<f64> {
        if self.training_time_secs > 0.0 && self.epochs_completed > 0 {
            Some(self.epochs_completed as f64 / self.training_time_secs)
        } else {
            None
        }
    }

    /// Get training duration as Duration
    pub fn training_duration(&self) -> Duration {
        Duration::from_secs_f64(self.training_time_secs)
    }

    /// Check if any loss component contains NaN or Inf
    pub fn has_numerical_issues(&self) -> bool {
        self.total_loss.iter().any(|&x| !x.is_finite())
            || self.data_loss.iter().any(|&x| !x.is_finite())
            || self.pde_loss.iter().any(|&x| !x.is_finite())
            || self.bc_loss.iter().any(|&x| !x.is_finite())
    }

    /// Compute loss reduction percentage from first to last epoch
    ///
    /// Reduction = (1 - L_final / L_initial) × 100%
    pub fn loss_reduction_percent(&self) -> Option<f64> {
        if (self.total_loss.shape()[0] * self.total_loss.shape()[1] * self.total_loss.shape()[2]) < 2 {
            return None;
        }

        let initial = self.total_loss[0];
        let final_loss = self.total_loss[(self.total_loss.shape()[0] * self.total_loss.shape()[1] * self.total_loss.shape()[2]) - 1];

        if initial.abs() < 1e-15 {
            return None;
        }

        Some((1.0 - final_loss / initial) * 100.0)
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::with_capacity(1000)
    }
}
