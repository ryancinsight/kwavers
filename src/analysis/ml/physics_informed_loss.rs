//! Physics-Informed Loss for Neural Beamforming
//!
//! This module implements advanced physics-informed loss functions that enforce
//! wave equation residuals and acoustic constraints during neural network training.
//!
//! ## Wave Equation Residuals
//!
//! The homogeneous Helmholtz equation governs acoustic wave propagation:
//! ```text
//! ∇²u + (ω/c)²u = 0
//! ```
//!
//! For a predicted beamformed image u(r), we compute the residual:
//! ```text
//! R(r) = ∇²u(r) + k²u(r)  where k = ω/c
//! ```
//!
//! Physics loss penalizes non-zero residuals:
//! ```text
//! L_physics = ||R(r)||²_2 = ∫ (∇²u + k²u)² dr
//! ```
//!
//! ## Loss Balancing Strategy
//!
//! The total loss combines data fit and physics constraints:
//! ```text
//! L_total = λ_data(t)·L_data + λ_physics(t)·L_physics
//! ```
//!
//! Where λ_data(t) and λ_physics(t) are adaptive weights:
//! - Epoch-based schedule (exponential decay)
//! - Adaptive tuning based on loss ratios
//! - Automatic normalization (sum = 1.0 always)
//!
//! ## Automatic Weight Scheduling
//!
//! Three scheduling strategies:
//! 1. **Exponential**: λ(t) = λ₀ · exp(-decay_rate · t)
//! 2. **Linear**: λ(t) = λ₀ · max(0, 1 - t/total_epochs)
//! 3. **Adaptive**: Based on loss component magnitudes
//!
//! ## References
//!
//! - Raissi et al. (2019): "Physics-informed neural networks"
//! - Linden et al. (2021): "Learning the solution operator of parametric PDEs"
//! - Jin et al. (2021): "NSFNet: Neural Speed Function Network for Physics-Informed"

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::{Array2, Array3};
use std::collections::VecDeque;

/// Weight scheduling strategy for loss balancing
#[derive(Debug, Clone, Copy)]
pub enum WeightSchedule {
    /// Exponential decay: λ(t) = λ₀ · exp(-decay_rate · t)
    Exponential { decay_rate: f64 },
    /// Linear decay: λ(t) = λ₀ · max(0, 1 - t/total_epochs)
    Linear { total_epochs: usize },
    /// Adaptive: Adjust based on loss component magnitudes
    Adaptive,
    /// Fixed weights (no scheduling)
    Fixed,
}

impl Default for WeightSchedule {
    fn default() -> Self {
        Self::Exponential { decay_rate: 0.01 }
    }
}

/// Physics loss configuration
#[derive(Debug, Clone)]
pub struct PhysicsLossConfig {
    /// Initial weight for data loss (MSE)
    pub lambda_data_init: f64,
    /// Initial weight for physics loss
    pub lambda_physics_init: f64,
    /// Wave speed (m/s) for wave equation residual computation
    pub sound_speed: f64,
    /// Frequency (Hz) for computing wave number k = ω/c
    pub frequency: f64,
    /// Gradient computation method
    pub gradient_method: GradientMethod,
    /// Weight scheduling strategy
    pub weight_schedule: WeightSchedule,
    /// Keep history of loss components for adaptive tuning
    pub track_history: bool,
    /// History window size for adaptive tuning
    pub history_window: usize,
}

impl Default for PhysicsLossConfig {
    fn default() -> Self {
        Self {
            lambda_data_init: 0.8,
            lambda_physics_init: 0.2,
            sound_speed: 343.0,
            frequency: 1_000_000.0,
            gradient_method: GradientMethod::FiniteDifference { delta: 0.001 },
            weight_schedule: WeightSchedule::Exponential { decay_rate: 0.01 },
            track_history: true,
            history_window: 20,
        }
    }
}

impl PhysicsLossConfig {
    /// Validate configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.lambda_data_init < 0.0 || self.lambda_physics_init < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Lambda weights must be non-negative".to_string(),
            ));
        }

        if (self.lambda_data_init + self.lambda_physics_init) < 1e-6 {
            return Err(KwaversError::InvalidInput(
                "At least one lambda weight must be positive".to_string(),
            ));
        }

        if self.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "sound_speed must be positive".to_string(),
            ));
        }

        if self.frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "frequency must be positive".to_string(),
            ));
        }

        if self.history_window == 0 {
            return Err(KwaversError::InvalidInput(
                "history_window must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Builder: set initial loss weights
    pub fn with_loss_weights(mut self, lambda_data: f64, lambda_physics: f64) -> Self {
        self.lambda_data_init = lambda_data;
        self.lambda_physics_init = lambda_physics;
        self
    }

    /// Builder: set wave parameters
    pub fn with_wave_params(mut self, sound_speed: f64, frequency: f64) -> Self {
        self.sound_speed = sound_speed;
        self.frequency = frequency;
        self
    }

    /// Builder: set weight schedule
    pub fn with_schedule(mut self, schedule: WeightSchedule) -> Self {
        self.weight_schedule = schedule;
        self
    }

    /// Builder: disable history tracking
    pub fn without_history(mut self) -> Self {
        self.track_history = false;
        self
    }
}

/// Gradient computation method
#[derive(Debug, Clone, Copy)]
pub enum GradientMethod {
    /// Finite difference approximation: ∇u ≈ (u(r+δ) - u(r-δ)) / 2δ
    FiniteDifference { delta: f64 },
    /// Sobel operator (3×3 kernel for 2D, 3×3×3 for 3D)
    Sobel,
}

impl Default for GradientMethod {
    fn default() -> Self {
        Self::FiniteDifference { delta: 0.001 }
    }
}

/// Physics loss component
///
/// Computes wave equation residuals and physics-informed constraints.
#[derive(Debug)]
pub struct PhysicsInformedLoss {
    config: PhysicsLossConfig,
    wave_number: f64,
    current_epoch: usize,
    loss_history: VecDeque<LossComponents>,
}

/// Loss components breakdown
#[derive(Debug, Clone, Copy)]
pub struct LossComponents {
    pub epoch: usize,
    pub data_loss: f64,
    pub physics_loss: f64,
    pub lambda_data: f64,
    pub lambda_physics: f64,
    pub total_loss: f64,
}

impl PhysicsInformedLoss {
    /// Create new physics-informed loss
    pub fn new(config: PhysicsLossConfig) -> KwaversResult<Self> {
        config.validate()?;

        // Compute wave number: k = 2πf/c
        let angular_frequency = 2.0 * std::f64::consts::PI * config.frequency;
        let wave_number = angular_frequency / config.sound_speed;
        let history_window = config.history_window;

        Ok(Self {
            config,
            wave_number,
            current_epoch: 0,
            loss_history: VecDeque::with_capacity(history_window),
        })
    }

    /// Compute wave equation residuals for 3D field
    ///
    /// Residual R(r) = ∇²u + k²u where:
    /// - ∇²u is the Laplacian (discretized via 7-point stencil)
    /// - k = ω/c is the wave number
    ///
    /// Returns MSE of residuals across domain.
    pub fn wave_equation_residual_3d(&self, field: &Array3<f64>) -> f64 {
        let (nx, ny, nz) = field.dim();

        // Interior points only (boundary = zero-flux)
        let mut residual_sum = 0.0;
        let mut count = 0usize;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let u_center = field[[i, j, k]];

                    // 7-point Laplacian stencil (3D)
                    let laplacian = field[[i - 1, j, k]]
                        + field[[i + 1, j, k]]
                        + field[[i, j - 1, k]]
                        + field[[i, j + 1, k]]
                        + field[[i, j, k - 1]]
                        + field[[i, j, k + 1]]
                        - 6.0 * u_center;

                    // Wave equation: R = ∇²u + k²u
                    let residual = laplacian + self.wave_number.powi(2) * u_center;

                    residual_sum += residual * residual;
                    count += 1;
                }
            }
        }

        if count == 0 {
            return 0.0;
        }

        residual_sum / (count as f64)
    }

    /// Compute wave equation residuals for 2D field
    pub fn wave_equation_residual_2d(&self, field: &Array2<f64>) -> f64 {
        let (nx, ny) = field.dim();

        let mut residual_sum = 0.0;
        let mut count = 0usize;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let u_center = field[[i, j]];

                // 5-point Laplacian stencil (2D)
                let laplacian =
                    field[[i - 1, j]] + field[[i + 1, j]] + field[[i, j - 1]] + field[[i, j + 1]]
                        - 4.0 * u_center;

                let residual = laplacian + self.wave_number.powi(2) * u_center;

                residual_sum += residual * residual;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        residual_sum / (count as f64)
    }

    /// Compute reciprocity loss (forward = reverse)
    ///
    /// For reciprocal systems: H(A→B) = H(B→A)
    /// Enforce via MSE between forward and reverse impulse responses.
    pub fn reciprocity_loss(forward: &Array2<f64>, reverse: &Array2<f64>) -> f64 {
        if forward.dim() != reverse.dim() {
            return f64::INFINITY;
        }

        let diff = forward - reverse;
        diff.iter().map(|x| x * x).sum::<f64>() / (forward.len() as f64)
    }

    /// Compute coherence loss (phase continuity)
    ///
    /// Adjacent spatial points should have continuous phase.
    /// Measure via sum of squared phase differences.
    pub fn coherence_loss(amplitudes: &Array2<f64>, phases: &Array2<f64>) -> f64 {
        if amplitudes.dim() != phases.dim() {
            return f64::INFINITY;
        }

        let (nx, ny) = phases.dim();
        let mut loss = 0.0;

        // Phase continuity in x direction
        for i in 0..nx - 1 {
            for j in 0..ny {
                let phase_diff = (phases[[i + 1, j]] - phases[[i, j]]).abs();
                let normalized = if phase_diff > std::f64::consts::PI {
                    2.0 * std::f64::consts::PI - phase_diff
                } else {
                    phase_diff
                };
                loss += normalized * normalized;
            }
        }

        // Phase continuity in y direction
        for i in 0..nx {
            for j in 0..ny - 1 {
                let phase_diff = (phases[[i, j + 1]] - phases[[i, j]]).abs();
                let normalized = if phase_diff > std::f64::consts::PI {
                    2.0 * std::f64::consts::PI - phase_diff
                } else {
                    phase_diff
                };
                loss += normalized * normalized;
            }
        }

        loss / (2.0 * ((nx - 1) * ny + nx * (ny - 1)) as f64)
    }

    /// Compute total loss with adaptive weighting
    ///
    /// L_total = λ_data(t)·L_data + λ_physics(t)·L_physics
    /// where λ values are scheduled based on epoch and loss components
    pub fn compute_total_loss(&mut self, data_loss: f64, physics_loss: f64) -> KwaversResult<f64> {
        // Get scheduled weights for current epoch
        let (lambda_data, lambda_physics) =
            self.compute_weight_schedule(data_loss, physics_loss)?;

        // Total loss
        let total = lambda_data * data_loss + lambda_physics * physics_loss;

        // Store in history
        if self.config.track_history {
            let components = LossComponents {
                epoch: self.current_epoch,
                data_loss,
                physics_loss,
                lambda_data,
                lambda_physics,
                total_loss: total,
            };

            self.loss_history.push_back(components);
            if self.loss_history.len() > self.config.history_window {
                self.loss_history.pop_front();
            }
        }

        self.current_epoch += 1;

        Ok(total)
    }

    /// Compute weight schedule based on current epoch and strategy
    fn compute_weight_schedule(
        &self,
        data_loss: f64,
        physics_loss: f64,
    ) -> KwaversResult<(f64, f64)> {
        let (mut lambda_data, mut lambda_physics) = (
            self.config.lambda_data_init,
            self.config.lambda_physics_init,
        );

        match self.config.weight_schedule {
            WeightSchedule::Exponential { decay_rate } => {
                // Physics loss decays over time: λ_physics(t) = λ₀·exp(-decay_rate·t)
                let decay_factor = (-decay_rate * self.current_epoch as f64).exp();
                lambda_physics = self.config.lambda_physics_init * decay_factor;
                lambda_data = self.config.lambda_data_init + lambda_physics * (1.0 - decay_factor);
            }
            WeightSchedule::Linear { total_epochs } => {
                // Linear decrease: focus on data loss later in training
                let epoch_frac = (self.current_epoch as f64) / (total_epochs as f64);
                let factor = 1.0 - epoch_frac.min(1.0);
                lambda_physics = self.config.lambda_physics_init * factor;
                lambda_data =
                    self.config.lambda_data_init + self.config.lambda_physics_init * (1.0 - factor);
            }
            WeightSchedule::Adaptive => {
                // Adjust based on loss magnitudes
                lambda_physics = self.adaptive_weight_schedule(data_loss, physics_loss)?;
                lambda_data = 1.0 - lambda_physics;
            }
            WeightSchedule::Fixed => {
                // No scheduling
            }
        }

        // Normalize to ensure sum = 1.0
        let sum = lambda_data + lambda_physics;
        if sum > 1e-10 {
            lambda_data /= sum;
            lambda_physics /= sum;
        } else {
            lambda_data = 1.0;
            lambda_physics = 0.0;
        }

        Ok((lambda_data, lambda_physics))
    }

    /// Adaptive weight schedule based on loss ratio
    ///
    /// If physics loss is much larger than data loss, increase its weight
    /// to encourage physics constraint satisfaction. Inverse if data loss dominates.
    fn adaptive_weight_schedule(&self, data_loss: f64, physics_loss: f64) -> KwaversResult<f64> {
        let epsilon = 1e-8;

        // Compute loss ratio
        let ratio = (physics_loss + epsilon) / (data_loss + epsilon);

        // If physics loss >> data loss, increase physics weight
        // If data loss >> physics loss, decrease physics weight
        let mut lambda_physics = self.config.lambda_physics_init;

        if ratio > 10.0 {
            // Physics loss dominates, reduce its weight
            lambda_physics *= 0.8;
        } else if ratio < 0.1 {
            // Data loss dominates, increase physics weight
            lambda_physics *= 1.2;
        }

        Ok(lambda_physics.min(1.0))
    }

    /// Get loss history
    pub fn loss_history(&self) -> Vec<LossComponents> {
        self.loss_history.iter().copied().collect()
    }

    /// Reset epoch counter and history
    pub fn reset(&mut self) {
        self.current_epoch = 0;
        self.loss_history.clear();
    }

    /// Get current epoch
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Get wave number k = ω/c
    pub fn wave_number(&self) -> f64 {
        self.wave_number
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use ndarray::Array;

    #[test]
    fn test_physics_loss_config_default() {
        let config = PhysicsLossConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_physics_loss_config_validation() {
        let mut config = PhysicsLossConfig::default();
        config.sound_speed = 0.0;
        assert!(config.validate().is_err());

        config.sound_speed = 343.0;
        config.frequency = -100.0;
        assert!(config.validate().is_err());

        config.frequency = 1_000_000.0;
        config.history_window = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_physics_loss_creation() {
        let config = PhysicsLossConfig::default();
        let loss = PhysicsInformedLoss::new(config);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_wave_equation_residual_2d() {
        let config = PhysicsLossConfig::default();
        let loss = PhysicsInformedLoss::new(config).unwrap();

        // Create a simple field (all zeros = solution to homogeneous equation)
        let field = Array2::<f64>::zeros((5, 5));
        let residual = loss.wave_equation_residual_2d(&field);
        assert!((residual - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_wave_equation_residual_3d() {
        let config = PhysicsLossConfig::default();
        let loss = PhysicsInformedLoss::new(config).unwrap();

        // Zero field should have zero residual
        let field = Array3::<f64>::zeros((5, 5, 5));
        let residual = loss.wave_equation_residual_3d(&field);
        assert!((residual - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_wave_number_computation() {
        let config = PhysicsLossConfig::default().with_wave_params(343.0, 1_000_000.0);
        let loss = PhysicsInformedLoss::new(config).unwrap();

        // k = 2πf/c = 2π·1e6/343 ≈ 18313
        let k_expected = 2.0 * std::f64::consts::PI * 1_000_000.0 / 343.0;
        assert!((loss.wave_number() - k_expected).abs() < 1.0);
    }

    #[test]
    fn test_reciprocity_loss() {
        let forward =
            Array2::<f64>::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let reverse = forward.clone();

        let loss = PhysicsInformedLoss::reciprocity_loss(&forward, &reverse);
        assert!((loss - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_reciprocity_loss_violation() {
        let forward = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let reverse = Array2::<f64>::from_shape_vec((2, 2), vec![1.0, 2.0, 4.0, 5.0]).unwrap();

        let loss = PhysicsInformedLoss::reciprocity_loss(&forward, &reverse);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_coherence_loss_uniform_field() {
        let amplitudes = Array2::<f64>::from_elem((5, 5), 1.0);
        let phases = Array2::<f64>::from_elem((5, 5), 0.0);

        let loss = PhysicsInformedLoss::coherence_loss(&amplitudes, &phases);
        assert!((loss - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_weight_schedule_exponential() {
        let config = PhysicsLossConfig::default()
            .with_schedule(WeightSchedule::Exponential { decay_rate: 0.1 });
        let mut loss = PhysicsInformedLoss::new(config).unwrap();

        let (_lambda_data1, lambda_physics1) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

        loss.current_epoch = 10;
        let (_lambda_data2, lambda_physics2) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

        // Physics weight should decrease over time
        assert!(lambda_physics2 < lambda_physics1);
    }

    #[test]
    fn test_weight_schedule_linear() {
        let config = PhysicsLossConfig::default()
            .with_schedule(WeightSchedule::Linear { total_epochs: 100 });
        let mut loss = PhysicsInformedLoss::new(config).unwrap();

        let (_lambda_data1, lambda_physics1) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

        loss.current_epoch = 50;
        let (_lambda_data2, lambda_physics2) = loss.compute_weight_schedule(1.0, 1.0).unwrap();

        // Physics weight decreases linearly
        assert!(lambda_physics2 < lambda_physics1);
    }

    #[test]
    fn test_weight_schedule_adaptive() {
        let config = PhysicsLossConfig::default().with_schedule(WeightSchedule::Adaptive);
        let loss = PhysicsInformedLoss::new(config).unwrap();

        // Physics loss dominates
        let (_, lambda_physics) = loss.compute_weight_schedule(1.0, 100.0).unwrap();
        assert!(lambda_physics < 0.3); // Should reduce

        // Data loss dominates
        let (_, lambda_physics) = loss.compute_weight_schedule(100.0, 1.0).unwrap();
        assert!(lambda_physics > 0.2); // Should increase
    }

    #[test]
    fn test_total_loss_computation() {
        let config = PhysicsLossConfig::default();
        let mut loss = PhysicsInformedLoss::new(config).unwrap();

        let total = loss.compute_total_loss(1.0, 1.0).unwrap();
        assert!(total > 0.0);
        assert!(total.is_finite());
    }

    #[test]
    fn test_loss_history_tracking() {
        let config = PhysicsLossConfig::default();
        let mut loss = PhysicsInformedLoss::new(config).unwrap();

        loss.compute_total_loss(1.0, 1.0).unwrap();
        loss.compute_total_loss(0.9, 0.8).unwrap();
        loss.compute_total_loss(0.8, 0.7).unwrap();

        let history = loss.loss_history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].epoch, 0);
        assert_eq!(history[1].epoch, 1);
        assert_eq!(history[2].epoch, 2);
    }

    #[test]
    fn test_loss_history_window() {
        let config = PhysicsLossConfig::default();
        let mut loss = PhysicsInformedLoss::new(config).unwrap();

        // Add more than window size
        for _ in 0..30 {
            loss.compute_total_loss(1.0, 1.0).unwrap();
        }

        let history = loss.loss_history();
        assert!(history.len() <= 20); // Should be capped at window size
    }

    #[test]
    fn test_reset() {
        let config = PhysicsLossConfig::default();
        let mut loss = PhysicsInformedLoss::new(config).unwrap();

        loss.compute_total_loss(1.0, 1.0).unwrap();
        loss.compute_total_loss(1.0, 1.0).unwrap();
        assert_eq!(loss.current_epoch(), 2);

        loss.reset();
        assert_eq!(loss.current_epoch(), 0);
        assert_eq!(loss.loss_history().len(), 0);
    }

    #[test]
    fn test_builder_pattern() {
        let config = PhysicsLossConfig::default()
            .with_loss_weights(0.7, 0.3)
            .with_wave_params(400.0, 2_000_000.0)
            .with_schedule(WeightSchedule::Linear { total_epochs: 50 })
            .without_history();

        assert!((config.lambda_data_init - 0.7).abs() < 1e-10);
        assert!((config.lambda_physics_init - 0.3).abs() < 1e-10);
        assert!((config.sound_speed - 400.0).abs() < 1e-10);
        assert!((config.frequency - 2_000_000.0).abs() < 1e-10);
        assert!(!config.track_history);
    }
}
