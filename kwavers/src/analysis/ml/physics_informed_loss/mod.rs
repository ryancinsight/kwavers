//! Physics-Informed Loss for Neural Beamforming
//!
//! Implements advanced physics-informed loss functions that enforce wave equation
//! residuals and acoustic constraints during neural network training.
//!
//! ## Wave Equation Residual
//! `R(r) = ∇²u(r) + k²u(r)` where `k = ω/c`.
//! Physics loss: `L_physics = ‖R(r)‖²₂`.
//!
//! ## Loss Balancing
//! `L_total = λ_data(t)·L_data + λ_physics(t)·L_physics`
//! with adaptive weights normalized so they always sum to 1.
//!
//! ## References
//! - Raissi et al. (2019): "Physics-informed neural networks"
//! - Jin et al. (2021): "NSFNet"

use crate::core::error::{KwaversError, KwaversResult};
use std::collections::VecDeque;

mod loss;
#[cfg(test)]
mod tests;

/// Weight scheduling strategy for loss balancing
#[derive(Debug, Clone, Copy)]
pub enum WeightSchedule {
    /// Exponential decay: λ(t) = λ₀ · exp(-decay_rate · t)
    Exponential { decay_rate: f64 },
    /// Linear decay: λ(t) = λ₀ · max(0, 1 - t/total_epochs)
    Linear { total_epochs: usize },
    /// Adaptive: adjust based on loss component magnitudes
    Adaptive,
    /// Fixed weights (no scheduling)
    Fixed,
}

impl Default for WeightSchedule {
    fn default() -> Self {
        Self::Exponential { decay_rate: 0.01 }
    }
}

/// Gradient computation method
#[derive(Debug, Clone, Copy)]
pub enum GradientMethod {
    /// Finite difference: ∇u ≈ (u(r+δ) - u(r-δ)) / 2δ
    FiniteDifference { delta: f64 },
    /// Sobel operator (3×3 kernel for 2D, 3×3×3 for 3D)
    Sobel,
}

impl Default for GradientMethod {
    fn default() -> Self {
        Self::FiniteDifference { delta: 0.001 }
    }
}

/// Physics loss configuration
#[derive(Debug, Clone)]
pub struct PhysicsLossConfig {
    pub lambda_data_init: f64,
    pub lambda_physics_init: f64,
    pub sound_speed: f64,
    pub frequency: f64,
    pub gradient_method: GradientMethod,
    pub weight_schedule: WeightSchedule,
    pub track_history: bool,
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
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.lambda_data_init < 0.0 || self.lambda_physics_init < 0.0 {
            return Err(KwaversError::InvalidInput(
                "Lambda weights must be non-negative".to_owned(),
            ));
        }
        if (self.lambda_data_init + self.lambda_physics_init) < 1e-6 {
            return Err(KwaversError::InvalidInput(
                "At least one lambda weight must be positive".to_owned(),
            ));
        }
        if self.sound_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "sound_speed must be positive".to_owned(),
            ));
        }
        if self.frequency <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "frequency must be positive".to_owned(),
            ));
        }
        if self.history_window == 0 {
            return Err(KwaversError::InvalidInput(
                "history_window must be positive".to_owned(),
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn with_loss_weights(mut self, lambda_data: f64, lambda_physics: f64) -> Self {
        self.lambda_data_init = lambda_data;
        self.lambda_physics_init = lambda_physics;
        self
    }

    #[must_use]
    pub fn with_wave_params(mut self, sound_speed: f64, frequency: f64) -> Self {
        self.sound_speed = sound_speed;
        self.frequency = frequency;
        self
    }

    #[must_use]
    pub fn with_schedule(mut self, schedule: WeightSchedule) -> Self {
        self.weight_schedule = schedule;
        self
    }

    #[must_use]
    pub fn without_history(mut self) -> Self {
        self.track_history = false;
        self
    }
}

/// Physics loss component — computes wave equation residuals and physics-informed constraints.
#[derive(Debug)]
pub struct PhysicsInformedLoss {
    pub(super) config: PhysicsLossConfig,
    pub(super) wave_number: f64,
    pub(super) current_epoch: usize,
    pub(super) loss_history: VecDeque<LossComponents>,
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
