//! PhysicsInformedLoss implementation.

use super::{
    PhysicsInformedLoss, PhysicsInformedLossComponents, PhysicsLossConfig, WeightSchedule,
};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use leto::{Array2, Array3};
use std::collections::VecDeque;

impl PhysicsInformedLoss {
    /// Create new physics-informed loss. Computes wave number `k = 2πf/c`.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(config: PhysicsLossConfig) -> KwaversResult<Self> {
        config.validate()?;

        let angular_frequency = TWO_PI * config.frequency;
        let wave_number = angular_frequency / config.sound_speed;
        let history_window = config.history_window;

        Ok(Self {
            config,
            wave_number,
            current_epoch: 0,
            loss_history: VecDeque::with_capacity(history_window),
        })
    }

    /// Compute wave equation MSE residual for 3D field using 7-point Laplacian stencil.
    ///
    /// `R(r) = ∇²u + k²u`, interior points only.
    /// The Laplacian is divided by dx² so both terms have units [field / m²].
    #[must_use]
    pub fn wave_equation_residual_3d(&self, field: &Array3<f64>) -> f64 {
        let [nx, ny, nz] = field.shape();
        let dx = self.config.grid_spacing;
        let inv_dx2 = 1.0 / (dx * dx);

        let mut residual_sum = 0.0;
        let mut count = 0usize;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let u_center = field[[i, j, k]];

                    let laplacian = inv_dx2
                        * 6.0f64.mul_add(
                            -u_center,
                            field[[i - 1, j, k]]
                                + field[[i + 1, j, k]]
                                + field[[i, j - 1, k]]
                                + field[[i, j + 1, k]]
                                + field[[i, j, k - 1]]
                                + field[[i, j, k + 1]],
                        );

                    let residual = self.wave_number.powi(2).mul_add(u_center, laplacian);

                    residual_sum += residual * residual;
                    count += 1;
                }
            }
        }

        if count == 0 {
            0.0
        } else {
            residual_sum / (count as f64)
        }
    }

    /// Compute wave equation MSE residual for 2D field using 5-point Laplacian stencil.
    /// The Laplacian is divided by dx² so both terms have units [field / m²].
    #[must_use]
    pub fn wave_equation_residual_2d(&self, field: &Array2<f64>) -> f64 {
        let [nx, ny] = field.shape();
        let dx = self.config.grid_spacing;
        let inv_dx2 = 1.0 / (dx * dx);

        let mut residual_sum = 0.0;
        let mut count = 0usize;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let u_center = field[[i, j]];

                let laplacian = inv_dx2
                    * 4.0f64.mul_add(
                        -u_center,
                        field[[i - 1, j]]
                            + field[[i + 1, j]]
                            + field[[i, j - 1]]
                            + field[[i, j + 1]],
                    );

                let residual = self.wave_number.powi(2).mul_add(u_center, laplacian);

                residual_sum += residual * residual;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            residual_sum / (count as f64)
        }
    }

    /// Reciprocity loss: MSE between forward and reverse impulse responses.
    #[must_use]
    pub fn reciprocity_loss(forward: &Array2<f64>, reverse: &Array2<f64>) -> f64 {
        if forward.shape() != reverse.shape() {
            return f64::INFINITY;
        }

        if forward.is_empty() {
            return 0.0;
        }
        let diff = forward - reverse;
        diff.iter().map(|x| x * x).sum::<f64>() / (forward.len() as f64)
    }

    /// Coherence loss: sum of squared phase differences between adjacent spatial points.
    #[must_use]
    pub fn coherence_loss(amplitudes: &Array2<f64>, phases: &Array2<f64>) -> f64 {
        if amplitudes.shape() != phases.shape() {
            return f64::INFINITY;
        }

        // Squared shortest-arc adjacent-pixel phase difference
        // |wrap_to_pi(Δφ)|² (SSOT wrap; see `math::signal::wrap_to_pi`).
        // Correct for any gradient magnitude, unlike a single π-fold of |Δφ|.
        use kwavers_math::signal::wrap_to_pi;
        let [nx, ny] = phases.shape();
        let mut loss = 0.0;

        for i in 0..nx - 1 {
            for j in 0..ny {
                let normalized = wrap_to_pi(phases[[i + 1, j]] - phases[[i, j]]);
                loss += normalized * normalized;
            }
        }

        for i in 0..nx {
            for j in 0..ny - 1 {
                let normalized = wrap_to_pi(phases[[i, j + 1]] - phases[[i, j]]);
                loss += normalized * normalized;
            }
        }

        loss / (2.0 * ((nx - 1) * ny + nx * (ny - 1)) as f64)
    }

    /// Compute total loss: `L_total = λ_data(t)·L_data + λ_physics(t)·L_physics`.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn compute_total_loss(&mut self, data_loss: f64, physics_loss: f64) -> KwaversResult<f64> {
        let (lambda_data, lambda_physics) =
            self.compute_weight_schedule(data_loss, physics_loss)?;

        let total = lambda_data.mul_add(data_loss, lambda_physics * physics_loss);

        if self.config.track_history {
            let components = PhysicsInformedLossComponents {
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
    /// Compute weight schedule.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub(super) fn compute_weight_schedule(
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
                let decay_factor = (-decay_rate * self.current_epoch as f64).exp();
                lambda_physics = self.config.lambda_physics_init * decay_factor;
                lambda_data =
                    lambda_physics.mul_add(1.0 - decay_factor, self.config.lambda_data_init);
            }
            WeightSchedule::Linear { total_epochs } => {
                let epoch_frac = (self.current_epoch as f64) / (total_epochs as f64);
                let factor = 1.0 - epoch_frac.min(1.0);
                lambda_physics = self.config.lambda_physics_init * factor;
                lambda_data = self
                    .config
                    .lambda_physics_init
                    .mul_add(1.0 - factor, self.config.lambda_data_init);
            }
            WeightSchedule::Adaptive => {
                lambda_physics = self.adaptive_weight_schedule(data_loss, physics_loss)?;
                lambda_data = 1.0 - lambda_physics;
            }
            WeightSchedule::Fixed => {}
        }

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

    fn adaptive_weight_schedule(&self, data_loss: f64, physics_loss: f64) -> KwaversResult<f64> {
        let epsilon = 1e-8;
        let ratio = (physics_loss + epsilon) / (data_loss + epsilon);

        let mut lambda_physics = self.config.lambda_physics_init;

        if ratio > 10.0 {
            lambda_physics *= 0.8;
        } else if ratio < 0.1 {
            lambda_physics *= 1.2;
        }

        Ok(lambda_physics.min(1.0))
    }

    #[must_use]
    pub fn loss_history(&self) -> Vec<PhysicsInformedLossComponents> {
        self.loss_history.iter().copied().collect()
    }

    pub fn reset(&mut self) {
        self.current_epoch = 0;
        self.loss_history.clear();
    }

    #[must_use]
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    #[must_use]
    pub fn wave_number(&self) -> f64 {
        self.wave_number
    }
}
