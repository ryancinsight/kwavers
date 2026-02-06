//! Learning rate scheduling for PINN training
//!
//! This module provides learning rate schedulers to adjust the learning rate
//! during training for improved convergence.

/// Learning rate scheduler configurations
#[derive(Debug, Clone)]
pub enum LearningRateScheduler {
    /// Constant learning rate
    Constant,
    /// Exponential decay: lr = initial_lr * decay_rate^epoch
    Exponential { decay_rate: f64 },
    /// Step decay: multiply by factor every step_size epochs
    Step { factor: f64, step_size: usize },
    /// Cosine annealing: cyclical learning rate decay
    CosineAnnealing { lr_min: f64 },
    /// Reduce on plateau: reduce LR when loss stops improving
    ReduceOnPlateau {
        factor: f64,
        patience: usize,
        threshold: f64,
    },
}

/// Learning rate scheduler
///
/// Adjusts learning rate during training based on configured schedule.
#[derive(Debug, Clone)]
pub struct LRScheduler {
    /// Initial learning rate
    pub initial_lr: f64,
    /// Current learning rate
    pub current_lr: f64,
    /// Scheduler configuration
    pub scheduler: LearningRateScheduler,
    /// Current epoch
    pub epoch: usize,
    /// Best loss (for ReduceOnPlateau)
    pub best_loss: f64,
    /// Epochs without improvement (for ReduceOnPlateau)
    pub plateau_count: usize,
}

impl LRScheduler {
    /// Create new scheduler
    pub fn new(initial_lr: f64, scheduler: LearningRateScheduler) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            scheduler,
            epoch: 0,
            best_loss: f64::INFINITY,
            plateau_count: 0,
        }
    }

    /// Create constant learning rate scheduler
    pub fn constant(lr: f64) -> Self {
        Self::new(lr, LearningRateScheduler::Constant)
    }

    /// Create exponential decay scheduler
    pub fn exponential(lr: f64, decay_rate: f64) -> Self {
        Self::new(lr, LearningRateScheduler::Exponential { decay_rate })
    }

    /// Create step decay scheduler
    pub fn step_decay(lr: f64, factor: f64, step_size: usize) -> Self {
        Self::new(lr, LearningRateScheduler::Step { factor, step_size })
    }

    /// Create cosine annealing scheduler
    pub fn cosine_annealing(lr: f64, lr_min: f64) -> Self {
        Self::new(lr, LearningRateScheduler::CosineAnnealing { lr_min })
    }

    /// Create reduce-on-plateau scheduler
    pub fn reduce_on_plateau(lr: f64, factor: f64, patience: usize, threshold: f64) -> Self {
        Self::new(
            lr,
            LearningRateScheduler::ReduceOnPlateau {
                factor,
                patience,
                threshold,
            },
        )
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Step the scheduler
    ///
    /// Updates learning rate based on epoch and optionally loss value.
    ///
    /// # Arguments
    ///
    /// * `loss` - Current loss value (for ReduceOnPlateau)
    pub fn step(&mut self, loss: Option<f64>) {
        self.epoch += 1;

        match &self.scheduler {
            LearningRateScheduler::Constant => {
                // No change
            }
            LearningRateScheduler::Exponential { decay_rate } => {
                self.current_lr = self.initial_lr * decay_rate.powf(self.epoch as f64);
            }
            LearningRateScheduler::Step { factor, step_size } => {
                if self.epoch.is_multiple_of(*step_size) {
                    self.current_lr *= factor;
                }
            }
            LearningRateScheduler::CosineAnnealing { lr_min } => {
                let progress = self.epoch as f64 / 1000.0; // Assume max 1000 epochs
                self.current_lr = lr_min
                    + (self.initial_lr - lr_min)
                        * 0.5
                        * (1.0 + (std::f64::consts::PI * progress).cos());
            }
            LearningRateScheduler::ReduceOnPlateau {
                factor,
                patience,
                threshold,
            } => {
                if let Some(current_loss) = loss {
                    if current_loss < self.best_loss - threshold {
                        self.best_loss = current_loss;
                        self.plateau_count = 0;
                    } else {
                        self.plateau_count += 1;
                        if self.plateau_count >= *patience {
                            self.current_lr *= factor;
                            self.plateau_count = 0;
                            tracing::info!(
                                "ReduceOnPlateau: reducing LR to {:.6e}",
                                self.current_lr
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_scheduler() {
        let mut scheduler = LRScheduler::constant(0.01);
        assert_eq!(scheduler.get_lr(), 0.01);

        scheduler.step(None);
        assert_eq!(scheduler.get_lr(), 0.01); // Should remain constant
    }

    #[test]
    fn test_exponential_scheduler() {
        let mut scheduler = LRScheduler::exponential(0.01, 0.9);
        assert_eq!(scheduler.get_lr(), 0.01);

        scheduler.step(None);
        assert!(scheduler.get_lr() < 0.01); // Should decay
    }

    #[test]
    fn test_step_scheduler() {
        let mut scheduler = LRScheduler::step_decay(0.01, 0.5, 2);
        assert_eq!(scheduler.get_lr(), 0.01);

        scheduler.step(None);
        assert_eq!(scheduler.get_lr(), 0.01); // No change on step 1

        scheduler.step(None);
        assert_eq!(scheduler.get_lr(), 0.005); // Reduced on step 2
    }

    #[test]
    fn test_reduce_on_plateau() {
        let mut scheduler = LRScheduler::reduce_on_plateau(0.01, 0.5, 2, 0.001);

        // Initial loss
        scheduler.step(Some(1.0));
        assert_eq!(scheduler.get_lr(), 0.01);

        // No improvement
        scheduler.step(Some(1.0));
        scheduler.step(Some(1.0)); // Plateau reached
        assert_eq!(scheduler.get_lr(), 0.005); // Should reduce LR
    }
}
