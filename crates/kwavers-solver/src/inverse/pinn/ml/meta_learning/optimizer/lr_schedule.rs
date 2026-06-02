//! Learning rate schedules for meta-optimization.
//!
//! Implements common LR scheduling strategies:
//! - Constant
//! - Step decay: `lr = lr₀ * factor^(epoch / step_size)`
//! - Exponential: `lr = lr₀ * e^(-λt)`
//! - Cosine annealing: `lr = lr_min + 0.5(lr_max - lr_min)(1 + cos(πt/T))`

/// Learning rate schedule for meta-optimization.
#[derive(Debug, Clone)]
pub enum MetaLrSchedule {
    /// Constant learning rate — no decay.
    Constant,

    /// Step decay: multiply by `factor` every `step_size` epochs.
    StepDecay { factor: f64, step_size: usize },

    /// Exponential decay: `lr = lr₀ * e^(-decay_rate * epoch)`.
    Exponential { decay_rate: f64 },

    /// Cosine annealing: smooth decay following a cosine curve.
    CosineAnnealing { lr_min: f64, total_epochs: usize },
}

impl MetaLrSchedule {
    /// Compute the learning rate for the given `epoch` starting from `lr_initial`.
    pub fn get_lr(&self, epoch: usize, lr_initial: f64) -> f64 {
        match self {
            MetaLrSchedule::Constant => lr_initial,

            MetaLrSchedule::StepDecay { factor, step_size } => {
                let num_decays = epoch / step_size;
                lr_initial * factor.powi(num_decays as i32)
            }

            MetaLrSchedule::Exponential { decay_rate } => {
                lr_initial * (-decay_rate * epoch as f64).exp()
            }

            MetaLrSchedule::CosineAnnealing {
                lr_min,
                total_epochs,
            } => {
                let progress = (epoch as f64 / *total_epochs as f64).min(1.0);
                lr_min
                    + 0.5 * (lr_initial - lr_min) * (1.0 + (std::f64::consts::PI * progress).cos())
            }
        }
    }
}
