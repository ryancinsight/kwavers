//! `ModelRegularizer1D` — regularization on 1D vector models.

use super::{config::RegularizationConfig, ops::for_each_pair_mut};
use ndarray::Array1;

/// 1D Regularizer for vector models
#[derive(Debug)]
pub struct ModelRegularizer1D {
    config: RegularizationConfig,
}

impl ModelRegularizer1D {
    /// Create new 1D regularizer
    #[must_use]
    pub fn new(config: RegularizationConfig) -> Self {
        Self { config }
    }

    /// Apply regularization to 1D gradient
    pub fn apply_to_gradient(&self, gradient: &mut Array1<f64>, model: &Array1<f64>) {
        if !self.config.is_active() {
            return;
        }

        if self.config.tikhonov_weight > 0.0 {
            self.apply_tikhonov(gradient, model);
        }

        if self.config.smoothness_weight > 0.0 {
            self.apply_smoothness(gradient);
        }

        if self.config.l1_weight > 0.0 {
            self.apply_l1(gradient, model);
        }
    }

    fn apply_tikhonov(&self, gradient: &mut Array1<f64>, model: &Array1<f64>) {
        let weight = self.config.tikhonov_weight;
        for_each_pair_mut(gradient, model, |g, m| *g += weight * m);
    }

    fn apply_smoothness(&self, gradient: &mut Array1<f64>) {
        let n = gradient.len();
        if n < 3 {
            return;
        }

        let mut laplacian = Array1::zeros(n);
        for i in 1..n - 1 {
            laplacian[i] = 2.0f64.mul_add(-gradient[i], gradient[i + 1] + gradient[i - 1]);
        }

        let weight = self.config.smoothness_weight;
        for_each_pair_mut(gradient, &laplacian, |g, lap| *g += weight * lap);
    }

    fn apply_l1(&self, gradient: &mut Array1<f64>, model: &Array1<f64>) {
        let weight = self.config.l1_weight;
        for_each_pair_mut(gradient, model, |g, m| *g += weight * m.signum());
    }
}
