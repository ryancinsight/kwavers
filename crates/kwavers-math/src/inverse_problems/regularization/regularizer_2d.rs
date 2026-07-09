//! `ModelRegularizer2D` — regularization on 2D spatial model arrays.

use super::{config::RegularizationConfig, ops::for_each_pair_mut};
use leto::Array2;

/// 2D Model Regularizer
///
/// Applies regularization to 2D model arrays (e.g., velocity models in 2D inversions).
#[derive(Debug)]
pub struct ModelRegularizer2D {
    config: RegularizationConfig,
}

impl ModelRegularizer2D {
    /// Create new 2D regularizer
    #[must_use]
    pub fn new(config: RegularizationConfig) -> Self {
        Self { config }
    }

    /// Apply regularization to 2D gradient
    pub fn apply_to_gradient(&self, gradient: &mut Array2<f64>, model: &Array2<f64>) {
        if !self.config.is_active() {
            return;
        }

        if self.config.tikhonov_weight > 0.0 {
            self.apply_tikhonov(gradient, model);
        }

        if self.config.tv_weight > 0.0 {
            self.apply_total_variation(gradient, model);
        }

        if self.config.smoothness_weight > 0.0 {
            self.apply_smoothness(gradient);
        }

        if self.config.l1_weight > 0.0 {
            self.apply_l1(gradient, model);
        }
    }

    fn apply_tikhonov(&self, gradient: &mut Array2<f64>, model: &Array2<f64>) {
        let weight = self.config.tikhonov_weight;
        for_each_pair_mut(gradient.view_mut(), model.view(), |g, m| *g += weight * m);
    }

    fn apply_total_variation(&self, gradient: &mut Array2<f64>, model: &Array2<f64>) {
        let [nx, ny] = model.shape();
        let eps = self.config.tv_epsilon;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let dx = model[[i + 1, j]] - model[[i, j]];
                let dy = model[[i, j + 1]] - model[[i, j]];

                let grad_norm = (dx.mul_add(dx, dy * dy) + eps).sqrt();
                let tv_term = (3.0f64.mul_add(model[[i, j]], -model[[i + 1, j]])
                    - model[[i, j + 1]])
                    / grad_norm;

                gradient[[i, j]] += self.config.tv_weight * tv_term;
            }
        }
    }

    fn apply_smoothness(&self, gradient: &mut Array2<f64>) {
        let [nx, ny] = gradient.shape();
        let mut laplacian = Array2::zeros(gradient.shape());

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                laplacian[[i, j]] = 4.0f64.mul_add(
                    -gradient[[i, j]],
                    gradient[[i + 1, j]]
                        + gradient[[i - 1, j]]
                        + gradient[[i, j + 1]]
                        + gradient[[i, j - 1]],
                );
            }
        }

        let weight = self.config.smoothness_weight;
        for_each_pair_mut(gradient.view_mut(), laplacian.view(), |g, lap| {
            *g += weight * lap
        });
    }

    fn apply_l1(&self, gradient: &mut Array2<f64>, model: &Array2<f64>) {
        let weight = self.config.l1_weight;
        for_each_pair_mut(gradient.view_mut(), model.view(), |g, m| {
            *g += weight * m.signum()
        });
    }
}
