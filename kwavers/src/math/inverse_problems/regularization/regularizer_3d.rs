//! `ModelRegularizer3D` — regularization on 3D spatial model arrays.

use super::config::RegularizationConfig;
use ndarray::{Array3, Zip};

/// 3D Model Regularizer
///
/// Applies regularization penalties to model updates/gradients in inverse problems.
/// Works with 3D arrays representing spatial model properties.
#[derive(Debug)]
pub struct ModelRegularizer3D {
    config: RegularizationConfig,
}

impl ModelRegularizer3D {
    /// Create new regularizer from configuration
    #[must_use] 
    pub fn new(config: RegularizationConfig) -> Self {
        Self { config }
    }

    /// Apply all active regularization to gradient field
    pub fn apply_to_gradient(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
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

    /// Apply Tikhonov (L2) regularization
    /// Penalizes large model values: grad_reg = λ·m
    fn apply_tikhonov(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        Zip::from(gradient).and(model).par_for_each(|g, &m| {
            *g += self.config.tikhonov_weight * m;
        });
    }

    /// Apply Total Variation regularization
    /// Edge-preserving penalty: grad_reg = λ·∇·(∇m/|∇m|)
    fn apply_total_variation(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        let (nx, ny, nz) = model.dim();
        let eps = self.config.tv_epsilon;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    let dx = model[[i + 1, j, k]] - model[[i, j, k]];
                    let dy = model[[i, j + 1, k]] - model[[i, j, k]];
                    let dz = model[[i, j, k + 1]] - model[[i, j, k]];

                    let grad_norm = (dz.mul_add(dz, dx.mul_add(dx, dy * dy)) + eps).sqrt();

                    let tv_term = (3.0f64.mul_add(model[[i, j, k]], -model[[i + 1, j, k]])
                        - model[[i, j + 1, k]]
                        - model[[i, j, k + 1]])
                        / grad_norm;

                    gradient[[i, j, k]] += self.config.tv_weight * tv_term;
                }
            }
        }
    }

    /// Apply smoothness regularization using Laplacian
    /// Penalizes second derivatives: grad_reg = λ·∇²m
    fn apply_smoothness(&self, gradient: &mut Array3<f64>) {
        let (nx, ny, nz) = gradient.dim();
        let mut laplacian = Array3::zeros(gradient.dim());

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    laplacian[[i, j, k]] = 6.0f64.mul_add(-gradient[[i, j, k]], gradient[[i + 1, j, k]]
                        + gradient[[i - 1, j, k]]
                        + gradient[[i, j + 1, k]]
                        + gradient[[i, j - 1, k]]
                        + gradient[[i, j, k + 1]] + gradient[[i, j, k - 1]]);
                }
            }
        }

        Zip::from(gradient).and(&laplacian).par_for_each(|g, &lap| {
            *g += self.config.smoothness_weight * lap;
        });
    }

    /// Apply L1 (Lasso) regularization
    /// Sparsity-promoting penalty: grad_reg = λ·sign(m)
    fn apply_l1(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        Zip::from(gradient).and(model).par_for_each(|g, &m| {
            *g += self.config.l1_weight * m.signum();
        });
    }
}
