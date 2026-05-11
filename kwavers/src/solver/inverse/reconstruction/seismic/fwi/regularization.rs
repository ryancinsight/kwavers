//! Regularization methods for FWI
//! Based on Tikhonov & Arsenin (1977): "Solutions of Ill-posed Problems"

use ndarray::{Array3, Zip};

/// Regularization methods for FWI
#[derive(Debug)]
pub struct Regularizer {
    /// Tikhonov regularization weight
    tikhonov_weight: f64,
    /// Total variation weight
    tv_weight: f64,
    /// Smoothness weight
    smoothness_weight: f64,
}

impl Default for Regularizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Regularizer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            tikhonov_weight: 0.01,
            tv_weight: 0.0,
            smoothness_weight: 0.0,
        }
    }

    /// Apply combined regularization to gradient
    pub fn apply_regularization(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        if self.tikhonov_weight > 0.0 {
            self.apply_tikhonov(gradient, model);
        }

        if self.tv_weight > 0.0 {
            self.apply_total_variation(gradient, model);
        }

        if self.smoothness_weight > 0.0 {
            self.apply_smoothness(gradient);
        }
    }

    /// Tikhonov (L2) regularization
    /// Penalizes large model values
    fn apply_tikhonov(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        let w = self.tikhonov_weight;
        Zip::from(gradient).and(model).par_for_each(|g, &m| {
            *g += w * m;
        });
    }

    /// Total Variation regularization
    /// Preserves edges while smoothing
    fn apply_total_variation(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        let (nx, ny, nz) = model.dim();
        let epsilon = 1e-8; // Small value to avoid division by zero

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Compute TV gradient
                    let dx = model[[i + 1, j, k]] - model[[i, j, k]];
                    let dy = model[[i, j + 1, k]] - model[[i, j, k]];
                    let dz = model[[i, j, k + 1]] - model[[i, j, k]];

                    let tv_norm = (dz.mul_add(dz, dx.mul_add(dx, dy * dy)) + epsilon).sqrt();

                    gradient[[i, j, k]] += self.tv_weight
                        * (3.0f64.mul_add(model[[i, j, k]], -model[[i + 1, j, k]])
                            - model[[i, j + 1, k]]
                            - model[[i, j, k + 1]])
                        / tv_norm;
                }
            }
        }
    }

    /// Smoothness regularization using Laplacian
    fn apply_smoothness(&self, gradient: &mut Array3<f64>) {
        let (nx, ny, nz) = gradient.dim();
        let mut laplacian = Array3::zeros(gradient.dim());

        // Compute Laplacian
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

        // Apply smoothness penalty
        let w = self.smoothness_weight;
        Zip::from(gradient).and(&laplacian).par_for_each(|g, &l| {
            *g -= w * l;
        });
    }

    /// Set regularization weights
    pub fn set_weights(&mut self, tikhonov: f64, tv: f64, smoothness: f64) {
        self.tikhonov_weight = tikhonov;
        self.tv_weight = tv;
        self.smoothness_weight = smoothness;
    }
}
