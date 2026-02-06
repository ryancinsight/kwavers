//! Regularization Methods for Ill-Posed Inverse Problems
//!
//! This module provides a comprehensive framework for regularization techniques
//! used in solving inverse problems. It serves as the Single Source of Truth (SSOT)
//! for all regularization implementations across the library.
//!
//! **Regularization Strategies**:
//!
//! 1. **Tikhonov Regularization (L2)**:
//!    - Penalizes large model values
//!    - Stabilizes solution by adding L2 norm constraint
//!    - Formulation: J(m) = ||Am - d||² + λ||m||²
//!
//! 2. **Total Variation (TV)**:
//!    - Edge-preserving regularization
//!    - Penalizes gradients while preserving discontinuities
//!    - Formulation: J(m) = ||Am - d||² + λ∫|∇m|
//!
//! 3. **Smoothness (Laplacian)**:
//!    - Encourages smooth variations in model
//!    - Penalizes second derivatives
//!    - Formulation: J(m) = ||Am - d||² + λ||∇²m||²
//!
//! 4. **L1 Regularization (Lasso)**:
//!    - Sparsity-promoting regularization
//!    - Encourages sparse solutions
//!    - Formulation: J(m) = ||Am - d||² + λ||m||₁
//!
//! 5. **Depth Weighting**:
//!    - Depth-dependent model weighting
//!    - Counteracts natural decay of sensitivity with depth
//!    - Formulation: J(m) = ||W(Am - d)||² + λ||D·m||²
//!
//! **References**:
//! - Tikhonov & Arsenin (1977): "Solutions of Ill-posed Problems"
//! - Rudin, Osher, Fatemi (1992): "Nonlinear total variation based noise removal"
//! - Hastie, Tibshirani, Wainwright (2015): "Statistical Learning with Sparsity"
//! - Constable et al. (1987): "Occam's inversion to generate smooth, two-dimensional models"

use ndarray::{Array1, Array2, Array3, Zip};

/// Configuration for regularization
#[derive(Debug, Clone, Copy)]
pub struct RegularizationConfig {
    /// Tikhonov (L2) regularization weight
    pub tikhonov_weight: f64,
    /// Total variation regularization weight
    pub tv_weight: f64,
    /// Smoothness (Laplacian) regularization weight
    pub smoothness_weight: f64,
    /// L1 regularization weight (sparsity)
    pub l1_weight: f64,
    /// Depth weighting exponent (0 = none, typical: 2.0)
    pub depth_weighting_exponent: f64,
    /// Small value for TV to avoid division by zero
    pub tv_epsilon: f64,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            tikhonov_weight: 0.01,
            tv_weight: 0.0,
            smoothness_weight: 0.0,
            l1_weight: 0.0,
            depth_weighting_exponent: 0.0,
            tv_epsilon: 1e-8,
        }
    }
}

impl RegularizationConfig {
    /// Create new configuration with no regularization
    pub fn none() -> Self {
        Self {
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            smoothness_weight: 0.0,
            l1_weight: 0.0,
            depth_weighting_exponent: 0.0,
            tv_epsilon: 1e-8,
        }
    }

    /// Enable Tikhonov (L2) regularization
    pub fn with_tikhonov(mut self, weight: f64) -> Self {
        self.tikhonov_weight = weight;
        self
    }

    /// Enable Total Variation regularization
    pub fn with_tv(mut self, weight: f64) -> Self {
        self.tv_weight = weight;
        self
    }

    /// Enable smoothness regularization
    pub fn with_smoothness(mut self, weight: f64) -> Self {
        self.smoothness_weight = weight;
        self
    }

    /// Enable L1 (Lasso) regularization
    pub fn with_l1(mut self, weight: f64) -> Self {
        self.l1_weight = weight;
        self
    }

    /// Enable depth weighting
    pub fn with_depth_weighting(mut self, exponent: f64) -> Self {
        self.depth_weighting_exponent = exponent;
        self
    }

    /// Check if any regularization is active
    pub fn is_active(&self) -> bool {
        self.tikhonov_weight > 0.0
            || self.tv_weight > 0.0
            || self.smoothness_weight > 0.0
            || self.l1_weight > 0.0
    }
}

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
        Zip::from(gradient).and(model).for_each(|g, &m| {
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
                    // Compute gradients
                    let dx = model[[i + 1, j, k]] - model[[i, j, k]];
                    let dy = model[[i, j + 1, k]] - model[[i, j, k]];
                    let dz = model[[i, j, k + 1]] - model[[i, j, k]];

                    let grad_norm = (dx * dx + dy * dy + dz * dz + eps).sqrt();

                    // TV gradient (divergence of normalized gradient)
                    let tv_term = (3.0 * model[[i, j, k]]
                        - model[[i + 1, j, k]]
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
                    laplacian[[i, j, k]] = gradient[[i + 1, j, k]]
                        + gradient[[i - 1, j, k]]
                        + gradient[[i, j + 1, k]]
                        + gradient[[i, j - 1, k]]
                        + gradient[[i, j, k + 1]]
                        + gradient[[i, j, k - 1]]
                        - 6.0 * gradient[[i, j, k]];
                }
            }
        }

        Zip::from(gradient).and(&laplacian).for_each(|g, &lap| {
            *g += self.config.smoothness_weight * lap;
        });
    }

    /// Apply L1 (Lasso) regularization
    /// Sparsity-promoting penalty: grad_reg = λ·sign(m)
    fn apply_l1(&self, gradient: &mut Array3<f64>, model: &Array3<f64>) {
        Zip::from(gradient).and(model).for_each(|g, &m| {
            *g += self.config.l1_weight * m.signum();
        });
    }
}

/// 2D Model Regularizer
///
/// Applies regularization to 2D model arrays (e.g., velocity models in 2D inversions).
#[derive(Debug)]
pub struct ModelRegularizer2D {
    config: RegularizationConfig,
}

impl ModelRegularizer2D {
    /// Create new 2D regularizer
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
        Zip::from(gradient).and(model).for_each(|g, &m| {
            *g += self.config.tikhonov_weight * m;
        });
    }

    fn apply_total_variation(&self, gradient: &mut Array2<f64>, model: &Array2<f64>) {
        let (nx, ny) = model.dim();
        let eps = self.config.tv_epsilon;

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                let dx = model[[i + 1, j]] - model[[i, j]];
                let dy = model[[i, j + 1]] - model[[i, j]];

                let grad_norm = (dx * dx + dy * dy + eps).sqrt();
                let tv_term =
                    (3.0 * model[[i, j]] - model[[i + 1, j]] - model[[i, j + 1]]) / grad_norm;

                gradient[[i, j]] += self.config.tv_weight * tv_term;
            }
        }
    }

    fn apply_smoothness(&self, gradient: &mut Array2<f64>) {
        let (nx, ny) = gradient.dim();
        let mut laplacian = Array2::zeros(gradient.dim());

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                laplacian[[i, j]] = gradient[[i + 1, j]]
                    + gradient[[i - 1, j]]
                    + gradient[[i, j + 1]]
                    + gradient[[i, j - 1]]
                    - 4.0 * gradient[[i, j]];
            }
        }

        Zip::from(gradient).and(&laplacian).for_each(|g, &lap| {
            *g += self.config.smoothness_weight * lap;
        });
    }

    fn apply_l1(&self, gradient: &mut Array2<f64>, model: &Array2<f64>) {
        Zip::from(gradient).and(model).for_each(|g, &m| {
            *g += self.config.l1_weight * m.signum();
        });
    }
}

/// 1D Regularizer for vector models
#[derive(Debug)]
pub struct ModelRegularizer1D {
    config: RegularizationConfig,
}

impl ModelRegularizer1D {
    /// Create new 1D regularizer
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
        Zip::from(gradient).and(model).for_each(|g, &m| {
            *g += self.config.tikhonov_weight * m;
        });
    }

    fn apply_smoothness(&self, gradient: &mut Array1<f64>) {
        let n = gradient.len();
        if n < 3 {
            return;
        }

        let mut laplacian = Array1::zeros(n);
        for i in 1..n - 1 {
            laplacian[i] = gradient[i + 1] + gradient[i - 1] - 2.0 * gradient[i];
        }

        Zip::from(gradient).and(&laplacian).for_each(|g, &lap| {
            *g += self.config.smoothness_weight * lap;
        });
    }

    fn apply_l1(&self, gradient: &mut Array1<f64>, model: &Array1<f64>) {
        Zip::from(gradient).and(model).for_each(|g, &m| {
            *g += self.config.l1_weight * m.signum();
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regularization_config_default() {
        let cfg = RegularizationConfig::default();
        assert_eq!(cfg.tikhonov_weight, 0.01);
        assert_eq!(cfg.tv_weight, 0.0);
        assert!(cfg.is_active());
    }

    #[test]
    fn test_regularization_config_builder() {
        let cfg = RegularizationConfig::none()
            .with_tikhonov(0.05)
            .with_tv(0.02)
            .with_smoothness(0.01);

        assert_eq!(cfg.tikhonov_weight, 0.05);
        assert_eq!(cfg.tv_weight, 0.02);
        assert_eq!(cfg.smoothness_weight, 0.01);
        assert!(cfg.is_active());
    }

    #[test]
    fn test_tikhonov_3d() {
        let mut gradient = Array3::zeros((3, 3, 3));
        let model = Array3::ones((3, 3, 3));

        let cfg = RegularizationConfig::default().with_tikhonov(0.5);
        let regularizer = ModelRegularizer3D::new(cfg);
        regularizer.apply_to_gradient(&mut gradient, &model);

        // Gradient should be modified (all ones now)
        assert!(gradient[[1, 1, 1]] > 0.0);
    }

    #[test]
    fn test_tv_3d_basic() {
        let mut gradient = Array3::zeros((3, 3, 3));
        let model = Array3::zeros((3, 3, 3));

        let cfg = RegularizationConfig::default().with_tv(0.1);
        let regularizer = ModelRegularizer3D::new(cfg);
        regularizer.apply_to_gradient(&mut gradient, &model);

        // Gradient of zero field should remain zero
        assert_eq!(gradient[[1, 1, 1]], 0.0);
    }

    #[test]
    fn test_smoothness_3d() {
        let mut gradient = Array3::zeros((5, 5, 5));
        gradient[[2, 2, 2]] = 1.0; // Spike

        let cfg = RegularizationConfig::default().with_smoothness(0.1);
        let regularizer = ModelRegularizer3D::new(cfg);
        let model = Array3::zeros((5, 5, 5));
        regularizer.apply_to_gradient(&mut gradient, &model);

        // Central point should be reduced by Laplacian smoothing
        assert!(gradient[[2, 2, 2]] < 1.0);
    }

    #[test]
    fn test_l1_3d() {
        let mut gradient = Array3::zeros((3, 3, 3));
        let model = Array3::ones((3, 3, 3));

        let cfg = RegularizationConfig::default().with_l1(0.5);
        let regularizer = ModelRegularizer3D::new(cfg);
        regularizer.apply_to_gradient(&mut gradient, &model);

        // Gradient should be modified by sign penalty
        assert!(gradient[[1, 1, 1]] > 0.0);
    }

    #[test]
    fn test_regularization_2d() {
        let mut gradient = Array2::zeros((3, 3));
        let model = Array2::ones((3, 3));

        let cfg = RegularizationConfig::default().with_tikhonov(0.2);
        let regularizer = ModelRegularizer2D::new(cfg);
        regularizer.apply_to_gradient(&mut gradient, &model);

        assert!(gradient[[1, 1]] > 0.0);
    }

    #[test]
    fn test_regularization_1d() {
        let mut gradient = Array1::zeros(5);
        let model = Array1::ones(5);

        let cfg = RegularizationConfig::default().with_tikhonov(0.3);
        let regularizer = ModelRegularizer1D::new(cfg);
        regularizer.apply_to_gradient(&mut gradient, &model);

        assert!(gradient[2] > 0.0);
    }

    #[test]
    fn test_combined_regularization() {
        let mut gradient = Array3::zeros((5, 5, 5));
        let model = Array3::ones((5, 5, 5));

        let cfg = RegularizationConfig::default()
            .with_tikhonov(0.01)
            .with_tv(0.01)
            .with_smoothness(0.01);

        let regularizer = ModelRegularizer3D::new(cfg);
        regularizer.apply_to_gradient(&mut gradient, &model);

        // Should have combined effect
        assert!(gradient[[2, 2, 2]] > 0.0);
    }

    #[test]
    fn test_no_regularization() {
        let mut gradient = Array3::ones((3, 3, 3));
        let model = Array3::ones((3, 3, 3));
        let original = gradient.clone();

        let cfg = RegularizationConfig::none();
        let regularizer = ModelRegularizer3D::new(cfg);
        regularizer.apply_to_gradient(&mut gradient, &model);

        // Should be unchanged
        assert_eq!(gradient, original);
    }
}
