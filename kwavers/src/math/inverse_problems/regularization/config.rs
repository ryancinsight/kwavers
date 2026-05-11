//! `RegularizationConfig` — weights and parameters for all regularization strategies.

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
    #[must_use] 
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
    #[must_use] 
    pub fn with_tikhonov(mut self, weight: f64) -> Self {
        self.tikhonov_weight = weight;
        self
    }

    /// Enable Total Variation regularization
    #[must_use] 
    pub fn with_tv(mut self, weight: f64) -> Self {
        self.tv_weight = weight;
        self
    }

    /// Enable smoothness regularization
    #[must_use] 
    pub fn with_smoothness(mut self, weight: f64) -> Self {
        self.smoothness_weight = weight;
        self
    }

    /// Enable L1 (Lasso) regularization
    #[must_use] 
    pub fn with_l1(mut self, weight: f64) -> Self {
        self.l1_weight = weight;
        self
    }

    /// Enable depth weighting
    #[must_use] 
    pub fn with_depth_weighting(mut self, exponent: f64) -> Self {
        self.depth_weighting_exponent = exponent;
        self
    }

    /// Check if any regularization is active
    #[must_use] 
    pub fn is_active(&self) -> bool {
        self.tikhonov_weight > 0.0
            || self.tv_weight > 0.0
            || self.smoothness_weight > 0.0
            || self.l1_weight > 0.0
    }
}
