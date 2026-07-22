//! `PolynomialFilterConfig`: configuration for polynomial regression clutter filter.

use kwavers_core::error::{KwaversError, KwaversResult};

/// Configuration for polynomial regression clutter filter.
#[derive(Debug, Clone)]
pub struct PolynomialFilterConfig {
    /// Order of the polynomial (typically 1-5).
    ///
    /// - Order 1: Linear fit (tissue motion assumed linear)
    /// - Order 2: Quadratic fit (tissue acceleration)
    /// - Order 3-5: Higher-order motion (respiration, cardiac)
    ///
    /// Higher orders can fit more complex tissue motion but may remove
    /// some blood signal and are more sensitive to noise.
    pub polynomial_order: usize,

    /// Temporal normalization for numerical stability.
    ///
    /// When true, time indices are normalized to [0, 1] range before
    /// polynomial fitting to improve numerical conditioning.
    pub normalize_time: bool,
}

impl Default for PolynomialFilterConfig {
    fn default() -> Self {
        Self {
            polynomial_order: 2,  // Quadratic fit (good balance)
            normalize_time: true, // Recommended for stability
        }
    }
}

impl PolynomialFilterConfig {
    /// Create configuration with specific polynomial order.
    #[must_use]
    pub fn with_order(order: usize) -> Self {
        Self {
            polynomial_order: order,
            ..Default::default()
        }
    }

    /// Validate configuration parameters.
    ///
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if polynomial order is 0 or exceeds 10.
    pub fn validate(&self) -> KwaversResult<()> {
        if self.polynomial_order == 0 {
            return Err(KwaversError::InvalidInput(
                "Polynomial order must be >= 1".to_owned(),
            ));
        }
        if self.polynomial_order > 10 {
            return Err(KwaversError::InvalidInput(format!(
                "Polynomial order {} is too high (max 10)",
                self.polynomial_order
            )));
        }
        Ok(())
    }
}