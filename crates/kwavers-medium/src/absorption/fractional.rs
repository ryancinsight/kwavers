//! Fractional Laplacian absorption model

use eunomia::Complex;
use kwavers_core::constants::numerical::TWO_PI;
use leto::Array3;

use kwavers_core::traversal::zip_mut;

/// Fractional Laplacian model for absorption
///
/// Implements the fractional wave equation:
/// ∂²p/∂t² = c² ∇²p - τ ∂^(y+1)/∂t^(y+1) ∇²p
#[derive(Debug, Clone)]
pub struct FractionalLaplacian {
    /// Power law exponent
    y: f64,
    /// Absorption parameter τ
    tau: f64,
    /// Reference sound speed
    c0: f64,
}

impl FractionalLaplacian {
    /// Create a new fractional Laplacian model
    #[must_use]
    pub fn new(y: f64, tau: f64, c0: f64) -> Self {
        Self { y, tau, c0 }
    }

    /// Calculate absorption coefficient
    #[must_use]
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        // α(ω) = τ * ω^y / (2 * c₀^3)
        let omega = TWO_PI * frequency;
        self.tau * omega.powf(self.y) / (2.0 * self.c0.powi(3))
    }

    /// Apply fractional Laplacian in k-space
    pub fn apply_k_space(
        &self,
        spectrum: &mut Array3<Complex<f64>>,
        k_squared: &Array3<f64>,
        frequency: f64,
    ) {
        let omega = TWO_PI * frequency;

        zip_mut(spectrum.view_mut(), k_squared.view(), |s, &k2| {
            if k2 > 0.0 {
                // Fractional Laplacian operator: (-k²)^(y/2)
                let fractional_term = k2.powf(self.y / 2.0);

                // Absorption factor
                let absorption = -self.tau * omega.powf(self.y + 1.0) * fractional_term;

                // Apply as exponential decay
                *s *= absorption.exp();
            }
        });
    }
}

mod derivative;
pub use derivative::FractionalDerivative;
