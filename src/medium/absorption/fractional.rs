//! Fractional Laplacian absorption model

use ndarray::{Array3, Zip};
use num_complex::Complex;

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
    pub fn new(y: f64, tau: f64, c0: f64) -> Self {
        Self { y, tau, c0 }
    }

    /// Calculate absorption coefficient
    pub fn absorption_at_frequency(&self, frequency: f64) -> f64 {
        // α(ω) = τ * ω^y / (2 * c₀^3)
        let omega = 2.0 * std::f64::consts::PI * frequency;
        self.tau * omega.powf(self.y) / (2.0 * self.c0.powi(3))
    }

    /// Apply fractional Laplacian in k-space
    pub fn apply_k_space(
        &self,
        spectrum: &mut Array3<Complex<f64>>,
        k_squared: &Array3<f64>,
        frequency: f64,
    ) {
        let omega = 2.0 * std::f64::consts::PI * frequency;

        Zip::from(spectrum).and(k_squared).for_each(|s, &k2| {
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

/// Fractional derivative calculator
pub struct FractionalDerivative {
    /// Fractional order
    order: f64,
    /// Grünwald-Letnikov coefficients
    gl_coefficients: Vec<f64>,
}

impl FractionalDerivative {
    /// Create a new fractional derivative calculator
    pub fn new(order: f64, max_terms: usize) -> Self {
        let gl_coefficients = Self::compute_gl_coefficients(order, max_terms);

        Self {
            order,
            gl_coefficients,
        }
    }

    /// Compute Grünwald-Letnikov coefficients
    fn compute_gl_coefficients(alpha: f64, n: usize) -> Vec<f64> {
        let mut coeffs = vec![1.0];

        for k in 1..=n {
            let coeff = coeffs[k - 1] * (k as f64 - 1.0 - alpha) / k as f64;
            coeffs.push(coeff);
        }

        coeffs
    }

    /// Apply fractional derivative using Grünwald-Letnikov method
    pub fn apply(&self, signal: &[f64], dt: f64) -> Vec<f64> {
        let n = signal.len();
        let mut result = vec![0.0; n];

        let dt_alpha = dt.powf(-self.order);

        for i in 0..n {
            let mut sum = 0.0;
            let max_k = (i + 1).min(self.gl_coefficients.len());

            for k in 0..max_k {
                sum += self.gl_coefficients[k] * signal[i - k];
            }

            result[i] = dt_alpha * sum;
        }

        result
    }

    /// Apply fractional derivative in frequency domain
    pub fn apply_frequency_domain(&self, spectrum: &mut [Complex<f64>], frequencies: &[f64]) {
        for (s, &freq) in spectrum.iter_mut().zip(frequencies.iter()) {
            if freq != 0.0 {
                // (iω)^α = |ω|^α * exp(i * α * π/2 * sign(ω))
                let omega = 2.0 * std::f64::consts::PI * freq;
                let magnitude = omega.abs().powf(self.order);
                let phase = self.order * std::f64::consts::PI / 2.0 * omega.signum();

                *s *= Complex::from_polar(magnitude, phase);
            }
        }
    }
}
