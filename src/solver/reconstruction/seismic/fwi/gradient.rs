//! Gradient computation for FWI
//!
//! Implements adjoint-state method for efficient gradient calculation.
//!
//! ## Literature Reference
//! - Plessix, R. E. (2006). "A review of the adjoint-state method for
//!   computing the gradient of a functional with geophysical applications."
//!   Geophysical Journal International, 167(2), 495-503.

use crate::error::KwaversResult;
use ndarray::{Array2, Array3, Zip};

/// Gradient calculator for FWI
#[derive(Debug))]
pub struct GradientCalculator {
    /// Number of time steps
    nt: usize,
    /// Time step size
    dt: f64,
}

impl GradientCalculator {
    /// Create new gradient calculator
    pub fn new(nt: usize, dt: f64) -> Self {
        Self { nt, dt }
    }

    /// Compute gradient using adjoint-state method
    ///
    /// The gradient is computed as:
    /// ∇J = -∫∫ (∂²u/∂t²) * λ dt dx
    ///
    /// where u is forward wavefield and λ is adjoint wavefield
    pub fn compute_gradient(
        &self,
        forward_wavefield: &Array3<f64>,
        adjoint_wavefield: &Array3<f64>,
        velocity_model: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut gradient = Array3::zeros(velocity_model.dim());

        // Compute second time derivative of forward wavefield (approximation)
        // In practice, this would be stored during forward modeling
        let d2u_dt2 = self.compute_time_derivative(forward_wavefield);

        // Zero-lag cross-correlation
        Zip::from(&mut gradient)
            .and(&d2u_dt2)
            .and(adjoint_wavefield)
            .and(velocity_model)
            .for_each(|grad, &d2u, &adj, &vel| {
                // Gradient kernel: -2/v³ * ∫(∂²u/∂t²) * λ dt
                *grad = -2.0 / (vel * vel * vel) * d2u * adj * self.dt;
            });

        Ok(gradient)
    }

    /// Compute approximate second time derivative
    fn compute_time_derivative(&self, wavefield: &Array3<f64>) -> Array3<f64> {
        // This is a placeholder - in practice, we'd store this during forward modeling
        // Using simple finite difference approximation
        let mut derivative = Array3::zeros(wavefield.dim());

        // For demonstration, use Laplacian as proxy (wave equation relates them)
        for i in 1..wavefield.shape()[0] - 1 {
            for j in 1..wavefield.shape()[1] - 1 {
                for k in 1..wavefield.shape()[2] - 1 {
                    let laplacian = wavefield[[i + 1, j, k]
                        + wavefield[[i - 1, j, k]
                        + wavefield[[i, j + 1, k]
                        + wavefield[[i, j - 1, k]
                        + wavefield[[i, j, k + 1]
                        + wavefield[[i, j, k - 1]
                        - 6.0 * wavefield[[i, j, k];

                    derivative[[i, j, k] = laplacian / (self.dt * self.dt);
                }
            }
        }

        derivative
    }

    /// Apply preconditioning to gradient (depth scaling)
    pub fn precondition_gradient(&self, gradient: &Array3<f64>) -> Array3<f64> {
        let mut preconditioned = gradient.clone();

        // Apply depth-based scaling to compensate for geometric spreading
        for k in 0..gradient.shape()[2] {
            let depth_scale = ((k + 1) as f64).sqrt();
            for i in 0..gradient.shape()[0] {
                for j in 0..gradient.shape()[1] {
                    preconditioned[[i, j, k] *= depth_scale;
                }
            }
        }

        preconditioned
    }

    /// Compute data residual (observed - predicted)
    pub fn compute_residual(&self, observed: &Array2<f64>, predicted: &Array2<f64>) -> Array2<f64> {
        observed - predicted
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_computation() {
        let calc = GradientCalculator::new(100, 0.001);

        let forward = Array3::ones((32, 32, 32));
        let adjoint = Array3::ones((32, 32, 32));
        let velocity = Array3::ones((32, 32, 32)) * 1500.0;

        let gradient = calc
            .compute_gradient(&forward, &adjoint, &velocity)
            .unwrap();

        // Gradient should be non-zero
        let grad_norm: f64 = gradient.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(grad_norm > 0.0);
    }

    #[test]
    fn test_preconditioning() {
        let calc = GradientCalculator::new(100, 0.001);
        let gradient = Array3::ones((32, 32, 32));

        let preconditioned = calc.precondition_gradient(&gradient);

        // Deeper parts should have larger values after preconditioning
        assert!(preconditioned[[16, 16, 31] > preconditioned[[16, 16, 0]);
    }
}
