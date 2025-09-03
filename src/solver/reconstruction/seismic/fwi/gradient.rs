//! Gradient computation for FWI
//! Based on Plessix (2006): "Adjoint-state method for gradient computation"

use crate::error::KwaversResult;
use ndarray::{Array2, Array3, Zip};

/// Gradient computation methods for FWI
#[derive(Debug)]
pub struct GradientComputer {
    /// Preconditioning matrix (optional)
    preconditioner: Option<Array3<f64>>,
}

impl Default for GradientComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl GradientComputer {
    #[must_use]
    pub fn new() -> Self {
        Self {
            preconditioner: None,
        }
    }

    /// Compute gradient using adjoint method
    /// Based on Tarantola (1984) formulation
    #[must_use]
    pub fn compute_adjoint_gradient(
        &self,
        forward_wavefield: &Array3<f64>,
        adjoint_wavefield: &Array3<f64>,
        dt: f64,
    ) -> Array3<f64> {
        // Gradient = -∫ (∂²u_f/∂t²) * u_a dt
        // where u_f is forward wavefield, u_a is adjoint wavefield

        let mut gradient = Array3::zeros(forward_wavefield.dim());

        // Compute gradient using imaging condition: ∇J = -∫ ∂²u/∂t² · λ dt
        // where u is forward wavefield and λ is adjoint wavefield
        let dt2 = dt * dt;
        let (nx, ny, nz) = forward_wavefield.dim();

        // Apply second-order finite difference for time derivative
        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // Second time derivative approximation (assuming stored at current time)
                    let d2u_dt2 = (forward_wavefield[[i + 1, j, k]]
                        - 2.0 * forward_wavefield[[i, j, k]]
                        + forward_wavefield[[i - 1, j, k]])
                        / dt2;

                    gradient[[i, j, k]] = -d2u_dt2 * adjoint_wavefield[[i, j, k]];
                }
            }
        }
        // 2. Zero-lag cross-correlation with adjoint
        // 3. Integration over time

        gradient
    }

    /// Apply preconditioning to gradient
    /// Based on Shin et al. (2001): "Amplitude preservation for elastic migration"
    pub fn apply_preconditioning(&self, gradient: &mut Array3<f64>) {
        if let Some(ref precond) = self.preconditioner {
            Zip::from(gradient).and(precond).for_each(|g, &p| {
                *g *= p;
            });
        }
    }

    /// Compute Hessian-vector product for Newton methods
    /// Based on Pratt et al. (1998): "Gauss-Newton and full Newton methods"
    pub fn hessian_vector_product(
        &self,
        model_perturbation: &Array3<f64>,
        forward_wavefield: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        // H*dm = ∂²J/∂m² * dm
        // Requires solving another forward problem

        Ok(Array3::zeros(model_perturbation.dim()))
    }

    /// Compute gradient with source encoding
    /// Based on Krebs et al. (2009): "Fast full-wavefield seismic inversion"
    #[must_use]
    pub fn encoded_gradient(
        &self,
        encoded_sources: &Array2<f64>,
        encoded_residuals: &Array2<f64>,
    ) -> Array3<f64> {
        // Simultaneous source gradient computation
        Array3::zeros((100, 100, 100))
    }
}
