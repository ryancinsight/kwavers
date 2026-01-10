//! Stability analysis for IMEX schemes

use super::IMEXSchemeType;
use crate::domain::core::error::KwaversResult;
use ndarray::Array3;
use std::f64::consts::PI;

/// Stability region information
#[derive(Debug, Clone)]
pub struct StabilityRegion {
    /// Maximum stable time step for explicit part
    pub explicit_dt_max: f64,
    /// Maximum stable time step for implicit part
    pub implicit_dt_max: f64,
    /// Combined stability limit
    pub combined_dt_max: f64,
    /// Stability angle (for A(α)-stable methods)
    pub stability_angle: Option<f64>,
}

/// IMEX stability analyzer
#[derive(Debug)]
pub struct IMEXStabilityAnalyzer {
    /// Number of test points for stability region
    n_test_points: usize,
    /// Safety factor
    safety_factor: f64,
}

impl IMEXStabilityAnalyzer {
    /// Create a new stability analyzer
    #[must_use]
    pub fn new() -> Self {
        Self {
            n_test_points: 100,
            safety_factor: 0.9,
        }
    }

    /// Set number of test points
    #[must_use]
    pub fn with_test_points(mut self, n: usize) -> Self {
        self.n_test_points = n;
        self
    }

    /// Set safety factor
    #[must_use]
    pub fn with_safety_factor(mut self, factor: f64) -> Self {
        self.safety_factor = factor;
        self
    }

    /// Compute stability region for a scheme
    #[must_use]
    pub fn compute_region(&self, scheme: &IMEXSchemeType) -> StabilityRegion {
        // For explicit part, estimate based on order
        let explicit_dt_max = match scheme.order() {
            1 => 2.0,
            2 => 2.0,
            3 => 2.51,
            4 => 2.78,
            _ => 1.0,
        };

        // For implicit part, L-stable methods have unlimited stability
        let implicit_dt_max = if scheme.is_l_stable() {
            f64::INFINITY
        } else if scheme.is_a_stable() {
            1000.0 // Large but finite
        } else {
            10.0
        };

        // Combined limit depends on the coupling
        let combined_dt_max = explicit_dt_max;

        // Stability angle for A(α)-stable methods
        let stability_angle = if scheme.is_a_stable() && !scheme.is_l_stable() {
            Some(PI / 2.0) // 90 degrees for A-stable
        } else {
            None
        };

        StabilityRegion {
            explicit_dt_max,
            implicit_dt_max,
            combined_dt_max,
            stability_angle,
        }
    }

    /// Estimate maximum stable time step
    pub fn max_stable_timestep<F, G>(
        &self,
        scheme: &IMEXSchemeType,
        field: &Array3<f64>,
        explicit_rhs: &F,
        implicit_rhs: &G,
    ) -> KwaversResult<f64>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // Estimate eigenvalues for stability
        let explicit_eigenvalue = self.estimate_max_eigenvalue(field, explicit_rhs)?;
        let implicit_eigenvalue = self.estimate_max_eigenvalue(field, implicit_rhs)?;

        // Get stability region
        let region = self.compute_region(scheme);

        // Compute time step limits
        let dt_explicit = if explicit_eigenvalue > 0.0 {
            region.explicit_dt_max / explicit_eigenvalue
        } else {
            f64::INFINITY
        };

        let dt_implicit = if implicit_eigenvalue > 0.0 && !scheme.is_l_stable() {
            region.implicit_dt_max / implicit_eigenvalue
        } else {
            f64::INFINITY
        };

        // Take minimum and apply safety factor
        Ok(self.safety_factor * dt_explicit.min(dt_implicit))
    }

    /// Estimate maximum eigenvalue magnitude
    fn estimate_max_eigenvalue<F>(&self, field: &Array3<f64>, rhs_fn: &F) -> KwaversResult<f64>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        // Use power iteration to estimate largest eigenvalue
        let epsilon = 1e-8;
        let mut v = Array3::from_elem(field.dim(), 1.0);

        // Normalize
        let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
        v.mapv_inplace(|x| x / norm);

        let mut eigenvalue = 0.0;

        for _ in 0..10 {
            // Apply Jacobian via finite differences
            let mut field_perturbed = field.clone();
            for (fp, &vi) in field_perturbed.iter_mut().zip(v.iter()) {
                *fp += epsilon * vi;
            }

            let f_base = rhs_fn(field)?;
            let f_pert = rhs_fn(&field_perturbed)?;

            // Jacobian-vector product
            let mut jv = Array3::zeros(field.dim());
            for ((j, &fb), &fp) in jv.iter_mut().zip(f_base.iter()).zip(f_pert.iter()) {
                *j = (fp - fb) / epsilon;
            }

            // Update eigenvalue estimate
            eigenvalue = jv.iter().zip(&v).map(|(&j, &vi)| j * vi).sum::<f64>();

            // Update vector
            v = jv;
            let norm: f64 = v.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm > 0.0 {
                v.mapv_inplace(|x| x / norm);
            } else {
                break;
            }
        }

        Ok(eigenvalue.abs())
    }

    /// Check if a time step is stable
    pub fn is_stable<F, G>(
        &self,
        scheme: &IMEXSchemeType,
        field: &Array3<f64>,
        dt: f64,
        explicit_rhs: &F,
        implicit_rhs: &G,
    ) -> KwaversResult<bool>
    where
        F: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
        G: Fn(&Array3<f64>) -> KwaversResult<Array3<f64>>,
    {
        let max_dt = self.max_stable_timestep(scheme, field, explicit_rhs, implicit_rhs)?;
        Ok(dt <= max_dt)
    }

    /// Compute stability function value at a point
    #[must_use]
    /// Compute combined IMEX stability function
    ///
    /// For additive IMEX schemes treating du/dt = F_E(u) + F_I(u),
    /// the stability function is a combination R(z_e, z_i).
    ///
    /// This uses the factorized approximation R ≈ R_explicit(z_e) * R_implicit(z_i)
    /// which is exact for linear operators and provides good estimates for nonlinear cases.
    ///
    /// References:
    /// - Ascher et al. (1997): "Implicit-explicit Runge-Kutta methods for time-dependent PDEs"
    /// - Kennedy & Carpenter (2003): "Additive Runge-Kutta schemes"
    pub fn stability_function_at_point(
        &self,
        scheme: &IMEXSchemeType,
        z_explicit: f64,
        z_implicit: f64,
    ) -> f64 {
        // Factorized stability function approximation
        let r_explicit = self.explicit_stability_function(scheme.order(), z_explicit);
        let r_implicit = scheme.stability_function(z_implicit);

        r_explicit * r_implicit
    }

    /// Explicit stability function (for RK methods)
    fn explicit_stability_function(&self, order: usize, z: f64) -> f64 {
        // Taylor series expansion of exp(z) up to given order
        let mut sum = 1.0;
        let mut term = 1.0;

        for k in 1..=order {
            term *= z / k as f64;
            sum += term;
        }

        sum
    }
}

impl Default for IMEXStabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
