//! Conservation enforcement for interface coupling

use super::InterfaceGeometry;
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use ndarray::Array3;

/// Conservation enforcer for interface coupling
#[derive(Debug)]
pub struct HybridCouplingConservationEnforcer {
    /// Conservation tolerance
    tolerance: f64,
}

impl HybridCouplingConservationEnforcer {
    /// Create a new conservation enforcer
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(_geometry: &InterfaceGeometry) -> Self {
        Self { tolerance: 1e-10 }
    }

    /// Enforce conservation laws on transferred fields
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn enforce(
        &self,
        interpolated: &Array3<f64>,
        target: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        if interpolated.dim() != target.dim() {
            return Err(KwaversError::Validation(
                ValidationError::DimensionMismatch {
                    expected: format!("{:?}", target.dim()),
                    actual: format!("{:?}", interpolated.dim()),
                },
            ));
        }

        let mut conserved = interpolated.clone();

        self.enforce_integral_and_energy(&mut conserved, interpolated, target);

        Ok(conserved)
    }

    /// Enforce target interface integral and quadratic energy.
    ///
    /// ## Theorem
    ///
    /// Let `u` be the interpolated pressure trace and `t` the target pressure
    /// trace over `N` interface points. Set `m_t = sum(t)/N`,
    /// `z = u - mean(u)`, and `v = m_t + alpha z`. Then
    /// `sum(v)=sum(t)` for every `alpha` because `sum(z)=0`. If
    /// `||z||_2 > 0`, choosing
    /// `alpha = sqrt((||t||_2^2 - N m_t^2) / ||z||_2^2)` also gives
    /// `||v||_2 = ||t||_2`. The numerator is non-negative by Cauchy's
    /// inequality. If `z` is zero, the least-distorting conservative field is
    /// the constant target mean.
    fn enforce_integral_and_energy(
        &self,
        fields: &mut Array3<f64>,
        interpolated: &Array3<f64>,
        target: &Array3<f64>,
    ) {
        let n = target.len() as f64;
        let source_sum: f64 = interpolated.iter().sum();
        let source_mean = source_sum / n;
        let target_sum: f64 = target.iter().sum();
        let target_mean = target_sum / n;
        let target_energy: f64 = target.iter().map(|x| x * x).sum();
        let centered_energy: f64 = interpolated
            .iter()
            .map(|&x| {
                let centered = x - source_mean;
                centered * centered
            })
            .sum();

        if centered_energy <= self.tolerance {
            fields.fill(target_mean);
            return;
        }

        let constant_energy = n * target_mean * target_mean;
        let variable_energy = (target_energy - constant_energy).max(0.0);
        let scale = (variable_energy / centered_energy).sqrt();

        fields.zip_mut_with(interpolated, |field_value, &source_value| {
            *field_value = target_mean + scale * (source_value - source_mean);
        });
    }

    /// Get conservation metrics
    #[must_use]
    pub fn get_metrics(&self) -> HybridConservationMetrics {
        HybridConservationMetrics {
            mass_error: 0.0,
            momentum_error: (0.0, 0.0, 0.0),
            energy_error: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geometry() -> InterfaceGeometry {
        InterfaceGeometry {
            normal_direction: 0,
            plane_position: 0.0,
            extent: (1.0, 1.0),
            area: 1.0,
            num_points: 4,
        }
    }

    fn sum(field: &Array3<f64>) -> f64 {
        field.iter().sum()
    }

    fn energy(field: &Array3<f64>) -> f64 {
        field.iter().map(|x| x * x).sum()
    }

    #[test]
    fn enforcement_matches_target_integral_and_energy() {
        let enforcer = HybridCouplingConservationEnforcer::new(&geometry());
        let interpolated = Array3::from_shape_vec((2, 2, 1), vec![1.0, 2.0, 4.0, 8.0]).unwrap();
        let target = Array3::from_shape_vec((2, 2, 1), vec![3.0, 5.0, 6.0, 10.0]).unwrap();

        let conserved = enforcer.enforce(&interpolated, &target).unwrap();

        assert!((sum(&conserved) - sum(&target)).abs() < 1e-12);
        assert!((energy(&conserved) - energy(&target)).abs() < 1e-12);
    }

    #[test]
    fn enforcement_preserves_identical_interface() {
        let enforcer = HybridCouplingConservationEnforcer::new(&geometry());
        let interpolated = Array3::from_shape_vec((2, 2, 1), vec![1.0, -2.0, 4.0, 8.0]).unwrap();

        let conserved = enforcer.enforce(&interpolated, &interpolated).unwrap();

        assert_eq!(conserved, interpolated);
    }

    #[test]
    fn enforcement_rejects_shape_mismatch() {
        let enforcer = HybridCouplingConservationEnforcer::new(&geometry());
        let interpolated = Array3::zeros((2, 2, 1));
        let target = Array3::zeros((2, 1, 1));

        let error = enforcer.enforce(&interpolated, &target).unwrap_err();

        assert!(format!("{error}").contains("Dimension mismatch"));
    }
}

/// Conservation metrics
#[derive(Debug, Clone)]
pub struct HybridConservationMetrics {
    /// Mass conservation error
    pub mass_error: f64,
    /// Momentum conservation error (x, y, z)
    pub momentum_error: (f64, f64, f64),
    /// Energy conservation error
    pub energy_error: f64,
}
