//! Electromagnetic Physics Domain for PINN
//!
//! This module implements Maxwell's equations for electromagnetic field simulation
//! using Physics-Informed Neural Networks. The implementation supports electrostatic,
//! magnetostatic, and quasi-static electromagnetic phenomena.

pub mod domain;
pub mod physics;
pub mod residuals;
pub mod types;

// Re-export main types
pub use domain::ElectromagneticDomain;
pub use types::{EMProblemType, ElectromagneticBoundarySpec};

#[cfg(test)]
mod tests {
    use super::*;

    type TestBackend = burn::backend::Autodiff<burn::backend::NdArray<f32>>;

    // We need to bring variants into scope or use full path
    use crate::analysis::ml::pinn::physics::BoundaryPosition;

    #[test]
    fn test_electromagnetic_domain_creation() {
        let domain: ElectromagneticDomain<TestBackend> = ElectromagneticDomain::new(
            EMProblemType::Electrostatic,
            8.854e-12,
            4e-7 * std::f64::consts::PI,
            0.0,
            vec![1.0, 1.0],
        );

        assert_eq!(domain.problem_type, EMProblemType::Electrostatic);
        assert!((domain.permittivity - 8.854e-12).abs() < 1e-15);
    }

    #[test]
    fn test_domain_validation() {
        let valid_domain: ElectromagneticDomain<TestBackend> =
            ElectromagneticDomain::<TestBackend>::default();
        assert!(valid_domain.validate().is_ok());

        let invalid_domain: ElectromagneticDomain<TestBackend> = ElectromagneticDomain::new(
            EMProblemType::Electrostatic,
            -1.0, // Invalid permittivity
            4e-7 * std::f64::consts::PI,
            0.0,
            vec![1.0, 1.0],
        );
        assert!(invalid_domain.validate().is_err());
    }

    #[test]
    fn test_boundary_condition_builder() {
        let domain = ElectromagneticDomain::<TestBackend>::default()
            .add_pec_boundary(BoundaryPosition::Left)
            .add_pmc_boundary(BoundaryPosition::Right);

        assert_eq!(domain.boundary_specs.len(), 2);

        match &domain.boundary_specs[0] {
            ElectromagneticBoundarySpec::PerfectElectricConductor { .. } => {
                // Correct type
            }
            _ => panic!("Expected PerfectElectricConductor"),
        }
    }
}
