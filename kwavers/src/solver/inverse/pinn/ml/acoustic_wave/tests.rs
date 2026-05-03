use super::domain::AcousticWaveDomain;
use super::types::{AcousticBoundarySpec, AcousticBoundaryType, AcousticProblemType};
use crate::solver::inverse::pinn::ml::physics::{BoundaryPosition, PhysicsDomain};
use burn::backend::NdArray;
use std::collections::HashMap;

type B = burn::backend::Autodiff<NdArray<f32>>;

#[test]
fn test_acoustic_wave_domain_creation() {
    let domain = AcousticWaveDomain::new(
        AcousticProblemType::Linear,
        1500.0, // m/s (typical for soft tissue)
        1000.0, // kg/m³ (water density)
        None,   // No nonlinearity
    );

    assert_eq!(
        <AcousticWaveDomain as PhysicsDomain<B>>::domain_name(&domain),
        "acoustic_wave"
    );
    assert_eq!(domain.wave_speed(), 1500.0);
    assert_eq!(domain.density(), 1000.0);
    assert!(domain.nonlinearity_coefficient().is_none());
    assert!(<AcousticWaveDomain as PhysicsDomain<B>>::supports_coupling(
        &domain
    ));
}

#[test]
fn test_nonlinear_acoustic_domain() {
    let domain = AcousticWaveDomain::new(
        AcousticProblemType::Nonlinear,
        1500.0,
        1000.0,
        Some(3.5), // Typical β for water
    );

    assert_eq!(domain.problem_type, AcousticProblemType::Nonlinear);
    assert_eq!(domain.nonlinearity_coefficient(), Some(3.5));
}

#[test]
fn test_boundary_conditions() {
    let mut domain = AcousticWaveDomain::new(AcousticProblemType::Linear, 1500.0, 1000.0, None);

    domain.add_boundary_condition(AcousticBoundarySpec {
        position: BoundaryPosition::Top,
        condition_type: AcousticBoundaryType::SoundSoft,
        parameters: HashMap::new(),
    });

    let bcs = <AcousticWaveDomain as PhysicsDomain<B>>::boundary_conditions(&domain);
    assert_eq!(bcs.len(), 1);
    assert!(!bcs.is_empty());
}

#[test]
fn test_validation_metrics() {
    let domain = AcousticWaveDomain::new(AcousticProblemType::Linear, 1500.0, 1000.0, None);

    let metrics = <AcousticWaveDomain as PhysicsDomain<B>>::validation_metrics(&domain);
    assert_eq!(metrics.len(), 3);
    assert_eq!(metrics[0].name, "wave_speed_accuracy");
    assert_eq!(metrics[1].name, "energy_conservation");
    assert_eq!(metrics[2].name, "nonlinearity_error");
}

#[test]
fn test_coupling_interfaces() {
    let domain = AcousticWaveDomain::new(AcousticProblemType::Linear, 1500.0, 1000.0, None);

    let interfaces = <AcousticWaveDomain as PhysicsDomain<B>>::coupling_interfaces(&domain);
    assert_eq!(interfaces.len(), 2);
    assert_eq!(interfaces[0].name, "acoustic_solid");
    assert_eq!(interfaces[1].name, "acoustic_thermal");
}
