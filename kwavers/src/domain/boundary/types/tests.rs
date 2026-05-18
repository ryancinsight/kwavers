//! Tests for canonical boundary condition types.

use super::*;

#[test]
fn test_boundary_type_creation() {
    let dirichlet = BoundaryType::Dirichlet;
    assert_eq!(dirichlet, BoundaryType::Dirichlet);

    let robin = BoundaryType::Robin {
        alpha: 1.0,
        beta: 2.0,
    };
    match robin {
        BoundaryType::Robin { alpha, beta } => {
            assert_eq!(alpha, 1.0);
            assert_eq!(beta, 2.0);
        }
        _ => panic!("Expected Robin boundary"),
    }
}

#[test]
fn test_boundary_face() {
    let face = BoundaryFace::XMin;
    assert_eq!(face, BoundaryFace::XMin);
}

#[test]
fn test_acoustic_to_general_conversion() {
    let sound_soft = AcousticBoundaryType::SoundSoft;
    let general: BoundaryType = sound_soft.into();
    assert_eq!(general, BoundaryType::Dirichlet);

    let impedance = AcousticBoundaryType::Impedance { impedance: 1500.0 };
    let general: BoundaryType = impedance.into();
    match general {
        BoundaryType::Robin { alpha, beta } => {
            assert_eq!(alpha, 1.0);
            assert_eq!(beta, 1500.0);
        }
        _ => panic!("Expected Robin boundary"),
    }
}

#[test]
fn test_electromagnetic_conversion() {
    let pec = ElectromagneticBoundaryType::PerfectElectricConductor;
    let general: BoundaryType = pec.into();
    assert_eq!(general, BoundaryType::Dirichlet);

    let pmc = ElectromagneticBoundaryType::PerfectMagneticConductor;
    let general: BoundaryType = pmc.into();
    assert_eq!(general, BoundaryType::Neumann);
}

#[test]
fn test_elastic_conversion() {
    let clamped = ElasticBoundaryType::Clamped;
    let general: BoundaryType = clamped.into();
    assert_eq!(general, BoundaryType::Dirichlet);

    let free = ElasticBoundaryType::Free;
    let general: BoundaryType = free.into();
    assert_eq!(general, BoundaryType::FreeSurface);
}

#[test]
fn test_boundary_spec() {
    let spec = BoundarySpec::new(
        BoundaryFace::XMin,
        BoundaryType::Dirichlet,
        FaceBoundaryComponent::All,
    );
    assert_eq!(spec.face, BoundaryFace::XMin);
    assert_eq!(spec.boundary_type, BoundaryType::Dirichlet);
    assert_eq!(spec.component, FaceBoundaryComponent::All);
    assert!(!spec.time_dependent);
}

#[test]
fn test_time_dependent_spec() {
    let spec = BoundarySpec::time_dependent(
        BoundaryFace::YMax,
        BoundaryType::Neumann,
        FaceBoundaryComponent::Normal,
    );
    assert!(spec.time_dependent);
}
