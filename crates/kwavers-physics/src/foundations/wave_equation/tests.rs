use super::domain::{Domain, SpatialDimension, WaveEquationBoundary};

#[test]
fn test_domain_1d() {
    let domain = Domain::new_1d(0.0, 1.0, 101, WaveEquationBoundary::Periodic);
    assert_eq!(domain.dimension, SpatialDimension::One);
    assert_eq!(domain.resolution.len(), 1);
    assert_eq!(domain.resolution[0], 101);
    let spacing = domain.spacing();
    assert!((spacing[0] - 0.01).abs() < 1e-10);
}

#[test]
fn test_domain_2d() {
    let domain = Domain::new_2d(
        -1.0,
        1.0,
        -2.0,
        2.0,
        51,
        101,
        WaveEquationBoundary::Absorbing { damping: 0.1 },
    );
    assert_eq!(domain.dimension, SpatialDimension::Two);
    assert_eq!(domain.resolution.len(), 2);
    let spacing = domain.spacing();
    assert!((spacing[0] - 0.04).abs() < 1e-10);
    assert!((spacing[1] - 0.04).abs() < 1e-10);
}

#[test]
fn test_boundary_condition_types() {
    let bc1 = WaveEquationBoundary::Dirichlet { value: 1.0 };
    let bc2 = WaveEquationBoundary::Neumann { flux: 0.5 };
    let bc3 = WaveEquationBoundary::Periodic;
    assert!(matches!(bc1, WaveEquationBoundary::Dirichlet { .. }));
    assert!(matches!(bc2, WaveEquationBoundary::Neumann { .. }));
    assert_eq!(bc3, WaveEquationBoundary::Periodic);
}
