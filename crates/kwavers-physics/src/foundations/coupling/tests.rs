//! Tests for multi-physics coupling types

use super::domain_decomposition::{DomainDecompTransmissionCondition, DomainDecomposition};
use super::*;

#[test]
fn test_interface_conditions() {
    let dirichlet = CouplingInterfaceCondition::Dirichlet {
        field_name: "pressure".to_string(),
    };
    let neumann = CouplingInterfaceCondition::Neumann {
        flux_name: "stress".to_string(),
    };

    match dirichlet {
        CouplingInterfaceCondition::Dirichlet { field_name } => assert_eq!(field_name, "pressure"),
        _ => panic!("Wrong condition type"),
    }

    match neumann {
        CouplingInterfaceCondition::Neumann { flux_name } => assert_eq!(flux_name, "stress"),
        _ => panic!("Wrong condition type"),
    }
}

#[test]
fn test_coupling_strength() {
    let strength = CouplingStrength {
        spatial_weight: Dimensionless::from_base(1.0),
        temporal_coefficient: ReciprocalTime::from_base(1e6),
        energy_efficiency: Dimensionless::from_base(0.9),
    };

    assert_eq!(strength.spatial_weight.into_base(), 1.0);
    assert_eq!(strength.temporal_coefficient.into_base(), 1e6);
    assert_eq!(strength.energy_efficiency.into_base(), 0.9);
}

#[test]
fn test_domain_decomposition() {
    let decomposition = DomainDecomposition {
        subdomain_bounds: vec![
            [0.0, 1.0].map(Length::from_base).to_vec(),
            [1.0, 2.0].map(Length::from_base).to_vec(),
        ],
        interface_conditions: vec![CouplingInterfaceCondition::Dirichlet {
            field_name: "velocity".to_string(),
        }],
        overlap_thickness: Length::from_base(0.1),
        transmission_conditions: vec![DomainDecompTransmissionCondition::Dirichlet {
            boundary_value: aequitas::systems::si::quantities::Pressure::from_base(0.0),
        }],
    };

    assert_eq!(decomposition.subdomain_bounds.len(), 2);
    assert_eq!(decomposition.interface_conditions.len(), 1);
    assert_eq!(decomposition.overlap_thickness.into_base(), 0.1);
}

#[test]
fn pressure_coefficients_preserve_orientation_and_finite_scale() {
    use aequitas::systems::si::quantities::AcousticImpedance;
    use tracing_subscriber::fmt::format::FmtSpan;

    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_span_events(FmtSpan::FULL)
            .with_test_writer()
            .finish(),
    );
    let samples = tracing::info_span!("arrange").in_scope(|| {
        [
            [1.0, 3.0],
            [3.0, 1.0],
            [f64::MAX, f64::MAX],
            [f64::MIN_POSITIVE, f64::MIN_POSITIVE],
        ]
        .map(|pair| pair.map(AcousticImpedance::from_base))
    });
    let coefficients = tracing::info_span!("act").in_scope(|| {
        samples.map(|[incident, transmitted]| {
            acoustic_elastic::pressure_coefficients(incident, transmitted)
                .map(Dimensionless::into_base)
        })
    });
    let _assert = tracing::info_span!("assert").entered();
    // Using powers of two for the analytic expectations makes the results
    // exactly representable. Normalization by 3 contributes one rounding.
    for (actual, expected) in
        coefficients
            .into_iter()
            .zip([[0.5, 1.5], [-0.5, 0.5], [0.0, 1.0], [0.0, 1.0]])
    {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert!((actual - expected).abs() <= f64::EPSILON);
        }
    }
}
