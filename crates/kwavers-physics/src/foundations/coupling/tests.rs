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
        spatial_coefficient: 1.0,
        temporal_coefficient: 1e6,
        energy_efficiency: 0.9,
    };

    assert_eq!(strength.spatial_coefficient, 1.0);
    assert_eq!(strength.temporal_coefficient, 1e6);
    assert_eq!(strength.energy_efficiency, 0.9);
}

#[test]
fn test_domain_decomposition() {
    let decomposition = DomainDecomposition {
        subdomain_bounds: vec![vec![0.0, 1.0], vec![1.0, 2.0]],
        interface_conditions: vec![CouplingInterfaceCondition::Dirichlet {
            field_name: "velocity".to_string(),
        }],
        overlap_thickness: 0.1,
        transmission_conditions: vec![DomainDecompTransmissionCondition::Dirichlet {
            boundary_value: 0.0,
        }],
    };

    assert_eq!(decomposition.subdomain_bounds.len(), 2);
    assert_eq!(decomposition.interface_conditions.len(), 1);
    assert_eq!(decomposition.overlap_thickness, 0.1);
}
