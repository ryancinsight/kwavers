use super::*;
use crate::domain::medium::properties::OpticalPropertyData;

#[test]
fn test_optical_properties_from_domain() {
    let domain_props = OpticalPropertyData::new(
        10.0, // absorption_coefficient
        100.0, // scattering_coefficient
        0.9, // anisotropy
        1.4 // refractive_index
    ).unwrap();

    let props = OpticalProperties::from_domain(domain_props);

    assert_eq!(props.absorption_coefficient, 10.0);
    // μₛ' = μₛ(1-g) = 100 * (1 - 0.9) = 10
    assert!((props.reduced_scattering_coefficient - 10.0).abs() < 1e-10, "Expected: 10.0, Got: {}", props.reduced_scattering_coefficient);
    assert_eq!(props.refractive_index, 1.4);
}

#[test]
fn test_diffusion_coefficient() {
    let props = OpticalProperties {
        absorption_coefficient: 10.0,
        reduced_scattering_coefficient: 90.0,
        refractive_index: 1.4,
    };

    // D = 1 / (3 * (μₐ + μₛ'))
    let expected_d = 1.0 / (3.0 * (10.0 + 90.0));
    assert_eq!(props.diffusion_coefficient(), expected_d);
}

#[test]
fn test_diffusion_approximation_validity() {
    let valid_props = OpticalProperties {
        absorption_coefficient: 1.0,
        reduced_scattering_coefficient: 15.0, // > 10x absorption
        refractive_index: 1.4,
    };
    assert!(valid_props.diffusion_approximation_valid());

    let invalid_props = OpticalProperties {
        absorption_coefficient: 10.0,
        reduced_scattering_coefficient: 20.0, // < 10x absorption
        refractive_index: 1.4,
    };
    assert!(!invalid_props.diffusion_approximation_valid());
}

#[test]
fn test_light_diffusion_solver_initialization() {
    let grid = crate::domain::grid::Grid::new(
        10, 10, 10, 
        0.001, 0.001, 0.001
    ).unwrap();

    let props = OpticalProperties::biological_tissue();

    let solver = LightDiffusion::new(&grid, props, false, false);

    assert_eq!(solver.fluence_rate.dim(), (1, 10, 10, 10));
    assert_eq!(solver.emission_spectrum.dim(), (10, 10, 10));
}
