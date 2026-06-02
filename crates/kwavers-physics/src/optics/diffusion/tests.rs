use super::*;
use kwavers_core::constants::optical::REFRACTIVE_INDEX_SOFT_TISSUE;
use kwavers_domain::medium::properties::OpticalPropertyData;

#[test]
fn test_optical_properties_from_domain() {
    let domain_props = OpticalPropertyData::new(
        10.0,                         // absorption_coefficient
        100.0,                        // scattering_coefficient
        0.9,                          // anisotropy
        REFRACTIVE_INDEX_SOFT_TISSUE, // refractive_index — SSOT: optical::REFRACTIVE_INDEX_SOFT_TISSUE
    )
    .unwrap();

    let props = DiffusionOpticalProperties::from_domain(domain_props);

    assert_eq!(props.absorption_coefficient, 10.0);
    // μₛ' = μₛ(1-g) = 100 * (1 - 0.9) = 10
    assert!(
        (props.reduced_scattering_coefficient - 10.0).abs() < 1e-10,
        "Expected: 10.0, Got: {}",
        props.reduced_scattering_coefficient
    );
    assert_eq!(props.refractive_index, REFRACTIVE_INDEX_SOFT_TISSUE);
}

#[test]
fn test_diffusion_coefficient() {
    let props = DiffusionOpticalProperties {
        absorption_coefficient: 10.0,
        reduced_scattering_coefficient: 90.0,
        refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
    };

    // D = 1 / (3 * (μₐ + μₛ'))
    let expected_d = 1.0 / (3.0 * (10.0 + 90.0));
    assert_eq!(props.diffusion_coefficient(), expected_d);
}

#[test]
fn test_diffusion_approximation_validity() {
    let valid_props = DiffusionOpticalProperties {
        absorption_coefficient: 1.0,
        reduced_scattering_coefficient: 15.0, // > 10x absorption
        refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
    };
    assert!(valid_props.diffusion_approximation_valid());

    let invalid_props = DiffusionOpticalProperties {
        absorption_coefficient: 10.0,
        reduced_scattering_coefficient: 20.0, // < 10x absorption
        refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
    };
    assert!(!invalid_props.diffusion_approximation_valid());
}

#[test]
fn uniform_fluence_decays_at_rate_c_mu_a() {
    // With uniform initial fluence the Laplacian is zero in the interior
    // and the source is zero, so the time-dependent photon diffusion
    // equation collapses to ∂φ/∂t = -c·μₐ·φ, with single-step
    // forward-Euler update φ_new = φ_old·(1 - dt·c·μₐ).
    //
    // The factor of c (≈ 2.14×10⁸ m/s in tissue) is required for
    // dimensional consistency: without it the predicted decay would be
    // ~9 orders of magnitude too slow.
    use kwavers_core::constants::fundamental::SPEED_OF_LIGHT;
    use kwavers_domain::field::indices::LIGHT_IDX;
    use crate::acoustics::traits::LightDiffusionModelTrait;
    use ndarray::Array4;

    let grid = kwavers_domain::grid::Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let props = DiffusionOpticalProperties {
        absorption_coefficient: 10.0,          // m⁻¹
        reduced_scattering_coefficient: 1.0e3, // m⁻¹, μₛ' ≫ μₐ
        refractive_index: REFRACTIVE_INDEX_SOFT_TISSUE,
    };
    let mut solver = LightDiffusion::new(&grid, props, false, false);

    let phi0 = 1.0_f64;
    let mut fields: Array4<f64> = Array4::from_elem((LIGHT_IDX + 1, 8, 8, 8), phi0);
    let source: ndarray::Array3<f64> = ndarray::Array3::zeros((8, 8, 8));
    let medium = kwavers_domain::medium::homogeneous::HomogeneousMedium::water(&grid);
    let dt = 1.0e-12_f64; // 1 ps; chosen so dt·c·μₐ ≪ 1 (≈ 2.14e-3).

    solver.update_light(&mut fields, &source, &grid, &medium, dt);

    let c_medium = SPEED_OF_LIGHT / props.refractive_index;
    let expected = phi0 * (1.0 - dt * c_medium * props.absorption_coefficient);
    let observed = fields[[LIGHT_IDX, 4, 4, 4]];
    let rel_err = (observed - expected).abs() / expected;
    assert!(
        rel_err < 1.0e-12,
        "uniform-fluence decay step expected {expected:.12}, got {observed:.12} (rel err {rel_err:.2e})"
    );
}

#[test]
fn test_light_diffusion_solver_initialization() {
    let grid = kwavers_domain::grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();

    let props = DiffusionOpticalProperties::biological_tissue();

    let solver = LightDiffusion::new(&grid, props, false, false);

    assert_eq!(solver.fluence_rate.dim(), (1, 10, 10, 10));
    assert_eq!(solver.emission_spectrum.dim(), (10, 10, 10));
}
