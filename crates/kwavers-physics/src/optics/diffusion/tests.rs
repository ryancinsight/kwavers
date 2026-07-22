use super::*;
use aequitas::systems::si::units::PerMeter;
use kwavers_core::constants::optical::REFRACTIVE_INDEX_SOFT_TISSUE;
use kwavers_medium::properties::OpticalPropertyData;

#[test]
fn material_aggregate_maps_to_hyperion_coefficients() {
    let domain_props = OpticalPropertyData::new(
        10.0,                         // absorption_coefficient
        100.0,                        // scattering_coefficient
        0.9,                          // anisotropy
        REFRACTIVE_INDEX_SOFT_TISSUE, // refractive_index — SSOT: optical::REFRACTIVE_INDEX_SOFT_TISSUE
    )
    .unwrap();

    let coefficients = domain_props.diffusion_coefficients().unwrap();

    assert_eq!(coefficients.absorption().in_unit::<PerMeter>(), 10.0);
    assert!(
        (coefficients.reduced_scattering().in_unit::<PerMeter>() - 10.0).abs() < 1e-10,
        "Expected: 10.0, Got: {}",
        coefficients.reduced_scattering().in_unit::<PerMeter>()
    );
    assert_eq!(
        domain_props.refractive_index(),
        REFRACTIVE_INDEX_SOFT_TISSUE
    );
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
    use crate::acoustics::traits::LightDiffusionModelTrait;
    use kwavers_core::constants::fundamental::SPEED_OF_LIGHT;
    use kwavers_field::indices::LIGHT_IDX;
    use leto::Array4;

    let grid = kwavers_grid::Grid::new(8, 8, 8, 1.0e-3, 1.0e-3, 1.0e-3).unwrap();
    let props = OpticalPropertyData::new(10.0, 1.0e4, 0.9, REFRACTIVE_INDEX_SOFT_TISSUE).unwrap();
    let mut solver = LightDiffusion::new(&grid, props, false, false).unwrap();

    let phi0 = 1.0_f64;
    let mut fields: Array4<f64> = Array4::from_elem((LIGHT_IDX + 1, 8, 8, 8), phi0);
    let source: leto::Array3<f64> = leto::Array3::zeros([8, 8, 8]);
    let medium = kwavers_medium::homogeneous::HomogeneousMedium::water(&grid);
    let dt = 1.0e-12_f64; // 1 ps; chosen so dt·c·μₐ ≪ 1 (≈ 2.14e-3).

    solver.update_light(&mut fields, &source, &grid, &medium, dt);

    let c_medium = SPEED_OF_LIGHT / props.refractive_index();
    let expected = phi0 * (1.0 - dt * c_medium * props.absorption_coefficient());
    let observed = fields[[LIGHT_IDX, 4, 4, 4]];
    let rel_err = (observed - expected).abs() / expected;
    assert!(
        rel_err < 1.0e-12,
        "uniform-fluence decay step expected {expected:.12}, got {observed:.12} (rel err {rel_err:.2e})"
    );
}

#[test]
fn test_light_diffusion_solver_initialization() {
    let grid = kwavers_grid::Grid::new(10, 10, 10, 0.001, 0.001, 0.001).unwrap();

    let props = OpticalPropertyData::soft_tissue();

    let solver = LightDiffusion::new(&grid, props, false, false).unwrap();

    assert_eq!(solver.fluence_rate.shape(), [1, 10, 10, 10]);
    assert_eq!(solver.emission_spectrum.shape(), [10, 10, 10]);
}
