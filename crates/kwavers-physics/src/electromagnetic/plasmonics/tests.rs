use super::*;
use num_complex::Complex;

#[test]
fn test_mie_theory_gold() {
    let mie = MieTheory::gold_in_water(15e-9); // 15 nm radius

    let eps_5209 = mie_theory::gold_dielectric_johnson_christy(520.9e-9);
    let expected_5209 = Complex::new(0.62 * 0.62 - 2.081 * 2.081, 2.0 * 0.62 * 2.081);
    assert!(
        (eps_5209 - expected_5209).norm() < 1e-12,
        "Johnson-Christy endpoint conversion must satisfy eps=(n+ik)^2; actual={eps_5209:?}, expected={expected_5209:?}"
    );

    let eps_midpoint = mie_theory::gold_dielectric_johnson_christy(534.75e-9);
    let n_midpoint = 0.5 * (0.62 + 0.43);
    let k_midpoint = 0.5 * (2.081 + 2.455);
    let expected_midpoint = Complex::new(
        n_midpoint * n_midpoint - k_midpoint * k_midpoint,
        2.0 * n_midpoint * k_midpoint,
    );
    assert!(
        (eps_midpoint - expected_midpoint).norm() < 1e-12,
        "Johnson-Christy affine interpolation mismatch: actual={eps_midpoint:?}, expected={expected_midpoint:?}"
    );

    // Test polarizability calculation at theoretical SPR wavelength
    let alpha = mie.polarizability(530e-9);
    assert!(alpha.re > 0.0);

    // Test fundamental cross-section conservation
    let sigma_scat = mie.scattering_cross_section(530e-9);
    let sigma_abs = mie.absorption_cross_section(530e-9);
    let sigma_ext = mie.extinction_cross_section(530e-9);

    assert!(sigma_scat > 0.0);
    assert!(sigma_abs > 0.0);
    assert_eq!(sigma_ext, sigma_scat + sigma_abs);
}

#[test]
fn test_plasmonic_enhancement() {
    let mie = MieTheory::gold_in_water(15e-9);
    let enhancement = PlasmonicEnhancementCalculator::new(mie, 1e20);

    // Verify near-field enhancement at particle boundary
    let surface_point = [15e-9, 0.0, 0.0];
    let factor = enhancement.field_enhancement_factor(530e-9, &surface_point);
    assert!(factor > 1.0);

    let spr_enhancement = enhancement.surface_plasmon_enhancement(530e-9);
    assert!(spr_enhancement > 1.0);
}

#[test]
fn test_nanoparticle_array() {
    let array = NanoparticleArray::linear_array(15e-9, 50e-9, 3);

    // Verify coherent coupling enhancement midway between adjacent elements
    let midpoint = [25e-9, 0.0, 0.0];
    let enhancement = array.collective_enhancement(530e-9, &midpoint);
    assert!(enhancement >= 1.0);

    // Verify hot spot extraction
    let hot_spots = array.hot_spots(530e-9);
    assert!(!hot_spots.is_empty());
    assert!(hot_spots[0].0 >= 1.0);
}

#[test]
fn test_maxwell_garnett_endpoint_and_closed_form() {
    let eps_particle = Complex::new(9.0, 0.6);
    let eps_host = Complex::new(2.25, 0.0);

    let at_zero = enhancement::maxwell_garnett_effective_dielectric(eps_particle, eps_host, 0.0);
    assert!(
        (at_zero - eps_host).norm() < 1e-14,
        "Maxwell-Garnett must return host at f=0; got {at_zero:?}"
    );

    let f = 0.12;
    let contrast = eps_particle - eps_host;
    let expected = eps_host * (eps_particle + 2.0 * eps_host + 2.0 * f * contrast)
        / (eps_particle + 2.0 * eps_host - f * contrast);
    let actual = enhancement::maxwell_garnett_effective_dielectric(eps_particle, eps_host, f);

    assert!(
        (actual - expected).norm() < 1e-14,
        "Maxwell-Garnett closed form mismatch: actual={actual:?}, expected={expected:?}"
    );
}

#[test]
fn test_bruggeman_endpoint_and_residual() {
    let eps_particle = Complex::new(9.0, 0.6);
    let eps_host = Complex::new(2.25, 0.0);

    let at_zero = enhancement::bruggeman_effective_dielectric(eps_particle, eps_host, 0.0);
    let at_one = enhancement::bruggeman_effective_dielectric(eps_particle, eps_host, 1.0);

    assert!(
        (at_zero - eps_host).norm() < 1e-14,
        "Bruggeman must return host at f=0; got {at_zero:?}"
    );
    assert!(
        (at_one - eps_particle).norm() < 1e-14,
        "Bruggeman must return particle at f=1; got {at_one:?}"
    );

    let f = 0.37;
    let eps_eff = enhancement::bruggeman_effective_dielectric(eps_particle, eps_host, f);
    let residual = f * (eps_particle - eps_eff) / (eps_particle + 2.0 * eps_eff)
        + (1.0 - f) * (eps_host - eps_eff) / (eps_host + 2.0 * eps_eff);

    assert!(
        residual.norm() < 1e-13,
        "Bruggeman effective dielectric must satisfy implicit mixture equation; residual={residual:?}"
    );
}
