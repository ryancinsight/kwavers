use super::*;
use crate::core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

#[test]
fn test_shock_formation_distance() {
    let params = NonlinearParameters::water();
    let p0 = MPA_TO_PA; // 1 MPa
    let f = MHZ_TO_HZ; // 1 MHz

    let l_s = shock::shock_formation_distance(p0, f, &params);

    // Theory check: l_s = rho0 * c0^3 / (beta * omega * P0)
    // Water: beta ~ 3.5, rho0 ~ 1000, c0 ~ 1500
    // l_s ~ (1000 * 3.375e9) / (3.5 * 2pi*1e6 * 1e6)
    // ~ 3.375e12 / (2.2e13) ~ 0.15 m
    assert!(
        l_s > 0.05 && l_s < 0.3,
        "Shock distance out of expected theoretical bounds: {}",
        l_s
    );
}

#[test]
fn test_second_harmonic_generation() {
    let params = NonlinearParameters::soft_tissue();
    let p0 = 2.0 * MPA_TO_PA;
    let f = 2.0 * MHZ_TO_HZ;
    let z = 0.05; // 5 cm

    let p2 = harmonics::second_harmonic_amplitude(p0, f, z, &params);

    // P2 should be significantly less than P0 initially
    assert!(p2 < p0);
    assert!(
        p2 > p0 * 0.01,
        "Second harmonic should be measurable at 5cm"
    );
}

#[test]
fn test_acoustic_saturation() {
    let params = NonlinearParameters::water();
    let f = MHZ_TO_HZ;
    let z = 0.1;

    let p_sat = saturation::acoustic_saturation_pressure(f, z, &params);

    // Should be a finite positive value
    assert!(p_sat > 0.0);
    assert!(p_sat.is_finite());
}

#[test]
fn test_burgers_equation() {
    let params = NonlinearParameters::soft_tissue();
    let p0 = MPA_TO_PA;
    let f = MHZ_TO_HZ;
    let z = 0.01; // Pre-shock distance

    let p_z = burgers::burgers_equation(p0, f, z, &params);
    assert!(p_z > 0.0 && p_z <= p0);
}

/// Tissue-specific nonlinear-parameter factories must route their density,
/// sound speed, and B/A through the SSOT constants and derive
/// `β = 1 + B/(2A)` exactly.
///
/// ## Theorem
/// For each tissue factory T ∈ {liver, kidney, brain, fat}, the returned
/// `NonlinearParameters` must satisfy:
/// (i) `T.density = DENSITY_T` (no hardcoded numeric literals),
/// (ii) `T.sound_speed = SOUND_SPEED_T`,
/// (iii) `T.b_over_a = B_OVER_A_T`,
/// (iv) `T.beta = 1 + B_OVER_A_T / 2` (Hamilton & Blackstock 1998 Eq. 3.7),
/// (v) attenuation exponent y is positive and finite,
/// (vi) attenuation at 1 MHz equals α₀·1^y = α₀ (sanity check on the
///      `attenuation_at_frequency` mapping).
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn tissue_nonlinear_parameter_factories_route_through_ssot() {
    use crate::core::constants::fundamental::{
        B_OVER_A_BRAIN, B_OVER_A_FAT, B_OVER_A_KIDNEY, B_OVER_A_LIVER, DENSITY_BRAIN, DENSITY_FAT,
        DENSITY_KIDNEY, DENSITY_LIVER, SOUND_SPEED_BRAIN, SOUND_SPEED_FAT, SOUND_SPEED_KIDNEY,
        SOUND_SPEED_LIVER,
    };

    let cases: [(NonlinearParameters, f64, f64, f64, &str); 4] = [
        (
            NonlinearParameters::liver(),
            DENSITY_LIVER,
            SOUND_SPEED_LIVER,
            B_OVER_A_LIVER,
            "liver",
        ),
        (
            NonlinearParameters::kidney(),
            DENSITY_KIDNEY,
            SOUND_SPEED_KIDNEY,
            B_OVER_A_KIDNEY,
            "kidney",
        ),
        (
            NonlinearParameters::brain(),
            DENSITY_BRAIN,
            SOUND_SPEED_BRAIN,
            B_OVER_A_BRAIN,
            "brain",
        ),
        (
            NonlinearParameters::fat(),
            DENSITY_FAT,
            SOUND_SPEED_FAT,
            B_OVER_A_FAT,
            "fat",
        ),
    ];

    for (params, expected_rho, expected_c, expected_b_over_a, label) in cases {
        assert_eq!(
            params.density, expected_rho,
            "{label}: density must route through SSOT (expected {expected_rho}, got {})",
            params.density
        );
        assert_eq!(
            params.sound_speed, expected_c,
            "{label}: sound speed must route through SSOT (expected {expected_c}, got {})",
            params.sound_speed
        );
        assert_eq!(
            params.b_over_a, expected_b_over_a,
            "{label}: B/A must route through SSOT (expected {expected_b_over_a}, got {})",
            params.b_over_a
        );
        let expected_beta = 1.0 + expected_b_over_a / 2.0;
        assert!(
            (params.beta - expected_beta).abs() < 1e-15,
            "{label}: β must equal 1 + B/(2A) exactly (expected {expected_beta}, got {})",
            params.beta
        );
        assert!(
            params.attenuation_exponent > 0.0 && params.attenuation_exponent.is_finite(),
            "{label}: attenuation exponent must be positive finite, got {}",
            params.attenuation_exponent
        );
        assert!(
            params.attenuation_coeff > 0.0 && params.attenuation_coeff.is_finite(),
            "{label}: attenuation coefficient must be positive finite, got {}",
            params.attenuation_coeff
        );
        // At 1 MHz the power-law collapses to α(1 MHz) = α₀·1^y = α₀ (Np/m).
        let alpha_1mhz = params.attenuation_at_frequency(MHZ_TO_HZ);
        assert!(
            (alpha_1mhz - params.attenuation_coeff).abs() < 1e-12 * params.attenuation_coeff,
            "{label}: α(1 MHz) must equal stored α₀; got {alpha_1mhz} vs {}",
            params.attenuation_coeff
        );
    }
}

/// Tissue-specific factories must produce literature-consistent β ordering:
/// fat is the most nonlinear soft tissue, kidney > liver ≳ brain among the
/// three solid abdominal/cranial targets, and water has the lowest β of all
/// (B/A = 5.0 → β = 3.5). Spot-checks the literature consensus and acts as
/// a regression guard against accidental SSOT swaps.
/// # Panics
/// - Panics on any assertion failure.
#[test]
fn tissue_beta_ordering_matches_published_consensus() {
    let water = NonlinearParameters::water().beta;
    let brain = NonlinearParameters::brain().beta;
    let liver = NonlinearParameters::liver().beta;
    let kidney = NonlinearParameters::kidney().beta;
    let fat = NonlinearParameters::fat().beta;

    assert!(
        water < brain,
        "β: water {water} should be less than brain {brain}"
    );
    assert!(
        brain < liver,
        "β: brain {brain} should be ≤ liver {liver} (Duck 1990 ordering)"
    );
    assert!(
        liver < kidney,
        "β: liver {liver} should be less than kidney {kidney}"
    );
    assert!(
        kidney < fat,
        "β: fat {fat} should be the largest β among soft tissues"
    );
    // Sanity: β stays in a plausible biological range [3, 7].
    for (label, value) in [
        ("water", water),
        ("brain", brain),
        ("liver", liver),
        ("kidney", kidney),
        ("fat", fat),
    ] {
        assert!(
            (3.0..=7.0).contains(&value),
            "{label}: β = {value} outside plausible biological range [3, 7]"
        );
    }
}

#[test]
fn test_parametric_array() {
    let params = NonlinearParameters::water();

    // Two high-frequency primaries
    let p1 = MPA_TO_PA;       // 1 MPa
    let p2 = MPA_TO_PA;       // 1 MPa
    let f1 = 2.0 * MHZ_TO_HZ;
    let f2 = 2.1 * MHZ_TO_HZ;
    let z = 1.0; // 1 meter far-field

    let p_diff = parametric::difference_frequency_amplitude(p1, p2, f1, f2, z, &params);

    // Difference frequency is 100 kHz.
    // Amplitude should be physical and non-zero.
    assert!(p_diff > 0.0);
    assert!(
        p_diff < p1 * 0.1,
        "Difference frequency amplitude is bounded by Demodulation/Westervelt limits"
    );
}
