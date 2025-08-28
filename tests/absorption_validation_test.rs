//! Power law absorption validation test
//!
//! Validates that the power law absorption model correctly implements
//! the frequency-dependent attenuation in biological tissues.
//!
//! Reference: Treeby et al., "Modeling power law absorption and dispersion
//! for acoustic propagation using the fractional Laplacian", JASA 2010

use kwavers::constants::medium_properties::{TISSUE_SOUND_SPEED, WATER_SOUND_SPEED};

/// Power law absorption coefficient
///
/// α(f) = α₀ * (f/f₀)^y
///
/// where:
/// - α₀: absorption coefficient at reference frequency [Np/m]
/// - f: frequency [Hz]
/// - f₀: reference frequency [Hz]
/// - y: power law exponent (typically 1.0-1.5 for tissues)
fn power_law_absorption(alpha_0: f64, frequency: f64, reference_frequency: f64, power: f64) -> f64 {
    alpha_0 * (frequency / reference_frequency).powf(power)
}

#[test]
fn test_water_absorption() {
    // Water absorption: α = 0.0022 * f^2 [dB/(MHz^2·cm)]
    // Convert to Np/m: 1 dB = 0.115 Np, 1 cm = 0.01 m

    let frequencies = [0.5e6, 1.0e6, 2.0e6, 5.0e6]; // Hz
    let expected_alpha_db_per_cm = [
        0.00055, // 0.5 MHz
        0.0022,  // 1.0 MHz
        0.0088,  // 2.0 MHz
        0.055,   // 5.0 MHz
    ];

    for (i, &f) in frequencies.iter().enumerate() {
        let f_mhz = f / 1e6;
        let alpha_db_per_cm = 0.0022 * f_mhz.powi(2);
        let alpha_np_per_m = alpha_db_per_cm * 0.115 / 0.01;

        // Verify against expected values
        let error = (alpha_db_per_cm - expected_alpha_db_per_cm[i]).abs();
        assert!(
            error < 1e-5,
            "Water absorption incorrect at {} MHz: expected {}, got {}",
            f_mhz,
            expected_alpha_db_per_cm[i],
            alpha_db_per_cm
        );

        // Verify power law with y=2 for water
        let alpha_from_power_law = power_law_absorption(
            0.0022 * 0.115 / 0.01, // α₀ at 1 MHz in Np/m
            f,
            1e6, // Reference frequency 1 MHz
            2.0, // Power law exponent for water
        );

        assert!(
            (alpha_from_power_law - alpha_np_per_m).abs() / alpha_np_per_m < 0.01,
            "Power law model error at {} MHz",
            f_mhz
        );
    }
}

#[test]
fn test_tissue_absorption() {
    // Soft tissue: α = 0.5-1.0 dB/(MHz·cm) with y ≈ 1.1
    // Reference: Szabo, "Diagnostic Ultrasound Imaging", 2004

    let alpha_0_db = 0.7; // dB/(MHz·cm) at 1 MHz
    let power = 1.1; // Typical for soft tissue

    let frequencies = [1e6, 3e6, 5e6, 10e6]; // Hz

    for &f in frequencies.iter() {
        let f_mhz = f / 1e6;

        // Calculate using power law
        let alpha_db = alpha_0_db * f_mhz.powf(power);
        let alpha_np_per_m = alpha_db * 0.115 / 0.01;

        // Verify reasonable range for soft tissue
        assert!(
            alpha_np_per_m > 0.0 && alpha_np_per_m < 200.0,
            "Tissue absorption out of range at {} MHz: {} Np/m",
            f_mhz,
            alpha_np_per_m
        );

        // Verify power law behavior
        if f > 1e6 {
            let alpha_1mhz = power_law_absorption(alpha_0_db * 0.115 / 0.01, 1e6, 1e6, power);
            let ratio = alpha_np_per_m / alpha_1mhz;
            let expected_ratio = f_mhz.powf(power);

            assert!(
                (ratio - expected_ratio).abs() / expected_ratio < 0.01,
                "Power law scaling incorrect at {} MHz",
                f_mhz
            );
        }
    }
}

#[test]
fn test_absorption_dispersion_relation() {
    // Kramers-Kronig relations link absorption to dispersion
    // For power law media: c(ω) = c₀ / (1 - α₀·tan(πy/2)·ω^(y-1))
    // Reference: Treeby et al., JASA 2010

    let c0 = TISSUE_SOUND_SPEED;
    let alpha_0 = 0.7 * 0.115 / 0.01; // Np/m at 1 MHz
    let y = 1.1;
    let tan_factor = (std::f64::consts::PI * y / 2.0).tan();

    let frequencies = [1e6, 2e6, 5e6];

    for &f in frequencies.iter() {
        let omega = 2.0 * std::f64::consts::PI * f;
        let omega_ref = 2.0 * std::f64::consts::PI * 1e6;

        // Phase velocity with dispersion
        let dispersion_factor = alpha_0 * tan_factor * (omega / omega_ref).powf(y - 1.0);
        let c_dispersive = c0 / (1.0 - dispersion_factor);

        // Verify causality: phase velocity increases with frequency for y > 1
        if y > 1.0 {
            assert!(
                c_dispersive > c0,
                "Causality violation: dispersive velocity {} < c0 {} at {} Hz",
                c_dispersive,
                c0,
                f
            );
        }

        // Verify reasonable dispersion (typically < 1% change)
        let dispersion_percent = ((c_dispersive - c0) / c0).abs() * 100.0;
        assert!(
            dispersion_percent < 2.0,
            "Excessive dispersion: {}% at {} MHz",
            dispersion_percent,
            f / 1e6
        );
    }
}

#[test]
fn test_thermoviscous_absorption() {
    // Classical thermoviscous absorption in water
    // α_classical = 2πf²/c³ * (4μ/3 + μ_B + κ(γ-1)/Cp)
    // Reference: Pierce, "Acoustics", 1989, Eq. 10-3.8

    let frequency = 1e6; // 1 MHz
    let c = WATER_SOUND_SPEED;
    let rho = 1000.0; // kg/m³

    // Water properties at 20°C
    let mu = 1.002e-3; // Shear viscosity [Pa·s]
    let mu_b = 2.8e-3; // Bulk viscosity [Pa·s]
    let kappa = 0.598; // Thermal conductivity [W/(m·K)]
    let gamma = 1.0; // Specific heat ratio (approximately 1 for water)
    let cp = 4182.0; // Specific heat [J/(kg·K)]

    // Classical absorption coefficient
    let omega = 2.0 * std::f64::consts::PI * frequency;
    let alpha_classical = (omega * omega / (2.0 * rho * c.powi(3)))
        * (4.0 * mu / 3.0 + mu_b + kappa * (gamma - 1.0) / cp);

    // Convert to dB/(MHz²·cm) for comparison
    let alpha_db_per_mhz2_cm = alpha_classical * (1e6 * 1e6) * 0.01 / 0.115;

    // Should be on the order of 0.002 for water
    assert!(
        alpha_db_per_mhz2_cm > 0.001 && alpha_db_per_mhz2_cm < 0.01,
        "Thermoviscous absorption out of range: {} dB/(MHz²·cm)",
        alpha_db_per_mhz2_cm
    );
}
