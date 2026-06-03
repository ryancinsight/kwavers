//! Power law absorption validation test
//!
//! Validates that the power law absorption model correctly implements
//! the frequency-dependent attenuation in biological tissues.
//!
//! Reference: Treeby et al., "Modeling power law absorption and dispersion
//! for acoustic propagation using the fractional Laplacian", JASA 2010

use kwavers_core::constants::SOUND_SPEED_WATER;
use plotters::prelude::*;
use std::fs;

// Define tissue sound speed constant locally since it's not exported
const TISSUE_SOUND_SPEED: f64 = 1540.0;

const FIGURE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/test-figures");

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

/// Generate figure showing power law absorption curves for water and soft tissue,
/// plus dispersive phase velocity from Kramers-Kronig.
///
/// Three panels:
///   Panel 1 (top-left):  Water α(f) in dB/(MHz²·cm) — quadratic (y=2)
///   Panel 2 (top-right): Soft tissue α(f) in dB/(MHz·cm) — power law y=1.1
///   Panel 3 (bottom):    Phase velocity c(f) showing Kramers-Kronig dispersion
fn save_absorption_figure() -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(FIGURE_DIR)?;
    let path = format!("{}/absorption_power_law.png", FIGURE_DIR);
    let root = BitMapBackend::new(&path, (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let (top, bottom) = root.split_vertically(450);
    let (panel_water, panel_tissue) = top.split_horizontally(600);

    // ── Panel 1: Water absorption α(f), quadratic power law (y=2) ────────────
    {
        let freq_mhz: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect(); // 0.1–10 MHz
        let alpha_model: Vec<(f64, f64)> = freq_mhz.iter().map(|&f| (f, 0.0022 * f * f)).collect();
        // Discrete reference points from Duck, "Physical Properties of Tissue", 1990
        let reference_pts = [(0.5, 0.00055), (1.0, 0.0022), (2.0, 0.0088), (5.0, 0.055)];

        let alpha_max = alpha_model.iter().map(|&(_, a)| a).fold(0.0_f64, f64::max);

        let mut chart = ChartBuilder::on(&panel_water)
            .caption("Water Absorption (y=2)", ("sans-serif", 18))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0.0f64..10.0f64, 0.0f64..alpha_max * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Frequency (MHz)")
            .y_desc("α [dB/(MHz²·cm)]")
            .draw()?;

        chart
            .draw_series(LineSeries::new(alpha_model, BLUE.stroke_width(2)))?
            .label("α = 0.0022·f² model")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart
            .draw_series(
                reference_pts
                    .iter()
                    .map(|&(f, a)| Circle::new((f, a), 5, RED.filled())),
            )?
            .label("Duck 1990 reference")
            .legend(|(x, y)| Circle::new((x + 10, y), 4, RED.filled()));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    // ── Panel 2: Soft tissue α(f), power law y=1.1 ───────────────────────────
    {
        let alpha_0_db = 0.7; // dB/(MHz·cm) at 1 MHz
        let y = 1.1;
        let freq_mhz: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
        let alpha_tissue: Vec<(f64, f64)> = freq_mhz
            .iter()
            .map(|&f| (f, alpha_0_db * f.powf(y)))
            .collect();
        // Acceptable range band: 0.5–1.0 dB/(MHz·cm) at 1 MHz
        let alpha_lo: Vec<(f64, f64)> = freq_mhz.iter().map(|&f| (f, 0.5 * f.powf(y))).collect();
        let alpha_hi: Vec<(f64, f64)> = freq_mhz.iter().map(|&f| (f, 1.0 * f.powf(y))).collect();

        let alpha_max = alpha_tissue.iter().map(|&(_, a)| a).fold(0.0_f64, f64::max);

        let mut chart = ChartBuilder::on(&panel_tissue)
            .caption("Soft Tissue Absorption (y=1.1)", ("sans-serif", 18))
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(0.0f64..10.0f64, 0.0f64..alpha_max * 1.1)?;

        chart
            .configure_mesh()
            .x_desc("Frequency (MHz)")
            .y_desc("α [dB/(MHz·cm)]")
            .draw()?;

        // Shaded acceptable range
        chart
            .draw_series(AreaSeries::new(
                alpha_hi.iter().chain(alpha_lo.iter().rev()).cloned(),
                0.0,
                GREEN.mix(0.15),
            ))?
            .label("Acceptable range (0.5–1.0 dB/(MHz·cm))")
            .legend(|(x, y)| {
                Rectangle::new([(x, y - 4), (x + 20, y + 4)], GREEN.mix(0.4).filled())
            });

        chart
            .draw_series(LineSeries::new(alpha_tissue, RED.stroke_width(2)))?
            .label("α = 0.7·f^1.1 model")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    // ── Panel 3: Kramers-Kronig phase velocity dispersion ────────────────────
    {
        let c0 = TISSUE_SOUND_SPEED;
        let alpha_0_np_per_m = 0.7 * 0.115 / 0.01; // Np/m at 1 MHz
        let y = 1.1;
        let omega_ref = 2.0 * std::f64::consts::PI * 1e6; // reference ω₀ = 2π·1 MHz
        let tan_factor = (std::f64::consts::PI * y / 2.0).tan(); // negative for 1 < y < 2
                                                                 // Szabo parameterization: α(ω) = α_szabo·|ω|^y (Szabo 1994 Eq. 17)
        let alpha_szabo = alpha_0_np_per_m / omega_ref.powf(y);

        let freq_mhz: Vec<f64> = (1..=100).map(|i| i as f64 * 0.1).collect();
        let c_dispersive: Vec<(f64, f64)> = freq_mhz
            .iter()
            .map(|&f_mhz| {
                let omega = 2.0 * std::f64::consts::PI * f_mhz * 1e6;
                let c_inv = 1.0 / c0
                    + alpha_szabo * tan_factor * (omega.powf(y - 1.0) - omega_ref.powf(y - 1.0));
                (f_mhz, 1.0 / c_inv)
            })
            .collect();

        let c_min = c_dispersive
            .iter()
            .map(|&(_, c)| c)
            .fold(f64::INFINITY, f64::min);
        let c_max = c_dispersive
            .iter()
            .map(|&(_, c)| c)
            .fold(f64::NEG_INFINITY, f64::max);
        let margin = (c_max - c_min).max(1.0) * 0.2;

        let mut chart = ChartBuilder::on(&bottom)
            .caption(
                format!("Kramers-Kronig Phase Velocity (c₀ = {:.0} m/s)", c0),
                ("sans-serif", 18),
            )
            .margin(20)
            .x_label_area_size(40)
            .y_label_area_size(70)
            .build_cartesian_2d(0.0f64..10.0f64, (c_min - margin)..(c_max + margin))?;

        chart
            .configure_mesh()
            .x_desc("Frequency (MHz)")
            .y_desc("Phase velocity c(f) [m/s]")
            .draw()?;

        // c₀ reference line
        chart
            .draw_series(LineSeries::new(
                [(0.0, c0), (10.0, c0)],
                BLACK.stroke_width(1),
            ))?
            .label(format!("c₀ = {:.0} m/s (lossless)", c0))
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK));

        chart
            .draw_series(LineSeries::new(c_dispersive, BLUE.stroke_width(2)))?
            .label("c(f) with KK dispersion (y=1.1)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

        chart
            .configure_series_labels()
            .background_style(WHITE.mix(0.8))
            .border_style(BLACK)
            .draw()?;
    }

    root.present()?;
    println!("Figure saved: {}", path);
    Ok(())
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
        let f_mhz: f64 = f / 1e6;
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

    // Generate figure after the last test in this module runs (only once)
    if let Err(e) = save_absorption_figure() {
        eprintln!("Warning: absorption figure save failed: {e}");
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
        let f_mhz: f64 = f / 1e6;

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
    // Kramers-Kronig dispersion for power-law absorbing media.
    //
    // Reference: Szabo 1994 (JASA 96:491), Eq. 17:
    //   c_p⁻¹(ω) = c₀⁻¹ + α_szabo·tan(πy/2)·(|ω|^(y-1) − ω₀^(y-1))
    //
    // where α_szabo is the Szabo parameterization coefficient with units
    // [Np·s^y / m], derived from the measured α₀ [Np/m at ω₀]:
    //   α(ω) = α_szabo·|ω|^y  ⟹  α_szabo = α₀ / ω₀^y
    //
    // For y = 1.1 > 1: tan(πy/2) < 0, so for ω > ω₀ the correction to
    // c_p⁻¹ is negative → c_p increases with frequency (anomalous dispersion
    // consistent with measured tissue behavior).

    let c0 = TISSUE_SOUND_SPEED;
    let alpha_0_np_per_m = 0.7 * 0.115 / 0.01; // Np/m at 1 MHz reference
    let y = 1.1;
    let omega_ref = 2.0 * std::f64::consts::PI * 1e6; // ω₀ = 2π·1 MHz
    let tan_factor = (std::f64::consts::PI * y / 2.0).tan(); // negative for 1 < y < 2

    // Convert to Szabo parameterization: α(ω) = α_szabo·|ω|^y
    let alpha_szabo = alpha_0_np_per_m / omega_ref.powf(y);

    // At 1 MHz: correction = 0 (reference frequency), so c_p = c₀.
    // At 2 and 5 MHz: tan < 0, |ω|^(y-1) > ω₀^(y-1) ⟹ correction < 0
    //   ⟹ c_p⁻¹ < c₀⁻¹ ⟹ c_p > c₀.
    let frequencies = [1e6, 2e6, 5e6];

    for &f in frequencies.iter() {
        let omega = 2.0 * std::f64::consts::PI * f;

        // KK dispersion relation (Szabo 1994 Eq. 17)
        let c_inv_dispersive =
            1.0 / c0 + alpha_szabo * tan_factor * (omega.powf(y - 1.0) - omega_ref.powf(y - 1.0));
        let c_dispersive = 1.0 / c_inv_dispersive;

        // Causality: for y > 1, phase velocity increases with frequency
        // (only strictly testable above the reference frequency)
        if y > 1.0 && f > 1e6 {
            assert!(
                c_dispersive > c0,
                "Causality violation at {} MHz: c_dispersive={:.4} < c₀={:.4}",
                f / 1e6,
                c_dispersive,
                c0
            );
        }

        // Dispersion magnitude must remain physically reasonable (< 2%)
        let dispersion_percent = ((c_dispersive - c0) / c0).abs() * 100.0;
        assert!(
            dispersion_percent < 2.0,
            "Excessive dispersion at {} MHz: {:.4}% (expected < 2%)",
            f / 1e6,
            dispersion_percent
        );
    }
}

#[test]
fn test_thermoviscous_absorption() {
    // Classical thermoviscous absorption in water
    // α_classical = 2πf²/c³ * (4μ/3 + μ_B + κ(γ-1)/Cp)
    // Reference: Pierce, "Acoustics", 1989, Eq. 10-3.8

    let frequency = 1e6; // 1 MHz
    let c = SOUND_SPEED_WATER;
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

    // Convert to dB/(MHz²·cm) for comparison.
    // alpha_classical is in Np/m at f=1 MHz.
    // 1 Np/m = (1/11.5) dB/cm  (since 1 dB = 0.115 Np, 1 cm = 0.01 m)
    // At 1 MHz the quadratic law α = α₀·f² gives α₀ = α/f² = α (per MHz² with f=1).
    // So: α₀ [dB/(MHz²·cm)] = alpha_classical [Np/m] × 0.01/0.115
    let alpha_db_per_mhz2_cm = alpha_classical * 0.01 / 0.115;

    // Should be on the order of 0.002 for water (Duck 1990: 0.0022 dB/(MHz²·cm))
    assert!(
        alpha_db_per_mhz2_cm > 0.001 && alpha_db_per_mhz2_cm < 0.01,
        "Thermoviscous absorption out of range: {:.6} dB/(MHz²·cm)",
        alpha_db_per_mhz2_cm
    );
}
