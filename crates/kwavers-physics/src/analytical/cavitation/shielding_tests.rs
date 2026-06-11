//! Value-semantic tests for the cavitation-shielding balance.
//!
//! The asserted properties follow from the ODE structure and the audited
//! sub-models: (i) no bubbles ⇒ tissue-only transmission and zero shielding
//! loss; (ii) the shielding law is monotonic in β (more bubbles ⇒ less
//! transmission); (iii) with the drive OFF, β strictly dissolves through the
//! interval (Epstein–Plesset clearance); (iv) a sweep crossing the cloud
//! resonance sees a lower band-average attenuation than a fixed tone on
//! resonance, so the combined pulsed+swept drive delivers the most focal energy.
//! Whether pulsing lowers the *time-averaged* β is deliberately not asserted —
//! it is regime-dependent (the literature reports an optimal PRF), so the tests
//! pin only the regime-independent mechanisms.

use super::*;
use crate::analytical::cavitation::SweepProfile;

fn medium() -> ShieldingMedium {
    // 2 µm residual bubble (Minnaert resonance ≈ 1.6 MHz), low tissue
    // attenuation so the delivered pressure clears the cavitation threshold.
    ShieldingMedium {
        c_liquid: 1540.0,
        rho_liquid: 1050.0,
        mu_liquid: 1.5e-3,
        p0_pa: 1.013e5,
        polytropic: 1.4,
        r0_m: 2.0e-6,
        alpha_tissue_np_m: 2.0,
        path_len_m: 0.04,
        saturation_fraction: 0.9,
    }
}

fn production() -> CavitationProduction {
    // Gentle build-up (sub-second to saturation) so a single millisecond pulse
    // deposits only a fraction of β_max — the regime where pulsing controls the
    // accumulated cloud rather than the cloud saturating within one burst.
    CavitationProduction {
        k_prod_per_s: 6.0,
        beta_max: 1.0e-2,
        p_threshold_pa: 1.0e6,
        p_ref_pa: 1.0e6,
        supralinearity: 3.0,
    }
}

/// Drive amplitude giving a moderate above-threshold excess at the focus.
const DRIVE_PA: f64 = 2.0e6;

fn sweep() -> FrequencySweep {
    // Wide band centred on the ~1.6 MHz cloud resonance, swept in 0.5 ms.
    FrequencySweep::new(1.2e6, 2.0e6, 0.5e-3, SweepProfile::Triangular).unwrap()
}

#[test]
fn pulse_protocol_gate_and_duty() {
    let cw = PulseProtocol::continuous();
    assert!(cw.is_on(0.0));
    assert!(cw.is_on(123.4));
    assert_eq!(cw.duty_cycle(), 1.0);

    let p = PulseProtocol::pulsed(1.0e-3, 9.0e-3); // 10% duty
    assert!((p.duty_cycle() - 0.1).abs() < 1e-12);
    assert!(p.is_on(0.0));
    assert!(p.is_on(0.9e-3));
    assert!(!p.is_on(1.5e-3));
    assert!(!p.is_on(9.5e-3));
    assert!(p.is_on(10.0e-3)); // next period
}

#[test]
fn drive_frequency_fixed_and_swept() {
    let fixed = DriveFrequency::Fixed(1.6e6);
    assert_eq!(fixed.at(0.0), 1.6e6);
    assert_eq!(fixed.at(1.0), 1.6e6);

    let s = sweep();
    let swept = DriveFrequency::Swept(s);
    // Triangular sweep starts at f_start and returns to it after a full period.
    assert!((swept.at(0.0) - s.f_start_hz).abs() < 1.0);
    assert!((swept.at(s.period_s) - s.f_start_hz).abs() < 1.0);
}

#[test]
fn no_production_gives_tissue_only_transmission_and_zero_shielding() {
    let mut prod = production();
    prod.k_prod_per_s = 0.0; // disable cavitation production ⇒ β stays 0
    let medium = medium();
    let cfg = ShieldingConfig {
        total_time_s: 0.2,
        dt_s: 1.0e-4,
    };
    let trace = simulate_shielding(
        DRIVE_PA,
        &DriveFrequency::Fixed(1.6e6),
        &PulseProtocol::continuous(),
        &prod,
        &medium,
        &cfg,
    );
    // β never grows.
    assert!(trace.peak_void_fraction < 1e-15);
    // Delivered fraction is exactly the tissue-only transmission exp(-α_t·L).
    let expected = (-medium.alpha_tissue_np_m * medium.path_len_m).exp();
    assert!((trace.mean_delivered_fraction_on - expected).abs() < 1e-9);
    // No cloud ⇒ no shielding loss.
    assert!(trace.shielding_loss_fraction < 1e-9);
}

#[test]
fn shielding_builds_up_under_cw_drive() {
    let trace = simulate_shielding(
        DRIVE_PA,
        &DriveFrequency::Fixed(1.6e6),
        &PulseProtocol::continuous(),
        &production(),
        &medium(),
        &ShieldingConfig {
            total_time_s: 1.0,
            dt_s: 5.0e-4,
        },
    );
    // The cloud accumulates and derates delivery: positive void fraction and a
    // real shielding loss.
    assert!(trace.peak_void_fraction > 1e-4);
    assert!(trace.shielding_loss_fraction > 0.05);
    // Delivered transmission is strictly below the unshielded tissue-only value.
    let tissue_only = (-medium().alpha_tissue_np_m * medium().path_len_m).exp();
    assert!(trace.mean_delivered_fraction_on < tissue_only);
}

#[test]
fn sweeping_reduces_shielding_versus_on_resonance_tone() {
    let cmp = compare_shielding_control(
        DRIVE_PA,
        &sweep(),
        &PulseProtocol::pulsed(5.0e-3, 0.4),
        &production(),
        &medium(),
        &ShieldingConfig {
            total_time_s: 2.0,
            dt_s: 5.0e-4,
        },
    );
    // Sweep-only beats fixed-only: a sweep crossing the cloud resonance sees a
    // lower band-average attenuation than a fixed tone on resonance.
    assert!(
        cmp.cw_swept.shielding_loss_fraction <= cmp.cw_fixed.shielding_loss_fraction,
        "swept loss {} should not exceed fixed loss {}",
        cmp.cw_swept.shielding_loss_fraction,
        cmp.cw_fixed.shielding_loss_fraction
    );
    // Combined control delivers the most energy of the four exposures.
    let best = cmp.pulsed_swept.delivered_energy;
    assert!(best >= cmp.cw_fixed.delivered_energy);
    assert!(best >= cmp.pulsed_fixed.delivered_energy);
    assert!(best >= cmp.cw_swept.delivered_energy);
}

#[test]
fn shielding_law_is_monotonic_in_void_fraction() {
    // The delivered focal pressure must fall monotonically as the cloud grows:
    // more bubbles ⇒ more Commander–Prosperetti scattering ⇒ less transmission.
    let medium = medium();
    let f = 1.6e6;
    let betas = [0.0, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4];
    let mut prev = f64::INFINITY;
    for &b in &betas {
        let p = delivered_pressure(DRIVE_PA, f, b, &medium);
        assert!(p > 0.0 && p <= DRIVE_PA);
        assert!(p < prev, "delivered pressure must decrease with β");
        prev = p;
    }
}

#[test]
fn off_interval_dissolves_residual_cloud() {
    // The mechanism behind ms-pulsing control: with the drive OFF, the residual
    // cloud dissolves (Epstein–Plesset), so β strictly decreases through the OFF
    // interval — the focus recovers transparency before the next pulse.
    let trace = simulate_shielding(
        DRIVE_PA,
        &DriveFrequency::Fixed(1.6e6),
        &PulseProtocol::pulsed(5.0e-3, 0.4),
        &production(),
        &medium(),
        &ShieldingConfig {
            total_time_s: 0.4,
            dt_s: 2.0e-4,
        },
    );
    // Indices safely inside the first OFF interval (t ∈ [5 ms, 405 ms]).
    let early = (0.02 / 2.0e-4) as usize; // t ≈ 20 ms
    let late = (0.39 / 2.0e-4) as usize; // t ≈ 390 ms
    let beta_early = trace.void_fraction[early];
    let beta_late = trace.void_fraction[late];
    assert!(
        beta_early > 0.0,
        "a cloud must have formed during the burst"
    );
    assert!(
        beta_late < beta_early,
        "OFF-interval dissolution must lower β: early {} late {}",
        beta_early,
        beta_late
    );
}
