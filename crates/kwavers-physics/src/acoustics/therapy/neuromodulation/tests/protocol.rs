//! Auto-organised from the original single-file `tests.rs` (split by concern).
//!
//! Pulse-train protocol, dosimetry, and ITRUSST safety — these reference
//! `super::super::protocol::*` directly, so no shared prelude is needed.

#[test]
fn protocol_derived_quantities_match_blackmore_definitions() {
    use super::super::protocol::PulseTrainProtocol;
    // A nested protocol: 0.5 ms pulses at 1500 Hz (BDC = 0.75), bursts of 0.2 s
    // every 0.5 s (burst-on fraction 0.4), 10 bursts.
    let p = PulseTrainProtocol {
        carrier_freq_hz: 500.0e3,
        pulse_length_s: 0.5e-3,
        pulse_repetition_freq_hz: 1500.0,
        burst_duration_s: 0.2,
        burst_interval_s: 0.3,
        num_bursts: 10,
    };
    assert!(p.is_valid());
    assert!((p.burst_duty_cycle() - 0.75).abs() < 1e-12, "BDC");
    assert!((p.burst_on_fraction() - 0.4).abs() < 1e-12, "burst-on");
    assert!((p.total_duty_cycle() - 0.30).abs() < 1e-12, "TDC = BDC·0.4");
    assert!(
        (p.burst_repetition_freq_hz() - 2.0).abs() < 1e-12,
        "BRF = 1/0.5"
    );
    assert!((p.total_time_s() - 5.0).abs() < 1e-12, "TT = 10·0.5");
}

#[test]
fn protocol_rejects_pulse_longer_than_period() {
    use super::super::protocol::PulseTrainProtocol;
    // PL = 1 ms but 1/PRF = 0.5 ms ⇒ pulse cannot fit in its repetition period.
    let p = PulseTrainProtocol {
        carrier_freq_hz: 500.0e3,
        pulse_length_s: 1.0e-3,
        pulse_repetition_freq_hz: 2000.0,
        burst_duration_s: 0.1,
        burst_interval_s: 0.0,
        num_bursts: 1,
    };
    assert!(!p.is_valid());
}

#[test]
fn protocol_pulse_envelope_gates_correctly() {
    use super::super::protocol::PulseTrainProtocol;
    // 1 ms pulses at 100 Hz (period 10 ms, BDC 0.1), single 50 ms burst.
    let p = PulseTrainProtocol {
        carrier_freq_hz: 500.0e3,
        pulse_length_s: 1.0e-3,
        pulse_repetition_freq_hz: 100.0,
        burst_duration_s: 0.05,
        burst_interval_s: 0.0,
        num_bursts: 1,
    };
    assert!(p.pulse_active(0.0005), "inside first pulse");
    assert!(!p.pulse_active(0.005), "in pulse-off gap");
    assert!(p.pulse_active(0.0105), "inside second pulse");
    assert!(!p.pulse_active(0.06), "after the burst ends");
    assert!(!p.pulse_active(-1.0), "before onset");
}

#[test]
fn theta_burst_atkinson_dosimetry_matches_paper() {
    use super::super::protocol::{tissue_dosimetry, PulseTrainProtocol};
    let p = PulseTrainProtocol::theta_burst_atkinson_2025();
    assert!(p.is_valid());
    // Paper: 10 % duty cycle, 80 s total.
    assert!((p.total_duty_cycle() - 0.10).abs() < 1e-12, "DC = 10%");
    assert!((p.total_time_s() - 80.0).abs() < 1e-9, "TT = 80 s");
    // Paper I_SPPA = 54.51 W/cm² ⇒ I_SPTA = I_SPPA·DC = 5.451 W/cm².
    // Invert: peak pressure for I_SPPA = 54.51 W/cm² in tissue, then check ratio.
    let z = 1000.0 * 1540.0;
    let p_peak = (2.0 * z * 54.51e4_f64).sqrt(); // 54.51 W/cm² = 54.51e4 W/m²
    let dose = tissue_dosimetry(&p, p_peak);
    assert!(
        (dose.isppa_w_cm2 - 54.51).abs() < 1e-2,
        "ISPPA {}",
        dose.isppa_w_cm2
    );
    assert!(
        (dose.ispta_w_cm2 - 5.451).abs() < 1e-2,
        "ISPTA {}",
        dose.ispta_w_cm2
    );
    assert!(
        (dose.ispta_w_cm2 / dose.isppa_w_cm2 - 0.10).abs() < 1e-9,
        "ISPTA/ISPPA must equal duty cycle"
    );
}

#[test]
fn itrusst_safety_assessment() {
    use super::super::protocol::itrusst_assess;
    // Within both criteria: MI 0.5, ΔT 1 °C ⇒ non-significant risk.
    let ok = itrusst_assess(0.5, 1.0, 0.0);
    assert!(ok.mechanical_ok && ok.thermal_ok && ok.overall_ok);
    // Thermal met via CEM43 even if ΔT slightly exceeds 2 °C.
    let by_dose = itrusst_assess(0.5, 2.5, 1.0);
    assert!(by_dose.thermal_ok && by_dose.overall_ok);
    // Mechanical breach (MI > 1.9) fails overall despite cool thermal.
    let mech_breach = itrusst_assess(2.5, 0.5, 0.0);
    assert!(!mech_breach.mechanical_ok && !mech_breach.overall_ok);
    // Thermal breach (ΔT > 2 °C and CEM43 > 2) fails overall.
    let therm_breach = itrusst_assess(0.5, 3.0, 5.0);
    assert!(!therm_breach.thermal_ok && !therm_breach.overall_ok);
}

#[test]
fn dosimetry_fda_limit_screening() {
    use super::super::protocol::{tissue_dosimetry, PulseTrainProtocol};
    let p = PulseTrainProtocol::theta_burst_atkinson_2025();
    // Low pressure (50 kPa) is comfortably within all FDA reference limits.
    let safe = tissue_dosimetry(&p, 50.0e3);
    assert!(
        safe.within_fda_limits(),
        "50 kPa should be within FDA limits"
    );
    // A very high peak pressure (5 MPa) breaches the MI ≤ 1.9 limit
    // (MI = 5/√0.5 ≈ 7.07).
    let unsafe_dose = tissue_dosimetry(&p, 5.0e6);
    assert!(
        !unsafe_dose.within_fda_limits(),
        "5 MPa should breach FDA limits"
    );
    assert!(unsafe_dose.mechanical_index > MI_LIMIT_REF);
}

/// FDA general-diagnostic MI limit (mirrors `MI_LIMIT_SOFT_TISSUE`).
const MI_LIMIT_REF: f64 = 1.9;
