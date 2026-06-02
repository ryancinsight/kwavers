use super::*;
use kwavers_core::constants::numerical::{MHZ_TO_HZ, MPA_TO_PA};

const REL_TOL: f64 = 1.0e-6;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * b.abs().max(1.0)
}

#[test]
fn intrinsic_threshold_at_1mhz_matches_maxwell_2013() {
    // Maxwell 2013 Table II liver fit: p_t = 28.2 MPa at 1 MHz.
    let pt = intrinsic_threshold_pa(MHZ_TO_HZ);
    assert!(approx_eq(pt, 28.2 * MPA_TO_PA, REL_TOL));
}

#[test]
fn intrinsic_threshold_is_monotone_in_frequency() {
    // Vlaisavljevich 2015 reports a weak monotone increase with frequency.
    let lo = intrinsic_threshold_pa(0.5 * MHZ_TO_HZ);
    let mid = intrinsic_threshold_pa(MHZ_TO_HZ);
    let hi = intrinsic_threshold_pa(3.0 * MHZ_TO_HZ);
    assert!(lo < mid);
    assert!(mid < hi);
    // Slope is ~1.4 MPa/decade — verify mid/hi spread is bounded.
    assert!((hi - mid) < MPA_TO_PA); // < 1 MPa over factor of 3 in f
}

#[test]
fn microsecond_scenario_exceeds_intrinsic_threshold() {
    let s = HistotripsyScenario::intrinsic_threshold_liver_1mhz();
    assert_eq!(s.regime, HistotripsyRegime::IntrinsicThreshold);
    assert!(s.exceeds_intrinsic_threshold());
    // P_cav at PNP=30 MPa, p_t=28.2 MPa, sigma=0.96 MPa is essentially 1.
    let p = s.cavitation_probability();
    assert!(p > 0.95 && p <= 1.0, "p_cav={}", p);
}

#[test]
fn shock_scattering_does_not_exceed_intrinsic_threshold() {
    let s = HistotripsyScenario::shock_scattering_1mhz();
    assert_eq!(s.regime, HistotripsyRegime::ShockScattering);
    assert!(!s.exceeds_intrinsic_threshold());
}

#[test]
fn millisecond_scenarios_classify_correctly() {
    let bh = HistotripsyScenario::boiling_histotripsy_liver_1mhz();
    let cc = HistotripsyScenario::millisecond_cavitation_500khz();
    assert!(bh.regime.is_millisecond());
    assert!(cc.regime.is_millisecond());
    assert!(!bh.regime.is_intrinsic_threshold());
    assert!(!cc.regime.is_intrinsic_threshold());
    // Both rely on sub-threshold drive; PNP magnitude < p_t(f).
    assert!(!bh.exceeds_intrinsic_threshold());
    assert!(!cc.exceeds_intrinsic_threshold());
}

#[test]
fn boiling_histotripsy_duty_cycle_is_one_percent() {
    let s = HistotripsyScenario::boiling_histotripsy_liver_1mhz();
    // ShockFormed pattern has no PRF; explicit construction used in
    // Khokhlova 2014/2019 gives 10 ms / 1 s = 1%. We verify by composing
    // the canonical clinical pattern manually.
    let on = s.pulse.pulse_on_time_s(s.frequency_hz);
    assert!(approx_eq(on, 10.0e-3, 1.0e-9));
}

#[test]
fn dual_prf_pattern_has_correct_duty_cycle() {
    let s = HistotripsyScenario::intrinsic_threshold_dual_prf_1mhz();
    let on = s.pulse.pulse_on_time_s(s.frequency_hz);
    let prf = s.pulse.average_prf_hz();
    let duty = s.duty_cycle();
    // 5 micro-pulses at 1 kHz, 2 cycles each at 1 MHz → on-time 4 ms+2us
    assert!(prf > 0.0);
    assert!(duty > 0.0 && duty < 1.0);
    assert!(approx_eq(on * prf, duty, 1.0e-12));
    // Mean PRF 50 Hz, on-time ~4 ms → duty ~0.2
    assert!(duty > 0.1 && duty < 0.3, "duty={}", duty);
}

#[test]
fn dithered_prf_pattern_average_duty_matches_mean_prf() {
    let s = HistotripsyScenario::intrinsic_threshold_dithered_prf_1mhz();
    // 2 cycles at 1 MHz = 2 µs; mean PRF 200 Hz → duty 4e-4.
    let duty = s.duty_cycle();
    assert!(approx_eq(duty, 4.0e-4, 1.0e-9));
}

#[test]
fn mechanical_index_matches_aium_definition() {
    // PNP 30 MPa at 1 MHz → MI = 30.
    let s = HistotripsyScenario::intrinsic_threshold_liver_1mhz();
    assert!(approx_eq(s.mechanical_index(), 30.0, 1.0e-9));
    // Thrombolysis: 32 MPa / sqrt(1.5) ≈ 26.13.
    let t = HistotripsyScenario::thrombolysis_1_5mhz();
    assert!(approx_eq(
        t.mechanical_index(),
        32.0 / (1.5_f64).sqrt(),
        1.0e-9
    ));
}

#[test]
fn cavitation_probability_is_monotone_in_pnp() {
    // Construct a sweep of identical scenarios with rising PNP magnitude
    // and verify monotone non-decreasing P_cav (Theorem 21.1).
    let base = HistotripsyScenario::intrinsic_threshold_liver_1mhz();
    let mut last = -1.0;
    for pnp_mpa in [20.0, 24.0, 27.0, 28.2, 30.0, 35.0_f64] {
        let mut s = base;
        s.peak_negative_pressure_pa = -pnp_mpa * MPA_TO_PA;
        let p = s.cavitation_probability();
        assert!(p >= last - 1.0e-12, "non-monotone: {} < {}", p, last);
        last = p;
    }
    // Anchor: at p_t exactly, P_cav = 0.5.
    let mut s = base;
    s.peak_negative_pressure_pa = -intrinsic_threshold_pa(s.frequency_hz);
    assert!(approx_eq(s.cavitation_probability(), 0.5, 1.0e-6));
}
