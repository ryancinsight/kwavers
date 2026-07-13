//! Value-semantic tests for coherence-factor weighting.
//!
//! Expected values are derived analytically from the closed-form definitions in
//! [`super`]; no tolerance is wider than the f64 round-off of the arithmetic.

use super::{
    delay_and_sum_coherence, phase_coherence_from_iq_aperture, phase_coherence_from_phases,
    CoherenceFactor,
};
use crate::signal_processing::beamforming::time_domain::das::delay_and_sum;
use crate::signal_processing::beamforming::time_domain::delay_reference::DelayReference;
use eunomia::Complex64;
use leto::{Array2, Array3, SliceArg};
use std::f64::consts::PI;

/// Build a single-sample aligned aperture column `(n_elements, 1)`.
fn column(values: &[f64]) -> Array2<f64> {
    Array2::from_shape_vec((values.len(), 1), values.to_vec()).expect("valid column shape")
}

// --- Amplitude coherence factor (Mallart & Fink 1994) ---

#[test]
fn amplitude_cf_is_unity_for_a_perfectly_coherent_aperture() {
    // All elements equal ⇒ CF = |N·a|² / (N · N·a²) = 1, independent of a.
    let aligned = column(&[2.0, 2.0, 2.0, 2.0]);
    let cf = CoherenceFactor::Amplitude
        .weights(&aligned)
        .expect("weights");
    assert!(
        (cf[0] - 1.0).abs() < 1e-12,
        "coherent CF should be 1, got {}",
        cf[0]
    );
}

#[test]
fn amplitude_cf_is_zero_for_a_zero_mean_aperture() {
    // Σxᵢ = 0 ⇒ coherent energy 0 ⇒ CF = 0.
    let aligned = column(&[1.0, -1.0, 1.0, -1.0]);
    let cf = CoherenceFactor::Amplitude
        .weights(&aligned)
        .expect("weights");
    assert!(
        cf[0].abs() < 1e-12,
        "incoherent CF should be 0, got {}",
        cf[0]
    );
}

#[test]
fn amplitude_cf_matches_closed_form_intermediate() {
    // column [1,1,1,0]: Σx = 3, Σx² = 3, N = 4 ⇒ CF = 9 / (4·3) = 0.75.
    let aligned = column(&[1.0, 1.0, 1.0, 0.0]);
    let cf = CoherenceFactor::Amplitude
        .weights(&aligned)
        .expect("weights");
    assert!((cf[0] - 0.75).abs() < 1e-12, "expected 0.75, got {}", cf[0]);
}

#[test]
fn amplitude_cf_is_zero_for_all_zero_column() {
    let aligned = column(&[0.0, 0.0, 0.0]);
    let cf = CoherenceFactor::Amplitude
        .weights(&aligned)
        .expect("weights");
    assert_eq!(cf[0], 0.0);
}

#[test]
fn amplitude_cf_stays_in_unit_interval() {
    // Cauchy–Schwarz bound: CF ∈ [0,1] for arbitrary data.
    let aligned = Array2::from_shape_vec(
        (4, 3),
        vec![
            3.0, -1.0, 0.5, //
            -2.0, 4.0, -7.0, //
            1.5, 0.0, 2.0, //
            -0.25, 9.0, -3.0,
        ],
    )
    .expect("valid shape");
    let cf = CoherenceFactor::Amplitude
        .weights(&aligned)
        .expect("weights");
    for &v in cf.iter() {
        assert!((0.0..=1.0).contains(&v), "CF out of [0,1]: {v}");
    }
}

// --- Sign coherence factor (Camacho et al. 2009) ---

#[test]
fn sign_cf_is_unity_for_uniform_sign() {
    // All positive ⇒ b̄ = 1 ⇒ SCF = (1 − 0)^p = 1.
    let aligned = column(&[0.5, 2.0, 0.1, 3.0]);
    let cf = CoherenceFactor::Sign { sensitivity: 1.0 }
        .weights(&aligned)
        .expect("weights");
    assert!(
        (cf[0] - 1.0).abs() < 1e-12,
        "uniform-sign SCF should be 1, got {}",
        cf[0]
    );
}

#[test]
fn sign_cf_is_zero_for_balanced_signs() {
    // Equal +/- counts ⇒ b̄ = 0 ⇒ SCF = (1 − 1)^p = 0.
    let aligned = column(&[1.0, -1.0, 2.0, -2.0]);
    let cf = CoherenceFactor::Sign { sensitivity: 1.0 }
        .weights(&aligned)
        .expect("weights");
    assert!(
        cf[0].abs() < 1e-12,
        "balanced-sign SCF should be 0, got {}",
        cf[0]
    );
}

#[test]
fn sign_cf_matches_closed_form_intermediate() {
    // 3 positive, 1 negative, N=4 ⇒ b̄ = 0.5 ⇒ SCF = 1 − √(1 − 0.25) = 1 − √0.75.
    let aligned = column(&[1.0, 1.0, 1.0, -1.0]);
    let expected = 1.0 - 0.75_f64.sqrt();
    let cf = CoherenceFactor::Sign { sensitivity: 1.0 }
        .weights(&aligned)
        .expect("weights");
    assert!(
        (cf[0] - expected).abs() < 1e-12,
        "expected {expected}, got {}",
        cf[0]
    );
}

#[test]
fn sign_cf_sensitivity_exponentiates() {
    // Same b̄ = 0.5; sensitivity p=2 ⇒ SCF = (1 − √0.75)².
    let aligned = column(&[1.0, 1.0, 1.0, -1.0]);
    let base = 1.0 - 0.75_f64.sqrt();
    let cf = CoherenceFactor::Sign { sensitivity: 2.0 }
        .weights(&aligned)
        .expect("weights");
    assert!(
        (cf[0] - base * base).abs() < 1e-12,
        "expected {}, got {}",
        base * base,
        cf[0]
    );
}

#[test]
fn sign_cf_rejects_invalid_sensitivity() {
    let aligned = column(&[1.0, 1.0]);
    assert!(CoherenceFactor::Sign { sensitivity: 0.5 }
        .weights(&aligned)
        .is_err());
    assert!(CoherenceFactor::Sign {
        sensitivity: f64::NAN
    }
    .weights(&aligned)
    .is_err());
}

// --- End-to-end DAS + coherence ---

#[test]
fn coherence_das_equals_das_times_cf() {
    // Zero delays ⇒ aligned = raw samples; verify y_cf[j] == cf[j]·y_das[j].
    let fs = 1.0;
    let delays = [0.0, 0.0, 0.0];
    let weights = [1.0, 0.5, 2.0];
    let data = Array3::from_shape_vec(
        (3, 1, 4),
        vec![
            1.0, -2.0, 3.0, 0.0, //
            0.5, 2.0, -1.0, 4.0, //
            -1.0, 1.0, 2.0, -2.0,
        ],
    )
    .expect("valid shape");

    let das =
        delay_and_sum(&data, fs, &delays, &weights, DelayReference::SensorIndex(0)).expect("das");
    let (cf_out, cf) = delay_and_sum_coherence(
        &data,
        fs,
        &delays,
        &weights,
        DelayReference::SensorIndex(0),
        CoherenceFactor::Amplitude,
    )
    .expect("coherence das");

    for j in 0..4 {
        let expected = das[[0, 0, j]] * cf[j];
        assert!(
            (cf_out[[0, 0, j]] - expected).abs() < 1e-12,
            "j={j}: expected {expected}, got {}",
            cf_out[[0, 0, j]]
        );
    }
}

#[test]
fn coherence_preserves_aligned_coherent_impulse() {
    // Two aligned impulses: at the focus column = [1,1] ⇒ CF = 1 ⇒ output preserved.
    let fs = 10.0;
    let n = 8usize;
    let delays = [1.0, 1.2];
    let weights = [1.0, 1.0];
    let mut x = Array3::<f64>::zeros((2, 1, n));
    x[[0, 0, 3]] = 1.0;
    x[[1, 0, 5]] = 1.0;

    let (out, cf) = delay_and_sum_coherence(
        &x,
        fs,
        &delays,
        &weights,
        DelayReference::SensorIndex(0),
        CoherenceFactor::Amplitude,
    )
    .expect("coherence das");

    assert!(
        (cf[3] - 1.0).abs() < 1e-12,
        "focus CF should be 1, got {}",
        cf[3]
    );
    assert!(
        (out[[0, 0, 3]] - 2.0).abs() < 1e-12,
        "focus output should be 2, got {}",
        out[[0, 0, 3]]
    );
}

// --- Generalized coherence factor (Li & Li 2003) ---

#[test]
fn gcf_m0_zero_equals_amplitude_cf() {
    // KEYSTONE: GCF with m0=0 counts only the DC bin, |X_0|² = |Σx|², over the
    // Parseval total N·Σx² — bit-for-bit the amplitude CF, for arbitrary data.
    let aligned = Array2::from_shape_vec(
        (5, 3),
        vec![
            3.0, -1.0, 0.5, //
            -2.0, 4.0, -7.0, //
            1.5, 0.0, 2.0, //
            -0.25, 9.0, -3.0, //
            2.25, -4.5, 1.0,
        ],
    )
    .expect("valid aperture");
    let gcf = CoherenceFactor::Generalized { m0: 0 }
        .weights(&aligned)
        .expect("gcf");
    let acf = CoherenceFactor::Amplitude.weights(&aligned).expect("acf");
    for j in 0..gcf.len() {
        assert!(
            (gcf[j] - acf[j]).abs() < 1e-12,
            "j={j}: GCF(m0=0) {} must equal amplitude CF {}",
            gcf[j],
            acf[j]
        );
    }
}

#[test]
fn gcf_is_unity_when_passband_covers_all_spatial_frequencies() {
    // m0 ≥ N/2 ⇒ the whole spectrum is "coherent" ⇒ GCF = 1 by Parseval.
    let aligned = column(&[3.0, -2.0, 1.5, -0.25, 2.25, -4.5]); // N=6
    for m0 in [3usize, 4, 10] {
        let gcf = CoherenceFactor::Generalized { m0 }
            .weights(&aligned)
            .expect("gcf");
        assert!(
            (gcf[0] - 1.0).abs() < 1e-12,
            "m0={m0}: GCF should be 1, got {}",
            gcf[0]
        );
    }
}

#[test]
fn gcf_is_monotonic_non_decreasing_in_m0() {
    // Admitting more spatial frequencies can only add non-negative energy to the
    // numerator (fixed denominator) ⇒ GCF non-decreasing in m0.
    let aligned = column(&[1.0, -3.0, 2.0, 0.5, -1.5, 4.0, -2.0, 0.25]); // N=8
    let mut prev = -1.0;
    for m0 in 0..=4usize {
        let gcf = CoherenceFactor::Generalized { m0 }
            .weights(&aligned)
            .expect("gcf")[0];
        assert!(
            gcf >= prev - 1e-12,
            "m0={m0}: GCF {gcf} dropped below {prev}"
        );
        assert!(
            (0.0..=1.0 + 1e-12).contains(&gcf),
            "m0={m0}: GCF {gcf} out of [0,1]"
        );
        prev = gcf;
    }
    assert!(
        (prev - 1.0).abs() < 1e-12,
        "m0=N/2 must reach 1, got {prev}"
    );
}

#[test]
fn gcf_matches_closed_form_for_a_pure_two_cycle_aperture() {
    // x_t = cos(2π·2·t/N), N=8: all energy sits in the k=±2 spatial-frequency
    // bins. So GCF(m0<2)=0 and GCF(m0≥2)=1 — an exact spectral-localization check.
    let n = 8usize;
    let vals: Vec<f64> = (0..n)
        .map(|t| (2.0 * PI * 2.0 * t as f64 / n as f64).cos())
        .collect();
    let aligned = column(&vals);
    for m0 in 0..2usize {
        let g = CoherenceFactor::Generalized { m0 }
            .weights(&aligned)
            .expect("gcf")[0];
        assert!(
            g.abs() < 1e-12,
            "m0={m0}: energy is at k=2, GCF should be 0, got {g}"
        );
    }
    for m0 in 2..=4usize {
        let g = CoherenceFactor::Generalized { m0 }
            .weights(&aligned)
            .expect("gcf")[0];
        assert!(
            (g - 1.0).abs() < 1e-12,
            "m0={m0}: k=2 captured, GCF should be 1, got {g}"
        );
    }
}

#[test]
fn gcf_is_zero_for_all_zero_column() {
    let aligned = column(&[0.0, 0.0, 0.0, 0.0]);
    let gcf = CoherenceFactor::Generalized { m0: 1 }
        .weights(&aligned)
        .expect("gcf");
    assert_eq!(gcf[0], 0.0);
}

// --- Phase coherence factor (Camacho, Parrilla & Fritsch 2009) ---
//
// σ₀ = π/√3 is the std of a phase uniform on [−π, π]; below it is reused as the
// analytic reference for every derived expected value.

/// `√3/π = 1/σ₀` — the PCF slope per unit phase std at sensitivity 1.
fn inv_sigma_ref() -> f64 {
    3.0_f64.sqrt() / PI
}

#[test]
fn pcf_is_unity_for_a_perfectly_coherent_aperture() {
    // All phases equal ⇒ σ(φ) = σ(ψ) = 0 ⇒ PCF = 1, independent of the phase.
    assert!((phase_coherence_from_phases(&[0.3, 0.3, 0.3, 0.3], 1.0) - 1.0).abs() < 1e-12);
}

#[test]
fn pcf_matches_closed_form_for_a_small_symmetric_spread() {
    // φ = [0.1, −0.1, 0.1, −0.1]: σ(φ) = 0.1; ψ shifts these near ±π so
    // σ(ψ) ≈ π−0.1 ≫ σ(φ) ⇒ s = 0.1 ⇒ PCF = 1 − 0.1/σ₀.
    let pcf = phase_coherence_from_phases(&[0.1, -0.1, 0.1, -0.1], 1.0);
    let expected = 1.0 - 0.1 * inv_sigma_ref();
    assert!((pcf - expected).abs() < 1e-12, "PCF {pcf} vs {expected}");
}

#[test]
fn pcf_auxiliary_phase_rescues_wraparound_coherence() {
    // KEYSTONE: a wavefront coherent but straddling the ±π branch cut. The naive
    // phase std σ(φ) ≈ π−0.05 (spuriously near-incoherent), but the auxiliary
    // phase ψ = φ − sign(φ)·π collapses to ±0.05 ⇒ σ(ψ) = 0.05 ⇒ PCF high.
    // This is exactly what distinguishes the phase CF from the sign CF.
    let phases = [PI - 0.05, -PI + 0.05, PI - 0.05, -PI + 0.05];
    let pcf = phase_coherence_from_phases(&phases, 1.0);
    let expected = 1.0 - 0.05 * inv_sigma_ref();
    assert!((pcf - expected).abs() < 1e-12, "PCF {pcf} vs {expected}");
    assert!(
        pcf > 0.97,
        "wraparound-coherent aperture must score high: {pcf}"
    );
}

#[test]
fn pcf_matches_closed_form_for_a_ninety_degree_spread() {
    // φ = [−π/2, π/2]: σ(φ) = π/2; ψ = [π/2, −π/2] ⇒ σ(ψ) = π/2 ⇒ s = π/2 ⇒
    // PCF = 1 − (π/2)/σ₀ = 1 − √3/2.
    let pcf = phase_coherence_from_phases(&[-PI / 2.0, PI / 2.0], 1.0);
    let expected = 1.0 - 3.0_f64.sqrt() / 2.0;
    assert!((pcf - expected).abs() < 1e-12, "PCF {pcf} vs {expected}");
}

#[test]
fn pcf_is_near_zero_for_an_evenly_spread_aperture() {
    // φ = [−3π/4, −π/4, π/4, 3π/4] (step π/2): an arithmetic sequence with
    // population std (π/2)·√((4²−1)/12) = (π/2)·√1.25, and ψ maps the set onto
    // itself ⇒ s identical ⇒ PCF = 1 − s/σ₀ ≈ 0.032 (near full incoherence).
    let phases = [-3.0 * PI / 4.0, -PI / 4.0, PI / 4.0, 3.0 * PI / 4.0];
    let s = (PI / 2.0) * (15.0_f64 / 12.0).sqrt();
    let expected = (1.0 - s * inv_sigma_ref()).max(0.0);
    let pcf = phase_coherence_from_phases(&phases, 1.0);
    assert!((pcf - expected).abs() < 1e-12, "PCF {pcf} vs {expected}");
    assert!(
        pcf < 0.05,
        "evenly-spread aperture must be near-incoherent: {pcf}"
    );
}

#[test]
fn pcf_sensitivity_scales_the_rejection_linearly() {
    // PCF(γ) = 1 − γ·s/σ₀; doubling γ doubles the deficit from 1.
    let phases = [0.1, -0.1, 0.1, -0.1];
    let p1 = phase_coherence_from_phases(&phases, 1.0);
    let p2 = phase_coherence_from_phases(&phases, 2.0);
    let expected2 = 1.0 - 2.0 * 0.1 * inv_sigma_ref();
    assert!(
        (p2 - expected2).abs() < 1e-12,
        "PCF(γ=2) {p2} vs {expected2}"
    );
    assert!(p2 < p1, "higher sensitivity must reject more: {p2} !< {p1}");
}

#[test]
fn pcf_is_zero_for_an_empty_aperture() {
    assert_eq!(phase_coherence_from_phases(&[], 1.0), 0.0);
}

#[test]
fn pcf_via_weights_is_unity_for_identical_rows() {
    // Identical RF rows ⇒ identical analytic-signal phase per column ⇒ σ = 0 ⇒
    // PCF = 1 at every sample, exactly (the column-path wiring check).
    let row = [0.0, 1.0, 0.5, -0.3, 0.8, -0.2, 0.1, -0.6];
    let mut aligned = Array2::<f64>::zeros((3, row.len()));
    for mut r in aligned.rows_mut().expect("rows_mut") {
        let slots = r.as_mut_slice().expect("contiguous row");
        for (slot, &v) in slots.iter_mut().zip(row.iter()) {
            *slot = v;
        }
    }
    let cf = CoherenceFactor::Phase { sensitivity: 1.0 }
        .weights(&aligned)
        .expect("pcf weights");
    for &v in cf.iter() {
        assert!((v - 1.0).abs() < 1e-12, "identical rows ⇒ PCF 1, got {v}");
    }
}

#[test]
fn pcf_via_weights_is_low_for_a_quadrature_spread_aperture() {
    // Four elements driven 90° apart: cos(ωt − iπ/2), i = 0..3. The analytic
    // phases at each sample are {θ, θ−π/2, θ−π, θ−3π/2} — the evenly-spread
    // π/2-step set ⇒ PCF ≈ 0.03 (cf. `pcf_is_near_zero_for_an_evenly_spread`).
    let n = 64usize;
    let omega = 2.0 * PI * 2.0 / n as f64; // 2 cycles
    let mut aligned = Array2::<f64>::zeros((4, n));
    for (i, mut r) in aligned
        .rows_mut()
        .expect("rows_mut")
        .into_iter()
        .enumerate()
    {
        let slots = r.as_mut_slice().expect("contiguous row");
        for (t, slot) in slots.iter_mut().enumerate() {
            *slot = (omega * t as f64 - i as f64 * PI / 2.0).cos();
        }
    }
    let cf = CoherenceFactor::Phase { sensitivity: 1.0 }
        .weights(&aligned)
        .expect("pcf weights");
    // Interior samples (away from Hilbert edge transients) are near-incoherent.
    for &v in cf
        .slice_with::<1>(&[SliceArg::Range {
            start: Some(16),
            end: Some(48),
            step: 1,
        }])
        .expect("slice cf[16..48]")
        .iter()
    {
        assert!((0.0..=1.0).contains(&v), "PCF out of [0,1]: {v}");
        assert!(
            v < 0.2,
            "quadrature-spread aperture must score low, got {v}"
        );
    }
}

#[test]
fn pcf_rejects_invalid_sensitivity() {
    let aligned = column(&[1.0, 1.0]);
    assert!(CoherenceFactor::Phase { sensitivity: -1.0 }
        .weights(&aligned)
        .is_err());
    assert!(CoherenceFactor::Phase {
        sensitivity: f64::NAN
    }
    .weights(&aligned)
    .is_err());
}

// --- Native IQ/baseband phase coherence path (COV-1 follow-up) ---

#[test]
fn pcf_iq_matches_the_phase_scalar_core() {
    // KEYSTONE: feeding e^{iφ} per element to the IQ path must equal the scalar
    // `phase_coherence_from_phases` on φ — the IQ path is just `arg` + the same core.
    let phases = [0.2_f64, -0.5, 1.1, -2.0, 0.7];
    let n = phases.len();
    let mut iq = Array2::<Complex64>::zeros((n, 1));
    for (i, &p) in phases.iter().enumerate() {
        iq[[i, 0]] = Complex64::from_polar(2.3, p); // magnitude is irrelevant to phase CF
    }
    let cf_iq = phase_coherence_from_iq_aperture(&iq, 1.0).expect("iq pcf");
    let cf_phase = phase_coherence_from_phases(&phases, 1.0);
    assert!(
        (cf_iq[0] - cf_phase).abs() < 1e-12,
        "IQ path {} must equal phase core {}",
        cf_iq[0],
        cf_phase
    );
}

#[test]
fn pcf_iq_is_unity_for_a_phase_aligned_aperture() {
    // All elements share the same complex phase ⇒ σ = 0 ⇒ PCF = 1 at every sample,
    // regardless of per-element magnitude.
    let mags = [1.0, 0.3, 5.0];
    let mut iq = Array2::<Complex64>::zeros((3, 4));
    for j in 0..4 {
        for (i, &m) in mags.iter().enumerate() {
            iq[[i, j]] = Complex64::from_polar(m, 0.8); // common phase 0.8 rad
        }
    }
    let cf = phase_coherence_from_iq_aperture(&iq, 1.0).expect("iq pcf");
    for &v in cf.iter() {
        assert!((v - 1.0).abs() < 1e-12, "phase-aligned IQ ⇒ PCF 1, got {v}");
    }
}

#[test]
fn pcf_iq_rejects_empty_aperture_and_bad_sensitivity() {
    let iq = Array2::<Complex64>::zeros((0, 4));
    assert!(phase_coherence_from_iq_aperture(&iq, 1.0).is_err());
    let ok = Array2::<Complex64>::from_elem((2, 1), Complex64::new(1.0, 0.0));
    assert!(phase_coherence_from_iq_aperture(&ok, -1.0).is_err());
    assert!(phase_coherence_from_iq_aperture(&ok, f64::NAN).is_err());
}
