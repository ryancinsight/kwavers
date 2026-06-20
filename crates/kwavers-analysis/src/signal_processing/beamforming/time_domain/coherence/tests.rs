//! Value-semantic tests for coherence-factor weighting.
//!
//! Expected values are derived analytically from the closed-form definitions in
//! [`super`]; no tolerance is wider than the f64 round-off of the arithmetic.

use super::{delay_and_sum_coherence, CoherenceFactor};
use crate::signal_processing::beamforming::time_domain::das::delay_and_sum;
use crate::signal_processing::beamforming::time_domain::delay_reference::DelayReference;
use ndarray::{Array2, Array3};
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
