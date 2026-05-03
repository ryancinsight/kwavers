use super::functions::{add_noise, create_cw_signals, next_pow2, pad_zeros, tone_burst_series};
use super::window::WindowType;
use proptest::prelude::*;
use std::f64::consts::PI;

#[test]
fn tone_burst_series_respects_offset_and_length() {
    let y = tone_burst_series(
        1_000.0,
        100.0,
        2.0,
        10,
        Some(40),
        WindowType::Hann,
        1.0,
        0.0,
    )
    .unwrap();
    assert_eq!(y.len(), 40);
    assert!(y[..10].iter().all(|&v| v == 0.0));
    assert!(y[10..].iter().any(|&v| v != 0.0));
}

#[test]
fn tone_burst_series_matches_kwave_gaussian_reference() {
    let y = tone_burst_series(
        10_000_000.0,
        1_000_000.0,
        3.0,
        0,
        None,
        WindowType::Gaussian,
        1.0,
        0.0,
    )
    .unwrap();

    let expected = [
        0.0,
        0.011662302880078558,
        0.03238105368860502,
        0.053387331984085955,
        0.05226681535262026,
        -4.3527146854640016e-17,
        -0.11632193676921493,
        -0.26442918636912427,
        -0.3569420662005188,
        -0.286105797573363,
        3.9015027664288013e-16,
        0.42681969495829636,
        0.794389177429922,
        0.877935816529585,
        0.5761463244863592,
        3.6739403974420594e-16,
        -0.5761463244863579,
        -0.8779358165295834,
        -0.7943891774299194,
        -0.42681969495829386,
        7.80300553285756e-16,
        0.2861057975733616,
        0.3569420662005161,
        0.2644291863691221,
        0.11632193676921358,
        8.286896071370835e-17,
        -0.0522668153526196,
        -0.053387331984085316,
        -0.032381053688604576,
        -0.011662302880078391,
        0.0,
    ];

    assert_eq!(y.len(), expected.len());
    for (index, (got, exp)) in y.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-12,
            "mismatch at index {index}: got {got}, expected {exp}"
        );
    }
}

#[test]
fn tone_burst_series_uses_floor_plus_one_sample_count() {
    let y = tone_burst_series(
        11_293_333.333_333_33,
        500_000.0,
        5.0,
        0,
        None,
        WindowType::Rectangular,
        1.0,
        0.0,
    )
    .unwrap();

    assert_eq!(y.len(), 113);
}

#[test]
fn add_noise_achieves_target_snr_reasonably() {
    let sample_rate_hz = 10_000.0;
    let f0 = 100.0;
    let n = 65_536;
    let dt = 1.0 / sample_rate_hz;
    let clean: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * f0 * (i as f64 * dt)).sin())
        .collect();

    let snr_db_target = 20.0;
    let noisy = add_noise(&clean, snr_db_target, Some(123)).unwrap();
    let noise: Vec<f64> = noisy
        .iter()
        .zip(clean.iter())
        .map(|(&y, &x)| y - x)
        .collect();

    let signal_power = clean.iter().map(|&x| x * x).sum::<f64>() / n as f64;
    let noise_power = noise.iter().map(|&x| x * x).sum::<f64>() / n as f64;
    let snr_db_measured = 10.0 * (signal_power / noise_power).log10();

    assert!((snr_db_measured - snr_db_target).abs() < 0.5);
}

#[test]
fn create_cw_signals_broadcasts_phase() {
    let t = [0.0, 0.25, 0.5, 0.75];
    let out = create_cw_signals(&t, 1.0, &[1.0, 2.0], &[0.0]).unwrap();
    assert_eq!(out.dim(), (2, 4));
    assert!((out[[0, 1]] - 1.0).abs() < 1e-12);
    assert!((out[[1, 1]] - 2.0).abs() < 1e-12);
}

proptest! {
    #[test]
    fn next_pow2_is_power_of_two_and_ge_n(n in 1usize..(1<<20)) {
        let p = next_pow2(n);
        prop_assert!(p >= n);
        prop_assert!(p.is_power_of_two());
    }

    #[test]
    fn pad_zeros_preserves_prefix(n in 0usize..2048, m in 0usize..2048) {
        let src: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let dst = pad_zeros(&src, m);
        prop_assert_eq!(dst.len(), m);
        let k = n.min(m);
        prop_assert_eq!(&dst[..k], &src[..k]);
    }
}
