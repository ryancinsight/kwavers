//! Analytical oracles for the QUS spectral pipeline.
//!
//! Every test here recovers a *known* quantity from synthesized data, rather
//! than asserting that a spectrum merely exists or is non-empty.

use super::*;
use leto::Array2;

const SAMPLE_RATE_HZ: f64 = 40.0e6;
const FFT_SIZE: usize = 256;

fn frequency_step() -> f64 {
    SAMPLE_RATE_HZ / FFT_SIZE as f64
}

/// Deterministic pseudo-random phase, so speckle realizations are reproducible
/// without a dependency and without wall-clock seeding.
fn phase(seed: u64) -> f64 {
    let mixed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let unit = ((mixed >> 33) as f64) / ((1_u64 << 31) as f64);
    unit * std::f64::consts::TAU
}

/// Build `[n_lines, n_samples]` RF whose power spectrum follows `amplitude(f)`,
/// with independent random phase per line so the lines are uncorrelated
/// realizations of the same underlying spectrum.
fn synth_rf(n_lines: usize, n_samples: usize, amplitude: impl Fn(f64) -> f64) -> Array2<f64> {
    let mut rf = Array2::zeros((n_lines, n_samples));
    let bins = FFT_SIZE / 2;
    for line in 0..n_lines {
        for sample in 0..n_samples {
            let t = sample as f64 / SAMPLE_RATE_HZ;
            let mut value = 0.0;
            for bin in 1..bins {
                let f = bin as f64 * frequency_step();
                let a = amplitude(f);
                if a > 0.0 {
                    let ph = phase((line as u64) << 20 | bin as u64);
                    value += a * (std::f64::consts::TAU * f * t + ph).cos();
                }
            }
            rf[[line, sample]] = value;
        }
    }
    rf
}

/// A flat-spectrum sample against a flat-spectrum reference must normalize to a
/// flat 0 dB, and its fitted slope must be zero. This is the pipeline's null
/// case: identical sample and reference means "this tissue looks exactly like
/// the phantom", which must not manufacture structure.
#[test]
fn identical_sample_and_reference_normalize_to_zero_db_with_zero_slope() {
    let rf = synth_rf(32, 512, |_| 1.0);
    let spectrum = gated_spectrum(rf.view(), 64..320, SAMPLE_RATE_HZ, FFT_SIZE).expect("spectrum");
    let normalized = normalize_to_reference(&spectrum, &spectrum).expect("normalize");

    let band = AnalysisBand::try_new(2.0e6, 8.0e6).expect("band");
    let params = backscatter_parameters(&normalized, frequency_step(), band).expect("fit");

    assert!(
        params.midband_db.abs() < 1.0e-12,
        "midband must be 0 dB, got {}",
        params.midband_db
    );
    assert!(
        params.slope_db_per_hz.abs() < 1.0e-15,
        "slope must be 0, got {}",
        params.slope_db_per_hz
    );
    assert!(params.bins_used >= 30, "band should cover many bins");
}

/// The central oracle: impose a known dB-per-Hz tilt on the sample relative to
/// the reference and recover it.
///
/// A spectrum whose amplitude is `10^(k·f/20)` has power `10^(k·f/10)`, so its
/// dB spectrum relative to a flat reference is exactly `k·f` — a straight line
/// of slope `k`. The fit must return `-k` (the negation convention).
#[test]
fn known_spectral_tilt_is_recovered() {
    // -1.5 dB per MHz, a realistic soft-tissue-like roll-off.
    let slope_db_per_hz = -1.5 / 1.0e6;
    let sample_rf = synth_rf(48, 512, |f| 10.0_f64.powf(slope_db_per_hz * f / 20.0));
    let reference_rf = synth_rf(48, 512, |_| 1.0);

    let sample = gated_spectrum(sample_rf.view(), 64..320, SAMPLE_RATE_HZ, FFT_SIZE).unwrap();
    let reference = gated_spectrum(reference_rf.view(), 64..320, SAMPLE_RATE_HZ, FFT_SIZE).unwrap();
    let normalized = normalize_to_reference(&sample, &reference).expect("normalize");

    let band = AnalysisBand::try_new(2.0e6, 8.0e6).expect("band");
    let params = backscatter_parameters(&normalized, frequency_step(), band).expect("fit");

    // Tolerance: the sample and reference are independent speckle realizations,
    // so each bin's normalized value carries the ratio of two chi-square-like
    // estimates averaged over 48 lines. The residual scatter is a few tenths of
    // a dB per bin; regressed over ~39 bins spanning 6 MHz, that leaves a slope
    // uncertainty well under 10% of the imposed value. Asserting 10% therefore
    // detects a wrong formula or a sign error while tolerating speckle.
    let recovered = params.slope_db_per_hz;
    let expected = -slope_db_per_hz; // negation convention
    assert!(
        (recovered - expected).abs() < 0.1 * expected.abs(),
        "slope: recovered {recovered:e}, expected {expected:e}"
    );
    assert!(
        recovered > 0.0,
        "a decreasing spectrum must give a positive (negated) slope, got {recovered}"
    );
}

/// Doubling the sample's power at every frequency must raise the midband fit by
/// exactly 10·log₁₀(2) ≈ 3.0103 dB and leave the slope unchanged. This
/// separates the intercept/midband axis from the slope axis: a formula that
/// conflated them would move both.
#[test]
fn uniform_gain_moves_midband_but_not_slope() {
    let reference_rf = synth_rf(32, 512, |_| 1.0);
    // Amplitude x sqrt(2) is power x 2.
    let sample_rf = synth_rf(32, 512, |_| std::f64::consts::SQRT_2);

    let sample = gated_spectrum(sample_rf.view(), 64..320, SAMPLE_RATE_HZ, FFT_SIZE).unwrap();
    let reference = gated_spectrum(reference_rf.view(), 64..320, SAMPLE_RATE_HZ, FFT_SIZE).unwrap();
    let normalized = normalize_to_reference(&sample, &reference).unwrap();

    let band = AnalysisBand::try_new(2.0e6, 8.0e6).unwrap();
    let params = backscatter_parameters(&normalized, frequency_step(), band).unwrap();

    let expected_db = 10.0 * 2.0_f64.log10();
    assert!(
        (params.midband_db - expected_db).abs() < 1.0e-9,
        "midband must rise by {expected_db} dB, got {}",
        params.midband_db
    );
    assert!(
        params.slope_db_per_hz.abs() < 1.0e-12,
        "a frequency-independent gain must not tilt the spectrum, got {}",
        params.slope_db_per_hz
    );
}

/// Bins where the reference carries no power are excluded, not floored: a fit
/// over a band that runs past the reference's support must use only the
/// supported bins and report that count.
#[test]
fn unsupported_reference_bins_are_excluded_from_the_fit() {
    let sample = Array1::from_vec([5], vec![1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    // Reference has no power in the last two bins.
    let reference = Array1::from_vec([5], vec![1.0, 1.0, 1.0, 0.0, 0.0]).unwrap();
    let normalized = normalize_to_reference(&sample, &reference).unwrap();

    assert!(normalized[[3]].is_infinite() && normalized[[3]] < 0.0);

    let step = 1.0e6;
    let band = AnalysisBand::try_new(0.0, 4.0e6).unwrap();
    let params = backscatter_parameters(&normalized, step, band).unwrap();
    assert_eq!(
        params.bins_used, 3,
        "only the reference-supported bins may enter the fit"
    );
    assert!(params.midband_db.abs() < 1.0e-15);
}

#[test]
fn fit_refuses_a_band_with_fewer_than_two_usable_bins() {
    let sample = Array1::from_vec([4], vec![1.0, 1.0, 1.0, 1.0]).unwrap();
    let reference = Array1::from_vec([4], vec![1.0, 0.0, 0.0, 0.0]).unwrap();
    let normalized = normalize_to_reference(&sample, &reference).unwrap();
    let band = AnalysisBand::try_new(0.0, 3.0e6).unwrap();
    assert!(backscatter_parameters(&normalized, 1.0e6, band).is_err());
}

#[test]
fn rejects_invalid_bands_gates_and_rates() {
    assert!(AnalysisBand::try_new(5.0e6, 5.0e6).is_err());
    assert!(AnalysisBand::try_new(6.0e6, 5.0e6).is_err());
    assert!(AnalysisBand::try_new(-1.0, 5.0e6).is_err());
    assert!(AnalysisBand::try_new(f64::NAN, 5.0e6).is_err());

    let rf = synth_rf(4, 128, |_| 1.0);
    // Empty gate.
    assert!(gated_spectrum(rf.view(), 10..10, SAMPLE_RATE_HZ, FFT_SIZE).is_err());
    // Gate past the end.
    assert!(gated_spectrum(rf.view(), 100..200, SAMPLE_RATE_HZ, FFT_SIZE).is_err());
    // Non-positive sample rate.
    assert!(gated_spectrum(rf.view(), 0..64, 0.0, FFT_SIZE).is_err());

    let a = Array1::from_vec([3], vec![1.0, 1.0, 1.0]).unwrap();
    let b = Array1::from_vec([2], vec![1.0, 1.0]).unwrap();
    assert!(normalize_to_reference(&a, &b).is_err());
}
