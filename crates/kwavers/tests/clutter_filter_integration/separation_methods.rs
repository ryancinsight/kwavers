//! Value assertions for every adaptive-filter separation method on the
//! synthetic functional-ultrasound ensemble of the parent suite.

use super::generate_fus_data;
use kwavers_analysis::signal_processing::clutter_filter::{
    AdaptiveFilter, AdaptiveFilterConfig, SubspaceSeparationMethod,
};
use kwavers_core::error::KwaversResult;
use leto::Array2;
use std::f64::consts::PI;

/// Output and per-pixel clutter-to-blood ratio history of one adaptive-filter run.
struct AdaptiveRun {
    filtered: Array2<f64>,
    cbr_history: Vec<f64>,
}

fn run_adaptive(
    data: &Array2<f64>,
    separation_method: SubspaceSeparationMethod,
) -> KwaversResult<AdaptiveRun> {
    let mut filter = AdaptiveFilter::new(AdaptiveFilterConfig {
        separation_method,
        ..Default::default()
    })?;
    let filtered = filter.filter(data)?;
    Ok(AdaptiveRun {
        filtered,
        cbr_history: filter.cbr_history().to_vec(),
    })
}

fn pixel_row(matrix: &Array2<f64>, pixel: usize) -> Vec<f64> {
    (0..matrix.shape()[1]).map(|t| matrix[[pixel, t]]).collect()
}

fn energy(signal: &[f64]) -> f64 {
    signal.iter().map(|v| v * v).sum()
}

fn row_energy(matrix: &Array2<f64>, pixel: usize) -> f64 {
    energy(&pixel_row(matrix, pixel))
}

fn output_gap(a: &AdaptiveRun, b: &AdaptiveRun, pixel: usize) -> f64 {
    max_abs_diff(
        &pixel_row(&a.filtered, pixel),
        &pixel_row(&b.filtered, pixel),
    )
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

/// Every separation method on one synthetic ensemble, asserted against the
/// eigen-structure its generator parameters imply.
///
/// # Derivation
///
/// Each pixel is `x = A sin(w_t k) + B (1 + i/10) sin(w_b k)` over `n = 120`
/// frames, `A = 5`, `B = 0.5`, `w_t = 2 pi 0.02`, `w_b = 2 pi 0.15`. The filter
/// decomposes the lag-product covariance `R[j, k] = r(|j - k|)`,
/// `r(l) = mean_t x_t x_(t+l)`. Two facts follow.
///
/// 1. `trace R = n r(0) = sum_t x_t^2 = |x|^2` exactly, so the eigenvalues sum
///    to the pixel energy.
/// 2. A sinusoid of power `P = a^2 / 2` contributes `P cos(w l)` to `r(l)`, a
///    rank-2 matrix with two eigenvalues `n P / 2 = n a^2 / 4`. The tissue pair
///    is `lambda_T = n A^2 / 4 = 750`; the blood pair is
///    `lambda_B(i) = n B^2 (1 + i/10)^2 / 4 = 7.5 (1 + i/10)^2`, from 7.5
///    (pixel 0) to 114 (pixel 29). The unbiased lag mean leaves a window
///    term `-(a^2/2) mean_t cos(w (2t + l))` in `r(l)`, relatively at most
///    `1 / ((n/2) sin w)`; for the tissue tone that is
///    `LEAKAGE = 1 / (60 sin(2 pi 0.02)) = 0.133`. It moves the tissue pair by
///    `LEAKAGE` and leaves tissue energy of at most `LEAKAGE |T|` outside the
///    top-2 eigenspace; the blood tone, 15.6 frequency bins away, leaks
///    negligibly.
///
/// The filter removes the projection onto the `r` leading eigenvectors, so
/// `|f|^2 = |x|^2 - sum_(i<r) <x, e_i>^2` falls as `r` grows: the filtered
/// output of a larger rank is the output of a smaller rank minus further
/// components.
///
/// ## FixedRank { 2 }
///
/// Rank 2 removes the tissue pair and passes the blood tone:
/// `0.5 |b|^2 <= |f|^2 <= (|b| + LEAKAGE |T|)^2` for every pixel, with `|b|^2`,
/// `|T|^2` the generator's blood and tissue energies. The upper bound is the
/// blood tone plus the tissue energy outside the top-2 eigenspace; the lower
/// bound is conservative against the blood energy the rotated subspace can
/// absorb, `(1 - |E| / gap)^2 ~ 0.7` with `|E| <= lambda_B(29) = 114` and
/// `gap >= 640`. Rank 1 leaves tissue energy above the upper bound at every
/// pixel (pixel 0: 14x `|b|^2` against a bound of 5.5x); rank 3 removes the
/// blood pair's leading component, so at pixels with a strong blood pair
/// (`i >= 15`) its energy falls below the lower bound (pixel 29: 0.24x
/// `|b|^2`). Energy falls
/// strictly with rank, and the removed and filtered parts are orthogonal,
/// `<x - f, f> = 0`, to the `n^2 eps` backward-error envelope of the
/// eigensolver.
///
/// ## AdaptiveThreshold { 0.1 }
///
/// The rank is the index of the first eigenvalue below `0.1 lambda_max`, where
/// `lambda_max ~ lambda_T` within `LEAKAGE`'s 15%: threshold 63.75 to 86.25.
/// The blood pair stays below it while `7.5 (1 + i/10)^2 < 63.75`, that is
/// pixels `i <= 19`, giving rank 2. It exceeds it, with a 10% allowance,
/// once `0.9 * 7.5 (1 + i/10)^2 > 86.25`, pixels `i >= 26`, giving rank 4
/// (the leaked tissue eigenvalue near 40 ends the run). Between, either
/// is possible, so the rank lies in `[2, 4]`.
///
/// ## CbrBased { 25 dB }
///
/// The rank is the smallest `r` with `CBR(r) = sum_(i<r) lambda_i /
/// sum_(i>=r) lambda_i <= 10^2.5 = 316`. By fact 1, `CBR(1) = lambda_1 /
/// (|x|^2 - lambda_1)` with `lambda_1` in `[0.85, 1.15] lambda_T`, which is
/// between 0.56 and 1.24 for every pixel, far below 316, so rank 1 is
/// selected everywhere and the recorded CBR is that value.
pub(super) fn check_all_methods() -> KwaversResult<()> {
    const N_PIXELS: usize = 30;
    const N_FRAMES: usize = 120;
    const TISSUE_AMPLITUDE: f64 = 5.0;
    const BLOOD_AMPLITUDE: f64 = 0.5;
    const TISSUE_FREQ: f64 = 0.02;
    const BLOOD_FREQ: f64 = 0.15;
    // Ensemble length as a float, and the n^2 eps eigensolver envelope.
    const N: f64 = 120.0;
    const ENVELOPE: f64 = N * N * f64::EPSILON;
    // lambda_T relative uncertainty from the window term (derivation above).
    const TISSUE_SPREAD: f64 = 0.15;
    const TARGET_CBR_DB: f64 = 25.0;

    let (data, tissue, blood) = generate_fus_data(
        N_PIXELS,
        N_FRAMES,
        TISSUE_AMPLITUDE,
        BLOOD_AMPLITUDE,
        TISSUE_FREQ,
        BLOOD_FREQ,
    );
    let leakage = 1.0 / (0.5 * N * (2.0 * PI * TISSUE_FREQ).sin());
    let lambda_tissue = N * TISSUE_AMPLITUDE * TISSUE_AMPLITUDE / 4.0;

    let rank1 = run_adaptive(
        &data,
        SubspaceSeparationMethod::FixedRank { clutter_rank: 1 },
    )?;
    let rank2 = run_adaptive(
        &data,
        SubspaceSeparationMethod::FixedRank { clutter_rank: 2 },
    )?;
    let rank4 = run_adaptive(
        &data,
        SubspaceSeparationMethod::FixedRank { clutter_rank: 4 },
    )?;
    let threshold = run_adaptive(
        &data,
        SubspaceSeparationMethod::AdaptiveThreshold { decay_factor: 0.1 },
    )?;
    let cbr_based = run_adaptive(
        &data,
        SubspaceSeparationMethod::CbrBased {
            target_cbr_db: TARGET_CBR_DB,
        },
    )?;

    for run in [&rank1, &rank2, &rank4, &threshold, &cbr_based] {
        assert_eq!(run.filtered.shape(), data.shape());
        assert_eq!(
            run.cbr_history.len(),
            N_PIXELS,
            "one CBR estimate per pixel"
        );
    }

    for run in [&rank1, &rank2, &rank4, &threshold, &cbr_based] {
        for pixel in 0..N_PIXELS {
            assert!(
                row_energy(&run.filtered, pixel) < row_energy(&data, pixel),
                "pixel {pixel}: filtering must reduce power"
            );
        }
    }

    // FixedRank { 2 }: the tissue pair is removed and the blood tone kept.
    for pixel in 0..N_PIXELS {
        let x_energy = row_energy(&data, pixel);
        let blood_energy = row_energy(&blood, pixel);
        let tissue_energy = row_energy(&tissue, pixel);
        let (f1, f2, f4) = (
            row_energy(&rank1.filtered, pixel),
            row_energy(&rank2.filtered, pixel),
            row_energy(&rank4.filtered, pixel),
        );
        assert!(
            f1 > f2 && f2 > f4,
            "pixel {pixel}: energy must fall with rank"
        );
        let upper = (blood_energy.sqrt() + leakage * tissue_energy.sqrt()).powi(2);
        assert!(
            (0.5 * blood_energy..=upper).contains(&f2),
            "pixel {pixel}: rank-2 energy {f2} outside [{}, {upper}]",
            0.5 * blood_energy
        );
        let x = pixel_row(&data, pixel);
        let filtered = pixel_row(&rank2.filtered, pixel);
        let removed_dot_filtered: f64 =
            x.iter().zip(&filtered).map(|(xv, fv)| (xv - fv) * fv).sum();
        assert!(
            removed_dot_filtered.abs() <= ENVELOPE * x_energy,
            "pixel {pixel}: removed and filtered parts must be orthogonal"
        );
    }

    // AdaptiveThreshold { 0.1 }: rank 2, or 4 once the blood pair clears the
    // threshold; nested projections bound the energy in between.
    for pixel in 0..N_PIXELS {
        let x_energy = row_energy(&data, pixel);
        let slack = ENVELOPE * x_energy;
        let selected = row_energy(&threshold.filtered, pixel);
        assert!(
            (row_energy(&rank4.filtered, pixel) - slack
                ..=row_energy(&rank2.filtered, pixel) + slack)
                .contains(&selected),
            "pixel {pixel}: adaptive rank outside [2, 4]"
        );
        let tolerance = ENVELOPE * x_energy.sqrt();
        if pixel <= 19 {
            assert!(
                output_gap(&threshold, &rank2, pixel) <= tolerance,
                "pixel {pixel}: threshold must select rank 2"
            );
        }
        if pixel >= 26 {
            assert!(
                output_gap(&threshold, &rank4, pixel) <= tolerance,
                "pixel {pixel}: threshold must select rank 4"
            );
        }
    }

    // CbrBased { 25 dB }: CBR(1) is far below the target, so rank 1.
    let target_cbr = 10.0_f64.powf(TARGET_CBR_DB / 10.0);
    let (lo, hi) = (
        (1.0 - TISSUE_SPREAD) * lambda_tissue,
        (1.0 + TISSUE_SPREAD) * lambda_tissue,
    );
    for pixel in 0..N_PIXELS {
        let x_energy = row_energy(&data, pixel);
        assert!(
            output_gap(&cbr_based, &rank1, pixel) <= ENVELOPE * x_energy.sqrt(),
            "pixel {pixel}: CBR selection must pick rank 1"
        );
        let cbr = cbr_based.cbr_history[pixel];
        assert!(
            (lo / (x_energy - lo)..=hi / (x_energy - hi)).contains(&cbr) && cbr <= target_cbr,
            "pixel {pixel}: CBR {cbr} outside the rank-1 interval"
        );
    }

    Ok(())
}
