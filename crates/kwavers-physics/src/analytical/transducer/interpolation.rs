//! Bandlimited interpolation stencil for transducer grid-to-field mapping.

use kwavers_core::constants::numerical::TWO_PI;
use std::f64::consts::PI;

/// Compute bandlimited interpolation (BLI) stencil weights for fractional
/// grid offsets δ ∈ [0, 1).
///
/// Each weight set of length `n_stencil` (must be even) is a windowed-sinc
/// kernel centered at the nearest grid point:
/// ```text
/// w_j(δ) = sinc(j − δ) · hamming_window(j, N_stencil)
/// ```
/// for j = −N/2, …, N/2 − 1.
///
/// # Arguments
/// * `delta` – fractional offsets [0, 1) for each output sample
/// * `n_stencil` – stencil length (must be even; typical values 4, 8, 16)
///
/// # Reference
/// Schafer & Rabiner (1973), *Proc. IEEE* 61, 692.
#[must_use]
pub fn bli_stencil_weights(delta: &[f64], n_stencil: usize) -> Vec<Vec<f64>> {
    assert!(n_stencil.is_multiple_of(2), "n_stencil must be even");
    let half = (n_stencil / 2) as i64;
    let nm1 = (n_stencil - 1) as f64;
    delta
        .iter()
        .map(|&d| {
            let mut w: Vec<f64> = (0..n_stencil)
                .map(|j| {
                    let j_off = (j as i64 - half) as f64; // relative sample index
                    let x = j_off - d;
                    let sinc = if x.abs() < 1e-12 {
                        1.0
                    } else {
                        (PI * x).sin() / (PI * x)
                    };
                    // Hamming window over the stencil
                    let window = 0.54 - 0.46 * (TWO_PI * j as f64 / nm1).cos();
                    sinc * window
                })
                .collect();
            // Normalise so weights sum to 1 (preserve DC)
            let sum: f64 = w.iter().sum();
            if sum.abs() > 1e-15 {
                w.iter_mut().for_each(|x| *x /= sum);
            }
            w
        })
        .collect()
}

/// RMS interpolation error curves for nearest-neighbour and BLI interpolation.
///
/// For each points-per-wavelength value, this evaluates the unit sinusoid
/// `sin(2πx / PPW)` at fractional offsets `delta`, compares nearest-neighbour
/// zero-order reconstruction against the ideal value, and compares the
/// BLI-reconstructed value from [`bli_stencil_weights`] against the same ideal.
///
/// Returns `(nearest_rms, bli_rms)`, each with length `ppw.len()`. Returns empty
/// vectors when axes are empty, `n_stencil` is odd/zero, or any sampled value is
/// non-finite/non-positive where positivity is required.
#[must_use]
pub fn bli_interpolation_error_curves(
    ppw: &[f64],
    delta: &[f64],
    n_stencil: usize,
) -> (Vec<f64>, Vec<f64>) {
    if ppw.is_empty()
        || delta.is_empty()
        || n_stencil == 0
        || !n_stencil.is_multiple_of(2)
        || ppw.iter().any(|&v| !v.is_finite() || v <= 0.0)
        || delta.iter().any(|&v| !v.is_finite())
    {
        return (Vec::new(), Vec::new());
    }

    let weights = bli_stencil_weights(delta, n_stencil);
    let half = (n_stencil / 2) as i64;
    let offsets: Vec<f64> = (0..n_stencil).map(|j| (j as i64 - half) as f64).collect();
    let inv_count = 1.0 / delta.len() as f64;

    let mut nearest_rms = Vec::with_capacity(ppw.len());
    let mut bli_rms = Vec::with_capacity(ppw.len());
    for &points_per_wavelength in ppw {
        let phase = TWO_PI / points_per_wavelength;
        let samples: Vec<f64> = offsets
            .iter()
            .map(|&offset| (phase * offset).sin())
            .collect();
        let mut nearest_sum_sq = 0.0_f64;
        let mut bli_sum_sq = 0.0_f64;

        for (&fractional_offset, row) in delta.iter().zip(weights.iter()) {
            let ideal = (phase * fractional_offset).sin();
            nearest_sum_sq += ideal * ideal;
            let reconstructed = row
                .iter()
                .zip(samples.iter())
                .map(|(&weight, &sample)| weight * sample)
                .sum::<f64>();
            let error = reconstructed - ideal;
            bli_sum_sq += error * error;
        }

        nearest_rms.push((nearest_sum_sq * inv_count).sqrt());
        bli_rms.push((bli_sum_sq * inv_count).sqrt());
    }

    (nearest_rms, bli_rms)
}
