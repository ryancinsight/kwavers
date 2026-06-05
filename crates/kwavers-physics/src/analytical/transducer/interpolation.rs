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
