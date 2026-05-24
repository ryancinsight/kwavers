//! Array-factor and directivity physics for transducer arrays.
//!
//! Covers: circular-piston directivity (O'Neil 1949), linear array factor,
//! grating-lobe prediction, and apodization windows.

use std::f64::consts::PI;

// ─── Directivity ──────────────────────────────────────────────────────────────

/// Baffled circular-piston directivity function.
///
/// ```text
/// D(θ) = 2·J₁(ka·sin θ) / (ka·sin θ)
/// ```
/// Normalised so D(0) = 1.  Uses the L'Hôpital limit at θ = 0.
///
/// # Arguments
/// * `theta_rad` – observation angles [rad]
/// * `ka` – wavenumber–radius product k·a (dimensionless)
///
/// # Reference
/// O'Neil (1949), *J. Acoust. Soc. Am.* 21, 516.
#[must_use]
pub fn circular_piston_directivity(theta_rad: &[f64], ka: f64) -> Vec<f64> {
    theta_rad
        .iter()
        .map(|&theta| {
            let s = theta.sin();
            let arg = ka * s;
            if arg.abs() < 1e-12 {
                1.0
            } else {
                2.0 * bessel_j1(arg) / arg
            }
        })
        .collect()
}

// ─── Array factor ─────────────────────────────────────────────────────────────

/// Normalised linear array factor for a uniform linear array of N elements.
///
/// ```text
/// AF(θ) = sin(N·ψ/2) / (N·sin(ψ/2))
/// ψ = k·d·(sin θ − sin θ_steer)
/// ```
/// Normalised to 1.0 at the steering angle (L'Hôpital limit applied).
///
/// # Arguments
/// * `theta_rad` – observation angles [rad]
/// * `k` – wavenumber [rad/m]
/// * `d_m` – element pitch [m]
/// * `n` – number of elements
/// * `steer_rad` – steering angle [rad]
///
/// # Reference
/// Van Trees (2002) *Optimum Array Processing*, §2.2.
#[must_use]
pub fn linear_array_factor(
    theta_rad: &[f64],
    k: f64,
    d_m: f64,
    n: usize,
    steer_rad: f64,
) -> Vec<f64> {
    let n_f = n as f64;
    theta_rad
        .iter()
        .map(|&theta| {
            let psi = k * d_m * (theta.sin() - steer_rad.sin());
            let half_psi = psi / 2.0;
            if half_psi.abs() < 1e-12 {
                1.0
            } else {
                (n_f * half_psi).sin() / (n_f * half_psi.sin())
            }
        })
        .collect()
}

/// Grating-lobe angles for a linear array steered to `steer_rad`.
///
/// Grating lobes occur where:
/// ```text
/// sin θ_g = sin θ_steer ± m·λ/d,  m = 1, 2, …
/// ```
/// Returns all angles in rad within the visible range [−π/2, π/2].
///
/// # Arguments
/// * `k` – wavenumber [rad/m]
/// * `d_m` – element pitch [m]
/// * `steer_rad` – steering angle [rad]
#[must_use]
pub fn grating_lobe_angles(k: f64, d_m: f64, steer_rad: f64) -> Vec<f64> {
    let lambda_over_d = 2.0 * PI / (k * d_m);
    let sin_steer = steer_rad.sin();
    let mut angles = Vec::new();
    for m in 1_i32..=10 {
        for sign in [-1_i32, 1_i32] {
            let sin_g = sin_steer + sign as f64 * m as f64 * lambda_over_d;
            if sin_g.abs() <= 1.0 {
                angles.push(sin_g.asin());
            }
        }
    }
    angles.sort_by(|a, b| a.total_cmp(b));
    angles.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
    angles
}

// ─── Apodization ──────────────────────────────────────────────────────────────

/// Apodization (window) weights for an N-element array.
///
/// Supported window types:
/// * `"uniform"` – all ones (rectangular window)
/// * `"hamming"` – Hamming window
/// * `"hann"` – Hann (von Hann) window
/// * `"blackman"` – Blackman window
/// * `"tukey25"` – Tukey window with α = 0.25
///
/// Unknown types fall back to uniform.
///
/// # Reference
/// Harris (1978), *Proc. IEEE* 66, 51.
#[must_use]
pub fn apodization_weights(n: usize, window_type: &str) -> Vec<f64> {
    let nm1 = (n - 1) as f64;
    match window_type {
        "uniform" => vec![1.0; n],
        "hamming" => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / nm1).cos())
            .collect(),
        "hann" => (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / nm1).cos()))
            .collect(),
        "blackman" => (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f64 / nm1;
                0.42 - 0.5 * t.cos() + 0.08 * (2.0 * t).cos()
            })
            .collect(),
        "tukey25" => {
            let alpha = 0.25_f64;
            (0..n)
                .map(|i| {
                    let x = i as f64 / nm1;
                    if x < alpha / 2.0 {
                        0.5 * (1.0 - (2.0 * PI * x / alpha).cos())
                    } else if x <= 1.0 - alpha / 2.0 {
                        1.0
                    } else {
                        0.5 * (1.0 - (2.0 * PI * (x - 1.0 + alpha / 2.0) / alpha).cos())
                    }
                })
                .collect()
        }
        _ => vec![1.0; n], // fallback: uniform
    }
}

// ─── Internal helper (used by directivity) ────────────────────────────────────

/// Bessel J₁(x) — Horner-evaluated Chebyshev rational approximation.
/// Error < 2e-9 for |x| ≤ 8; Hankel expansion elsewhere.
pub(super) fn bessel_j1(x: f64) -> f64 {
    let ax = x.abs();
    let r = if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72_362_614_232.0
                + y * (-7_895_059_235.0
                    + y * (242_396_853.1
                        + y * (-2_972_611.439 + y * (15_704.482_60 + y * (-30.160_366_06))))));
        let den = 144_725_228_442.0
            + y * (2_300_535_178.0
                + y * (18_583_304.74 + y * (99_447.433_94 + y * (376.999_139_7 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_490_2;
        let p = 1.0
            + y * (0.183_105e-2
                + y * (-3.516_396_496e-5 + y * (2.457_520_174e-5 - y * 2.400_505_341e-7)));
        let q = 0.046_874_999_95
            + y * (-2.002_690_873e-4
                + y * (8.449_199_096e-5 + y * (-8.822_898_7e-5 + y * 1.050_343_160e-6)));
        (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    };
    if x < 0.0 { -r } else { r }
}
