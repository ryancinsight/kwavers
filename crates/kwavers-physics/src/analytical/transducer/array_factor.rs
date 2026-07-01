//! Array-factor and directivity physics for transducer arrays.
//!
//! Covers: circular-piston directivity (O'Neil 1949), linear array factor,
//! grating-lobe prediction, and apodization windows.

use kwavers_core::constants::numerical::TWO_PI;
use kwavers_math::fft::fft_1d_array;
use kwavers_math::signal::ApodizationType;
use kwavers_math::special::bessel::j1;
use ndarray::Array1;

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
                2.0 * j1(arg) / arg
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

/// Full far-field beam-pattern magnitude via the pattern-multiplication theorem.
///
/// The radiation pattern of an array of identical directional elements is the
/// product of the element directivity and the array factor:
/// ```text
/// |P(θ)| = |D(θ)| · |AF(θ)|
/// ```
/// where `D` is the baffled circular-piston directivity ([`circular_piston_directivity`])
/// with parameter `ka_elem = k·a_elem`, and `AF` is the linear array factor
/// ([`linear_array_factor`]). The returned magnitude is normalised to its peak
/// across the supplied angle set.
///
/// # Arguments
/// * `theta_rad` – observation angles [rad]
/// * `k` – wavenumber [rad/m]
/// * `d_m` – element pitch [m]
/// * `n` – number of elements
/// * `steer_rad` – steering angle [rad]
/// * `ka_elem` – element directivity parameter k·a_elem (dimensionless)
///
/// # Returns
/// Normalised pattern magnitude (linear), peak = 1.
///
/// # Reference
/// Steinberg (1976) *Principles of Aperture and Array System Design*, §2 (pattern
/// multiplication); Van Trees (2002) §2.2.
#[must_use]
pub fn beam_pattern_magnitude(
    theta_rad: &[f64],
    k: f64,
    d_m: f64,
    n: usize,
    steer_rad: f64,
    ka_elem: f64,
) -> Vec<f64> {
    let af = linear_array_factor(theta_rad, k, d_m, n, steer_rad);
    let d = circular_piston_directivity(theta_rad, ka_elem);
    let mut mag: Vec<f64> = af
        .iter()
        .zip(d.iter())
        .map(|(&a, &de)| (a * de).abs())
        .collect();
    let peak = mag.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    for m in &mut mag {
        *m /= peak;
    }
    mag
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
    let lambda_over_d = TWO_PI / (k * d_m);
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
/// Supported window keys:
/// * `"uniform"` – all ones (rectangular window)
/// * `"hamming"` – Hamming window
/// * `"hann"` – Hann (von Hann) window
/// * `"blackman"` – Blackman window
/// * `"tukey25"` – Tukey window with cosine fraction r = 0.25
///
/// Unknown keys fall back to uniform.
///
/// The window math is **not** reimplemented here: this is a thin string-keyed
/// wrapper over the canonical SSOT `kwavers_math::signal::ApodizationType`
/// (which in turn delegates to `kwavers_math::signal::window`), so the
/// coefficients live in exactly one place and the symmetric (N−1) convention,
/// endpoint tapering, and `n ≤ 1` handling are shared with the rest of the
/// workspace.
///
/// # Reference
/// Harris (1978), *Proc. IEEE* 66, 51.
#[must_use]
pub fn apodization_weights(n: usize, window_type: &str) -> Vec<f64> {
    let window = match window_type {
        "hamming" => ApodizationType::Hamming,
        "hann" => ApodizationType::Hanning,
        "blackman" => ApodizationType::Blackman,
        "tukey25" => ApodizationType::Tukey { r: 0.25 },
        // "uniform" and any unrecognised key → rectangular window.
        _ => ApodizationType::Uniform,
    };
    window.weights(n)
}

/// Apodization weights and normalized spatial-frequency response.
///
/// Returns apodization weights, cycles per aperture, and normalized response,
/// where `response_db = 20·log10(|FFT(w)| / max(|FFT(w)|) + 1e-12)` after FFT-shift.
/// The frequency axis matches the Chapter 4 plotting convention:
/// `linspace(-0.5, 0.5, nfft) * n_elements`.
///
/// # Errors
/// Returns an error if the number of elements is zero, the FFT length is zero,
/// or the FFT length is shorter than the apodization window.
#[derive(Debug, Clone, PartialEq)]
pub struct ApodizationWindowResponse {
    /// Apodization weights.
    pub weights: Vec<f64>,
    /// Shifted spatial frequency axis in cycles per aperture.
    pub cycles_per_aperture: Vec<f64>,
    /// Normalized frequency response in decibels.
    pub response_db: Vec<f64>,
}

pub fn apodization_window_response(
    n_elements: usize,
    window_type: &str,
    nfft: usize,
) -> Result<ApodizationWindowResponse, String> {
    if n_elements == 0 {
        return Err("n_elements must be positive".to_owned());
    }
    if nfft == 0 {
        return Err("nfft must be positive".to_owned());
    }
    if nfft < n_elements {
        return Err("nfft must be at least n_elements".to_owned());
    }

    let weights = apodization_weights(n_elements, window_type);
    let mut padded = vec![0.0; nfft];
    padded[..n_elements].copy_from_slice(&weights);
    let spectrum = fft_1d_array(&Array1::from_vec(padded));
    let shift = nfft / 2;
    let magnitudes: Vec<f64> = (0..nfft)
        .map(|idx| spectrum[(idx + shift) % nfft].norm())
        .collect();
    let peak = magnitudes
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1.0e-300);
    let response_db = magnitudes
        .iter()
        .map(|&magnitude| 20.0 * ((magnitude / peak) + 1.0e-12).log10())
        .collect();
    let denom = (nfft.saturating_sub(1)).max(1) as f64;
    let cycles_per_aperture = (0..nfft)
        .map(|idx| (-0.5 + idx as f64 / denom) * n_elements as f64)
        .collect();

    Ok(ApodizationWindowResponse {
        weights,
        cycles_per_aperture,
        response_db,
    })
}

// J₁ for circular-piston directivity is the workspace SSOT
// `kwavers_math::special::bessel::j1` (imported above).
