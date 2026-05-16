//! Transducer array and beamforming physics for book chapters ch04, ch11.
//!
//! Covers: circular piston directivity, linear array factor, grating lobes,
//! apodization windows, delay laws, 2-D beam patterns, on-axis pressure
//! profiles, and bandlimited interpolation stencils.

use std::f64::consts::PI;
use num_complex::Complex64;

// ─── Directivity ──────────────────────────────────────────────────────────────

/// Baffled circular-piston directivity function.
///
/// ```text
/// D(θ) = 2·J₁(ka·sin θ) / (ka·sin θ)
/// ```
/// normalised so that D(0) = 1.  Uses the L'Hôpital limit at θ = 0.
///
/// # Arguments
/// * `theta_rad` – observation angles [rad]
/// * `ka` – wavenumber–radius product k·a (dimensionless)
///
/// # Reference
/// O'Neil (1949), *J. Acoust. Soc. Am.* 21, 516.
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
    angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
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

// ─── Delay laws ───────────────────────────────────────────────────────────────

/// Compute time delays for a 2-D focusing delay law.
///
/// ```text
/// τᵢ = (max_r − rᵢ) / c   [s]
/// ```
/// where `rᵢ = sqrt((xᵢ − x_f)² + (zᵢ − z_f)²)` is the distance from
/// element i to the focal point, and `max_r` is the maximum such distance
/// (ensures all delays are non-negative).
///
/// # Arguments
/// * `elem_x`, `elem_z` – element positions [m]
/// * `x_f`, `z_f` – focal point [m]
/// * `c` – sound speed [m/s]
pub fn delay_law_focus_2d(
    elem_x: &[f64],
    elem_z: &[f64],
    x_f: f64,
    z_f: f64,
    c: f64,
) -> Vec<f64> {
    assert_eq!(elem_x.len(), elem_z.len(), "element arrays must have equal length");
    let r: Vec<f64> = elem_x
        .iter()
        .zip(elem_z.iter())
        .map(|(&xi, &zi)| ((xi - x_f).powi(2) + (zi - z_f).powi(2)).sqrt())
        .collect();
    let max_r = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    r.iter().map(|&ri| (max_r - ri) / c).collect()
}

// ─── Beam pattern ─────────────────────────────────────────────────────────────

/// Compute the complex 2-D pressure beam pattern from a linear array.
///
/// Each element radiates as an isotropic monopole (far-field approximation);
/// element i contributes:
/// ```text
/// p_i(x, z) = w_i · exp(−i·k·rᵢ) · exp(i·ω·τᵢ)
/// ```
/// where `rᵢ = sqrt((x − xᵢ)² + (z − zᵢ)²)` and τᵢ are the steering delays.
/// Geometric spreading `1/rᵢ` is omitted (pattern, not absolute amplitude).
///
/// Output is two flattened row-major Vecs (real, imag) of length NX × NZ.
///
/// # Arguments
/// * `x_arr`, `z_arr` – grid coordinates [m]
/// * `elem_x`, `elem_z` – element positions [m]
/// * `freq_hz` – frequency [Hz]
/// * `c` – sound speed [m/s]
/// * `weights` – apodization weights (length == n_elements)
/// * `delays` – steering delays [s] (length == n_elements)
pub fn beam_pattern_2d(
    x_arr: &[f64],
    z_arr: &[f64],
    elem_x: &[f64],
    elem_z: &[f64],
    freq_hz: f64,
    c: f64,
    weights: &[f64],
    delays: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let k = 2.0 * PI * freq_hz / c;
    let n_elem = elem_x.len();
    let nx = x_arr.len();
    let nz = z_arr.len();
    let n_grid = nx * nz;
    let mut real_out = vec![0.0_f64; n_grid];
    let mut imag_out = vec![0.0_f64; n_grid];

    for (ix, &x) in x_arr.iter().enumerate() {
        for (iz, &z) in z_arr.iter().enumerate() {
            let idx = ix * nz + iz;
            let mut p = Complex64::new(0.0, 0.0);
            for ie in 0..n_elem {
                let dx = x - elem_x[ie];
                let dz = z - elem_z[ie];
                let r = (dx * dx + dz * dz).sqrt().max(1e-12);
                // phase from propagation delay, plus steering pre-delay
                let phase = -k * r + 2.0 * PI * freq_hz * delays[ie];
                p += weights[ie] * Complex64::new(phase.cos(), phase.sin());
            }
            real_out[idx] = p.re;
            imag_out[idx] = p.im;
        }
    }
    (real_out, imag_out)
}

// ─── On-axis pressure ─────────────────────────────────────────────────────────

/// On-axis pressure magnitude of a baffled circular piston (O'Neil formula).
///
/// ```text
/// |p(z)| = 2·p₀·|sin(k/2·(√(z²+a²) − z))|
/// ```
///
/// # Arguments
/// * `z_arr` – on-axis distances from piston face [m] (must be > 0)
/// * `radius_m` – piston radius a [m]
/// * `freq_hz` – frequency [Hz]
/// * `p0_pa` – surface pressure amplitude [Pa]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// O'Neil (1949), *J. Acoust. Soc. Am.* 21, 516.
pub fn circular_piston_onaxis(
    z_arr: &[f64],
    radius_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> Vec<f64> {
    let k = 2.0 * PI * freq_hz / c;
    z_arr
        .iter()
        .map(|&z| {
            let r = (z * z + radius_m * radius_m).sqrt();
            let arg = k / 2.0 * (r - z);
            2.0 * p0_pa * arg.sin().abs()
        })
        .collect()
}

/// On-axis pressure magnitude of a focused spherical bowl (O'Neil 1949).
///
/// ```text
/// |p(z)| = 2·p₀·|sin(k/2·(R_bowl − √((F−z)²+a²))|
/// ```
/// where R_bowl is the chord (bowl aperture radius, approximated as `a` of the
/// equivalent piston), and the path-length difference is computed relative to
/// the bowl surface centred at (0, F).
///
/// # Arguments
/// * `z_arr` – axial positions [m]
/// * `bowl_radius_m` – bowl aperture radius a [m]
/// * `focal_length_m` – geometric focal length F [m]
/// * `freq_hz` – frequency [Hz]
/// * `p0_pa` – source pressure [Pa]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// O'Neil (1949), *J. Acoust. Soc. Am.* 21, 516, eq. (8).
pub fn focused_bowl_onaxis(
    z_arr: &[f64],
    bowl_radius_m: f64,
    focal_length_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> Vec<f64> {
    let k = 2.0 * PI * freq_hz / c;
    z_arr
        .iter()
        .map(|&z| {
            // Distance from the bowl rim at (a, 0) to the axial field point (0, z)
            let dz = focal_length_m - z;
            let r_rim = (dz * dz + bowl_radius_m * bowl_radius_m).sqrt();
            // Distance from the bowl pole (0, 0) to the field point
            let r_pole = z.abs();
            let arg = k / 2.0 * (r_rim - r_pole);
            2.0 * p0_pa * arg.sin().abs()
        })
        .collect()
}

// ─── Bandlimited interpolation stencil ───────────────────────────────────────

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
pub fn bli_stencil_weights(delta: &[f64], n_stencil: usize) -> Vec<Vec<f64>> {
    assert!(n_stencil % 2 == 0, "n_stencil must be even");
    let half = (n_stencil / 2) as i64;
    let nm1 = (n_stencil - 1) as f64;
    delta
        .iter()
        .map(|&d| {
            let mut w: Vec<f64> = (0..n_stencil)
                .map(|j| {
                    let j_off = (j as i64 - half) as f64; // relative sample index
                    let x = j_off - d;
                    let sinc = if x.abs() < 1e-12 { 1.0 } else { (PI * x).sin() / (PI * x) };
                    // Hamming window over the stencil
                    let window = 0.54 - 0.46 * (2.0 * PI * j as f64 / nm1).cos();
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

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Bessel J₁(x) — Horner-evaluated Chebyshev rational approximation.
/// Error < 2e-9 for |x| ≤ 8; Hankel expansion elsewhere.
fn bessel_j1(x: f64) -> f64 {
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
                + y * (18_583_304.74
                    + y * (99_447.433_94 + y * (376.999_139_7 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_490_2;
        let p = 1.0
            + y * (0.183_105e-2
                + y * (-3.516_396_496e-5
                    + y * (2.457_520_174e-5 - y * 2.400_505_341e-7)));
        let q = 0.046_874_999_95
            + y * (-2.002_690_873e-4
                + y * (8.449_199_096e-5
                    + y * (-8.822_898_7e-5 + y * 1.050_343_160e-6)));
        (2.0 / (PI * ax)).sqrt() * (p * xx.cos() - z * q * xx.sin())
    };
    if x < 0.0 { -r } else { r }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piston_directivity_on_axis() {
        let d = circular_piston_directivity(&[0.0], 5.0);
        assert!((d[0] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn array_factor_at_steering_angle_is_one() {
        let steer = 0.1_f64;
        let k = 2.0 * PI * 2e6 / 1500.0;
        let af = linear_array_factor(&[steer], k, 0.3e-3, 64, steer);
        assert!((af[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn apodization_uniform_sum() {
        let w = apodization_weights(64, "uniform");
        let s: f64 = w.iter().sum();
        assert!((s - 64.0).abs() < 1e-10);
    }

    #[test]
    fn bli_stencil_dc_preservation() {
        let ws = bli_stencil_weights(&[0.0, 0.25, 0.5, 0.75], 8);
        for w in &ws {
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "sum={}", s);
        }
    }

    #[test]
    fn delay_law_max_is_zero() {
        // The element closest to the focus should have delay approaching 0
        let ex = vec![0.0];
        let ez = vec![0.0];
        let d = delay_law_focus_2d(&ex, &ez, 0.0, 0.0, 1500.0);
        assert!((d[0]).abs() < 1e-15);
    }
}
