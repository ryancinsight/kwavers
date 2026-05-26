//! Beam pattern and on-axis pressure computation for transducer arrays.
//!
//! Covers: 2-D complex beam pattern (far-field monopole), focusing delay laws,
//! on-axis pressure of circular pistons, and focused spherical bowls.

use num_complex::Complex64;
use crate::core::constants::numerical::{TWO_PI};

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
#[must_use]
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
#[must_use]
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
    let k = TWO_PI * freq_hz / c;
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
                let phase = -k * r + TWO_PI * freq_hz * delays[ie];
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
#[must_use]
pub fn circular_piston_onaxis(
    z_arr: &[f64],
    radius_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> Vec<f64> {
    let k = TWO_PI * freq_hz / c;
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
#[must_use]
pub fn focused_bowl_onaxis(
    z_arr: &[f64],
    bowl_radius_m: f64,
    focal_length_m: f64,
    freq_hz: f64,
    p0_pa: f64,
    c: f64,
) -> Vec<f64> {
    let k = TWO_PI * freq_hz / c;
    z_arr
        .iter()
        .map(|&z| {
            let dz = focal_length_m - z;
            let r_rim = (dz * dz + bowl_radius_m * bowl_radius_m).sqrt();
            let r_pole = z.abs();
            let arg = k / 2.0 * (r_rim - r_pole);
            2.0 * p0_pa * arg.sin().abs()
        })
        .collect()
}
