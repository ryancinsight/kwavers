//! Beam pattern and on-axis pressure computation for transducer arrays.
//!
//! Covers: 2-D complex beam pattern (far-field monopole), focusing delay laws,
//! on-axis pressure of circular pistons, and focused spherical bowls.

use kwavers_core::constants::numerical::TWO_PI;
use num_complex::Complex64;
use rayon::prelude::*;

use super::array_factor::bessel_j1;

// ─── Array geometry ────────────────────────────────────────────────────────────

/// Generate element positions for a centred 2-D linear (1-D) array.
///
/// The array is laid along the x-axis, centred on the origin, radiating into
/// +z. Element i is placed at:
/// ```text
/// xᵢ = (i − (N−1)/2)·d,   zᵢ = 0
/// ```
///
/// # Arguments
/// * `n` – number of elements
/// * `pitch_m` – inter-element pitch d [m]
///
/// # Returns
/// `(elem_x, elem_z)` – element coordinates [m], each of length `n`.
#[must_use]
pub fn linear_array_positions(n: usize, pitch_m: f64) -> (Vec<f64>, Vec<f64>) {
    let center = (n as f64 - 1.0) / 2.0;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 - center) * pitch_m).collect();
    let z = vec![0.0_f64; n];
    (x, z)
}

/// Fresnel (near-field) transition distance — the *natural focus* of an
/// unfocused aperture.
///
/// For a uniformly excited aperture of full width `D` radiating at wavelength
/// `λ = c/f`, the on-axis pressure exhibits its last axial maximum (the
/// near-field/far-field transition) at:
/// ```text
/// N = D² / (4λ)
/// ```
/// This is the deepest axial point at which the unfocused aperture naturally
/// concentrates energy; electronic focusing is effective only at ranges
/// `z ≲ N` (focusing beyond the natural focus cannot tighten the beam).
///
/// # Arguments
/// * `aperture_m` – full aperture width D [m]
/// * `freq_hz` – frequency [Hz]
/// * `c` – sound speed [m/s]
///
/// # Returns
/// Natural-focus (near-field transition) range N [m].
///
/// # Reference
/// Szabo (2014) *Diagnostic Ultrasound Imaging*, §6.5; Cobbold (2007) §6.
#[must_use]
pub fn near_field_distance(aperture_m: f64, freq_hz: f64, c: f64) -> f64 {
    let lambda = c / freq_hz;
    aperture_m * aperture_m / (4.0 * lambda)
}

/// Map a steering angle and focal range onto a Cartesian focal point on the
/// *natural-focus arc*.
///
/// The steered focal point lies on the circular arc of radius `focal_range_m`
/// (the natural focal radius) at polar angle `steer_rad` measured from the
/// array normal (+z):
/// ```text
/// x_f = R·sin θ_s,   z_f = R·cos θ_s
/// ```
/// Steering at fixed `R` traces the focus along the natural-focus arc, keeping
/// the focal range constant while the lateral position changes.
///
/// # Arguments
/// * `focal_range_m` – focal range R (e.g. the natural focus N) [m]
/// * `steer_rad` – steering angle from the array normal [rad]
///
/// # Returns
/// `(x_f, z_f)` – focal point [m].
#[must_use]
pub fn steering_focus_point(focal_range_m: f64, steer_rad: f64) -> (f64, f64) {
    (
        focal_range_m * steer_rad.sin(),
        focal_range_m * steer_rad.cos(),
    )
}

// ─── Delay laws ───────────────────────────────────────────────────────────────

/// Compute the steering+focusing delay law for a focus on the natural-focus arc.
///
/// Combines [`steering_focus_point`] and [`delay_law_focus_2d`]: the focus is
/// placed at range `focal_range_m` and angle `steer_rad`, then per-element
/// time delays are computed so all elements arrive in phase at that point.
/// Driving `focal_range_m` equal to the array's natural focus
/// ([`near_field_distance`]) implements focusing *around the natural focus*.
///
/// # Arguments
/// * `elem_x`, `elem_z` – element positions [m]
/// * `focal_range_m` – focal range R [m]
/// * `steer_rad` – steering angle from the array normal [rad]
/// * `c` – sound speed [m/s]
///
/// # Returns
/// Per-element delay [s], same length as `elem_x`.
#[must_use]
pub fn delay_law_steer_2d(
    elem_x: &[f64],
    elem_z: &[f64],
    focal_range_m: f64,
    steer_rad: f64,
    c: f64,
) -> Vec<f64> {
    let (x_f, z_f) = steering_focus_point(focal_range_m, steer_rad);
    delay_law_focus_2d(elem_x, elem_z, x_f, z_f, c)
}

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
pub fn delay_law_focus_2d(elem_x: &[f64], elem_z: &[f64], x_f: f64, z_f: f64, c: f64) -> Vec<f64> {
    assert_eq!(
        elem_x.len(),
        elem_z.len(),
        "element arrays must have equal length"
    );
    let r: Vec<f64> = elem_x
        .iter()
        .zip(elem_z.iter())
        .map(|(&xi, &zi)| ((xi - x_f).powi(2) + (zi - z_f).powi(2)).sqrt())
        .collect();
    let max_r = r.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    r.iter().map(|&ri| (max_r - ri) / c).collect()
}

/// Element coordinates for a focused spherical bowl with beam axis `+x`.
///
/// The bowl surface is sampled by concentric rings of equal radial spacing.
/// Every element lies on the spherical surface centered at the geometric focus
/// `(focal_length_m, 0, 0)`, so each element-to-focus distance is exactly
/// `roc_m`. The aperture radius is the projected radius in the `y-z` plane.
///
/// Output is row-major `(n_elem, 3)` flattened as `[x0, y0, z0, ...]`.
#[must_use]
pub fn focused_bowl_element_positions_3d(
    n_rings: usize,
    elements_per_ring: usize,
    aperture_radius_m: f64,
    roc_m: f64,
    focal_length_m: f64,
) -> Vec<f64> {
    if n_rings == 0
        || elements_per_ring == 0
        || aperture_radius_m <= 0.0
        || roc_m <= aperture_radius_m
    {
        return Vec::new();
    }
    let mut out = Vec::with_capacity((1 + n_rings * elements_per_ring) * 3);
    out.extend_from_slice(&[focal_length_m - roc_m, 0.0, 0.0]);
    for ir in 1..=n_rings {
        let r = aperture_radius_m * (ir as f64) / (n_rings as f64);
        let x = focal_length_m - (roc_m * roc_m - r * r).sqrt();
        let phase_offset = if ir % 2 == 0 {
            0.0
        } else {
            std::f64::consts::PI / elements_per_ring as f64
        };
        for ia in 0..elements_per_ring {
            let phi = phase_offset + TWO_PI * (ia as f64) / (elements_per_ring as f64);
            out.extend_from_slice(&[x, r * phi.cos(), r * phi.sin()]);
        }
    }
    out
}

/// 3-D focusing delay law for arbitrary element and focus positions.
///
/// `elem_pos` is flattened `(n_elem, 3)`. Delays are nonnegative and align all
/// element phases at `focus_xyz`.
#[must_use]
pub fn delay_law_focus_3d(elem_pos: &[f64], focus_xyz: [f64; 3], c: f64) -> Vec<f64> {
    if elem_pos.len() % 3 != 0 || c <= 0.0 {
        return Vec::new();
    }
    let n = elem_pos.len() / 3;
    let mut ranges = Vec::with_capacity(n);
    for i in 0..n {
        let j = 3 * i;
        let dx = elem_pos[j] - focus_xyz[0];
        let dy = elem_pos[j + 1] - focus_xyz[1];
        let dz = elem_pos[j + 2] - focus_xyz[2];
        ranges.push((dx * dx + dy * dy + dz * dz).sqrt());
    }
    let max_r = ranges.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    ranges.into_iter().map(|r| (max_r - r) / c).collect()
}

/// Narrowband steered aperture pressure at arbitrary 3-D sample points.
///
/// Each element contributes the causal Green-function term
/// `w_i exp(-alpha r_i) exp(i(omega tau_i - k r_i)) / r_i`.
/// The returned magnitude is normalized by the computed magnitude at
/// `focus_xyz` and multiplied by `focus_pressure_pa`; therefore the commanded
/// focus has the requested pressure unless destructive interference makes the
/// focus field zero.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn steered_aperture_pressure_3d(
    points_xyz: &[f64],
    elem_pos: &[f64],
    weights: &[f64],
    delays_s: &[f64],
    focus_xyz: [f64; 3],
    freq_hz: f64,
    c: f64,
    alpha_np_m: f64,
    focus_pressure_pa: f64,
) -> Vec<f64> {
    if points_xyz.len() % 3 != 0
        || elem_pos.len() % 3 != 0
        || freq_hz <= 0.0
        || c <= 0.0
        || focus_pressure_pa < 0.0
    {
        return Vec::new();
    }
    let n_elem = elem_pos.len() / 3;
    if weights.len() != n_elem || delays_s.len() != n_elem {
        return Vec::new();
    }
    let k = TWO_PI * freq_hz / c;
    let omega = TWO_PI * freq_hz;
    let eval = |p: [f64; 3]| -> f64 {
        let mut acc = Complex64::new(0.0, 0.0);
        for ie in 0..n_elem {
            let j = 3 * ie;
            let dx = p[0] - elem_pos[j];
            let dy = p[1] - elem_pos[j + 1];
            let dz = p[2] - elem_pos[j + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-12);
            let phase = omega * delays_s[ie] - k * r;
            let amp = weights[ie] * (-alpha_np_m.max(0.0) * r).exp() / r;
            acc += amp * Complex64::new(phase.cos(), phase.sin());
        }
        acc.norm()
    };
    let focus_mag = eval(focus_xyz).max(1.0e-300);
    let n_points = points_xyz.len() / 3;
    let mut out = Vec::with_capacity(n_points);
    for ip in 0..n_points {
        let j = 3 * ip;
        let p = [points_xyz[j], points_xyz[j + 1], points_xyz[j + 2]];
        out.push(focus_pressure_pa * eval(p) / focus_mag);
    }
    out
}

/// Focused-bowl steered transverse pressure profile.
///
/// This is the source/transducer SSOT for chapter-level therapy planning: the
/// bowl element surface, element weights, 3-D focus delay law, and radial sample
/// coordinates are all constructed in Rust. Returned values are normalized by
/// `focus_pressure_pa`, so the first on-focus sample is one when `radius_m[0]=0`.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn focused_bowl_steered_pressure_profile(
    radius_m: &[f64],
    focus_xyz: [f64; 3],
    focus_pressure_pa: f64,
    n_rings: usize,
    elements_per_ring: usize,
    aperture_radius_m: f64,
    roc_m: f64,
    focal_length_m: f64,
    freq_hz: f64,
    c: f64,
    alpha_np_m: f64,
) -> Vec<f64> {
    if focus_pressure_pa <= 0.0 {
        return vec![0.0; radius_m.len()];
    }
    let elem = focused_bowl_element_positions_3d(
        n_rings,
        elements_per_ring,
        aperture_radius_m,
        roc_m,
        focal_length_m,
    );
    if elem.is_empty() {
        return Vec::new();
    }
    let delays = delay_law_focus_3d(&elem, focus_xyz, c);
    let weights = vec![1.0; delays.len()];
    let mut points = Vec::with_capacity(radius_m.len() * 3);
    for &r in radius_m {
        points.extend_from_slice(&[focus_xyz[0], focus_xyz[1] + r, focus_xyz[2]]);
    }
    steered_aperture_pressure_3d(
        &points,
        &elem,
        &weights,
        &delays,
        focus_xyz,
        freq_hz,
        c,
        alpha_np_m,
        focus_pressure_pa,
    )
    .into_iter()
    .map(|p| p / focus_pressure_pa)
    .collect()
}

/// Per-element geometric focusing delay laws for a set of focal sub-spots.
///
/// Multi-spot therapy — multi-focus histotripsy and multi-target blood–brain
/// barrier (BBB) opening — addresses several focal points `r_f^{(j)}` either
/// simultaneously (parallel multi-focus synthesis) or by rapid electronic
/// switching between sub-spots. For each sub-spot `j` and element `i` this
/// returns the geometric (straight-ray, homogeneous-medium) focusing delay
/// ```text
/// τ_{ij} = (max_i r_{ij} − r_{ij}) / c,   r_{ij} = ‖rᵢ − r_f^{(j)}‖,
/// ```
/// so that every element arrives in phase at sub-spot `j` and all delays are
/// non-negative (the farthest element of each sub-aperture fires first).
///
/// Output is a row-major flat `Vec` of length `n_spots × n_elem`; row `j`
/// (offset `j·n_elem`) holds the full-aperture delay law that focuses on
/// sub-spot `j`. The straight rays `rᵢ → r_f^{(j)}` and these delays are the
/// ray-path/delay-to-target description of a multi-spot treatment plan.
///
/// # Arguments
/// * `elem_x`, `elem_z` – element positions [m]
/// * `spot_x`, `spot_z` – focal sub-spot positions [m] (equal length)
/// * `c` – sound speed [m/s]
///
/// # Panics
/// Panics if `elem_x`/`elem_z` or `spot_x`/`spot_z` differ in length.
#[must_use]
pub fn multi_focus_delay_laws_2d(
    elem_x: &[f64],
    elem_z: &[f64],
    spot_x: &[f64],
    spot_z: &[f64],
    c: f64,
) -> Vec<f64> {
    assert_eq!(
        spot_x.len(),
        spot_z.len(),
        "sub-spot arrays must have equal length"
    );
    let n_elem = elem_x.len();
    let mut out = Vec::with_capacity(spot_x.len() * n_elem);
    for (&xf, &zf) in spot_x.iter().zip(spot_z.iter()) {
        out.extend(delay_law_focus_2d(elem_x, elem_z, xf, zf, c));
    }
    out
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
#[allow(clippy::too_many_arguments)]
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

/// Magnitude of the 2-D complex beam pattern, normalised to its peak.
///
/// Computes `|p(x, z)|` from [`beam_pattern_2d`] and divides by the field
/// maximum, returning a flattened row-major (NX × NZ) magnitude field in
/// `[0, 1]`. The absolute-value reduction and peak normalisation are the
/// physical field magnitude, kept on the Rust side so callers receive a
/// ready-to-display field.
///
/// # Arguments
/// See [`beam_pattern_2d`].
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn beam_pattern_2d_magnitude(
    x_arr: &[f64],
    z_arr: &[f64],
    elem_x: &[f64],
    elem_z: &[f64],
    freq_hz: f64,
    c: f64,
    weights: &[f64],
    delays: &[f64],
) -> Vec<f64> {
    let (re, im) = beam_pattern_2d(x_arr, z_arr, elem_x, elem_z, freq_hz, c, weights, delays);
    let mut mag: Vec<f64> = re
        .iter()
        .zip(im.iter())
        .map(|(&r, &i)| r.hypot(i))
        .collect();
    let peak = mag.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    for m in &mut mag {
        *m /= peak;
    }
    mag
}

/// Simultaneous multi-focus CW field magnitude via phase-conjugation synthesis.
///
/// Each element `i` is driven with the complex weight obtained by superposing
/// the phase-conjugate (time-reversed) field that focuses the aperture on each
/// sub-spot `j`:
/// ```text
/// w_i = Σ_j a_j · exp(+i·k·r_{ij}),   r_{ij} = ‖rᵢ − r_f^{(j)}‖,
/// ```
/// where `a_j` is the per-spot drive amplitude. The continuous-wave field at
/// grid point `(x, z)` is the monopole superposition
/// ```text
/// p(x, z) = Σ_i w_i · exp(−i·k·rᵢ(x, z)),
/// ```
/// and the returned magnitude `|p|` is normalised to its peak (row-major
/// `NX × NZ`, values in `[0, 1]`). With `n_spots ≥ 1` this synthesises
/// simultaneous foci at every `r_f^{(j)}`, the field model for parallel
/// multi-spot histotripsy and multi-target BBB opening. Phase conjugation is
/// the exact narrowband time-reversal solution for focusing in a homogeneous
/// medium; the absolute-value reduction and peak normalisation are kept on the
/// Rust side so callers receive a ready-to-display field.
///
/// # Arguments
/// * `x_arr`, `z_arr` – grid coordinates [m]
/// * `elem_x`, `elem_z` – element positions [m]
/// * `spot_x`, `spot_z` – focal sub-spot positions [m]
/// * `spot_amp` – per-spot drive amplitudes `a_j` (length == `n_spots`)
/// * `freq_hz` – frequency [Hz]
/// * `c` – sound speed [m/s]
///
/// # Panics
/// Panics if `spot_x`/`spot_z`/`spot_amp` differ in length or if
/// `elem_x`/`elem_z` differ in length.
///
/// # Reference
/// Fink (1992) *IEEE Trans. UFFC* 39(5):555–566 (time-reversal focusing);
/// Ebbini & Cain (1989) *IEEE Trans. UFFC* 36(5):540–548 (multi-focus
/// pattern synthesis).
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn multi_focus_field_magnitude_2d(
    x_arr: &[f64],
    z_arr: &[f64],
    elem_x: &[f64],
    elem_z: &[f64],
    spot_x: &[f64],
    spot_z: &[f64],
    spot_amp: &[f64],
    freq_hz: f64,
    c: f64,
) -> Vec<f64> {
    assert_eq!(
        elem_x.len(),
        elem_z.len(),
        "element arrays must have equal length"
    );
    assert_eq!(
        spot_x.len(),
        spot_z.len(),
        "sub-spot arrays must have equal length"
    );
    assert_eq!(
        spot_x.len(),
        spot_amp.len(),
        "sub-spot amplitude array must match sub-spot count"
    );
    let k = TWO_PI * freq_hz / c;
    let n_elem = elem_x.len();

    // Phase-conjugation element weights: w_i = Σ_j a_j·exp(+i·k·r_ij).
    let weights: Vec<Complex64> = (0..n_elem)
        .map(|ie| {
            let mut w = Complex64::new(0.0, 0.0);
            for js in 0..spot_x.len() {
                let dx = elem_x[ie] - spot_x[js];
                let dz = elem_z[ie] - spot_z[js];
                let r = (dx * dx + dz * dz).sqrt();
                let phase = k * r;
                w += spot_amp[js] * Complex64::new(phase.cos(), phase.sin());
            }
            w
        })
        .collect();

    let nx = x_arr.len();
    let nz = z_arr.len();
    let mut mag = vec![0.0_f64; nx * nz];
    for (ix, &x) in x_arr.iter().enumerate() {
        for (iz, &z) in z_arr.iter().enumerate() {
            let mut p = Complex64::new(0.0, 0.0);
            for ie in 0..n_elem {
                let dx = x - elem_x[ie];
                let dz = z - elem_z[ie];
                let r = (dx * dx + dz * dz).sqrt().max(1e-12);
                let phase = -k * r;
                p += weights[ie] * Complex64::new(phase.cos(), phase.sin());
            }
            mag[ix * nz + iz] = p.norm();
        }
    }
    let peak = mag.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
    for m in &mut mag {
        *m /= peak;
    }
    mag
}

// ─── Steering envelope (grating-lobe limited) ──────────────────────────────────

/// Aperiodic ("sparse") linear element positions — same aperture and element
/// count as a uniform array, but with the periodic grid broken by a
/// deterministic low-discrepancy dither.
///
/// The array lies along x, centred on the origin, spanning the full aperture
/// `aperture_m`; the two endpoint elements are anchored at `±aperture/2`, so
/// the physical aperture (hence the diffraction-limited main-lobe width) is
/// identical to the uniform array. Each interior element is displaced from its
/// grid position by a fraction `jitter_frac` of the element pitch using a
/// golden-ratio additive recurrence (`jitter_frac = 0` reproduces the uniform
/// layout; `≈0.7` strongly suppresses grating lobes). Destroying the spatial
/// periodicity scatters what would otherwise be a single coherent grating lobe
/// into a low, incoherent pedestal, so the same element count can be steered
/// further before any secondary lobe reaches the −6 dB safety limit. Only the
/// *placement* changes — frequency, aperture and element count are fixed — so
/// the comparison isolates the element-activation pattern.
///
/// # Arguments
/// * `n` – number of elements (matched to the uniform array)
/// * `aperture_m` – full aperture span D [m]
/// * `jitter_frac` – dither amplitude as a fraction of the element pitch
///
/// # Returns
/// Element x-positions [m], length `n`.
///
/// # Reference
/// Steinberg (1976) *Principles of Aperture and Array System Design* (thinned
/// aperiodic arrays and grating-lobe suppression).
#[must_use]
pub fn linear_array_aperiodic_positions(n: usize, aperture_m: f64, jitter_frac: f64) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0.0];
    }
    // Golden-ratio additive recurrence → low-discrepancy dither in [0, 1).
    let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
    let pitch = aperture_m / (n as f64 - 1.0);
    let half = 0.5 * aperture_m;
    (0..n)
        .map(|i| {
            if i == 0 {
                -half
            } else if i == n - 1 {
                half
            } else {
                let base = -half + pitch * (i as f64);
                let u = ((i as f64 + 1.0) * golden).fract();
                base + (u - 0.5) * jitter_frac * pitch
            }
        })
        .collect()
}

/// Baffled circular-piston element directivity `D(θ) = 2 J₁(ka·sinθ)/(ka·sinθ)`,
/// normalised so `D(0) = 1`. Avoids the per-call allocation of
/// [`circular_piston_directivity`] in the hot beam-pattern loops.
#[inline]
fn piston_directivity(theta: f64, ka: f64) -> f64 {
    let arg = ka * theta.sin();
    if arg.abs() < 1e-12 {
        1.0
    } else {
        2.0 * bessel_j1(arg) / arg
    }
}

/// Steered far-field beam pattern of a linear array.
///
/// The `N` elements lie along x at positions `elem_x` and radiate broadside
/// (normal `+z`); the array is phased to steer its main lobe to `steer_theta`
/// (measured from broadside). The far-field response at observation angle `θ`
/// is the product of the element directivity and the array factor,
/// ```text
/// P(θ) = D(θ) · | (1/N) Σ_i exp[ i·k·x_i·(sinθ − sin θ_s) ] |,
/// ```
/// where `D(θ) = 2 J₁(ka·sinθ)/(ka·sinθ)` is the baffled circular-piston
/// element factor with parameter `ka_elem = k·a_elem`. At `θ = θ_s` the array
/// factor is unity and `P = D(θ_s)` (the main lobe); coherent secondary peaks
/// where the array factor returns to unity at other angles are grating lobes.
///
/// # Arguments
/// * `elem_x` – element x-positions [m]
/// * `obs_theta` – observation angles [rad], from broadside
/// * `k` – wavenumber 2πf/c [rad/m]
/// * `steer_theta` – steering angle [rad], from broadside
/// * `ka_elem` – element directivity parameter k·a_elem
///
/// # Returns
/// Beam-pattern magnitude at each `obs_theta`.
///
/// # Reference
/// Van Trees (2002) *Optimum Array Processing*, §2.2; O'Neil (1949) (element
/// directivity).
#[must_use]
pub fn steered_beam_pattern_1d(
    elem_x: &[f64],
    obs_theta: &[f64],
    k: f64,
    steer_theta: f64,
    ka_elem: f64,
) -> Vec<f64> {
    let n = (elem_x.len() as f64).max(1.0);
    let sin_s = steer_theta.sin();
    obs_theta
        .iter()
        .map(|&th| {
            let dsin = th.sin() - sin_s;
            let mut acc = Complex64::new(0.0, 0.0);
            for &x in elem_x {
                acc += Complex64::from_polar(1.0, k * x * dsin);
            }
            piston_directivity(th, ka_elem) * (acc.norm() / n)
        })
        .collect()
}

/// Grating-lobe ratio versus steering angle — the basis of the *steering
/// envelope* at a fixed frequency.
///
/// For each steering angle `θ_s` the array is phased to that angle and its
/// beam pattern ([`steered_beam_pattern_1d`]) is searched for the strongest
/// lobe outside the main lobe (a `±mainlobe_halfwidth_rad` window about `θ_s`).
/// The returned value is the grating-lobe ratio
/// ```text
/// G(θ_s) = max_{|θ − θ_s| > Δ} P(θ) / P(θ_s),
/// ```
/// with `P(θ_s) = D(θ_s)` the main-lobe peak. The safe **steering envelope**
/// is the set `{ θ_s : G(θ_s) ≤ 0.5 }`, where no secondary lobe exceeds half
/// (−6 dB) the main-lobe pressure.
///
/// Holding frequency, aperture and element count fixed, a uniform array raises
/// a coherent grating lobe once `θ_s` exceeds the threshold set by its element
/// pitch — `G` jumps up — whereas an aperiodic array
/// ([`linear_array_aperiodic_positions`]) keeps `G` low over a much wider
/// steering range. The distinction is purely the element-activation pattern.
///
/// # Arguments
/// * `elem_x` – element x-positions [m]
/// * `steer_theta` – steering-angle grid [rad]
/// * `obs_theta` – observation-angle grid [rad] for the lobe search
/// * `k` – wavenumber 2πf/c [rad/m]
/// * `ka_elem` – element directivity parameter k·a_elem
/// * `mainlobe_halfwidth_rad` – half-width of the main-lobe exclusion window [rad]
///
/// # Returns
/// Grating-lobe ratio at each `steer_theta`.
///
/// # Reference
/// Steinberg (1976) *Principles of Aperture and Array System Design*; Pernot
/// et al. (2003) *Ultrasound Med. Biol.* 29(11):1559–1565 (electronic-steering
/// envelope and grating lobes of therapy arrays).
#[must_use]
pub fn steering_grating_lobe_ratio_1d(
    elem_x: &[f64],
    steer_theta: &[f64],
    obs_theta: &[f64],
    k: f64,
    ka_elem: f64,
    mainlobe_halfwidth_rad: f64,
) -> Vec<f64> {
    steer_theta
        .par_iter()
        .map(|&ts| {
            let pat = steered_beam_pattern_1d(elem_x, obs_theta, k, ts, ka_elem);
            let main = piston_directivity(ts, ka_elem).max(1e-300);
            let mut secondary = 0.0_f64;
            for (i, &th) in obs_theta.iter().enumerate() {
                if (th - ts).abs() <= mainlobe_halfwidth_rad {
                    continue; // inside the main lobe
                }
                if pat[i] > secondary {
                    secondary = pat[i];
                }
            }
            secondary / main
        })
        .collect()
}

/// Safe steering half-angle — the largest steering excursion from broadside
/// over which the grating-lobe ratio stays at or below a safety threshold.
///
/// Starting from the steering angle closest to broadside (`θ_s = 0`), the safe
/// region is expanded outward to both sides while `G ≤ threshold`; the returned
/// value is the symmetric half-angle `min(θ_right, |θ_left|)` of that
/// contiguous run. With `threshold = 0.5` this is the −6 dB grating-lobe-safe
/// steering half-angle; the ratio of an aperiodic to a uniform half-angle
/// quantifies the steering-envelope expansion from sparse activation. Returns
/// `0` if broadside itself is unsafe.
///
/// # Arguments
/// * `steer_theta` – steering-angle grid [rad] (monotonically increasing)
/// * `glr` – grating-lobe ratio at each `steer_theta`
/// * `threshold` – grating-lobe safety threshold (e.g. 0.5)
///
/// # Returns
/// Safe steering half-angle [rad].
///
/// # Panics
/// Panics if `steer_theta` and `glr` differ in length.
#[must_use]
pub fn safe_steering_halfangle(steer_theta: &[f64], glr: &[f64], threshold: f64) -> f64 {
    assert_eq!(
        steer_theta.len(),
        glr.len(),
        "steer_theta and glr length mismatch"
    );
    if steer_theta.is_empty() {
        return 0.0;
    }
    // Index closest to broadside (θ = 0).
    let i0 = (0..steer_theta.len())
        .min_by(|&a, &b| {
            steer_theta[a]
                .abs()
                .partial_cmp(&steer_theta[b].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();
    if glr[i0] > threshold {
        return 0.0;
    }
    let mut hi = i0;
    while hi + 1 < glr.len() && glr[hi + 1] <= threshold {
        hi += 1;
    }
    let mut lo = i0;
    while lo > 0 && glr[lo - 1] <= threshold {
        lo -= 1;
    }
    steer_theta[hi].abs().min(steer_theta[lo].abs())
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
