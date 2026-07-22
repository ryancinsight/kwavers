use super::*;
use eunomia::Complex64;
use kwavers_core::constants::numerical::TWO_PI;

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
/// * `elem_x`, `elem_z` – element positions `m`
/// * `focal_range_m` – focal range R `m`
/// * `steer_rad` – steering angle from the array normal `rad`
/// * `c` – sound speed [m/s]
///
/// # Returns
/// Per-element delay `s`, same length as `elem_x`.
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
/// τᵢ = (max_r − rᵢ) / c   `s`
/// ```
/// where `rᵢ = sqrt((xᵢ − x_f)² + (zᵢ − z_f)²)` is the distance from
/// element i to the focal point, and `max_r` is the maximum such distance
/// (ensures all delays are non-negative).
///
/// # Arguments
/// * `elem_x`, `elem_z` – element positions `m`
/// * `x_f`, `z_f` – focal point `m`
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
    if !elem_pos.len().is_multiple_of(3) || c <= 0.0 {
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
    if !points_xyz.len().is_multiple_of(3)
        || !elem_pos.len().is_multiple_of(3)
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
/// * `elem_x`, `elem_z` – element positions `m`
/// * `spot_x`, `spot_z` – focal sub-spot positions `m` (equal length)
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
