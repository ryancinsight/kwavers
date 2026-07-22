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
/// * `pitch_m` – inter-element pitch d `m`
///
/// # Returns
/// `(elem_x, elem_z)` – element coordinates `m`, each of length `n`.
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
/// * `aperture_m` – full aperture width D `m`
/// * `freq_hz` – frequency `Hz`
/// * `c` – sound speed [m/s]
///
/// # Returns
/// Natural-focus (near-field transition) range N `m`.
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
/// * `focal_range_m` – focal range R (e.g. the natural focus N) `m`
/// * `steer_rad` – steering angle from the array normal `rad`
///
/// # Returns
/// `(x_f, z_f)` – focal point `m`.
#[must_use]
pub fn steering_focus_point(focal_range_m: f64, steer_rad: f64) -> (f64, f64) {
    (
        focal_range_m * steer_rad.sin(),
        focal_range_m * steer_rad.cos(),
    )
}
