use kwavers_core::constants::numerical::TWO_PI;
use eunomia::Complex64;

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

