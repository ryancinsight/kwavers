use kwavers_core::constants::numerical::TWO_PI;

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
