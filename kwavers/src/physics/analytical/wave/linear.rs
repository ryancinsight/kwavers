/// Compute the pressure field of a 1-D standing wave.
///
/// ```text
/// p(x, t) = p₀ · sin(k·x) · cos(ω·t)   [Pa]
/// ```
///
/// # Arguments
/// * `p0` – peak pressure amplitude [Pa]
/// * `k` – wavenumber [rad/m]
/// * `x_arr` – spatial positions [m]
/// * `omega_t` – phase `ω·t` [rad]
#[must_use]
#[inline]
pub fn standing_wave_1d(p0: f64, k: f64, x_arr: &[f64], omega_t: f64) -> Vec<f64> {
    let cos_wt = omega_t.cos();
    x_arr.iter().map(|&x| p0 * (k * x).sin() * cos_wt).collect()
}

/// Compute the pressure field of a 1-D plane wave.
///
/// ```text
/// p(x, t) = A · cos(k·x − ω·t)   [Pa]
/// ```
///
/// # Arguments
/// * `amplitude` – peak amplitude [Pa]
/// * `k` – wavenumber [rad/m]
/// * `x_arr` – spatial positions [m]
/// * `omega_t` – `ω·t` [rad]
#[must_use]
#[inline]
pub fn plane_wave_pressure_1d(amplitude: f64, k: f64, x_arr: &[f64], omega_t: f64) -> Vec<f64> {
    x_arr
        .iter()
        .map(|&x| amplitude * (k * x - omega_t).cos())
        .collect()
}

/// Real part of the spherical-wave Green's function (far-field).
///
/// ```text
/// p(r) = A · cos(k·r) / r   [Pa]
/// ```
///
/// Singularity at r = 0 is guarded: returns `f64::INFINITY` there.
///
/// # Reference
/// Pierce (1989) *Acoustics*, §1.6.
#[must_use]
#[inline]
pub fn spherical_wave_pressure(amplitude: f64, k: f64, r_arr: &[f64]) -> Vec<f64> {
    r_arr
        .iter()
        .map(|&r| {
            if r == 0.0 {
                f64::INFINITY
            } else {
                amplitude * (k * r).cos() / r
            }
        })
        .collect()
}

/// Plane-wave pressure reflection coefficient at a normal-incidence interface.
///
/// ```text
/// R = (Z₂ − Z₁) / (Z₂ + Z₁)
/// ```
///
/// # Reference
/// Kinsler et al. (2000) *Fundamentals of Acoustics*, §6.3.
#[must_use]
#[inline]
pub fn reflection_pressure_coeff(z1: f64, z2: f64) -> f64 {
    (z2 - z1) / (z2 + z1)
}

/// Plane-wave pressure transmission coefficient at a normal-incidence interface.
///
/// ```text
/// T = 2·Z₂ / (Z₂ + Z₁)
/// ```
///
/// # Reference
/// Kinsler et al. (2000) *Fundamentals of Acoustics*, §6.3.
#[must_use]
#[inline]
pub fn transmission_pressure_coeff(z1: f64, z2: f64) -> f64 {
    2.0 * z2 / (z2 + z1)
}

/// Power-law attenuation in Nepers/m.
///
/// ```text
/// α(f) = α₀ · f^y   [Np/m]
/// ```
///
/// # Arguments
/// * `f_hz` – frequencies [Hz]
/// * `alpha0` – attenuation coefficient [Np/m/Hz^y]
/// * `y` – power-law exponent (typically 1–2)
///
/// # Reference
/// Szabo (1994), *J. Acoust. Soc. Am.* 96, 491.
#[must_use]
#[inline]
pub fn power_law_attenuation_np_m(f_hz: &[f64], alpha0: f64, y: f64) -> Vec<f64> {
    f_hz.iter().map(|&f| alpha0 * f.powf(y)).collect()
}

/// Power-law attenuation in dB/cm.
///
/// ```text
/// α(f) = α₀ · f^y   [dB/cm],  f in MHz
/// ```
///
/// # Arguments
/// * `f_mhz` – frequencies [MHz]
/// * `alpha0` – attenuation coefficient [dB/(cm·MHz^y)]
/// * `y` – power-law exponent
#[must_use]
#[inline]
pub fn absorption_power_law_db_cm(f_mhz: &[f64], alpha0: f64, y: f64) -> Vec<f64> {
    f_mhz.iter().map(|&f| alpha0 * f.powf(y)).collect()
}
