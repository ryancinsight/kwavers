//! Far-field acoustic emission radiated by a pulsating microbubble.
//!
//! A passive cavitation detector (PCD) records the pressure radiated by an
//! oscillating bubble, not its radius. For a spherical bubble of radius `R(t)`
//! the far-field (1/r) scattered pressure at observation distance `r_obs` is the
//! second time-derivative of the bubble volume (Leighton 1994 §3.2.1,
//! Neppiras 1980):
//! ```text
//!   p_sc(r_obs, t) = ρ_L / (4π r_obs) · V̈(t)
//!   V  = (4/3)π R³
//!   V̇  = 4π R² Ṙ
//!   V̈  = 4π R (2Ṙ² + R R̈)
//!   ⇒ p_sc(r_obs, t) = (ρ_L R / r_obs) · (2Ṙ² + R R̈)
//! ```
//! This is the time series whose power spectrum the harmonic/subharmonic/
//! broadband cavitation-dose decomposition operates on.

/// Far-field scattered (emitted) pressure from a microbubble radius history.
///
/// Computes `p_sc(r_obs, t) = (ρ_L R / r_obs)·(2Ṙ² + R R̈)` from the radius and
/// wall-velocity series produced by [`super::super::keller_miksis_rk4`] or
/// [`super::super::rayleigh_plesset_rk4`]. The wall acceleration `R̈` is obtained
/// by second-order central differences of `Ṙ` (forward/backward at the ends),
/// which matches the RK4 grid the integrator advanced on.
///
/// # Arguments
/// * `r_arr`    – bubble radius series `R(t)` `m`
/// * `rdot_arr` – wall velocity series `Ṙ(t)` [m/s] (same length as `r_arr`)
/// * `dt_s`     – uniform time step `s`
/// * `rho`      – liquid density [kg/m³]
/// * `r_obs_m`  – PCD observation distance from the bubble `m`
///
/// Returns the emitted-pressure series `p_sc(t)` `Pa` with the same length as the
/// inputs. Returns an empty vector if the inputs are shorter than 2 samples,
/// of unequal length, or if `dt_s`/`r_obs_m` are non-positive.
///
/// # Reference
/// Leighton T.G. (1994) *The Acoustic Bubble*, Academic Press, §3.2.1.
/// Neppiras E.A. (1980) *Phys. Rep.* 61, 159.
#[must_use]
pub fn bubble_acoustic_emission_pressure(
    r_arr: &[f64],
    rdot_arr: &[f64],
    dt_s: f64,
    rho: f64,
    r_obs_m: f64,
) -> Vec<f64> {
    let n = r_arr.len();
    if n < 2
        || rdot_arr.len() != n
        || !(dt_s.is_finite() && dt_s > 0.0)
        || !(r_obs_m.is_finite() && r_obs_m > 0.0)
        || !rho.is_finite()
    {
        return Vec::new();
    }
    let mut p = vec![0.0_f64; n];
    let inv_2dt = 0.5 / dt_s;
    for i in 0..n {
        // Central difference for R̈; one-sided at the boundaries.
        let rddot = if i == 0 {
            (rdot_arr[1] - rdot_arr[0]) / dt_s
        } else if i == n - 1 {
            (rdot_arr[n - 1] - rdot_arr[n - 2]) / dt_s
        } else {
            (rdot_arr[i + 1] - rdot_arr[i - 1]) * inv_2dt
        };
        let r = r_arr[i];
        let rdot = rdot_arr[i];
        p[i] = rho * r / r_obs_m * (2.0 * rdot * rdot + r * rddot);
    }
    p
}
