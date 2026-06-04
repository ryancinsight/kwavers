//! Histotripsy therapy-delivery helpers shared by PyO3 chapter orchestration.
//!
//! These routines keep geometric safety, measured-spectrum scaling, and delivered
//! dose-response accounting in the Rust physics crate. Python callers may build
//! arrays and plot results, but the therapy semantics live here.

use super::histotripsy_kill_fraction;
use crate::analytical::wave::{shock_formation_distance, shock_heat_source_density};

/// Lateral semi-axis that keeps an anisotropic focal ellipsoid within an
/// isotropic clearance constraint.
///
/// If a boiling-histotripsy focus has axial/lateral semi-axis ratio `a_z/a_r`,
/// then a clearance bound applies to the largest semi-axis, not only the
/// transverse radius. This function returns `min(natural_lateral, clearance /
/// max(1, a_z/a_r))`.
#[must_use]
pub fn clipped_lateral_radius_for_clearance(
    natural_lateral_radius_m: f64,
    clearance_m: f64,
    axial_to_lateral_ratio: f64,
) -> f64 {
    if !(natural_lateral_radius_m.is_finite()
        && clearance_m.is_finite()
        && axial_to_lateral_ratio.is_finite())
        || natural_lateral_radius_m <= 0.0
        || clearance_m <= 0.0
    {
        return 0.0;
    }
    let ratio = axial_to_lateral_ratio.max(1.0);
    natural_lateral_radius_m.min(clearance_m / ratio).max(0.0)
}

/// Check that every voxel inside a focal ellipsoid is inside an allowed mask.
///
/// `allowed_mask` is row-major with dimensions `(nx, ny, nz)`. The beam axis is
/// `x`; the focal ellipsoid uses `axial_radius_m` on x and `lateral_radius_m`
/// on y/z. The function returns false when the center or any part of the
/// ellipsoid exits the grid or overlaps a false mask voxel.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn ellipsoid_respects_allowed_mask(
    allowed_mask: &[bool],
    nx: usize,
    ny: usize,
    nz: usize,
    center_x: usize,
    center_y: usize,
    center_z: usize,
    lateral_radius_m: f64,
    axial_radius_m: f64,
    dx_m: f64,
) -> bool {
    if allowed_mask.len() != nx.saturating_mul(ny).saturating_mul(nz)
        || nx == 0
        || ny == 0
        || nz == 0
        || center_x >= nx
        || center_y >= ny
        || center_z >= nz
        || lateral_radius_m <= 0.0
        || axial_radius_m <= 0.0
        || dx_m <= 0.0
    {
        return false;
    }

    let rx = (axial_radius_m / dx_m).ceil() as isize;
    let rr = (lateral_radius_m / dx_m).ceil() as isize;
    let cx = center_x as isize;
    let cy = center_y as isize;
    let cz = center_z as isize;

    let x0 = cx - rx;
    let x1 = cx + rx;
    let y0 = cy - rr;
    let y1 = cy + rr;
    let z0 = cz - rr;
    let z1 = cz + rr;
    if x0 < 0 || y0 < 0 || z0 < 0 || x1 >= nx as isize || y1 >= ny as isize || z1 >= nz as isize {
        return false;
    }

    let inv_ax = 1.0 / axial_radius_m;
    let inv_lat = 1.0 / lateral_radius_m;
    for ix in x0..=x1 {
        let dxn = ((ix - cx) as f64 * dx_m * inv_ax).powi(2);
        for iy in y0..=y1 {
            let dyn_ = ((iy - cy) as f64 * dx_m * inv_lat).powi(2);
            for iz in z0..=z1 {
                let dzn = ((iz - cz) as f64 * dx_m * inv_lat).powi(2);
                if dxn + dyn_ + dzn <= 1.0 {
                    let idx = ((ix as usize) * ny + iy as usize) * nz + iz as usize;
                    if !allowed_mask[idx] {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// Apply receive-path and tissue-state scaling to a passive cavitation PSD.
///
/// The same factors that scale the measured scalar cavitation signal must scale
/// the plotted spectrum. `receive_fraction` is an amplitude/energy transfer
/// fraction for the passive path; `susceptibility` accounts for local lesion
/// memory and interface-enhanced cavitation source strength.
#[must_use]
pub fn scale_measured_emission_spectrum(
    psd: &[f64],
    receive_fraction: f64,
    susceptibility: f64,
) -> Vec<f64> {
    let scale = receive_fraction.max(0.0) * susceptibility.max(0.0);
    psd.iter().map(|v| v.max(0.0) * scale).collect()
}

/// Convert a cumulative delivered histotripsy dose series into kill fractions.
///
/// Dose samples are clamped to nonnegative values. The returned value uses the
/// Weibull survival law implemented by [`histotripsy_kill_fraction`].
#[must_use]
pub fn delivered_histotripsy_progress(dose: &[f64], d0: f64, weibull_k: f64) -> Vec<f64> {
    dose.iter()
        .map(|&d| histotripsy_kill_fraction(d.max(0.0), d0, weibull_k))
        .collect()
}

/// Result of boiling-histotripsy lesion sizing from a resolved pressure profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BoilingLesionPlan {
    /// Number of boiling pulses required to reach the requested coverage.
    pub pulses: usize,
    /// Lateral lesion semi-axis [m].
    pub lateral_radius_m: f64,
    /// Axial lesion semi-axis [m].
    pub axial_radius_m: f64,
    /// Single-pulse duration [ms].
    pub pulse_ms: f64,
}

/// Size a boiling-histotripsy lesion from pressure samples generated by the
/// active transmit model.
///
/// `radius_m` and `normalized_pressure` are paired radial samples in the focal
/// transverse plane. The shock heat source, boiling time, conformal clearance
/// clipping, and per-spot pulse count are computed here so Python callers do not
/// own therapy-domain logic.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn boiling_lesion_from_pressure_profile(
    radius_m: &[f64],
    normalized_pressure: &[f64],
    focal_pressure_pa: f64,
    focal_depth_m: f64,
    freq_hz: f64,
    c_m_s: f64,
    rho_kg_m3: f64,
    beta_nonlinearity: f64,
    alpha_np_m: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
    tau_max_s: f64,
    axial_to_lateral_ratio: f64,
    clearance_m: f64,
    coverage_target: f64,
) -> Option<BoilingLesionPlan> {
    let n = radius_m.len().min(normalized_pressure.len());
    if n < 2
        || focal_pressure_pa <= 0.0
        || focal_depth_m <= 0.0
        || freq_hz <= 0.0
        || c_m_s <= 0.0
        || rho_kg_m3 <= 0.0
        || alpha_np_m < 0.0
        || heat_capacity_j_kg_k <= 0.0
        || delta_t_k <= 0.0
        || tau_max_s <= 0.0
        || clearance_m <= 0.0
        || !(0.0..1.0).contains(&coverage_target)
    {
        return None;
    }

    let z_shock = shock_formation_distance(
        focal_pressure_pa,
        freq_hz,
        c_m_s,
        rho_kg_m3,
        beta_nonlinearity,
    )
    .max(1.0e-12);
    let mut p_local = Vec::with_capacity(n);
    let mut sigma = Vec::with_capacity(n);
    for &b in normalized_pressure.iter().take(n) {
        let bn = b.max(0.0);
        p_local.push(focal_pressure_pa * bn);
        sigma.push((focal_depth_m / z_shock) * bn);
    }
    let q_heat = shock_heat_source_density(&p_local, &sigma, alpha_np_m, rho_kg_m3, c_m_s);
    let mut natural_radius = 0.0_f64;
    for i in 0..n {
        if q_heat[i] <= 0.0 {
            continue;
        }
        let t_boil = rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / q_heat[i];
        if t_boil <= tau_max_s {
            natural_radius = natural_radius.max(radius_m[i]);
        }
    }
    if natural_radius <= 0.0 {
        return None;
    }
    let lateral =
        clipped_lateral_radius_for_clearance(natural_radius, clearance_m, axial_to_lateral_ratio);
    if lateral <= 0.0 {
        return None;
    }
    let pulse_s = boiling_time_at_radius(
        radius_m,
        &q_heat,
        lateral,
        rho_kg_m3,
        heat_capacity_j_kg_k,
        delta_t_k,
    )
    .min(tau_max_s);
    let fraction_per_pulse = (0.10 + 0.25 * pulse_s / tau_max_s).clamp(0.05, 0.40);
    let pulses = ((1.0 - coverage_target).ln() / (1.0 - fraction_per_pulse).ln()).ceil();
    Some(BoilingLesionPlan {
        pulses: pulses.max(1.0) as usize,
        lateral_radius_m: lateral,
        axial_radius_m: lateral * axial_to_lateral_ratio.max(1.0),
        pulse_ms: pulse_s * 1.0e3,
    })
}

/// Boiling-onset time profile from normalized pressure samples.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn boiling_time_profile_from_pressure(
    normalized_pressure: &[f64],
    focal_pressure_pa: f64,
    focal_depth_m: f64,
    freq_hz: f64,
    c_m_s: f64,
    rho_kg_m3: f64,
    beta_nonlinearity: f64,
    alpha_np_m: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
) -> Vec<f64> {
    if focal_pressure_pa <= 0.0
        || focal_depth_m <= 0.0
        || freq_hz <= 0.0
        || c_m_s <= 0.0
        || rho_kg_m3 <= 0.0
        || heat_capacity_j_kg_k <= 0.0
        || delta_t_k <= 0.0
    {
        return vec![f64::INFINITY; normalized_pressure.len()];
    }
    let z_shock = shock_formation_distance(
        focal_pressure_pa,
        freq_hz,
        c_m_s,
        rho_kg_m3,
        beta_nonlinearity,
    )
    .max(1.0e-12);
    let mut p_local = Vec::with_capacity(normalized_pressure.len());
    let mut sigma = Vec::with_capacity(normalized_pressure.len());
    for &b in normalized_pressure {
        let bn = b.max(0.0);
        p_local.push(focal_pressure_pa * bn);
        sigma.push((focal_depth_m / z_shock) * bn);
    }
    shock_heat_source_density(&p_local, &sigma, alpha_np_m, rho_kg_m3, c_m_s)
        .into_iter()
        .map(|q| {
            if q > 0.0 {
                rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / q
            } else {
                f64::INFINITY
            }
        })
        .collect()
}

fn boiling_time_at_radius(
    radius_m: &[f64],
    q_heat: &[f64],
    target_radius_m: f64,
    rho_kg_m3: f64,
    heat_capacity_j_kg_k: f64,
    delta_t_k: f64,
) -> f64 {
    let n = radius_m.len().min(q_heat.len());
    if n == 0 {
        return f64::INFINITY;
    }
    if target_radius_m <= radius_m[0] {
        return rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / q_heat[0].max(1.0e-300);
    }
    for i in 0..n - 1 {
        if radius_m[i] <= target_radius_m && target_radius_m <= radius_m[i + 1] {
            let denom = (radius_m[i + 1] - radius_m[i]).max(1.0e-300);
            let w = (target_radius_m - radius_m[i]) / denom;
            let qi = q_heat[i] * (1.0 - w) + q_heat[i + 1] * w;
            return rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / qi.max(1.0e-300);
        }
    }
    rho_kg_m3 * heat_capacity_j_kg_k * delta_t_k / q_heat[n - 1].max(1.0e-300)
}

/// Propagate a cavitation source PSD to every passive receiver channel.
///
/// The PSD scaling uses the squared magnitude of the attenuating acoustic
/// Green function, `exp(-2 alpha r)/(4 pi r)^2`, per receiver. Output is
/// flattened row-major `(n_receivers, n_freq)`.
#[must_use]
pub fn receiver_channel_psd_from_source(
    source_psd: &[f64],
    source_xyz: [f64; 3],
    receiver_xyz: &[f64],
    alpha_np_m: f64,
) -> Vec<f64> {
    if receiver_xyz.len() % 3 != 0 {
        return Vec::new();
    }
    let n_recv = receiver_xyz.len() / 3;
    let mut out = vec![0.0; n_recv * source_psd.len()];
    for ir in 0..n_recv {
        let j = 3 * ir;
        let dx = receiver_xyz[j] - source_xyz[0];
        let dy = receiver_xyz[j + 1] - source_xyz[1];
        let dz = receiver_xyz[j + 2] - source_xyz[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt().max(1.0e-9);
        let amp = (-alpha_np_m.max(0.0) * r).exp() / (4.0 * std::f64::consts::PI * r);
        let psd_gain = amp * amp;
        for (ifreq, &v) in source_psd.iter().enumerate() {
            out[ir * source_psd.len() + ifreq] = v.max(0.0) * psd_gain;
        }
    }
    out
}

/// Sum receiver-channel PSDs into a measured array spectrum.
#[must_use]
pub fn integrate_channel_psd(channel_psd: &[f64], n_receivers: usize, n_freq: usize) -> Vec<f64> {
    if channel_psd.len() != n_receivers.saturating_mul(n_freq) {
        return Vec::new();
    }
    let mut out = vec![0.0; n_freq];
    for ir in 0..n_receivers {
        for ifreq in 0..n_freq {
            out[ifreq] += channel_psd[ir * n_freq + ifreq].max(0.0);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clearance_clips_by_largest_ellipsoid_axis() {
        let r = clipped_lateral_radius_for_clearance(2.0e-3, 3.0e-3, 5.0);
        assert!((r - 0.6e-3).abs() < 1e-12);
        let unconstrained = clipped_lateral_radius_for_clearance(2.0e-3, 30.0e-3, 5.0);
        assert!((unconstrained - 2.0e-3).abs() < 1e-12);
    }

    #[test]
    fn ellipsoid_rejects_false_allowed_voxel() {
        let nx = 7;
        let ny = 7;
        let nz = 7;
        let mut mask = vec![true; nx * ny * nz];
        let idx = (3 * ny + 3) * nz + 4;
        mask[idx] = false;
        assert!(!ellipsoid_respects_allowed_mask(
            &mask, nx, ny, nz, 3, 3, 3, 1.1, 1.1, 1.0,
        ));
    }

    #[test]
    fn measured_spectrum_scaling_preserves_value_semantics() {
        let out = scale_measured_emission_spectrum(&[1.0, -2.0, 3.0], 0.5, 2.0);
        assert_eq!(out, vec![1.0, 0.0, 3.0]);
    }

    #[test]
    fn delivered_progress_is_monotone() {
        let k = delivered_histotripsy_progress(&[0.0, 1.0, 2.0], 1.0, 2.0);
        assert_eq!(k[0], 0.0);
        assert!(k[1] > k[0]);
        assert!(k[2] > k[1]);
    }

    #[test]
    fn boiling_lesion_from_profile_returns_physical_axes() {
        let r = [0.0, 0.5e-3, 1.0e-3, 1.5e-3];
        let b = [1.0, 0.9, 0.6, 0.2];
        let out = boiling_lesion_from_pressure_profile(
            &r, &b, 80.0e6, 80.0e-3, 1.0e6, 1540.0, 1060.0, 4.0, 5.0, 3600.0, 63.0, 20.0e-3, 4.0,
            10.0e-3, 0.95,
        )
        .expect("profile should boil");
        assert!(out.pulses >= 1);
        assert!(out.lateral_radius_m > 0.0);
        assert!((out.axial_radius_m / out.lateral_radius_m - 4.0).abs() < 1.0e-12);
        assert!(out.pulse_ms > 0.0 && out.pulse_ms <= 20.0);
    }

    #[test]
    fn boiling_time_profile_increases_as_pressure_falls() {
        let b = [1.0, 0.75, 0.5];
        let t = boiling_time_profile_from_pressure(
            &b, 80.0e6, 80.0e-3, 1.0e6, 1540.0, 1060.0, 4.0, 5.0, 3600.0, 63.0,
        );
        assert_eq!(t.len(), b.len());
        assert!(t[0] < t[1]);
        assert!(t[1] < t[2]);
    }

    #[test]
    fn receiver_channel_psd_decreases_with_range() {
        let psd = [2.0, 3.0];
        let recv = [0.01, 0.0, 0.0, 0.02, 0.0, 0.0];
        let ch = receiver_channel_psd_from_source(&psd, [0.0, 0.0, 0.0], &recv, 0.0);
        assert_eq!(ch.len(), 4);
        assert!(ch[0] > ch[2]);
        assert!(ch[1] > ch[3]);
        let sum = integrate_channel_psd(&ch, 2, 2);
        assert_eq!(sum.len(), 2);
        assert!((sum[0] - (ch[0] + ch[2])).abs() < 1.0e-12);
    }
}
