//! Bioheat and HIFU thermal physics for book chapter ch06.
//!
//! Covers: Pennes bioheat focal temperature rise (lumped ODE model),
//! HIFU focal pressure gain, and Gaussian acoustic power deposition.

use crate::core::constants::numerical::FOUR_PI;
use std::f64::consts::PI;

// ─── Bioheat focal temperature ────────────────────────────────────────────────

/// Lumped focal temperature rise from the Pennes bioheat equation.
///
/// At a focal hot-spot of volume V, the spatially averaged Pennes equation
/// reduces to a first-order ODE:
/// ```text
/// ρ_t·c_t·V·dT/dt = Q_dep − (k_t/L²)·(T − T_body)·V − w_b·ρ_b·c_b·(T − T_body)·V
/// ```
/// where L is the effective half-dimension of the focal volume (L = (3V/4π)^{1/3}
/// for a sphere).  The steady-state temperature rise is:
/// ```text
/// ΔT_ss = Q_w / (w_b·ρ_b·c_b + k_t/L²)
/// ```
/// and the transient response is exponential:
/// ```text
/// ΔT(t) = ΔT_ss·(1 − exp(−t/τ))
/// τ = ρ_t·c_t / (w_b·ρ_b·c_b + k_t/L²)
/// ```
///
/// # Arguments
/// * `t_arr` – time points [s]
/// * `acoustic_power_w` – absorbed acoustic power in the focal volume [W]
/// * `focal_volume_m3` – focal volume V [m³]
/// * `k_tissue` – tissue thermal conductivity k_t [W/(m·K)]
/// * `rho_tissue` – tissue density ρ_t [kg/m³]
/// * `cp_tissue` – tissue specific heat c_t [J/(kg·K)]
/// * `wb_perfusion` – blood perfusion rate w_b [kg/(m³·s)]
/// * `rho_blood` – blood density ρ_b [kg/m³]
/// * `cb_blood` – blood specific heat c_b [J/(kg·K)]
/// * `t_body_c` – body temperature [°C]
///
/// Returns absolute temperature [°C] at each time point.
///
/// # Reference
/// Pennes (1948), *J. Appl. Physiol.* 1, 93.
#[must_use]
pub fn bioheat_focal_temperature_rise(
    t_arr: &[f64],
    acoustic_power_w: f64,
    focal_volume_m3: f64,
    k_tissue: f64,
    rho_tissue: f64,
    cp_tissue: f64,
    wb_perfusion: f64,
    rho_blood: f64,
    cb_blood: f64,
    t_body_c: f64,
) -> Vec<f64> {
    // Effective half-dimension for spherical focal volume
    let l = (3.0 * focal_volume_m3 / (FOUR_PI)).powf(1.0 / 3.0);
    let perfusion_term = wb_perfusion * rho_blood * cb_blood; // [W/(m³·K)]
    let conduction_term = k_tissue / (l * l); // [W/(m³·K)]
    let loss = perfusion_term + conduction_term;
    let q_density = acoustic_power_w / focal_volume_m3; // [W/m³]
    let t_ss_rise = q_density / loss; // [K] steady-state above body temp
    let tau = rho_tissue * cp_tissue / loss; // [s]

    t_arr
        .iter()
        .map(|&t| t_body_c + t_ss_rise * (1.0 - (-t / tau).exp()))
        .collect()
}

// ─── HIFU focal gain ──────────────────────────────────────────────────────────

/// HIFU focal pressure gain for a focused transducer.
///
/// Simplified directivity-theory result (O'Neil 1949; Hynynen 1991):
/// ```text
/// G_p = π·D·f / (4·c·F#)   [dimensionless]
/// ```
/// where D is the aperture diameter, F# = focal_length/aperture, and
/// G_p is the ratio of focal to source pressure amplitude.
///
/// Derivation: for a spherical-cap of radius a = D/2 and focal length F,
/// the paraxial peak pressure gain is G = k·a²/(2F) = π·f·D²/(4·c·F).
/// Using F# = F/D gives G = π·f·D/(4·c·F#).
///
/// # Arguments
/// * `aperture_m` – transducer aperture diameter D [m]
/// * `f_number` – F-number (focal_length / aperture)
/// * `freq_hz` – frequency [Hz]
/// * `c` – sound speed [m/s]
///
/// # Reference
/// O'Neil HT (1949), *J. Acoust. Soc. Am.* 21, 516–526.
/// Hynynen K (1991), *Ultrasound Med. Biol.* 17, 157–169.
#[must_use]
#[inline]
pub fn hifu_focal_pressure_gain(aperture_m: f64, f_number: f64, freq_hz: f64, c: f64) -> f64 {
    PI * aperture_m * freq_hz / (c * 4.0 * f_number)
}

// ─── Gaussian power deposition ────────────────────────────────────────────────

/// 2-D Gaussian acoustic power deposition density.
///
/// Models the absorbed power density in a focused Gaussian beam:
/// ```text
/// Q(r, z) = 2·α·I(r, z)
/// I(r, z) = I₀·(w₀/w(z))²·exp(−2r²/w(z)²)·exp(−2α|z−z_f|)
/// I₀ = p₀²/(2·ρ·c)   [W/m²]
/// w(z) = w₀·√(1 + ((z−z_f)/z_R)²),  z_R = π·w₀²·f/c
/// ```
///
/// Output is a flattened row-major Vec of size `NR × NZ` [W/m³].
///
/// # Arguments
/// * `r_arr` – radial positions [m]
/// * `z_arr` – axial positions [m]
/// * `freq_hz` – frequency [Hz]
/// * `z_focus_m` – axial focal position [m]
/// * `p0_pa` – source pressure amplitude [Pa]
/// * `c` – sound speed [m/s]
/// * `rho` – density [kg/m³]
/// * `alpha_np_m` – attenuation at fundamental [Np/m]
/// * `w0_m` – beam waist radius at focus [m]
///
/// # Reference
/// O'Neil (1949); Soneson (2011), *J. Acoust. Soc. Am.* 130, EL158.
#[must_use]
pub fn gaussian_power_deposition_2d(
    r_arr: &[f64],
    z_arr: &[f64],
    freq_hz: f64,
    z_focus_m: f64,
    p0_pa: f64,
    c: f64,
    rho: f64,
    alpha_np_m: f64,
    w0_m: f64,
) -> Vec<f64> {
    let i0 = p0_pa * p0_pa / (2.0 * rho * c); // [W/m²]
    let z_r = PI * w0_m * w0_m * freq_hz / c; // Rayleigh range [m]
    let nr = r_arr.len();
    let nz = z_arr.len();
    let mut out = vec![0.0_f64; nr * nz];

    for (ir, &r) in r_arr.iter().enumerate() {
        for (iz, &z) in z_arr.iter().enumerate() {
            let dz = z - z_focus_m;
            let w = w0_m * (1.0 + (dz / z_r).powi(2)).sqrt();
            let intensity = i0
                * (w0_m / w).powi(2)
                * (-2.0 * r * r / (w * w)).exp()
                * (-2.0 * alpha_np_m * dz.abs()).exp();
            out[ir * nz + iz] = 2.0 * alpha_np_m * intensity;
        }
    }
    out
}

// ─── Beer-Lambert acoustic intensity depth profile ────────────────────────────

/// Acoustic intensity depth profile for a plane wave with Beer-Lambert attenuation.
///
/// One-way intensity attenuation of a propagating acoustic beam:
/// ```text
/// I(z) = I₀ · exp(−2·α·z)
/// ```
/// The factor of 2 arises because intensity is proportional to pressure squared;
/// amplitude attenuation coefficient α [Np/m] produces intensity attenuation 2α.
///
/// # Arguments
/// * `z_arr` – depth positions [m], z ≥ 0
/// * `alpha_np_m` – amplitude attenuation coefficient [Np/m]
/// * `surface_intensity` – I₀ at z = 0 [W/m² or normalised]
///
/// # Reference
/// Duck (1990) *Physical Properties of Tissue*, §2.1. Academic Press.
#[must_use]
pub fn acoustic_intensity_depth_profile(
    z_arr: &[f64],
    alpha_np_m: f64,
    surface_intensity: f64,
) -> Vec<f64> {
    z_arr
        .iter()
        .map(|&z| surface_intensity * (-2.0 * alpha_np_m * z).exp())
        .collect()
}

/// Volumetric acoustic power deposition depth profile (Beer-Lambert heat source).
///
/// Absorbed power density Q [W/m³] at depth z for a propagating plane wave:
/// ```text
/// Q(z) = 2·α · I(z) = 2·α·I₀ · exp(−2·α·z)
/// ```
/// This is the thermal source term Q entering the Pennes bioheat equation.
///
/// # Arguments
/// * `z_arr` – depth positions [m], z ≥ 0
/// * `alpha_np_m` – amplitude attenuation coefficient [Np/m]
/// * `surface_intensity` – surface intensity I₀ at z = 0 [W/m²]
///
/// # Reference
/// Duck (1990) *Physical Properties of Tissue*, §2.1. Academic Press.
/// Pennes (1948), *J. Appl. Physiol.* 1, 93.
#[must_use]
pub fn acoustic_power_deposition_depth_profile(
    z_arr: &[f64],
    alpha_np_m: f64,
    surface_intensity: f64,
) -> Vec<f64> {
    z_arr
        .iter()
        .map(|&z| 2.0 * alpha_np_m * surface_intensity * (-2.0 * alpha_np_m * z).exp())
        .collect()
}

// ─── Acoustic heat-source density from a 3-D pressure field ──────────────────

/// Convert a 3-D (or arbitrary-shape) acoustic pressure field to the
/// corresponding volumetric heat-source density for the Pennes bioheat equation.
///
/// For a CW or time-averaged pressure field p(x,y,z) in a medium with amplitude
/// attenuation α [Np/m], density ρ [kg/m³], and speed of sound c [m/s]:
/// ```text
/// Q(x,y,z) = α · p(x,y,z)² / (ρ · c)
/// ```
/// Derivation:
///   Time-averaged intensity:  I = p² / (2·ρ·c)
///   Absorbed power density:   Q = 2α · I = α · p² / (ρ · c)
///
/// The factor of 2 in `2α·I` accounts for amplitude (pressure) attenuation
/// producing intensity attenuation at rate 2α, but when expressed directly in
/// terms of pressure the factor cancels against the 1/(2) in I = p²/(2ρc).
///
/// The input `p_field` is a **flattened, row-major** slice of the pressure
/// amplitude array in [Pa]; the output is a `Vec<f64>` of the same length,
/// with the same linear index ordering, in [W/m³].  Reshaping to (nx,ny,nz)
/// is the caller's responsibility.
///
/// # Arguments
/// * `p_field`    – pressure amplitude field [Pa], arbitrary shape, flattened
/// * `alpha_np_m` – amplitude attenuation coefficient [Np/m]
/// * `rho`        – medium density [kg/m³]
/// * `c`          – medium speed of sound [m/s]
///
/// # Reference
/// Pennes (1948), *J. Appl. Physiol.* 1, 93.
/// Duck (1990) *Physical Properties of Tissue*, §5.2. Academic Press.
#[must_use]
pub fn acoustic_heat_source_density(
    p_field: &[f64],
    alpha_np_m: f64,
    rho: f64,
    c: f64,
) -> Vec<f64> {
    let inv_rhoc = alpha_np_m / (rho * c);
    p_field.iter().map(|&p| p * p * inv_rhoc).collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use crate::core::constants::numerical::{FOUR_PI, MHZ_TO_HZ, MPA_TO_PA};
    use crate::core::constants::thermodynamic::BODY_TEMPERATURE_C;
    use crate::core::constants::tissue_acoustics::DENSITY_BLOOD;
    use crate::core::constants::tissue_thermal::{SPECIFIC_HEAT_BLOOD, SPECIFIC_HEAT_TISSUE};

    #[test]
    fn bioheat_at_t0_is_body_temp() {
        let t = bioheat_focal_temperature_rise(
            &[0.0],
            10.0,
            1e-6,
            0.5,
            DENSITY_BLOOD,
            SPECIFIC_HEAT_TISSUE,
            5.0,
            DENSITY_BLOOD,
            SPECIFIC_HEAT_BLOOD,
            BODY_TEMPERATURE_C,
        );
        assert!((t[0] - BODY_TEMPERATURE_C).abs() < 1e-8);
    }

    #[test]
    fn bioheat_monotone_increasing() {
        let tvec: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let temp = bioheat_focal_temperature_rise(
            &tvec,
            10.0,
            1e-6,
            0.5,
            DENSITY_BLOOD,
            SPECIFIC_HEAT_TISSUE,
            5.0,
            DENSITY_BLOOD,
            SPECIFIC_HEAT_BLOOD,
            BODY_TEMPERATURE_C,
        );
        for i in 1..temp.len() {
            assert!(
                temp[i] >= temp[i - 1],
                "T[{}]={} < T[{}]={}",
                i,
                temp[i],
                i - 1,
                temp[i - 1]
            );
        }
    }

    #[test]
    fn bioheat_approaches_steady_state() {
        let t_long = 3600.0_f64; // 1 hour — far beyond τ
        let t = bioheat_focal_temperature_rise(
            &[0.0, t_long],
            10.0,
            1e-6,
            0.5,
            DENSITY_BLOOD,
            SPECIFIC_HEAT_TISSUE,
            5.0,
            DENSITY_BLOOD,
            SPECIFIC_HEAT_BLOOD,
            BODY_TEMPERATURE_C,
        );
        // Should saturate: T(∞) < T_body + some bound
        assert!(t[1] > BODY_TEMPERATURE_C && t[1] < 200.0);
        // Verify saturation: T(t_long) ≈ T(t_long/2)
        let t_half = bioheat_focal_temperature_rise(
            &[t_long / 2.0],
            10.0,
            1e-6,
            0.5,
            DENSITY_BLOOD,
            SPECIFIC_HEAT_TISSUE,
            5.0,
            DENSITY_BLOOD,
            SPECIFIC_HEAT_BLOOD,
            BODY_TEMPERATURE_C,
        );
        assert!((t[1] - t_half[0]).abs() / t[1].abs() < 0.01);
    }

    #[test]
    fn beer_lambert_intensity_at_zero_is_surface() {
        let i = acoustic_intensity_depth_profile(&[0.0], 7.0, 1.0);
        assert!((i[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn beer_lambert_intensity_monotone_decreasing() {
        let z: Vec<f64> = vec![0.0, 0.01, 0.02, 0.04];
        let i = acoustic_intensity_depth_profile(&z, 7.0, 1.0);
        for k in 1..i.len() {
            assert!(i[k] < i[k - 1], "I[{}]={} ≥ I[{}]={}", k, i[k], k - 1, i[k - 1]);
        }
    }

    #[test]
    fn beer_lambert_power_deposition_peak_at_surface() {
        // Q(0) = 2·α·I₀ is the maximum; Q(z) < Q(0) for z > 0
        let z: Vec<f64> = vec![0.0, 0.01];
        let q = acoustic_power_deposition_depth_profile(&z, 7.0, 1.0);
        assert!((q[0] - 2.0 * 7.0 * 1.0).abs() < 1e-12, "Q(0)={}", q[0]);
        assert!(q[1] < q[0]);
    }

    #[test]
    fn hifu_gain_positive() {
        let g = hifu_focal_pressure_gain(0.1, 1.5, MHZ_TO_HZ, SOUND_SPEED_WATER_SIM);
        assert!(g > 1.0, "g={}", g);
    }

    #[test]
    fn gaussian_deposition_peak_at_focus() {
        let r = vec![0.0];
        let z: Vec<f64> = vec![-5e-3, 0.0, 5e-3];
        let q = gaussian_power_deposition_2d(
            &r,
            &z,
            MHZ_TO_HZ,
            0.0,
            MPA_TO_PA,
            SOUND_SPEED_WATER_SIM,
            DENSITY_BLOOD,
            1.0,
            1e-3,
        );
        // Q at focus (z=0) should exceed Q at z=±5mm
        assert!(q[1] > q[0] && q[1] > q[2]);
    }

    #[test]
    fn heat_source_density_zero_pressure_is_zero() {
        let q = acoustic_heat_source_density(&[0.0], 7.0, 1060.0, 1540.0);
        assert!(q[0].abs() < 1e-30, "Q(0)={}", q[0]);
    }

    #[test]
    fn heat_source_density_formula_matches_analytic() {
        // Q = α·p²/(ρ·c)
        // For p = 1 MPa, α = 7 Np/m, ρ = 1060 kg/m³, c = 1540 m/s:
        // Q = 7 × (1e6)² / (1060 × 1540) ≈ 4283.5 W/m³  (per Pascal²)
        let p = 1.0e6_f64;
        let alpha = 7.0_f64;
        let rho = 1060.0_f64;
        let c = 1540.0_f64;
        let expected = alpha * p * p / (rho * c);
        let q = acoustic_heat_source_density(&[p], alpha, rho, c);
        assert!(
            (q[0] - expected).abs() / expected < 1e-12,
            "Q={} expected={}",
            q[0],
            expected
        );
    }

    #[test]
    fn heat_source_density_quadratic_in_pressure() {
        // Doubling pressure → quadrupling Q
        let alpha = 5.0_f64;
        let rho = 1040.0_f64;
        let c = 1543.0_f64;
        let q1 = acoustic_heat_source_density(&[1.0e5], alpha, rho, c)[0];
        let q2 = acoustic_heat_source_density(&[2.0e5], alpha, rho, c)[0];
        assert!(
            (q2 / q1 - 4.0).abs() < 1e-10,
            "quadratic scaling violated: q2/q1={}",
            q2 / q1
        );
    }
}
