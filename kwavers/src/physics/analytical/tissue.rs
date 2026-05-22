//! Tissue acoustical properties for book chapter ch12.
//!
//! Covers: water sound speed and density vs temperature, B/A nonlinearity
//! parameters, power-law tissue absorption, Kramers–Kronig dispersion, and
//! a canonical tissue-property database.

use crate::core::constants::fundamental::{DENSITY_BLOOD, DENSITY_TISSUE, SOUND_SPEED_WATER};
use std::f64::consts::PI;

// ─── Water properties ─────────────────────────────────────────────────────────

/// Sound speed in water vs temperature using the Del Grosso–Mader polynomial.
///
/// ```text
/// c(T) = 1402.7 + 4.8·T − 0.0477·T² + 0.000248·T³   [m/s]
/// ```
/// Valid for T ∈ [0, 100] °C.
///
/// # Reference
/// Del Grosso & Mader (1972), *J. Acoust. Soc. Am.* 52, 1442.
pub fn water_sound_speed_temperature(t_celsius: &[f64]) -> Vec<f64> {
    t_celsius
        .iter()
        .map(|&t| 1402.7 + 4.8 * t - 0.0477 * t * t + 0.000248 * t * t * t)
        .collect()
}

/// Water density vs temperature (simplified quadratic fit).
///
/// ```text
/// ρ(T) = 1000.0 − 0.003975·(T − 4.0)²   [kg/m³]
/// ```
/// Error < 0.5 kg/m³ for T ∈ [0, 50] °C.
///
/// # Reference
/// Kell (1975), *J. Chem. Eng. Data* 20, 97.
pub fn water_density_temperature(t_celsius: &[f64]) -> Vec<f64> {
    t_celsius
        .iter()
        .map(|&t| {
            let dt = t - 4.0;
            1000.0 - 0.003975 * dt * dt
        })
        .collect()
}

// ─── Nonlinearity parameter ───────────────────────────────────────────────────

/// B/A nonlinearity parameter for common biological media.
///
/// Values from tabulated experimental data:
/// | Medium | B/A |
/// |---|---|
/// | water | 5.2 |
/// | blood | 6.1 |
/// | fat | 10.0 |
/// | liver | 7.6 |
/// | muscle | 7.4 |
/// | bone | 12.0 |
/// | brain | 6.8 |
/// | kidney | 7.8 |
/// | cartilage | 8.5 |
///
/// Returns 5.2 (water) for unknown tissue names.
///
/// # Reference
/// Hamilton & Blackstock (1998) *Nonlinear Acoustics*, App. B.
pub fn ba_parameter(medium: &str) -> f64 {
    match medium {
        "water" => 5.2,
        "blood" => 6.1,
        "fat" => 10.0,
        "liver" => 7.6,
        "muscle" => 7.4,
        "bone" => 12.0,
        "brain" => 6.8,
        "kidney" => 7.8,
        "cartilage" => 8.5,
        _ => 5.2,
    }
}

// ─── Tissue absorption ────────────────────────────────────────────────────────

/// Power-law tissue absorption at multiple frequencies.
///
/// Uses tissue-specific (α₀, y) pairs from the duck database:
/// ```text
/// α(f) = α₀ · f^y   [dB/cm],  f in MHz
/// ```
///
/// | Tissue | α₀ [dB/(cm·MHz^y)] | y |
/// |---|---|---|
/// | water | 0.002 | 2.0 |
/// | liver | 0.5 | 1.05 |
/// | muscle | 0.57 | 1.0 |
/// | fat | 0.48 | 1.0 |
/// | skull | 13.0 | 1.2 |
/// | blood | 0.14 | 1.21 |
/// | brain | 0.43 | 1.3 |
/// | kidney | 1.0 | 1.0 |
/// | cartilage | 2.0 | 1.5 |
///
/// # Reference
/// Duck (1990) *Physical Properties of Tissue*, Academic Press.
pub fn tissue_absorption_db_cm(f_mhz: &[f64], tissue: &str) -> Vec<f64> {
    let (alpha0, y) = tissue_absorption_params(tissue);
    f_mhz.iter().map(|&f| alpha0 * f.powf(y)).collect()
}

// ─── Kramers–Kronig dispersion ────────────────────────────────────────────────

/// Sound speed dispersion from Kramers–Kronig relations for power-law media.
///
/// For a medium with attenuation α(f) = α₀·f^y [Np/m]:
/// ```text
/// c(f) = c_ref + (α₀·c_ref²/π)·tan(πy/2)·(f^(y−1) − f_ref^(y−1))   [m/s]
/// ```
/// Valid for y ≠ 1; at y = 1 the dispersion is zero.
///
/// # Arguments
/// * `f_hz` – frequencies [Hz]
/// * `alpha0_np_m_hzy` – attenuation prefactor [Np/m/Hz^y]
/// * `y` – power-law exponent
/// * `f_ref_hz` – reference frequency at which c = c_ref [Hz]
/// * `c_ref` – sound speed at reference frequency [m/s]
///
/// # Reference
/// Szabo & Wu (2000), *J. Acoust. Soc. Am.* 107, 2437.
pub fn kramers_kronig_sound_speed(
    f_hz: &[f64],
    alpha0_np_m_hzy: f64,
    y: f64,
    f_ref_hz: f64,
    c_ref: f64,
) -> Vec<f64> {
    // For y = 1 the integral produces a logarithmic correction; tan(π/2) diverges.
    // The KK correction is typically small; for y ≈ 1 use first-order expansion.
    if (y - 1.0).abs() < 1e-6 {
        // Logarithmic KK for y = 1: c(f) ≈ c_ref (no dispersion in first order)
        return vec![c_ref; f_hz.len()];
    }
    let tan_term = (PI * y / 2.0).tan();
    let f_ref_pow = f_ref_hz.powf(y - 1.0);
    f_hz.iter()
        .map(|&f| {
            let f_pow = f.powf(y - 1.0);
            c_ref + alpha0_np_m_hzy * c_ref * c_ref / PI * tan_term * (f_pow - f_ref_pow)
        })
        .collect()
}

// ─── Tissue property database ─────────────────────────────────────────────────

/// Return canonical acoustic and nonlinear tissue properties.
///
/// Returns `(c [m/s], ρ [kg/m³], α₀ [dB/cm/MHz^y], y, B/A)`.
///
/// Sources: Duck (1990) and Szabo (2014) *Diagnostic Ultrasound Imaging*.
pub fn tissue_properties(tissue: &str) -> (f64, f64, f64, f64, f64) {
    // (c [m/s], rho [kg/m3], alpha0 [dB/cm/MHz^y], y, B/A)
    match tissue {
        "water" => (SOUND_SPEED_WATER, 998.0, 0.002, 2.0, 5.2),
        "liver" => (1578.0, DENSITY_BLOOD, 0.5, 1.05, 7.6),
        "muscle" => (1580.0, DENSITY_TISSUE, 0.57, 1.0, 7.4),
        "fat" => (1450.0, 950.0, 0.48, 1.0, 10.0),
        "skull" => (2900.0, 1900.0, 13.0, 1.2, 12.0),
        "blood" => (1584.0, DENSITY_BLOOD, 0.14, 1.21, 6.1),
        "brain" => (1560.0, 1040.0, 0.43, 1.3, 6.8),
        "kidney" => (1560.0, DENSITY_TISSUE, 1.0, 1.0, 7.8),
        "cartilage" => (1700.0, 1100.0, 2.0, 1.5, 8.5),
        _ => (1540.0, 1000.0, 0.5, 1.0, 6.0),
    }
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn tissue_absorption_params(tissue: &str) -> (f64, f64) {
    match tissue {
        "water" => (0.002, 2.0),
        "liver" => (0.5, 1.05),
        "muscle" => (0.57, 1.0),
        "fat" => (0.48, 1.0),
        "skull" => (13.0, 1.2),
        "blood" => (0.14, 1.21),
        "brain" => (0.43, 1.3),
        "kidney" => (1.0, 1.0),
        "cartilage" => (2.0, 1.5),
        _ => (0.5, 1.0),
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_speed_at_20c() {
        let c = water_sound_speed_temperature(&[20.0]);
        // Del Grosso value at 20°C ≈ 1482.4 m/s
        assert!((c[0] - 1482.4).abs() < 5.0, "c={}", c[0]);
    }

    #[test]
    fn water_density_at_4c_is_max() {
        let rho4 = water_density_temperature(&[4.0]);
        let rho10 = water_density_temperature(&[10.0]);
        assert!(rho4[0] > rho10[0]);
        assert!((rho4[0] - 1000.0).abs() < 0.5);
    }

    #[test]
    fn ba_water_known() {
        assert!((ba_parameter("water") - 5.2).abs() < 1e-10);
    }

    #[test]
    fn tissue_absorption_monotone_liver() {
        let f: Vec<f64> = vec![1.0, 2.0, 5.0];
        let a = tissue_absorption_db_cm(&f, "liver");
        assert!(a[0] < a[1] && a[1] < a[2]);
    }

    #[test]
    fn tissue_properties_water_sound_speed() {
        let (c, _, _, _, _) = tissue_properties("water");
        assert!((c - SOUND_SPEED_WATER).abs() < 1.0);
    }

    #[test]
    fn kk_dispersion_no_dispersion_at_y1() {
        let f = vec![0.5e6, 1e6, 2e6];
        let c = kramers_kronig_sound_speed(&f, 1.0, 1.0, 1e6, 1540.0);
        // At y=1 dispersion is zero; all values equal c_ref
        assert!(c.iter().all(|&v| (v - 1540.0).abs() < 1.0));
    }
}
