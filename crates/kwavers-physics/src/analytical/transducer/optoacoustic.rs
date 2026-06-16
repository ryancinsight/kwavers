//! Optically-generated focused ultrasound (OFUS) from a soft optoacoustic pad
//! (SOAP), after Li et al., *Light: Sci. Appl.* **11**, 321 (2022).
//!
//! A SOAP is a spherically curved optoacoustic emitter: a nanosecond laser
//! pulse illuminates a thin light-absorbing nanocomposite layer coated on a
//! curved PDMS surface, and the thermoelastic stress launches an ultrasound
//! wave from every surface element simultaneously. Because the surface is a
//! spherical cap, the wavefronts converge at the geometric centre, producing a
//! focus *by geometry alone* — no electronic delays. The high numerical
//! aperture achievable with a soft (crack-free) absorber gives a far tighter
//! focus and higher focal gain than a piezoelectric transducer at the same
//! frequency.
//!
//! This module provides the closed-form design relations: the spherical-cap
//! geometry ⇄ numerical aperture ⇄ f-number conversions, the focal pressure
//! gain `G` (the paper's Eq. 2), and the acoustic-resolution lateral resolution
//! (the paper's Eq. 1). The optoacoustic *source amplitude* `p₀ = Γ·μ_a·F` and
//! the absorber materials live in `kwavers_optics` and `kwavers_medium`.

use kwavers_core::constants::numerical::TWO_PI;

/// Numerical aperture of a spherical-cap SOAP from its geometry.
///
/// For a cap of radius of curvature `r` and transverse aperture diameter `D_t`,
/// the aperture half-angle `θ` satisfies `sin θ = (D_t/2)/r`, so
/// `NA = sin θ = D_t / (2 r)` (in the surrounding medium, refractive index 1).
///
/// Returns the value clamped to `[0, 1]`; `NA = 1` is the physical limit (full
/// hemispherical aperture is `NA = 1` only in the `D_t = 2r` limit).
///
/// # Reference
/// Li et al. (2022), *Light: Sci. Appl.* 11, 321.
#[must_use]
#[inline]
pub fn numerical_aperture_from_geometry(radius_m: f64, transverse_diameter_m: f64) -> f64 {
    if radius_m <= 0.0 {
        return 0.0;
    }
    (transverse_diameter_m / (2.0 * radius_m)).clamp(0.0, 1.0)
}

/// f-number of a focusing aperture from its numerical aperture.
///
/// The SOAP f-number is the ratio of the radius of curvature to the transverse
/// diameter, `f_N = r / D_t`. Since `NA = D_t/(2r)`, this is exactly
/// `f_N = 1 / (2·NA)`. A full hemisphere (`NA → 1`) gives the minimum
/// `f_N = 0.5`.
#[must_use]
#[inline]
pub fn f_number_from_na(na: f64) -> f64 {
    if na <= 0.0 {
        return f64::INFINITY;
    }
    1.0 / (2.0 * na)
}

/// Numerical aperture from f-number, the inverse of [`f_number_from_na`].
#[must_use]
#[inline]
pub fn na_from_f_number(f_number: f64) -> f64 {
    if f_number <= 0.0 {
        return 1.0;
    }
    (1.0 / (2.0 * f_number)).clamp(0.0, 1.0)
}

/// Focal pressure gain `G` of a spherically focused source (paper Eq. 2):
///
/// ```text
/// G = (2πf/c₀)·r·(1 − √(1 − 1/(4 f_N²)))
/// ```
///
/// the ratio of the on-axis focal pressure to the pressure on the emitting
/// surface. With `f_N = 1/(2·NA)` the radicand is `1 − NA²`, so
/// `G = (2πf/c₀)·r·(1 − √(1 − NA²)) = k·h`, where `k = 2πf/c₀` is the wavenumber
/// and `h = r(1 − cos θ) = r(1 − √(1 − NA²))` is the sagitta (depth) of the cap.
/// This is the same focal-gain limit as the O'Neil focused-bowl on-axis result
/// (`focused_bowl_onaxis`), confirming the two derivations agree.
///
/// This is the lossless geometric gain; medium attenuation over the path `r`
/// reduces it by a few percent (e.g. ≈3 % for water at 15 MHz over 6.35 mm,
/// giving the paper's reported `G_max ≈ 280`).
///
/// Requires `f_N ≥ 0.5` (a spherical cap cannot exceed a hemisphere); returns
/// `0` for non-physical inputs (`f_N < 0.5`, non-positive `c₀`/`r`/`f`).
///
/// # Reference
/// Li et al. (2022), Eq. 2; O'Neil (1949).
#[must_use]
pub fn soap_focal_gain(freq_hz: f64, c0: f64, radius_m: f64, f_number: f64) -> f64 {
    if c0 <= 0.0 || radius_m <= 0.0 || freq_hz <= 0.0 || f_number < 0.5 {
        return 0.0;
    }
    let k = TWO_PI * freq_hz / c0;
    let radicand = 1.0 - 1.0 / (4.0 * f_number * f_number);
    k * radius_m * (1.0 - radicand.max(0.0).sqrt())
}

/// Acoustic-resolution lateral resolution (−6 dB focal width), paper Eq. 1:
///
/// ```text
/// R_L = 0.71 · ν / (NA · f)
/// ```
///
/// where `ν` is the ambient sound speed and `f` the acoustic centre frequency.
/// This is the standard acoustic-resolution photoacoustic-microscopy lateral
/// resolution; it is inversely proportional to `NA`, so the high-NA SOAP
/// achieves a far tighter focus than a low-NA piezo transducer at the same `f`.
///
/// At the paper's CS-PDMS operating point (`ν = 1500 m/s`, `f = 15 MHz`) this
/// reduces to the reported empirical fit `R_L[µm] ≈ 71.5 / NA`
/// (0.71·1500/15 = 71.0), so the fit and Eq. 1 are the same relation.
///
/// # Reference
/// Li et al. (2022), Eq. 1; Yao & Wang (2013), acoustic-resolution PAM.
#[must_use]
#[inline]
pub fn acoustic_resolution_lateral(sound_speed: f64, na: f64, freq_hz: f64) -> f64 {
    if na <= 0.0 || freq_hz <= 0.0 {
        return f64::INFINITY;
    }
    0.71 * sound_speed / (na * freq_hz)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analytical::transducer::focused_bowl_onaxis;
    use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
    use kwavers_core::constants::numerical::MHZ_TO_HZ;

    // CS-PDMS SOAP device geometry (Li et al. 2022).
    const R: f64 = 6.35e-3; // radius of curvature [m]
    const D_T: f64 = 12.1e-3; // transverse aperture diameter [m]
    const F: f64 = 15.0 * MHZ_TO_HZ; // 15 MHz centre frequency
    const C0: f64 = SOUND_SPEED_WATER_SIM; // 1500 m/s

    #[test]
    fn numerical_aperture_matches_device() {
        // D_t/(2r) = 12.1/(2·6.35) = 0.953 ⇒ the paper's NA ≈ 0.95–0.96.
        let na = numerical_aperture_from_geometry(R, D_T);
        assert!((na - 0.9528).abs() < 1e-3, "NA = {na}");
        assert!((0.95..=0.96).contains(&na));
    }

    #[test]
    fn na_f_number_round_trip() {
        // Device f_N = r/D_t = 6.35/12.1 = 0.525, and f_N = 1/(2·NA).
        let na = numerical_aperture_from_geometry(R, D_T);
        let f_n = f_number_from_na(na);
        assert!((f_n - R / D_T).abs() < 1e-6, "f_N = {f_n}");
        assert!((f_n - 0.525).abs() < 2e-3);
        // round-trip NA → f_N → NA
        assert!((na_from_f_number(f_n) - na).abs() < 1e-12);
    }

    #[test]
    fn focal_gain_matches_paper_and_oneil() {
        let na = numerical_aperture_from_geometry(R, D_T);
        let f_n = f_number_from_na(na);
        let g = soap_focal_gain(F, C0, R, f_n);
        // Precise device geometry (f_N = 0.525) gives G = 277.8; the paper's
        // "G_max ≈ 280" rounds f_N → 0.52 (lossless 289, ≈280 after water
        // attenuation). 277.8 matches "≈280" to < 1 %.
        assert!((275.0..=290.0).contains(&g), "G = {g}");

        // Internal consistency: G = k·h must equal the O'Neil focused-bowl
        // focal-gain limit |p(F)|/p₀ from focused_bowl_onaxis (aperture radius
        // a = D_t/2, focal length = radius of curvature r).
        let a = D_T / 2.0;
        let p_focus = focused_bowl_onaxis(&[R], a, R, F, 1.0, C0)[0];
        assert!(
            (g - p_focus).abs() / g < 1e-6,
            "Eq.2 gain {g} != O'Neil k·h {p_focus}"
        );
    }

    #[test]
    fn focal_gain_rejects_subhemispherical_fnumber() {
        // f_N < 0.5 is geometrically impossible for a spherical cap.
        assert_eq!(soap_focal_gain(F, C0, R, 0.49), 0.0);
        // f_N = 0.5 (NA = 1, hemisphere): G = k·r (full sagitta = r).
        let g = soap_focal_gain(F, C0, R, 0.5);
        assert!((g - TWO_PI * F / C0 * R).abs() / g < 1e-12);
    }

    #[test]
    fn lateral_resolution_matches_paper() {
        let na = numerical_aperture_from_geometry(R, D_T);
        let r_l = acoustic_resolution_lateral(C0, na, F);
        // Paper: ~75 µm at NA ≈ 0.95, 15 MHz.
        assert!((r_l - 74.5e-6).abs() < 2e-6, "R_L = {} µm", r_l * 1e6);

        // Inversely proportional to NA (the empirical R = 71.5/NA fit): halving
        // NA doubles the resolution width.
        let r_half = acoustic_resolution_lateral(C0, na / 2.0, F);
        assert!((r_half - 2.0 * r_l).abs() / r_l < 1e-12);

        // Fit constant: R_L[µm]·NA = 0.71·c/f = 71.0 µm at the device point.
        assert!((r_l * na * 1e6 - 71.0).abs() < 0.5, "fit = {}", r_l * na * 1e6);
    }
}
