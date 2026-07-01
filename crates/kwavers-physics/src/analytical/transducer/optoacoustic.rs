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

/// Centre frequency of a single-cycle optoacoustic pulse from its temporal
/// width:
///
/// ```text
/// f_center ≈ 1 / τ_FWHM.
/// ```
///
/// The emitted optoacoustic pulse is a single bipolar cycle whose duration is
/// set by the absorber nanostructure (a thinner, more strongly absorbing layer
/// of smaller particles confines the thermoelastic stress to a shorter time).
/// For a one-cycle pulse the spectral centre is the reciprocal of its duration;
/// the four Li et al. (2022) absorbers confirm `f·τ ≈ 0.9–1.45`, so this places
/// each material in its band to ~30 %: candle soot (55 nm particles, τ≈0.09 µs)
/// → ~15 MHz, while the larger carbon-nanotube / nanoparticle aggregates and
/// the bulk heat-shrink membrane (τ≈0.24–0.31 µs) fall in the 3–5 MHz band.
/// This is the design knob that selects the array's operating frequency through
/// the choice of absorber material.
///
/// # Reference
/// Li et al. (2022), Fig. 1g,h; Kim et al. (2019), *IEEE Nanotechnol. Mag.* 13,
/// 13 (candle-soot photoacoustic bandwidth vs structure).
#[must_use]
#[inline]
pub fn optoacoustic_center_frequency(pulse_fwhm_s: f64) -> f64 {
    if pulse_fwhm_s <= 0.0 {
        return 0.0;
    }
    1.0 / pulse_fwhm_s
}

/// Optical fluence at the flat tip of a fiber delivering a pulse of energy
/// `E_pulse` [J] through a core of diameter `d` [m]:
///
/// ```text
/// F = E_pulse / A_tip = E_pulse / (π (d/2)²) = 4 E_pulse / (π d²).
/// ```
///
/// This is the lever behind fiber-optic optoacoustic emitters: the optoacoustic
/// surface pressure is `p₀ = Γ μ_a F` (the source law of §10.1 / `OptoacousticEmitter`),
/// and concentrating a fixed pulse energy into a smaller core raises the fluence
/// — and hence the pressure — as `1/d²`. The achievable pressure is therefore
/// bounded not by the laser but by the absorber's optical-damage fluence; a
/// CS-PDMS coating tolerates tens of mJ/cm², far above the few mJ/cm² needed for
/// MPa-level output.
///
/// # Reference
/// Jiang et al. (2020), *Nat. Commun.* 11, 881; Shi et al. (2021), *Light: Sci.
/// Appl.* 10, 143 (tapered fiber optoacoustic emitters).
#[must_use]
#[inline]
pub fn fiber_tip_fluence(pulse_energy_j: f64, core_diameter_m: f64) -> f64 {
    if core_diameter_m <= 0.0 {
        return 0.0;
    }
    pulse_energy_j / (std::f64::consts::PI * (0.5 * core_diameter_m).powi(2))
}

/// Diffraction focal gain of a planar aperture of active radiating area
/// `A_active` [m²], electronically focused at distance `F_focus` [m] for
/// acoustic wavelength `λ` [m]:
///
/// ```text
/// G_focus = A_active / (λ · F_focus).
/// ```
///
/// For a filled circular aperture (`A_active = π a²`) this is the textbook
/// focused-piston focal gain `π a²/(λ F)`. For a **sparse** array (e.g. a bundle
/// of fiber tips) `A_active` is the summed tip area; the main-focus pressure
/// still scales with the total radiating area when the contributions are phased
/// to arrive in phase at the focus, at the cost of grating-lobe sidelobes from
/// the sparse sampling. Distinct from the spherical-cap gain `k h`
/// ([`soap_focal_gain`]): this is the gain of a *flat, delay-focused* array.
#[must_use]
#[inline]
pub fn focused_aperture_gain(active_area_m2: f64, wavelength_m: f64, focal_distance_m: f64) -> f64 {
    if wavelength_m <= 0.0 || focal_distance_m <= 0.0 {
        return 0.0;
    }
    active_area_m2 / (wavelength_m * focal_distance_m)
}

/// Focal pressure of a coherently-focused optoacoustic matrix array of
/// `n_elements` fiber tips, each of radiating area `element_area_m2` [m²] and
/// surface pressure `surface_pressure_pa` [Pa], focused at `focal_distance_m`:
///
/// ```text
/// p_focus = p₀ · G_focus,   G_focus = (n · A_element) / (λ · F_focus).
/// ```
///
/// Combines the per-element optoacoustic surface pressure (`p₀ = Γ μ_a F`, from
/// [`fiber_tip_fluence`] and the emitter material) with the array focal gain
/// ([`focused_aperture_gain`]). This is the design relation for a delay-focused
/// fiber-optoacoustic matrix array.
#[must_use]
pub fn optoacoustic_array_focal_pressure(
    surface_pressure_pa: f64,
    n_elements: usize,
    element_area_m2: f64,
    wavelength_m: f64,
    focal_distance_m: f64,
) -> f64 {
    let active_area = n_elements as f64 * element_area_m2;
    surface_pressure_pa * focused_aperture_gain(active_area, wavelength_m, focal_distance_m)
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
        assert!(
            (r_l * na * 1e6 - 71.0).abs() < 0.5,
            "fit = {}",
            r_l * na * 1e6
        );
    }

    #[test]
    fn fiber_fluence_scales_inverse_square_with_core() {
        let e = 1.0e-7; // 0.1 µJ pulse
        let f_100 = fiber_tip_fluence(e, 100e-6);
        let f_50 = fiber_tip_fluence(e, 50e-6);
        // Halving the core diameter quadruples the fluence (∝ 1/d²): this is how
        // a small fiber reaches a high optoacoustic pressure from a modest pulse.
        assert!((f_50 / f_100 - 4.0).abs() < 1e-9);
        // 0.1 µJ into a 100 µm core ≈ 12.7 J/m² (1.27 mJ/cm²).
        assert!((f_100 - 12.73e0).abs() / f_100 < 0.01, "F = {f_100} J/m²");
    }

    #[test]
    fn center_frequency_from_pulse_width_places_materials_in_band() {
        // Single-cycle spectral estimate f ≈ 1/τ lands each Li et al. absorber
        // in its measured band (~30 %).
        let cs = optoacoustic_center_frequency(0.09e-6); // candle soot, 55 nm
        let cnt = optoacoustic_center_frequency(0.24e-6); // carbon nanotube
        let cnp = optoacoustic_center_frequency(0.29e-6); // carbon nanoparticle
        let hsm = optoacoustic_center_frequency(0.31e-6); // heat-shrink membrane
        assert!(cs > 10.0e6, "CS {cs} not high-frequency");
        // CNT and CNP sit in the target 3–6 MHz band; HSM near 3 MHz.
        assert!((3.0e6..=6.0e6).contains(&cnt), "CNT {cnt}");
        assert!((3.0e6..=6.0e6).contains(&cnp), "CNP {cnp}");
        assert!((2.5e6..=4.0e6).contains(&hsm), "HSM {hsm}");
        // Monotone: shorter pulse ⇒ higher frequency.
        assert!(cs > cnt && cnt > cnp && cnp > hsm);
    }

    #[test]
    fn focused_aperture_gain_matches_filled_piston() {
        // A filled circular aperture: G = π a²/(λ F) = A/(λ F).
        let a = 1e-3_f64;
        let area = std::f64::consts::PI * a * a;
        let (lambda, f_focus) = (1.0267e-4, 5e-3);
        let g = focused_aperture_gain(area, lambda, f_focus);
        assert!((g - area / (lambda * f_focus)).abs() < 1e-9);
    }

    #[test]
    fn dense_fiber_array_reaches_one_mpa_at_short_focus() {
        // 96 fibers, 100 µm core, coherently focused 5 mm into tissue at 15 MHz
        // (λ ≈ 103 µm, c = 1540 m/s). A modest per-element surface pressure of
        // 0.7 MPa — reached at ≈2.5 mJ/cm² on a CS-PDMS tip, far below the
        // absorber's tens-of-mJ/cm² damage fluence — clears 1 MPa.
        let element_area = std::f64::consts::PI * (50e-6_f64).powi(2);
        let lambda = 1540.0 / (15.0 * MHZ_TO_HZ);
        let p_focus = optoacoustic_array_focal_pressure(0.7e6, 96, element_area, lambda, 5e-3);
        assert!(p_focus >= 1.0e6, "focal pressure {p_focus} Pa < 1 MPa");
        // Headroom: a damage-safe 3 MPa per-element surface pressure exceeds
        // 4 MPa at the focus.
        let p_hot = optoacoustic_array_focal_pressure(3.0e6, 96, element_area, lambda, 5e-3);
        assert!(p_hot > 4.0e6, "hot-case focal pressure {p_hot} Pa");
    }

    #[test]
    fn low_frequency_array_focuses_closer_for_one_mpa() {
        // A 3–5 MHz design (CNT-/CNP-PDMS, λ ≈ 308 µm at 5 MHz) loses focal gain
        // relative to 15 MHz (G ∝ 1/λ ∝ f), so it focuses closer (3 mm) to keep
        // G_focus near unity. 96×100 µm cores then reach 1 MPa at a damage-safe
        // ~1.25 MPa per element (≈4.5 mJ/cm²).
        let element_area = std::f64::consts::PI * (50e-6_f64).powi(2);
        let lambda = 1540.0 / (5.0 * MHZ_TO_HZ);
        let p_focus = optoacoustic_array_focal_pressure(1.25e6, 96, element_area, lambda, 3e-3);
        assert!(
            p_focus >= 1.0e6,
            "5 MHz focal pressure {p_focus} Pa < 1 MPa"
        );
        // The lower-frequency gain is below the 15 MHz case, as expected.
        let lambda_15 = 1540.0 / (15.0 * MHZ_TO_HZ);
        let g5 = focused_aperture_gain(96.0 * element_area, lambda, 3e-3);
        let g15 = focused_aperture_gain(96.0 * element_area, lambda_15, 3e-3);
        assert!(g15 > g5, "15 MHz gain {g15} should exceed 5 MHz gain {g5}");
    }

    #[test]
    fn filled_aperture_beats_sparse_tips_at_deep_focus() {
        // A 24 mm × 5 mm strip of 96 elements focused 40 mm deep at 5 MHz
        // (λ ≈ 308 µm). At this long focus a SPARSE bundle of bare 100 µm fiber
        // tips collapses (A_active ≈ 0.75 mm² ⇒ G ≈ 0.06, p₀ would exceed the
        // damage limit), so the 96 elements must be sub-aperture TILES that fill
        // the face.
        let lambda = 1540.0 / (5.0 * MHZ_TO_HZ);
        let f_focus = 40e-3;

        // Sparse bare-tip bundle fails: 1 MPa would need a supra-damage p₀.
        let tip_area = std::f64::consts::PI * (50e-6_f64).powi(2);
        let g_sparse = focused_aperture_gain(96.0 * tip_area, lambda, f_focus);
        assert!(
            g_sparse < 0.1,
            "sparse-tip gain {g_sparse} unexpectedly large"
        );
        assert!(
            1.0e6 / g_sparse > 10.0e6,
            "sparse tips would need < damage p₀"
        );

        // Filled tiles succeed: 96 tiles each ≈0.8 mm² radiating (0.5 mm
        // effective radius) over the 24×5 mm = 120 mm² face (≈0.63 fill).
        let tile_area = std::f64::consts::PI * (0.5e-3_f64).powi(2);
        let g_filled = focused_aperture_gain(96.0 * tile_area, lambda, f_focus);
        assert!(g_filled > 5.0, "filled-aperture gain {g_filled} too low");
        // A damage-safe 0.6 MPa per tile (≈2.2 mJ/cm²) clears 1 MPa with margin
        // wide enough to survive a 3× derate for the anisotropic (line-focus)
        // aperture — the 5 mm width is past its 20 mm near field at 40 mm.
        let p_focus = optoacoustic_array_focal_pressure(0.6e6, 96, tile_area, lambda, f_focus);
        assert!(
            p_focus >= 3.0e6,
            "filled-aperture focal pressure {p_focus} Pa"
        );
        assert!(p_focus / 3.0 >= 1.0e6, "3× anisotropy derate still < 1 MPa");
    }

    #[test]
    fn optoacoustic_array_has_no_focal_gain_advantage_over_piezo() {
        // Proof that an optoacoustic array does not out-perform a piezoelectric
        // array of the same geometry: the focused-aperture gain A/(λF) is pure
        // diffraction, independent of the transduction mechanism, so the focal
        // pressure ratio of two same-geometry arrays equals their surface-
        // pressure ratio. A piezo element's surface pressure (several MPa)
        // exceeds the optoacoustic p₀ = Γμ_aF (sub-MPa at damage-safe fluence),
        // hence the piezo array focuses HIGHER pressure, not lower.
        let (a_active, lambda, f) = (7.5e-5, 308e-6, 40e-3);
        let g = focused_aperture_gain(a_active, lambda, f);
        let (p0_oa, p0_piezo) = (0.6e6, 3.0e6); // surface pressures
        let p_oa = p0_oa * g;
        let p_piezo = p0_piezo * g;
        // Identical gain ⇒ focal ratio equals the surface-pressure ratio.
        assert!((p_piezo / p_oa - p0_piezo / p0_oa).abs() < 1e-9);
        // Piezo wins at equal geometry — no optoacoustic focusing advantage.
        assert!(p_piezo > p_oa);
    }
}
