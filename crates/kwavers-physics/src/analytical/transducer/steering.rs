//! Electronic-steering off-focus efficiency for focused phased arrays.
//!
//! When a focused bowl/array steers its focus electronically away from the
//! geometric (mechanical) focus, the achievable focal pressure drops because of
//! (1) per-element directivity loss (∝ cos θ; Hand 2009), (2) projected-aperture
//! loss (∝ cos θ), and (3) the Gaussian roll-off as the steered focus
//! approaches grating-lobe onset. The combined derating is modelled as a
//! separable Gaussian in the lateral and axial steering offsets, calibrated to
//! a 50 mm-aperture / 120 mm-ROC bowl at 1 MHz (Pernot 2003; Hand 2009;
//! Vlaisavljevich 2017 in-vivo steering range).

/// Off-focus electronic-steering efficiency `ε ∈ (0, 1]`.
///
/// ```text
///   ε(Δ_lat, Δ_ax) = exp[ −(Δ_lat / R_lat)² − (Δ_ax / R_ax)² ]
///   R_lat = R_lat,1MHz · (λ / λ_1MHz),   R_ax = R_ax,1MHz · (λ / λ_1MHz)
/// ```
/// The characteristic compensable ranges scale linearly with wavelength
/// `λ = c / f₀` (longer wavelength → wider steering window). With apodization
/// (re-weighting elements by cos θ toward the steered focus) the projected-
/// aperture loss is largely recovered, extending the lateral/axial 1/e ranges
/// from 5/15 mm to 7/21 mm at 1 MHz (≈ √2 wider); modern clinical systems
/// apodize by default.
///
/// # Arguments
/// * `dr_lat_m` – lateral steering offset from the mechanical focus [m]
/// * `dr_ax_m`  – axial steering offset from the mechanical focus [m]
/// * `f0_hz`    – drive frequency [Hz]
/// * `c_m_s`    – medium sound speed [m/s]
/// * `apodized` – whether cos θ apodization is applied
///
/// Returns `ε ∈ (0, 1]`; `1.0` at the mechanical focus (zero offset).
/// Returns `0.0` for non-finite or non-positive `f0`/`c`.
///
/// # Reference
/// Pernot M. et al. (2003) *Ultrasound Med. Biol.* 29, 1525.
/// Hand J.W. et al. (2009) *Med. Phys.* 36, 2107.
#[must_use]
pub fn electronic_steering_efficiency(
    dr_lat_m: f64,
    dr_ax_m: f64,
    f0_hz: f64,
    c_m_s: f64,
    apodized: bool,
) -> f64 {
    if !(f0_hz.is_finite() && f0_hz > 0.0 && c_m_s.is_finite() && c_m_s > 0.0) {
        return 0.0;
    }
    // 1 MHz reference wavelength in water-equivalent tissue (1540 m/s).
    const LAMBDA_1MHZ_M: f64 = 1.54e-3;
    let lambda_m = c_m_s / f0_hz;
    let scale = lambda_m / LAMBDA_1MHZ_M;
    let (base_lat, base_ax) = if apodized {
        (7.0e-3, 21.0e-3)
    } else {
        (5.0e-3, 15.0e-3)
    };
    let r_lat = base_lat * scale;
    let r_ax = base_ax * scale;
    let arg = (dr_lat_m / r_lat).powi(2) + (dr_ax_m / r_ax).powi(2);
    (-arg).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn efficiency_is_unity_at_mechanical_focus() {
        let e = electronic_steering_efficiency(0.0, 0.0, 1.0e6, 1540.0, true);
        assert!((e - 1.0).abs() < 1e-12, "ε(0,0)={e} ≠ 1");
    }

    #[test]
    fn efficiency_decreases_with_offset() {
        let near = electronic_steering_efficiency(3.0e-3, 0.0, 1.0e6, 1540.0, true);
        let far = electronic_steering_efficiency(10.0e-3, 0.0, 1.0e6, 1540.0, true);
        assert!(
            near > far,
            "ε must fall with lateral offset: near={near}, far={far}"
        );
        assert!((0.0..=1.0).contains(&far));
    }

    #[test]
    fn apodization_widens_steering_window() {
        // At a fixed offset apodization recovers projected-aperture loss → higher ε.
        let apod = electronic_steering_efficiency(7.0e-3, 0.0, 1.0e6, 1540.0, true);
        let bare = electronic_steering_efficiency(7.0e-3, 0.0, 1.0e6, 1540.0, false);
        assert!(apod > bare, "apodized ε={apod} should exceed bare ε={bare}");
    }

    #[test]
    fn one_over_e_at_characteristic_lateral_range() {
        // Apodized lateral 1/e range is 7 mm at 1 MHz.
        let e = electronic_steering_efficiency(7.0e-3, 0.0, 1.0e6, 1540.0, true);
        assert!((e - (-1.0_f64).exp()).abs() < 1e-9, "ε(7mm)={e} ≠ 1/e");
    }

    #[test]
    fn longer_wavelength_widens_window() {
        // Lower frequency (longer λ) → wider steering window → higher ε at fixed offset.
        let hi = electronic_steering_efficiency(7.0e-3, 0.0, 1.0e6, 1540.0, true);
        let lo = electronic_steering_efficiency(7.0e-3, 0.0, 0.5e6, 1540.0, true);
        assert!(
            lo > hi,
            "lower f₀ should steer more efficiently: lo={lo}, hi={hi}"
        );
    }

    #[test]
    fn rejects_bad_input() {
        assert_eq!(
            electronic_steering_efficiency(1e-3, 0.0, 0.0, 1540.0, true),
            0.0
        );
        assert_eq!(
            electronic_steering_efficiency(1e-3, 0.0, 1e6, -1.0, true),
            0.0
        );
    }
}
