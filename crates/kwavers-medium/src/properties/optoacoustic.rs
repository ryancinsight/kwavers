//! Optoacoustic emitter materials: laser-absorbing nanoparticle composites that
//! convert a nanosecond laser pulse into an ultrasound pulse.
//!
//! These are the active layer of a **soft optoacoustic pad** (SOAP): a thin
//! light-absorbing nanocomposite coated on a curved elastic (PDMS) surface. A
//! nanosecond laser pulse is absorbed by the filler, the absorbed energy heats
//! the host, and thermoelastic expansion launches an ultrasound pulse from the
//! surface. On a spherical cap the wavefronts converge at the geometric centre,
//! producing an optically-generated focused ultrasound (OFUS) focus with no
//! electronic delays.
//!
//! Each material carries (a) the **acoustic** properties (host-dominated density,
//! sound speed, absorption) needed to place the layer in a heterogeneous medium
//! for full-wave propagation, and (b) the **optoacoustic conversion** properties
//! needed to drive the source: the Grüneisen parameter of the expansion host, the
//! optical absorption coefficient of the composite, the measured optoacoustic
//! sensitivity (peak surface pressure per unit laser fluence), and the centre
//! frequency and temporal width of the emitted photoacoustic pulse.
//!
//! The focal pressure of a SOAP is the **surface** pressure of the emitter times
//! the geometric focal gain `G` of the cap (see
//! `kwavers_physics::analytical::transducer::soap_focal_gain`):
//! `p_focus = G · p_surface = G · S · F`.
//!
//! # References
//! - Li et al. (2022). "Optically-generated focused ultrasound for noninvasive
//!   brain stimulation with ultrahigh precision." *Light: Sci. Appl.* 11, 321.
//!   Reported device: CS-PDMS SOAP, 48 MPa focal pressure at 0.62 mJ/cm², NA 0.95.
//! - Wang & Wu (2007). *Biomedical Optics: Principles and Imaging.* (Γ = β c²/C_p)

use super::AcousticPropertyData;

/// A laser-driven optoacoustic emitter material (light-absorbing filler in an
/// elastic host, e.g. candle soot in PDMS).
///
/// The optoacoustic sensitivity `S` is the *measured* peak surface pressure per
/// unit laser fluence \[Pa·m²/J\]. It is the load-bearing quantity for the source
/// amplitude and is taken from measurement; it is **not** equal to the naive
/// volumetric `Γ·μ_a` because a thin surface absorber with an acoustically soft
/// (air) backing builds up and out-couples stress differently than a bulk
/// absorber. `gruneisen` and `optical_absorption` are carried as documented
/// physical properties for first-principles light-transport / thermoelastic
/// workflows; they do not by themselves define the radiated pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptoacousticEmitter {
    /// Human-readable material name.
    pub name: &'static str,
    /// Host-dominated mass density ρ \[kg/m³\].
    pub density: f64,
    /// Host-dominated longitudinal sound speed c \[m/s\].
    pub sound_speed: f64,
    /// Power-law acoustic absorption prefactor α₀ \[dB/(cm·MHz^y)\].
    pub absorption_coefficient: f64,
    /// Acoustic absorption power-law exponent y (dimensionless).
    pub absorption_power: f64,
    /// Grüneisen parameter Γ of the thermoelastic host (dimensionless).
    pub gruneisen: f64,
    /// Optical absorption coefficient μ_a of the composite at the excitation
    /// wavelength \[1/m\].
    pub optical_absorption: f64,
    /// Measured optoacoustic sensitivity S: peak surface pressure per unit laser
    /// fluence \[Pa·m²/J\] (equivalently Pa per (J/m²)). Linear in fluence.
    pub optoacoustic_sensitivity: f64,
    /// Centre frequency of the emitted photoacoustic pulse \[Hz\].
    pub center_frequency: f64,
    /// Temporal full-width-at-half-maximum of the emitted pressure pulse \[s\].
    pub pulse_fwhm: f64,
}

impl OptoacousticEmitter {
    /// Peak optoacoustic **surface** pressure for a laser fluence (linear regime):
    /// `p_surface = S · F`.
    ///
    /// `fluence_j_m2` is the absorbed-relevant laser fluence \[J/m²\]
    /// (1 mJ/cm² = 10 J/m²). Returns Pa.
    #[must_use]
    #[inline]
    pub fn surface_pressure(&self, fluence_j_m2: f64) -> f64 {
        self.optoacoustic_sensitivity * fluence_j_m2
    }

    /// Focal pressure of a SOAP made from this emitter: `p_focus = G · S · F`,
    /// where `G` is the geometric focal gain of the cap (from
    /// `kwavers_physics::analytical::transducer::soap_focal_gain`).
    #[must_use]
    #[inline]
    pub fn focal_pressure(&self, fluence_j_m2: f64, focal_gain: f64) -> f64 {
        focal_gain * self.surface_pressure(fluence_j_m2)
    }

    /// Whether the material actually generates ultrasound (has an absorber).
    /// A bare host (e.g. transparent PDMS) has zero sensitivity.
    #[must_use]
    #[inline]
    pub fn is_active(&self) -> bool {
        self.optoacoustic_sensitivity > 0.0
    }

    /// Fractional bandwidth of the emitted pulse, estimated from its temporal
    /// FWHM via the time–bandwidth relation of a Gaussian pulse
    /// (`Δf·Δt ≈ 0.441`): `BW = Δf / f_c = 0.441 / (f_c · Δt)`.
    ///
    /// Returns 0 for an inactive emitter (no pulse).
    #[must_use]
    pub fn fractional_bandwidth(&self) -> f64 {
        if !self.is_active() || self.center_frequency <= 0.0 || self.pulse_fwhm <= 0.0 {
            return 0.0;
        }
        const GAUSSIAN_TIME_BANDWIDTH: f64 = 0.441;
        GAUSSIAN_TIME_BANDWIDTH / (self.center_frequency * self.pulse_fwhm)
    }

    /// Acoustic-property view for placing the emitter layer in a heterogeneous
    /// medium. Uses a representative nonlinearity for a soft polymer host.
    ///
    /// # Panics
    /// Never on the built-in presets (all carry physical positive values); the
    /// underlying validating constructor only rejects non-physical inputs.
    #[must_use]
    pub fn to_acoustic_properties(&self) -> AcousticPropertyData {
        // B/A ≈ 6 is representative for a soft silicone/polyolefin host; it is not
        // load-bearing for the linear optoacoustic source.
        const HOST_NONLINEARITY: f64 = 6.0;
        AcousticPropertyData::new(
            self.density,
            self.sound_speed,
            self.absorption_coefficient,
            self.absorption_power,
            HOST_NONLINEARITY,
        )
        .expect("invariant: optoacoustic emitter presets carry physical acoustic values")
    }
}

// ─── Presets (Li et al. 2022) ───────────────────────────────────────────────
//
// The four absorbers were fabricated on SOAPs of identical geometry, so the
// measured focal-pressure ratios equal the surface-sensitivity ratios:
// CS : CNT : CNP : HSM = 1 : 1/6 : 1/6 : 1/30  (paper p. 4). The CS-PDMS
// reference sensitivity is back-calculated from the reported 48 MPa focal
// pressure at 0.62 mJ/cm² (= 6.2 J/m²) divided by the device focal gain
// G_max ≈ 280:  S_CS = 48 MPa / (280 · 6.2 J/m²) ≈ 2.765e4 Pa·m²/J.

/// Reference optoacoustic surface sensitivity of CS-PDMS \[Pa·m²/J\].
/// Back-calculated from the paper's 48 MPa focal pressure at 6.2 J/m² and
/// G_max ≈ 280: `48e6 / (280 · 6.2)`.
const S_CS: f64 = 48.0e6 / (280.0 * 6.2);

/// Grüneisen parameter of the cured-PDMS thermoelastic host (Sylgard 184).
/// Γ = β c²/C_p with β_vol ≈ 9.2e-4 /K, c ≈ 1030 m/s, C_p ≈ 1460 J/(kg·K).
const GRUENEISEN_PDMS: f64 = 0.90;

/// Representative power-law acoustic absorption of the soft polymer host
/// \[dB/(cm·MHz^y)\], y = 1. Polymer hosts attenuate far more than water; the
/// value is representative and not reference-validated.
const ALPHA_HOST: f64 = 1.0;
const ALPHA_POWER_HOST: f64 = 1.0;

/// **PDMS** — the bare elastic host (Sylgard 184), transparent: no absorber, so
/// it generates no ultrasound. Carried for the medium and as the optoacoustic
/// matrix reference. ρ ≈ 1030 kg/m³, c ≈ 1030 m/s.
pub const PDMS: OptoacousticEmitter = OptoacousticEmitter {
    name: "PDMS (Sylgard 184)",
    density: 1030.0,
    sound_speed: 1030.0,
    absorption_coefficient: ALPHA_HOST,
    absorption_power: ALPHA_POWER_HOST,
    gruneisen: GRUENEISEN_PDMS,
    optical_absorption: 0.0,
    optoacoustic_sensitivity: 0.0,
    center_frequency: 0.0,
    pulse_fwhm: 0.0,
};

/// **CS-PDMS** — candle-soot nanoparticles (≈55 nm) in PDMS. The most efficient
/// absorber in the paper: broadband, ≈98 % optical absorption over a ≈2.7 µm
/// layer (⇒ μ_a ≈ 1.45e6 /m), highest centre frequency (≈15 MHz) and tightest
/// focus. Reference for the 48 MPa / 0.62 mJ/cm² device.
pub const CS_PDMS: OptoacousticEmitter = OptoacousticEmitter {
    name: "CS-PDMS (candle soot)",
    density: 1040.0,
    sound_speed: 1030.0,
    absorption_coefficient: ALPHA_HOST,
    absorption_power: ALPHA_POWER_HOST,
    gruneisen: GRUENEISEN_PDMS,
    optical_absorption: 1.45e6, // ln(1/0.02)/2.7 µm ≈ 1.45e6 /m
    optoacoustic_sensitivity: S_CS,
    center_frequency: 15.0e6,
    pulse_fwhm: 0.09e-6,
};

/// **CNT-PDMS** — multi-wall carbon nanotubes (5 wt%) in PDMS. ≈1/6 the surface
/// sensitivity of CS-PDMS, centre frequency ≈5 MHz. μ_a is a representative
/// broadband-carbon value (not the load-bearing quantity; sensitivity is
/// measured).
pub const CNT_PDMS: OptoacousticEmitter = OptoacousticEmitter {
    name: "CNT-PDMS (carbon nanotube)",
    density: 1040.0,
    sound_speed: 1040.0,
    absorption_coefficient: ALPHA_HOST,
    absorption_power: ALPHA_POWER_HOST,
    gruneisen: GRUENEISEN_PDMS,
    optical_absorption: 2.0e5,
    optoacoustic_sensitivity: S_CS / 6.0,
    center_frequency: 5.0e6,
    pulse_fwhm: 0.24e-6,
};

/// **CNP-PDMS** — carbon nanoparticles in PDMS. Same sensitivity level as
/// CNT-PDMS (≈1/6 of CS), centre frequency ≈5 MHz, slightly broader pulse.
pub const CNP_PDMS: OptoacousticEmitter = OptoacousticEmitter {
    name: "CNP-PDMS (carbon nanoparticle)",
    density: 1040.0,
    sound_speed: 1040.0,
    absorption_coefficient: ALPHA_HOST,
    absorption_power: ALPHA_POWER_HOST,
    gruneisen: GRUENEISEN_PDMS,
    optical_absorption: 2.0e5,
    optoacoustic_sensitivity: S_CS / 6.0,
    center_frequency: 5.0e6,
    pulse_fwhm: 0.29e-6,
};

/// **HSM** — heat-shrink membrane (black polyolefin), the absorber and expansion
/// host in one. Lowest sensitivity (≈1/30 of CS, i.e. 1/5 of CNT/CNP), lowest
/// centre frequency (≈3 MHz). ρ ≈ 920 kg/m³, c ≈ 1950 m/s (polyolefin).
pub const HSM: OptoacousticEmitter = OptoacousticEmitter {
    name: "HSM (heat-shrink polyolefin)",
    density: 920.0,
    sound_speed: 1950.0,
    absorption_coefficient: ALPHA_HOST,
    absorption_power: ALPHA_POWER_HOST,
    gruneisen: 0.85,
    optical_absorption: 1.0e5,
    optoacoustic_sensitivity: S_CS / 30.0,
    center_frequency: 3.0e6,
    pulse_fwhm: 0.31e-6,
};

/// All built-in optoacoustic emitter presets (active absorbers, excluding the
/// bare PDMS host), ordered by decreasing optoacoustic sensitivity.
#[must_use]
pub fn all_absorbers() -> [OptoacousticEmitter; 4] {
    [CS_PDMS, CNT_PDMS, CNP_PDMS, HSM]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 0.62 mJ/cm² laser fluence expressed in J/m² (1 mJ/cm² = 10 J/m²).
    const F_PAPER: f64 = 6.2;
    /// Device focal gain G_max ≈ 280 (Li et al. 2022, NA 0.95).
    const G_MAX: f64 = 280.0;

    #[test]
    fn cs_pdms_reproduces_paper_focal_pressure() {
        // p_focus = G · S · F must reproduce the reported 48 MPa at 0.62 mJ/cm².
        let p_focus = CS_PDMS.focal_pressure(F_PAPER, G_MAX);
        assert!(
            (p_focus - 48.0e6).abs() < 1.0e3,
            "CS-PDMS focal pressure {p_focus} Pa != 48 MPa"
        );
        // Surface pressure is the focal pressure divided by the gain (~0.17 MPa).
        let p_surface = CS_PDMS.surface_pressure(F_PAPER);
        assert!(
            (p_surface - 48.0e6 / G_MAX).abs() < 1.0,
            "surface {p_surface} != 48 MPa / G"
        );
        // Linear in fluence: doubling fluence doubles pressure.
        assert!((CS_PDMS.surface_pressure(2.0 * F_PAPER) - 2.0 * p_surface).abs() < 1e-6);
    }

    #[test]
    fn absorber_sensitivity_ratios_match_paper() {
        // CS : CNT : CNP : HSM = 1 : 1/6 : 1/6 : 1/30 (focal pressures 48 : 8 : 8 : 1.6 MPa).
        assert!((CS_PDMS.focal_pressure(F_PAPER, G_MAX) - 48.0e6).abs() < 1.0e3);
        assert!((CNT_PDMS.focal_pressure(F_PAPER, G_MAX) - 8.0e6).abs() < 1.0e3);
        assert!((CNP_PDMS.focal_pressure(F_PAPER, G_MAX) - 8.0e6).abs() < 1.0e3);
        assert!((HSM.focal_pressure(F_PAPER, G_MAX) - 1.6e6).abs() < 1.0e3);
        // CS is 6× CNT and 30× HSM.
        assert!(
            (CS_PDMS.optoacoustic_sensitivity / CNT_PDMS.optoacoustic_sensitivity - 6.0).abs()
                < 1e-9
        );
        assert!(
            (CS_PDMS.optoacoustic_sensitivity / HSM.optoacoustic_sensitivity - 30.0).abs() < 1e-9
        );
    }

    #[test]
    fn center_frequencies_ordered_cs_highest() {
        // The tighter (higher-frequency) focus follows the more efficient, faster
        // absorber: CS (15 MHz) > CNT = CNP (5 MHz) > HSM (3 MHz).
        assert!(CS_PDMS.center_frequency > CNT_PDMS.center_frequency);
        assert!((CNT_PDMS.center_frequency - CNP_PDMS.center_frequency).abs() < 1e-3);
        assert!(CNP_PDMS.center_frequency > HSM.center_frequency);
        assert_eq!(CS_PDMS.center_frequency, 15.0e6);
    }

    #[test]
    fn shorter_pulse_has_higher_center_frequency_and_broad_bandwidth() {
        // CS-PDMS: 0.09 µs pulse → BW = 0.441 / (15e6 · 0.09e-6) ≈ 0.327. The
        // paper reports a broad −6 dB band (5–35 MHz, ~200 % fractional); the
        // Gaussian time–bandwidth estimate is a conservative lower bound.
        // CS: 0.441/(15e6·0.09e-6) ≈ 0.327; HSM: 0.441/(3e6·0.31e-6) ≈ 0.474.
        let bw_cs = CS_PDMS.fractional_bandwidth();
        let bw_hsm = HSM.fractional_bandwidth();
        assert!((bw_cs - 0.327).abs() < 0.01, "CS BW {bw_cs}");
        assert!((bw_hsm - 0.474).abs() < 0.01, "HSM BW {bw_hsm}");
    }

    #[test]
    fn bare_pdms_host_is_inactive() {
        assert!(!PDMS.is_active());
        assert_eq!(PDMS.surface_pressure(F_PAPER), 0.0);
        assert_eq!(PDMS.fractional_bandwidth(), 0.0);
        // The absorbers are all active.
        assert!(all_absorbers().iter().all(OptoacousticEmitter::is_active));
    }

    #[test]
    fn acoustic_view_is_physical() {
        for e in all_absorbers() {
            let a = e.to_acoustic_properties();
            assert_eq!(a.density, e.density);
            assert_eq!(a.sound_speed, e.sound_speed);
            assert!(a.density > 0.0 && a.sound_speed > 0.0);
        }
    }
}
