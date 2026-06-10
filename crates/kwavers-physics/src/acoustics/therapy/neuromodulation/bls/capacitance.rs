//! Bilayer-sonophore capacitance source (Plaksin et al. 2014, Eq. 8).
//!
//! This source uses the *exact* curved-dome membrane-capacitance geometry of the
//! NICE model — Eq. (8) of Plaksin et al. (2014) — to map a leaflet deflection
//! `Z` to the specific capacitance `C_m(Z)`. As the intramembrane cavity inflates
//! during the rarefactional half of the acoustic cycle the leaflets bow outward
//! (`Z ≥ 0`), the average plate separation widens, and the capacitance falls;
//! the resulting *asymmetric* `C_m(t)` waveform — unlike a symmetric sinusoid —
//! lets the membrane's leak current rectify into a net charge accumulation, which
//! is the basis of the post-stimulus depolarisation in the NICE mechanism.
//!
//! # Capacitance geometry (exact, Eq. 8)
//!
//! For a circular bilayer patch of radius `a`, rest gap `Δ`, and central
//! deflection `Z ≥ 0`:
//!
//! ```text
//! C_m(Z) = (C_m0·Δ / a²) · [ Z + ((a² − Z² − Z·Δ)/(2Z))·ln((2Z + Δ)/Δ) ]
//! ```
//!
//! with the removable `Z → 0` limit `C_m(0) = C_m0` (series `C_m ≈ C_m0·(1 −
//! Z/Δ)` for `Z ≪ Δ`). The bracket carries units of length and `Δ/a²` of inverse
//! length, so the factor multiplying `C_m0` is dimensionless; lengths are passed
//! in metres and `C_m0` in µF/cm².
//!
//! # Leaflet deflection (kinematic surrogate)
//!
//! The full bilayer-sonophore mechanical ODE (Plaksin Eq. 2, with the molecular
//! attraction-repulsion pressure of Krasovitski et al. 2011) requires
//! leaflet-tension, viscosity and molecular-force parameters that are not given
//! in the open preprint. Here the leaflet deflection is therefore *prescribed*
//! as a non-negative, once-per-cycle expansion driven by the carrier:
//!
//! ```text
//! Z(t) = Z_max · (1 − cos(ω t)) / 2 ≥ 0      ω = 2π·1000·f_MHz  [rad/ms]
//! ```
//!
//! **Evidence tier:** the capacitance geometry (Eq. 8) and the displacement-
//! current coupling (Eq. 1, in [`super::super::nice`]) are reproduced exactly from the
//! reference; the deflection *shape* is a documented kinematic surrogate for the
//! BLS mechanical oscillation, and the peak deflection `Z_max` is an input
//! parameter (its calibration to acoustic pressure requires solving Eq. 2). The
//! qualitative behaviour is validated against the reference's stated results:
//! membrane hyperpolarisation *during* sonication and a charge-accumulation
//! depolarisation *after* it (see the module tests).
//!
//! # References
//!
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). Intramembrane cavitation as a
//!   predictive bio-piezoelectric mechanism for ultrasonic brain stimulation.
//!   *Phys. Rev. X* 4, 011004 (arXiv:1307.7701), Eqs. (1), (2), (8).
//! - Krasovitski, B. et al. (2011). *PNAS* 108(8), 3258-3263 (original BLS).

use super::super::intramembrane_cavitation::CapacitanceSource;
use std::f64::consts::PI;

/// Sonophore radius `a` [m]: half the 64 nm transmembrane-protein interspacing
/// (Plaksin et al. 2014).
pub const SONOPHORE_RADIUS_M: f64 = 32.0e-9;

/// Rest inter-leaflet gap `Δ` [m] (Plaksin et al. 2014).
pub const LEAFLET_GAP_M: f64 = 1.26e-9;

/// Curved-dome bilayer membrane capacitance `C_m(Z)` [µF/cm²] (Plaksin Eq. 8).
///
/// `z`, `a`, `delta` are lengths in metres; `cm0` is the rest specific
/// capacitance in µF/cm². The formula is applied for any `Z` with a positive
/// inter-leaflet gap `2Z + Δ > 0` (expansion `Z > 0` **and** compression
/// `Z < 0`, matching the reference, which special-cases only `Z = 0`): during
/// compression the gap narrows and `C_m` rises above `C_m0`. For `|Z| ≪ Δ` a
/// series limit (valid for both signs) avoids the `1/Z` cancellation; the two
/// branches agree to first order at the threshold. At the steric limit
/// (`2Z + Δ ≤ 0`) the parallel-plate capacitance diverges, so a large finite
/// value is returned (the integrator keeps `Z ≥ Z_min`, where the gap stays
/// positive).
///
/// # Examples
///
/// ```
/// use kwavers_physics::acoustics::therapy::neuromodulation::{
///     bls_capacitance, LEAFLET_GAP_M, SONOPHORE_RADIUS_M,
/// };
/// // A flat membrane (Z = 0) has the rest capacitance; expansion lowers it and
/// // compression raises it.
/// let cm0 = 1.0;
/// assert!((bls_capacitance(0.0, cm0, SONOPHORE_RADIUS_M, LEAFLET_GAP_M) - cm0).abs() < 1e-9);
/// assert!(bls_capacitance(2.0e-9, cm0, SONOPHORE_RADIUS_M, LEAFLET_GAP_M) < cm0);
/// assert!(bls_capacitance(-0.1e-9, cm0, SONOPHORE_RADIUS_M, LEAFLET_GAP_M) > cm0);
/// ```
#[must_use]
pub fn bls_capacitance(z: f64, cm0: f64, a: f64, delta: f64) -> f64 {
    let gap = 2.0 * z + delta;
    if gap <= 0.0 {
        return cm0 * 1.0e3; // steric limit: gap → 0 ⇒ C_m → ∞ (capped)
    }
    if z.abs() < 1.0e-4 * delta {
        // Series: g(Z) = a²/Δ − a²Z/Δ² + O(Z²) ⇒ C_m ≈ C_m0·(1 − Z/Δ); the
        // `1 − Z/Δ` form is correct for both signs (compression Z<0 ⇒ C_m > C_m0).
        return cm0 * (1.0 - z / delta);
    }
    let factor = cm0 * delta / (a * a);
    let bracket = z + ((a * a - z * z - z * delta) / (2.0 * z)) * (gap / delta).ln();
    factor * bracket
}

/// Bilayer-sonophore capacitance source: prescribed leaflet deflection mapped
/// through the exact curved-dome capacitance [`bls_capacitance`].
#[derive(Debug, Clone, Copy)]
pub struct BilayerSonophore {
    /// Rest specific capacitance C_m0 [µF/cm²].
    pub cm0_uf_cm2: f64,
    /// Sonophore radius a [m].
    pub radius_a_m: f64,
    /// Rest inter-leaflet gap Δ [m].
    pub gap_delta_m: f64,
    /// Peak leaflet deflection Z_max [m].
    pub deflection_amp_m: f64,
    /// Angular carrier frequency ω = 2π f [rad/ms].
    pub omega_rad_ms: f64,
}

impl BilayerSonophore {
    /// Construct from rest capacitance [µF/cm²], carrier frequency [MHz], and
    /// peak leaflet deflection [m], using the canonical sonophore geometry
    /// ([`SONOPHORE_RADIUS_M`], [`LEAFLET_GAP_M`]).
    #[must_use]
    pub fn new(cm0_uf_cm2: f64, freq_mhz: f64, deflection_amp_m: f64) -> Self {
        Self {
            cm0_uf_cm2,
            radius_a_m: SONOPHORE_RADIUS_M,
            gap_delta_m: LEAFLET_GAP_M,
            deflection_amp_m,
            omega_rad_ms: 2.0 * PI * 1.0e3 * freq_mhz,
        }
    }

    /// Leaflet deflection Z(t) [m]: a non-negative once-per-cycle expansion
    /// `Z_max·(1 − cos ωt)/2`.
    #[inline]
    #[must_use]
    pub fn deflection(&self, t_ms: f64) -> f64 {
        0.5 * self.deflection_amp_m * (1.0 - (self.omega_rad_ms * t_ms).cos())
    }

    /// Carrier period [ms].
    #[inline]
    #[must_use]
    fn period_ms(&self) -> f64 {
        2.0 * PI / self.omega_rad_ms
    }
}

impl CapacitanceSource for BilayerSonophore {
    #[inline]
    fn capacitance(&self, t_ms: f64) -> f64 {
        bls_capacitance(
            self.deflection(t_ms),
            self.cm0_uf_cm2,
            self.radius_a_m,
            self.gap_delta_m,
        )
    }

    /// dC_m/dt by central finite difference of the analytic `C_m(Z(t))`
    /// (step = carrier period / 5000), robust through the `Z → 0` limit.
    fn capacitance_rate(&self, t_ms: f64) -> f64 {
        let h = self.period_ms() / 5000.0;
        (self.capacitance(t_ms + h) - self.capacitance(t_ms - h)) / (2.0 * h)
    }

    #[inline]
    fn baseline_capacitance(&self) -> f64 {
        self.cm0_uf_cm2
    }

    #[inline]
    fn carrier_omega_rad_ms(&self) -> f64 {
        self.omega_rad_ms
    }

    #[inline]
    fn is_source_valid(&self) -> bool {
        self.cm0_uf_cm2 > 0.0
            && self.radius_a_m > 0.0
            && self.gap_delta_m > 0.0
            && self.deflection_amp_m >= 0.0
            && self.omega_rad_ms > 0.0
    }
}
