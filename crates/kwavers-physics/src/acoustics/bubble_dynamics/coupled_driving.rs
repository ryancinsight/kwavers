//! Foldy–Lax self-consistent coupled bubble driving
//! (Build 15 / WS-B, KW-PHYS-013).
//!
//! Every other cloud path in this programme drove each bubble with
//! the incident field alone and summed the received field linearly.
//! This module removes that approximation: bubble `i` is driven by
//!
//! ```text
//! p_i(t) = p_inc(t) + Σ_{j≠i} (ρ / 4π d_ij) · V̈_j(t − d_ij / c)
//! ```
//!
//! with `V = 4πR³/3` the bubble volume, `d_ij` the pair distance and
//! `c` the propagation sound speed (Foldy–Lax leading order:
//! monopole coupling only — no translation/Bjerknes drift, no
//! coalescence, no thermal coupling). Each bubble's radial dynamics
//! are integrated with the existing Keller–Miksis model
//! ([`super::keller_miksis::KellerMiksisModel`], via the
//! [`AdaptiveBubbleModel`](super::adaptive_integration::AdaptiveBubbleModel) contract — the KM equation itself is
//! **not** duplicated here). The drive contract is the KM contract:
//! the *instantaneous* far-field pressure `p∞(t) − p0` and its time
//! derivative (Keller & Miksis 1980, Eq. 3.4); the total drive is
//! precomputed on the integration grid per Picard pass and its
//! derivative taken by centred differences of the total, exactly as
//! in the Python reference (`kwavers-work/b15/km_coupled.py`).
//!
//! # Picard iteration and damping
//!
//! Pass 0 is independent driving (coupling field zero). Each later
//! pass precomputes the coupling field from the previous history
//! and re-integrates every bubble. **Plain Picard oscillates** on
//! strongly coupled close pairs (WS-B diagnostic at 0.123 MPa:
//! segment band power +13.4 dB then −8.4 dB on successive passes),
//! so the history fed to the next pass is damped,
//! `V̈ ← (1 − θ)·V̈_prev + θ·V̈_new` with
//! [`PICARD_DAMPING_THETA`] `θ = 0.5`. Damping changes only the
//! path to the fixed point, never the fixed point itself: the
//! coupling field is linear in the `V̈` history, so history
//! blending is exact. With θ = 0.5 the diagnostic run converged
//! monotonically (−6.6 / −2.1 / −0.53 / −0.14 dB per pass).
//!
//! # Validation (Build 15 / WS-B, Python reference)
//!
//! * **Dilute limit:** at density ×0.008 (cloud radius ×5) the
//!   coupled solver reproduces independent driving — 1.5f₀ band
//!   power **−0.11 dB**, centre drive +0.18 dB, iteration converged
//!   to 0.007 dB by pass 3. (On the Kuji reduction *statistic* the
//!   same physical difference reads +1.24 dB at fc 3.0 MHz — the
//!   statistic divides by a near-floor 3P image and amplifies
//!   sub-0.2 dB emission changes; the 0.11 dB emission figure is
//!   the solver tolerance.)
//! * **KM cross-check:** the reference RHS (transcribed from this
//!   crate's `keller_miksis/equation.rs`) matches pykwavers
//!   `solve_keller_miksis` 1.5f₀ band power within **2.7 %** at
//!   0.123 / 0.268 MPa and **3.5 %** at 0.729 MPa
//!   (R(t) correlation ≥ 0.998); at 1.273 MPa the band max of the
//!   spiky collapse spectrum is trajectory-chaotic and only
//!   Rmax (within 3.5 %) is cited.
//!
//! # Measured effect size (WS-B, formation density ≈ 32 bubbles/mm³)
//!
//! Coupling does **not** close the Kuji cavitation-side gap. At
//! fc 3.0 MHz the cavitation-reduction change (coupled −
//! independent) is **+0.19 / +0.83 / +2.44 / −1.94 dB** at
//! 0.123 / 0.268 / 0.729 / 1.273 MPa — **+0.83 dB of the 10.5 dB
//! gap (≈ 8 %) at 0.26 MPa**, sign-reversing at the top PNP, and
//! PNP-dependent (4.4 dB spread) where the paper is nearly flat.
//! The raw bubble *emission* rises more (+2.0 to +6.6 dB in the
//! 1.5f₀ band) — a rare close-pair coherent-enhancement effect
//! (at 24 µm separation a neighbour's radiated field is comparable
//! to the incident field) — but most of it does not survive the
//! 3P/1P reduction statistic. Across a 0.25×–4× density bracket
//! the metric effect is not monotone and never exceeds ≈ +1 dB
//! at 0.268 MPa / fc 3.0 MHz.
//!
//! # No shadowing — and a warning about extinction
//!
//! At imaging conditions there is **no shadowing**: the total
//! drive at the cloud centre is *enhanced*, **+0.9 to +2.6 dB RMS**
//! relative to the incident tone — neighbours' fields arrive there
//! substantially in phase with the incident wave.
//!
//! **Warning:** the extinction coefficient α of
//! [`super::cloud_attenuation`] (KW-PHYS-006, Xia 2020 energy
//! method) is an *incoherent energy* quantity — each bubble does
//! remove energy (`W > 0`), and at imaging conditions α evaluates
//! to 8–68 Np/m (`αL ≤ 0.08` over a cloud radius). It must **not**
//! be imported as a *local drive reduction*: the coherent field
//! sum at a point can have the opposite sign (the centre-drive
//! measurement implies an effective α of −80 to −246 Np/m, a
//! gain). α is a propagation/energy quantity, not a local-drive
//! quantity.
//!
//! # Regime
//!
//! **Validated at imaging conditions only:** 2.45 MHz,
//! PNP 0.12–1.27 MPa, R0 ≈ 2–2.8 µm, densities to
//! ≈ 128 bubbles/mm³ (the 4× bracket point), void fraction
//! ≈ 2.9e-6 at the formation density. **NOT validated at trigger
//! pressures** (10–20 MPa): nothing here has been checked against
//! a coupled benchmark in the formation regime, and the KM solves
//! themselves leave their validated envelope there. Use
//! [`coupled_regime_reasons`] to check a proposed operating point
//! against the validated envelope. Monopole coupling also requires
//! pair distances ≫ R0 (the WS-B draw's minimum pair distance was
//! 24 µm ≈ 8.6 R0).

mod coupling;
mod pass;
mod solve;

pub use coupling::{coupling_field, pair_geometry};
pub use solve::{segment_band_power, solve_coupled_cloud, CoupledSolution};

/// Picard damping factor θ (WS-B): the history fed to the next
/// pass is `(1 − θ)·previous + θ·new`. θ = 0.5 converges
/// monotonically where plain Picard (θ = 1) oscillates; the fixed
/// point is θ-independent because the coupling field is linear in
/// the `V̈` history.
pub const PICARD_DAMPING_THETA: f64 = 0.5;

/// Validated drive frequency [Hz] (Kuji imaging, WS-B).
pub const COUPLED_VALIDATED_FREQUENCY_HZ: f64 = 2.45e6;

/// Validated incident PNP range [Pa] (WS-B sweep).
pub const COUPLED_VALIDATED_PNP_PA: (f64, f64) = (0.12e6, 1.27e6);

/// Highest density [bubbles/mm³] in the WS-B bracket (4× the
/// ≈ 32 bubbles/mm³ formation density).
pub const COUPLED_VALIDATED_MAX_DENSITY_PER_MM3: f64 = 128.0;

/// 1.5f₀ diagnostic band [Hz] at 2.45 MHz (prototype convention).
pub const BAND_1P5_HZ: (f64, f64) = (3.425e6, 3.925e6);

/// One bubble of the coupled cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledBubble {
    /// Equilibrium radius `R0` [m].
    pub radius0_m: f64,
    /// Position [m] (3D; coupling distances are the full 3D
    /// distances, as in WS-B).
    pub position_m: [f64; 3],
}

/// Numerical configuration of the coupled solve. Defaults are the
/// WS-B production conventions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledCloudConfig {
    /// Liquid density `ρ` [kg/m³] in the monopole coupling weight
    /// `ρ/(4πd)` (WS-B: 998).
    pub density_kg_m3: f64,
    /// Propagation sound speed [m/s] for the retardation delays
    /// `d/c` (WS-B: 1540, the prototype imaging convention — the
    /// KM-internal sound speed stays each model's own `c_liquid`).
    pub propagation_sound_speed_m_s: f64,
    /// Picard damping θ (see [`PICARD_DAMPING_THETA`]).
    pub damping_theta: f64,
    /// Integration steps per drive cycle (the WS-B SPC ladder:
    /// 1000 at ≤ 0.27 MPa, 4000 at 0.73 MPa, 8000 at 1.27 MPa).
    pub steps_per_cycle: usize,
    /// Total drive cycles integrated (WS-B: 12).
    pub n_cycles: usize,
    /// First cycle of the steady segment used by the band
    /// diagnostic (WS-B: cycles 6–12).
    pub segment_start_cycle: usize,
}

impl Default for CoupledCloudConfig {
    fn default() -> Self {
        Self {
            density_kg_m3: 998.0,
            propagation_sound_speed_m_s: 1540.0,
            damping_theta: PICARD_DAMPING_THETA,
            steps_per_cycle: 1000,
            n_cycles: 12,
            segment_start_cycle: 6,
        }
    }
}

/// Reasons a proposed operating point lies outside the validated
/// envelope of this component (empty = inside). Checked against
/// the WS-B validation: imaging frequency and PNPs, bracketed
/// density. Trigger pressures are outside by construction.
#[must_use]
pub fn coupled_regime_reasons(
    frequency_hz: f64,
    pnp_pa: f64,
    density_per_mm3: f64,
) -> Vec<&'static str> {
    let mut reasons = Vec::new();
    if (frequency_hz - COUPLED_VALIDATED_FREQUENCY_HZ).abs() > 0.01 * COUPLED_VALIDATED_FREQUENCY_HZ
    {
        reasons.push("drive frequency outside the validated 2.45 MHz imaging point");
    }
    if pnp_pa < COUPLED_VALIDATED_PNP_PA.0 || pnp_pa > COUPLED_VALIDATED_PNP_PA.1 {
        reasons.push("PNP outside the validated 0.12–1.27 MPa imaging range (not validated at trigger pressures)");
    }
    if density_per_mm3 > COUPLED_VALIDATED_MAX_DENSITY_PER_MM3 {
        reasons.push("density above the validated 128 bubbles/mm³ bracket");
    }
    reasons
}

#[cfg(test)]
mod tests;
