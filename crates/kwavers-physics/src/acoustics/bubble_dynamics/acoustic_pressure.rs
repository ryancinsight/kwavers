//! Typed acoustic-drive pressures for bubble dynamics.
//!
//! The Keller–Miksis, Gilmore, Rayleigh–Plesset and shell equations are
//! written in terms of a general time-dependent far-field pressure `p_∞(t)`
//! (Keller & Miksis 1980, JASA 68(2):628–633, Eq. 3.4; Gilmore 1952,
//! Caltech Hydro. Lab. Report 26-4; Marmottant et al. 2005, JASA
//! 118(6):3499–3505). A harmonic drive is the specialisation
//! `p_∞(t) = p0 ± P_A·sin(ωt + φ)` — not part of the equations.
//!
//! An instantaneous pressure and a pressure amplitude share the SI
//! dimension (M L⁻¹ T⁻²), so a plain Aequitas `Pressure` cannot tell them
//! apart — and passing an amplitude where an instantaneous pressure belongs
//! (or vice versa) was a real bug: before Nov 18, 2026 the models
//! multiplied their `p_acoustic` argument by `sin(ωt)` internally,
//! double-modulating every caller that already passed an instantaneous
//! pressure. These newtypes make that mix-up a compile error:
//!
//! * [`InstantaneousPressure`] — the instantaneous far-field acoustic
//!   pressure `p_∞(t) − p0`, the drive argument every bubble model takes.
//! * [`PressureAmplitude`] — the peak pressure `P_A` of a harmonic drive.
//!   Amplitudes never reach a model directly; callers synthesise the tone
//!   with [`PressureAmplitude::sample`].
//!
//! Conversion to raw pascals happens only at formula boundaries
//! ([`InstantaneousPressure::as_pascals`], `PressureRate::as_base`);
//! dense pressure arrays stay scalar numerical storage (ADR 070) and are
//! wrapped one sample at a time where a model consumes them.

use std::ops::{Add, Sub};

use aequitas::systems::si::quantities::{Pressure, PressureRate};

/// Instantaneous far-field acoustic pressure `p_∞(t) − p0` in pascals.
///
/// This is the drive argument of every bubble model in this module: the
/// value of the far-field pressure at one instant, already carrier- and
/// envelope-modulated by the caller. It is NOT an amplitude — an amplitude
/// is [`PressureAmplitude`], and the two cannot be interchanged: there is
/// no conversion between them, only the tone synthesis
/// [`PressureAmplitude::sample`].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct InstantaneousPressure(Pressure);

impl InstantaneousPressure {
    /// Zero acoustic pressure (the bubble sits at ambient `p0`).
    pub const ZERO: Self = Self(Pressure::from_base(0.0));

    /// Construct from a value in pascals.
    #[must_use]
    pub const fn from_pascals(pascals: f64) -> Self {
        Self(Pressure::from_base(pascals))
    }

    /// Construct from an Aequitas pressure quantity — the bridge from an
    /// already-typed boundary (e.g. the microbubble service, which takes
    /// `Pressure<f64>`).
    #[must_use]
    pub const fn from_pressure(pressure: Pressure) -> Self {
        Self(pressure)
    }

    /// The wrapped Aequitas quantity.
    #[must_use]
    pub const fn as_pressure(&self) -> Pressure {
        self.0
    }

    /// Value in pascals — the formula-boundary extraction.
    #[must_use]
    pub fn as_pascals(&self) -> f64 {
        *self.0.as_base()
    }
}

impl Add for InstantaneousPressure {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl Sub for InstantaneousPressure {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

/// Peak pressure `P_A` of a harmonic acoustic drive, in pascals.
///
/// An amplitude is a property of a *tone*, not of an instant: it reaches a
/// bubble model only through the tone it synthesises
/// ([`PressureAmplitude::sample`]). There is no conversion from an
/// amplitude to an [`InstantaneousPressure`] other than sampling the tone
/// at a time — that is the point.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct PressureAmplitude(Pressure);

impl PressureAmplitude {
    /// Construct from a value in pascals.
    #[must_use]
    pub const fn from_pascals(pascals: f64) -> Self {
        Self(Pressure::from_base(pascals))
    }

    /// Construct from an Aequitas pressure quantity.
    #[must_use]
    pub const fn from_pressure(pressure: Pressure) -> Self {
        Self(pressure)
    }

    /// The wrapped Aequitas quantity.
    #[must_use]
    pub const fn as_pressure(&self) -> Pressure {
        self.0
    }

    /// Value in pascals.
    #[must_use]
    pub fn as_pascals(&self) -> f64 {
        *self.0.as_base()
    }

    /// Instantaneous pressure of the tone `p(t) = A·sin(ωt + φ)` at time
    /// `t`, with `phase` in radians.
    #[must_use]
    pub fn instantaneous_at(&self, omega: f64, t: f64, phase: f64) -> InstantaneousPressure {
        InstantaneousPressure::from_pascals(self.as_pascals() * (omega * t + phase).sin())
    }

    /// Instantaneous pressure at an arbitrary absolute phase `φ`:
    /// `p = A·sin φ`. For a fixed-frequency tone φ = ωt + φ₀ (see
    /// [`Self::instantaneous_at`]); for a swept drive φ is the integrated
    /// phase (see `FrequencySweep::pressure`).
    #[must_use]
    pub fn instantaneous_at_phase(&self, phase: f64) -> InstantaneousPressure {
        InstantaneousPressure::from_pascals(self.as_pascals() * phase.sin())
    }

    /// Time derivative at an arbitrary absolute phase: `dp/dt = A·ω·cos φ`,
    /// where `ω` is the instantaneous angular frequency at that moment.
    #[must_use]
    pub fn rate_at_phase(&self, omega: f64, phase: f64) -> PressureRate {
        PressureRate::from_base(self.as_pascals() * omega * phase.cos())
    }

    /// The `(p, dp/dt)` forcing pair of the tone `p(t) = A·sin(ωt + φ)`
    /// at time `t`, with `phase` in radians — the pair every bubble model
    /// takes as its drive. `dp/dt = A·ω·cos(ωt + φ)`.
    #[must_use]
    pub fn sample(&self, omega: f64, t: f64, phase: f64) -> (InstantaneousPressure, PressureRate) {
        let (sin, cos) = (omega * t + phase).sin_cos();
        let amp = self.as_pascals();
        (
            InstantaneousPressure::from_pascals(amp * sin),
            PressureRate::from_base(amp * omega * cos),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two newtypes are distinct at compile time: this test only
    /// compiles if each type is used in its own role, and the sampled
    /// tone matches the closed form.
    #[test]
    fn amplitude_samples_to_instantaneous_tone() {
        let amp = PressureAmplitude::from_pascals(0.26e6);
        let omega = 2.0 * std::f64::consts::PI * 1.74e6;
        let (p, dp_dt) = amp.sample(omega, 0.25e-6, 0.0);
        assert!((p.as_pascals() - 0.26e6 * (omega * 0.25e-6).sin()).abs() < 1e-9);
        assert!((dp_dt.as_base() - 0.26e6 * omega * (omega * 0.25e-6).cos()).abs() < 1e-3);
        // Zero phase at t = 0: zero pressure, maximum rate.
        let (p0, dp0) = amp.sample(omega, 0.0, 0.0);
        assert_eq!(p0.as_pascals(), 0.0);
        assert!((dp0.as_base() - 0.26e6 * omega).abs() < 1e-3);
        // A 120° phase shifts the sampled pressure exactly.
        let (p120, _) = amp.sample(omega, 0.0, 120.0_f64.to_radians());
        assert!((p120.as_pascals() - 0.26e6 * 120.0_f64.to_radians().sin()).abs() < 1e-9);
    }

    /// Instantaneous pressures add (field + secondary Bjerknes), and the
    /// sum keeps the type.
    #[test]
    fn instantaneous_pressures_add() {
        let a = InstantaneousPressure::from_pascals(1.0e5);
        let b = InstantaneousPressure::from_pascals(2.0e4);
        assert_eq!((a + b).as_pascals(), 1.2e5);
        assert_eq!((a - b).as_pascals(), 8.0e4);
        assert_eq!(InstantaneousPressure::ZERO.as_pascals(), 0.0);
    }

    /// The `from_pressure` bridge round-trips: an Aequitas quantity built
    /// in pascals (base or explicit unit) carries the same value.
    #[test]
    fn amplitude_from_pressure_round_trips() {
        use aequitas::systems::si::units::Pascal;
        let from_pa = PressureAmplitude::from_pascals(260_000.0);
        let from_quantity =
            PressureAmplitude::from_pressure(Pressure::from_unit::<Pascal>(260_000.0));
        assert_eq!(from_pa.as_pascals(), from_quantity.as_pascals());
        assert_eq!(
            from_pa.as_pressure(),
            PressureAmplitude::from_pressure(Pressure::from_base(260_000.0)).as_pressure()
        );
        let omega = 2.0 * std::f64::consts::PI * 1.74e6;
        assert_eq!(
            from_pa.sample(omega, 0.3e-6, 0.0).0.as_pascals(),
            from_quantity.sample(omega, 0.3e-6, 0.0).0.as_pascals()
        );
    }
}
