//! Frequency-swept (chirp) acoustic drive waveform.
//!
//! A swept drive sweeps its instantaneous frequency across a band so the
//! resonance condition `f(t) = f_Minnaert(R₀)` is met, in turn, by a wide range
//! of bubble sizes — the mechanism by which a chirp engages a broader fraction
//! of the nuclei population than a single tone (see [`super::engagement`]).
//!
//! The waveform is defined by its exact instantaneous-phase integral
//! `φ(t) = 2π ∫₀ᵗ f(t') dt'`, so the drive pressure `p(t) = A·sin φ(t)` and its
//! time derivative `ṗ(t) = A·2π f(t)·cos φ(t)` are analytic — required by the
//! chirped Keller–Miksis forcing in [`super::chirped_dynamics`], which evaluates
//! both at the retarded time.

use kwavers_core::constants::numerical::TWO_PI;

/// Shape of the frequency ramp within one sweep period.
///
/// Direction (up vs down) is encoded by the ordering of `f_start` / `f_end`; the
/// profile selects how the instantaneous frequency traverses the band.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SweepProfile {
    /// Monotonic sawtooth ramp `f_start → f_end` over each `period`, then resets
    /// (the phase is continuous; only the instantaneous frequency steps back).
    Linear,
    /// Symmetric up-then-down ramp: `f_start → f_end` over the first half of the
    /// period and `f_end → f_start` over the second half. Both the frequency and
    /// the phase are continuous — the natural "sweep up and down" waveform.
    Triangular,
}

/// A frequency-swept acoustic drive.
///
/// `instantaneous_frequency` is the carrier frequency at time `t`; `phase` is the
/// exact accumulated phase; `pressure`/`pressure_derivative` give the drive and
/// its derivative for the bubble-dynamics forcing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrequencySweep {
    /// Frequency at the start of each sweep period [Hz].
    pub f_start_hz: f64,
    /// Frequency at the turn of each sweep period [Hz].
    pub f_end_hz: f64,
    /// Sweep period [s] (one full `Linear` ramp, or one full up-down `Triangular`).
    pub period_s: f64,
    /// Ramp shape.
    pub profile: SweepProfile,
}

impl FrequencySweep {
    /// Construct a sweep, returning `None` for non-physical parameters
    /// (non-finite, non-positive frequency, or non-positive period).
    #[must_use]
    pub fn new(
        f_start_hz: f64,
        f_end_hz: f64,
        period_s: f64,
        profile: SweepProfile,
    ) -> Option<Self> {
        if !(f_start_hz.is_finite()
            && f_end_hz.is_finite()
            && period_s.is_finite()
            && f_start_hz > 0.0
            && f_end_hz > 0.0
            && period_s > 0.0)
        {
            return None;
        }
        Some(Self {
            f_start_hz,
            f_end_hz,
            period_s,
            profile,
        })
    }

    /// Mean frequency over one period [Hz] — `(f_start + f_end)/2` for both
    /// profiles (the time-average of a linear ramp and of a symmetric triangle
    /// are equal). One full period advances the phase by `2π·f̄·period`.
    #[must_use]
    #[inline]
    pub fn mean_frequency_hz(&self) -> f64 {
        0.5 * (self.f_start_hz + self.f_end_hz)
    }

    /// Bandwidth `|f_end − f_start|` [Hz].
    #[must_use]
    #[inline]
    pub fn bandwidth_hz(&self) -> f64 {
        (self.f_end_hz - self.f_start_hz).abs()
    }

    /// Instantaneous carrier frequency `f(t)` [Hz].
    #[must_use]
    pub fn instantaneous_frequency(&self, t_s: f64) -> f64 {
        let phase_frac = (t_s / self.period_s).rem_euclid(1.0); // position in [0,1)
        match self.profile {
            SweepProfile::Linear => {
                self.f_start_hz + (self.f_end_hz - self.f_start_hz) * phase_frac
            }
            SweepProfile::Triangular => {
                if phase_frac < 0.5 {
                    // up ramp over the first half
                    self.f_start_hz + (self.f_end_hz - self.f_start_hz) * (phase_frac / 0.5)
                } else {
                    // down ramp over the second half
                    self.f_end_hz + (self.f_start_hz - self.f_end_hz) * ((phase_frac - 0.5) / 0.5)
                }
            }
        }
    }

    /// Exact accumulated phase `φ(t) = 2π ∫₀ᵗ f(t') dt'` [rad].
    ///
    /// Closed form per profile: `n` complete periods each contribute
    /// `2π·f̄·period`, plus the partial-period integral evaluated analytically.
    #[must_use]
    pub fn phase(&self, t_s: f64) -> f64 {
        let n_periods = (t_s / self.period_s).floor();
        let tau = t_s - n_periods * self.period_s; // elapsed time within current period, [0, period)
        let full = TWO_PI * self.mean_frequency_hz() * self.period_s * n_periods;
        full + TWO_PI * self.partial_phase_cycles(tau)
    }

    /// Phase in *cycles* accumulated from the period start to elapsed `tau`
    /// (`∫₀^τ f dt'`), evaluated analytically for the active profile.
    fn partial_phase_cycles(&self, tau: f64) -> f64 {
        let t = self.period_s;
        let (f0, f1) = (self.f_start_hz, self.f_end_hz);
        match self.profile {
            SweepProfile::Linear => {
                // f(t') = f0 + k t', k = (f1-f0)/T  ⇒  ∫ = f0 τ + ½ k τ²
                let k = (f1 - f0) / t;
                f0 * tau + 0.5 * k * tau * tau
            }
            SweepProfile::Triangular => {
                let half = 0.5 * t;
                let k_up = (f1 - f0) / half; // rate over the up half
                if tau < half {
                    f0 * tau + 0.5 * k_up * tau * tau
                } else {
                    // full up half + partial down half
                    let up = f0 * half + 0.5 * k_up * half * half;
                    let s = tau - half; // elapsed within the down half
                    let k_dn = (f0 - f1) / half;
                    up + f1 * s + 0.5 * k_dn * s * s
                }
            }
        }
    }

    /// Drive pressure `p(t) = A·sin φ(t)` [Pa].
    #[must_use]
    #[inline]
    pub fn pressure(&self, t_s: f64, amplitude_pa: f64) -> f64 {
        amplitude_pa * self.phase(t_s).sin()
    }

    /// Bare drive-pressure time derivative `ṗ(t) = A·2π f(t)·cos φ(t)` [Pa/s]
    /// (without the Keller–Miksis `(1 + Ṙ/c)` compressibility factor, which the
    /// integrator applies).
    #[must_use]
    #[inline]
    pub fn pressure_derivative(&self, t_s: f64, amplitude_pa: f64) -> f64 {
        amplitude_pa * TWO_PI * self.instantaneous_frequency(t_s) * self.phase(t_s).cos()
    }

    /// The `(p_ac, ṗ_ac)` forcing pair at time `t` — the closure consumed by the
    /// chirped Keller–Miksis core.
    #[must_use]
    #[inline]
    pub fn forcing(&self, t_s: f64, amplitude_pa: f64) -> (f64, f64) {
        (
            self.pressure(t_s, amplitude_pa),
            self.pressure_derivative(t_s, amplitude_pa),
        )
    }

    /// Frequency band `(f_lo, f_hi)` actually traversed by the sweep within the
    /// window `[0, window_s]` — the cycle-budget gate. A pulse shorter than the
    /// sweep period covers only part of the band; a pulse spanning ≥ one period
    /// covers the full `[min(f_start,f_end), max(f_start,f_end)]`. This is why a
    /// µs (≈ single-cycle) pulse realizes ≈ zero swept bandwidth while a ms pulse
    /// realizes the full band (see [`super::engagement`]).
    #[must_use]
    pub fn covered_band_hz(&self, window_s: f64) -> (f64, f64) {
        let lo_full = self.f_start_hz.min(self.f_end_hz);
        let hi_full = self.f_start_hz.max(self.f_end_hz);
        if !(window_s.is_finite() && window_s > 0.0) {
            return (self.f_start_hz, self.f_start_hz);
        }
        // A triangular period reaches the turn (f_end) at T/2; a linear period
        // reaches f_end at T. Once the window covers that fraction, the full band
        // is traversed.
        let full_at = match self.profile {
            SweepProfile::Linear => self.period_s,
            SweepProfile::Triangular => 0.5 * self.period_s,
        };
        if window_s >= full_at {
            return (lo_full, hi_full);
        }
        // Partial coverage: the instantaneous frequency ramps linearly from
        // f_start toward f_end; sample the endpoint reached at `window_s`.
        let f_at_window = self.instantaneous_frequency(window_s);
        (
            self.f_start_hz.min(f_at_window),
            self.f_start_hz.max(f_at_window),
        )
    }
}
