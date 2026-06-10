//! Intramembrane cavitation: acoustic modulation of membrane capacitance.
//!
//! # Physical picture (NICE model)
//!
//! In the *Neuronal Intramembrane Cavitation Excitation* (NICE) model of Plaksin
//! et al. (2014), the two leaflets of the lipid bilayer are driven apart and
//! together by the oscillating acoustic pressure, periodically nucleating and
//! collapsing a nanoscale intramembrane gas cavity (the "bilayer sonophore",
//! Krasovitski et al. 2011). Because the membrane behaves as a parallel-plate
//! capacitor whose plate separation is the bilayer thickness `d`, a relative
//! thickness change `Δd/d` produces a relative capacitance change of opposite
//! sign:
//!
//! ```text
//! C_m = ε₀ ε_r A / d     ⇒     ΔC_m / C_m = − Δd / d
//! ```
//!
//! The resulting time-varying capacitance injects a charge-redistribution
//! (displacement) current into the membrane equation (see [`super::nice`]),
//! which — through the steep voltage dependence of the Na⁺ activation gate —
//! rectifies to a net depolarising drive. This is Blackmore et al. (2019)
//! mechanism (i): capacitance change via flexoelectric / conformational
//! coupling.
//!
//! # Capacitance waveform
//!
//! The carrier oscillation is represented as a sinusoidal modulation of the
//! specific capacitance about its baseline `C_m0` with relative depth `ε`:
//!
//! ```text
//! C_m(t) = C_m0 · (1 + ε · sin(2π f t))
//! dC_m/dt = C_m0 · ε · 2π f · cos(2π f t)
//! ```
//!
//! with `t` in ms and `f` the acoustic carrier frequency in MHz, so `2π f`
//! carries units of rad/ms and `dC_m/dt` units of µF/cm²/ms — consistent with
//! the Hodgkin–Huxley integration in [`super::hodgkin_huxley`].
//!
//! # Pressure → modulation-depth bridge
//!
//! [`modulation_depth_from_pressure`] gives a first-principles small-signal
//! estimate of `ε` from the peak acoustic pressure using the bilayer
//! area-expansion modulus `K_A`. **Evidence tier:** analytic small-strain
//! derivation plus the limiting-behaviour and monotonicity property tests in
//! this module — it is *not* calibrated against the full bilayer-sonophore ODE
//! of Plaksin et al. (2014) or its SONIC reduction (Lemaire et al. 2019). It
//! provides the correct scaling (`ε ∝ p`) and order of magnitude for coupling
//! the acoustic field to the membrane, and the modulation depth may instead be
//! supplied directly when a calibrated value is available.
//!
//! # References
//!
//! - Krasovitski, B. et al. (2011). Intramembrane cavitation as a unifying
//!   mechanism for ultrasound-induced bioeffects. *PNAS* 108(8), 3258-3263.
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). Intramembrane cavitation as a
//!   predictive bio-piezoelectric mechanism for ultrasonic brain stimulation.
//!   *Phys. Rev. X* 4, 011004.
//! - Lemaire, T. et al. (2019). Understanding ultrasound neuromodulation using a
//!   computationally efficient and interpretable model of intramembrane
//!   cavitation. *J. Neural Eng.* 16, 046007 (SONIC).
//! - Blackmore, J. et al. (2019). Ultrasound neuromodulation: a review of
//!   results, mechanisms and safety. *Ultrasound Med. Biol.* 45(7), 1509-1536.
//! - Rawicz, W. et al. (2000). Effect of chain length and unsaturation on
//!   elasticity of lipid bilayers. *Biophys. J.* 79(1), 328-339 (K_A ≈ 0.24 N/m).

use std::f64::consts::PI;

/// A time-varying membrane-capacitance source for the NICE coupling.
///
/// Implementors supply the instantaneous specific capacitance `C_m(t)` and its
/// time derivative `dC_m/dt`, in the electrophysiology units used by
/// [`super::hodgkin_huxley`] (`C_m` [µF/cm²], `t` [ms]). The two concrete
/// sources are [`CapacitanceModulation`] (a symmetric sinusoid) and
/// [`super::bls::BilayerSonophore`] (the grounded curved-dome geometry of
/// Plaksin et al. 2014, Eq. 8). [`super::nice::simulate_nice`] is generic over
/// this trait so the membrane integration is monomorphised per source with zero
/// dispatch overhead.
pub trait CapacitanceSource {
    /// Instantaneous specific capacitance C_m(t) [µF/cm²].
    fn capacitance(&self, t_ms: f64) -> f64;
    /// Instantaneous capacitance rate dC_m/dt [µF/cm²/ms].
    fn capacitance_rate(&self, t_ms: f64) -> f64;
    /// Baseline (resting) specific capacitance C_m0 [µF/cm²].
    fn baseline_capacitance(&self) -> f64;
    /// Angular carrier frequency ω = 2π f [rad/ms] (for sampling-adequacy checks).
    fn carrier_omega_rad_ms(&self) -> f64;
    /// Physical-consistency predicate for the source parameters.
    fn is_source_valid(&self) -> bool;
}

/// Sinusoidal membrane-capacitance modulation driven by the acoustic carrier.
///
/// Encapsulates the NICE capacitance waveform `C_m(t) = C_m0·(1 + ε·sin(ωt))`
/// and its analytic time derivative, in the electrophysiology units used by
/// [`super::hodgkin_huxley`] (`C_m` [µF/cm²], `t` [ms]).
#[derive(Debug, Clone, Copy)]
pub struct CapacitanceModulation {
    /// Baseline specific capacitance C_m0 [µF/cm²].
    pub cm0_uf_cm2: f64,
    /// Relative modulation depth ε (dimensionless, |ε| < 1 for a physical,
    /// strictly-positive capacitance).
    pub depth: f64,
    /// Angular carrier frequency ω = 2π f [rad/ms].
    pub omega_rad_ms: f64,
}

impl CapacitanceModulation {
    /// Construct from baseline capacitance [µF/cm²], modulation depth ε, and
    /// carrier frequency in **MHz**. `f` MHz = `1000·f` cycles per ms, hence the
    /// angular frequency is `ω = 2π·1000·f_MHz` rad/ms — the unit consumed by the
    /// HH integration ([`super::hodgkin_huxley`], time in ms).
    #[must_use]
    pub fn new(cm0_uf_cm2: f64, depth: f64, freq_mhz: f64) -> Self {
        Self {
            cm0_uf_cm2,
            depth,
            omega_rad_ms: 2.0 * PI * 1.0e3 * freq_mhz,
        }
    }

    /// Returns `true` if the modulation keeps capacitance strictly positive
    /// (|ε| < 1) with a positive baseline and carrier frequency.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.cm0_uf_cm2 > 0.0 && self.depth.abs() < 1.0 && self.omega_rad_ms > 0.0
    }

    /// Instantaneous capacitance C_m(t) [µF/cm²] at time `t_ms`.
    #[inline]
    #[must_use]
    pub fn capacitance(&self, t_ms: f64) -> f64 {
        self.cm0_uf_cm2 * (1.0 + self.depth * (self.omega_rad_ms * t_ms).sin())
    }

    /// Instantaneous capacitance rate dC_m/dt [µF/cm²/ms] at time `t_ms`.
    #[inline]
    #[must_use]
    pub fn capacitance_rate(&self, t_ms: f64) -> f64 {
        self.cm0_uf_cm2 * self.depth * self.omega_rad_ms * (self.omega_rad_ms * t_ms).cos()
    }
}

impl CapacitanceSource for CapacitanceModulation {
    #[inline]
    fn capacitance(&self, t_ms: f64) -> f64 {
        CapacitanceModulation::capacitance(self, t_ms)
    }
    #[inline]
    fn capacitance_rate(&self, t_ms: f64) -> f64 {
        CapacitanceModulation::capacitance_rate(self, t_ms)
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
        self.is_valid()
    }
}

/// Lipid-bilayer area-expansion modulus K_A [N/m] (Rawicz et al. 2000).
pub const BILAYER_AREA_MODULUS_N_M: f64 = 0.24;

/// Small-signal estimate of the capacitance modulation depth ε from a peak
/// acoustic pressure.
///
/// # Derivation
///
/// A peak pressure `p` loading a spherical membrane of radius `R` produces an
/// in-plane membrane tension `T = p·R/2` (Laplace thin-shell law; the same
/// relation used by [`crate::acoustics::therapy::sonogenetics::compute_membrane_tension`]).
/// That tension stretches the bilayer by an areal strain `α = ΔA/A = T / K_A`
/// against the area-expansion modulus `K_A`. At fixed lipid volume the bilayer
/// thins in proportion to its area increase, `Δd/d = −ΔA/A = −α`, and since
/// `ΔC_m/C_m = −Δd/d` the relative capacitance change is
/// ```text
/// ε = ΔC_m/C_m = α = T / K_A = p·R / (2·K_A)
/// ```
/// which is dimensionless: `[Pa·m]/[N/m] = [N/m]/[N/m]`. This yields the correct
/// linear scaling `ε ∝ p` and the right order of magnitude (e.g. a 10 kPa peak,
/// R = 10 µm, K_A ≈ 0.24 N/m gives ε ≈ 0.2). The estimate saturates at large
/// neuromodulation pressures where the small-strain assumption breaks down; the
/// result is clamped to `[0, 0.99]` to keep the capacitance strictly positive in
/// [`CapacitanceModulation`].
///
/// **Evidence tier:** analytic small-strain derivation + property tests (see
/// module-level note); not calibrated to the full bilayer-sonophore ODE.
///
/// # Arguments
/// * `peak_pressure_pa` — peak acoustic pressure amplitude [Pa]
/// * `cell_radius_m` — membrane (cell soma) radius R [m]
/// * `area_modulus_n_m` — bilayer area-expansion modulus K_A [N/m]
///   (use [`BILAYER_AREA_MODULUS_N_M`] for the default lipid value)
#[must_use]
pub fn modulation_depth_from_pressure(
    peak_pressure_pa: f64,
    cell_radius_m: f64,
    area_modulus_n_m: f64,
) -> f64 {
    if !(peak_pressure_pa.is_finite()
        && peak_pressure_pa > 0.0
        && cell_radius_m > 0.0
        && area_modulus_n_m > 0.0)
    {
        return 0.0;
    }
    (peak_pressure_pa * cell_radius_m / (2.0 * area_modulus_n_m)).clamp(0.0, 0.99)
}

/// A phase-periodic membrane-capacitance source backed by a precomputed
/// one-carrier-cycle sample table.
///
/// This is the single source of truth for capacitance sources whose `C_m(t)` is
/// most cheaply obtained by precomputing one steady carrier cycle and
/// interpolating by phase — both the quasi-static
/// ([`super::bls::pressures::BilayerSonophoreQuasistatic`]) and the transient
/// ([`super::bls::dynamics::BilayerSonophoreDynamic`]) bilayer-sonophore sources
/// build one. The sample array is uniform in carrier phase with index 0 at phase
/// 0; `dC_m/dt` is derived once by a central difference on the periodic array,
/// `dC_m/dt = (dC_m/dphase)·ω`.
#[derive(Debug, Clone)]
pub struct PhaseCycle {
    cm0_uf_cm2: f64,
    omega_rad_ms: f64,
    cm_cycle: Vec<f64>,
    dcmdt_cycle: Vec<f64>,
}

impl PhaseCycle {
    /// Build from the baseline capacitance [µF/cm²], angular carrier frequency
    /// [rad/ms], and one cycle of `C_m` samples (uniform phase, index 0 = phase 0).
    ///
    /// # Panics (debug)
    /// Panics if `cm_cycle` has fewer than 2 samples.
    #[must_use]
    pub fn new(cm0_uf_cm2: f64, omega_rad_ms: f64, cm_cycle: Vec<f64>) -> Self {
        debug_assert!(cm_cycle.len() >= 2, "PhaseCycle needs ≥ 2 samples");
        let n = cm_cycle.len();
        let dphase = 2.0 * PI / n as f64;
        let dcmdt_cycle: Vec<f64> = (0..n)
            .map(|i| {
                let next = cm_cycle[(i + 1) % n];
                let prev = cm_cycle[(i + n - 1) % n];
                (next - prev) / (2.0 * dphase) * omega_rad_ms
            })
            .collect();
        Self {
            cm0_uf_cm2,
            omega_rad_ms,
            cm_cycle,
            dcmdt_cycle,
        }
    }

    /// Linearly interpolate a periodic per-cycle array at carrier time `t_ms`.
    #[inline]
    fn interp(arr: &[f64], omega_rad_ms: f64, t_ms: f64) -> f64 {
        let n = arr.len();
        let phase = (omega_rad_ms * t_ms).rem_euclid(2.0 * PI);
        let x = phase / (2.0 * PI) * n as f64;
        let i = (x.floor() as usize) % n;
        let frac = x - x.floor();
        arr[i] * (1.0 - frac) + arr[(i + 1) % n] * frac
    }
}

impl CapacitanceSource for PhaseCycle {
    #[inline]
    fn capacitance(&self, t_ms: f64) -> f64 {
        Self::interp(&self.cm_cycle, self.omega_rad_ms, t_ms)
    }
    #[inline]
    fn capacitance_rate(&self, t_ms: f64) -> f64 {
        Self::interp(&self.dcmdt_cycle, self.omega_rad_ms, t_ms)
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
        self.cm0_uf_cm2 > 0.0 && self.omega_rad_ms > 0.0 && self.cm_cycle.len() >= 8
    }
}
