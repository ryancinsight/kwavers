//! Time-resolved cavitation-shielding control under pulsed and frequency-swept
//! drives.
//!
//! A bubble cloud that accumulates at the focus attenuates the incoming drive by
//! resonant scattering (Commander–Prosperetti): the delivered focal pressure
//! `p_focus = p_drive·exp(−(α_tissue + α_gas(f,β))·L)` *falls* as the void
//! fraction `β` grows, so cavitation production self-limits — the **shielding**
//! that caps HIFU/histotripsy efficacy. Two exposure controls suppress it, and
//! both emerge from one void-fraction balance rather than ad-hoc switches:
//!
//! * **Millisecond pulsing.** During the OFF interval the residual cloud
//!   dissolves by gas diffusion (Epstein–Plesset, time constant `τ_diss`); `β`
//!   strictly decreases through every OFF interval. An OFF interval comparable to
//!   `τ_diss` resets `β` toward zero each cycle, so the next pulse sees a
//!   transparent focus. The PRF is the control knob and there is an *optimum* —
//!   too short an OFF accumulates the cloud, too long wastes treatment time — so
//!   the benefit over a continuous drive is regime-dependent, not universal.
//! * **Frequency sweeping.** The accumulated cloud scatters most strongly at its
//!   Minnaert resonance. A swept drive spends most of each period *off* that
//!   resonance, so the instantaneous `α_gas(f(t),β)` it experiences is smaller
//!   than a fixed tone parked on resonance — less self-shielding, more delivered
//!   pressure. This is captured exactly by evaluating the same C–P attenuation at
//!   the instantaneous swept frequency; no separate de-coherence factor is
//!   introduced.
//!
//! The two compose: a swept *and* pulsed drive both lowers the per-pulse
//! attenuation and clears the residual between pulses, the regime the references
//! below identify as optimal (short sweep time + large sweep range, low duty).
//!
//! # Model tier and scope
//! This is a reduced-order **phenomenological balance** for the focal void
//! fraction, not a bubble-by-bubble cloud simulation. The shielding law
//! (`α_gas`) and the inter-pulse dissolution (`τ_diss`) are the audited
//! first-principles pieces ([`commander_prosperetti_attenuation`],
//! [`residual_dissolution_time_s`]); the production term is a threshold
//! supralinear source calibrated by its parameters. Claims rest on the ODE
//! structure (property-tested limiting cases) plus those audited sub-models.
//!
//! # References
//! * Wang M. (2017), *HIFU Ablation Using the Frequency Sweeping Excitation*,
//!   PhD thesis, NTU — chirp enhances both stable and inertial cavitation; short
//!   sweep time + large sweep range preferred (0.9–1.1 MHz, 300 kPa surface).
//! * Zhang et al. (2015), *Ultrason. Sonochem.* 27, 437 — enhancement/quenching
//!   of HIFU cavitation via short frequency-sweep gaps.
//! * Pulsed-ultrasound OFF-time lets sub-resonant residual bubbles dissolve,
//!   resetting the cloud and removing the shielding layer (LWT 2021).

use super::swept_frequency::{residual_dissolution_time_s, tissue_gas_diffusion, FrequencySweep};
use crate::acoustics::bubble_dynamics::commander_prosperetti_attenuation;

/// Drive-frequency program over the exposure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DriveFrequency {
    /// Constant carrier `f₀` [Hz] (the conventional fixed-frequency drive).
    Fixed(f64),
    /// Frequency-swept (chirp) carrier; the instantaneous frequency moves across
    /// the band each sweep period.
    Swept(FrequencySweep),
}

impl DriveFrequency {
    /// Instantaneous carrier frequency at time `t_s` [Hz].
    #[must_use]
    #[inline]
    pub fn at(&self, t_s: f64) -> f64 {
        match self {
            DriveFrequency::Fixed(f) => *f,
            DriveFrequency::Swept(s) => s.instantaneous_frequency(t_s),
        }
    }
}

/// ON/OFF pulse gate. `pulse_off_s = 0` is a continuous (CW) drive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PulseProtocol {
    /// ON duration per pulse [s] (the millisecond burst).
    pub pulse_on_s: f64,
    /// OFF interval between pulses [s] (`0` ⇒ continuous wave).
    pub pulse_off_s: f64,
}

impl PulseProtocol {
    /// Continuous-wave drive (no OFF interval).
    #[must_use]
    #[inline]
    pub fn continuous() -> Self {
        Self {
            pulse_on_s: 1.0,
            pulse_off_s: 0.0,
        }
    }

    /// Pulsed drive with the given ON/OFF durations.
    #[must_use]
    #[inline]
    pub fn pulsed(pulse_on_s: f64, pulse_off_s: f64) -> Self {
        Self {
            pulse_on_s: pulse_on_s.max(0.0),
            pulse_off_s: pulse_off_s.max(0.0),
        }
    }

    /// Fraction of time the drive is ON ∈ (0, 1].
    #[must_use]
    #[inline]
    pub fn duty_cycle(&self) -> f64 {
        let period = self.pulse_on_s + self.pulse_off_s;
        if period > 0.0 {
            (self.pulse_on_s / period).clamp(0.0, 1.0)
        } else {
            1.0
        }
    }

    /// Whether the drive is ON at time `t_s`.
    #[must_use]
    #[inline]
    pub fn is_on(&self, t_s: f64) -> bool {
        let period = self.pulse_on_s + self.pulse_off_s;
        if period <= 0.0 || self.pulse_off_s <= 0.0 {
            return true; // continuous
        }
        let phase = t_s.rem_euclid(period);
        phase < self.pulse_on_s
    }
}

/// Threshold–supralinear cavitation production source for the void-fraction
/// balance: above the cavitation threshold the production rate grows as a power
/// of the pressure excess and saturates as `β → β_max`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CavitationProduction {
    /// Maximum void-fraction production rate `k_prod` [1/s] at full drive.
    pub k_prod_per_s: f64,
    /// Saturation void fraction `β_max` (logistic ceiling).
    pub beta_max: f64,
    /// Cavitation pressure threshold `p_thr` [Pa]; no production below it.
    pub p_threshold_pa: f64,
    /// Pressure scale `p_ref` [Pa] normalising the supralinear excess term.
    pub p_ref_pa: f64,
    /// Supralinearity exponent `n ≥ 1` of the production–pressure law.
    pub supralinearity: f64,
}

impl Default for CavitationProduction {
    /// Histotripsy-scale defaults: ~MPa threshold, MPa reference scale, cubic
    /// supralinearity, percent-level saturation void fraction.
    fn default() -> Self {
        Self {
            k_prod_per_s: 50.0,
            beta_max: 1.0e-2,
            p_threshold_pa: 1.0e6,
            p_ref_pa: 1.0e6,
            supralinearity: 3.0,
        }
    }
}

/// Focal-region medium and path properties for the shielding evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShieldingMedium {
    /// Liquid sound speed `c` [m/s].
    pub c_liquid: f64,
    /// Liquid density `ρ` [kg/m³].
    pub rho_liquid: f64,
    /// Liquid dynamic viscosity `μ` [Pa·s].
    pub mu_liquid: f64,
    /// Ambient pressure `P₀` [Pa].
    pub p0_pa: f64,
    /// Gas polytropic exponent `κ`.
    pub polytropic: f64,
    /// Representative residual/cloud bubble radius `R₀` [m] (sets both the C–P
    /// resonance and the Epstein–Plesset dissolution time).
    pub r0_m: f64,
    /// Tissue power-law attenuation along the path `α_tissue` [Np/m].
    pub alpha_tissue_np_m: f64,
    /// Proximal→focus path length `L` [m] over which the cloud shields.
    pub path_len_m: f64,
    /// Dissolved-gas saturation fraction `f_sat` (`<1` ⇒ residual bubbles
    /// dissolve in the OFF interval; sets `τ_diss`).
    pub saturation_fraction: f64,
}

impl ShieldingMedium {
    /// Soft-tissue defaults with a 2 µm residual bubble, mild undersaturation,
    /// and a 4 cm focal path.
    #[must_use]
    pub fn soft_tissue() -> Self {
        Self {
            c_liquid: 1540.0,
            rho_liquid: 1050.0,
            mu_liquid: 1.5e-3,
            p0_pa: 1.013e5,
            polytropic: 1.4,
            r0_m: 2.0e-6,
            alpha_tissue_np_m: 5.0,
            path_len_m: 0.04,
            saturation_fraction: 0.9,
        }
    }

    /// Epstein–Plesset residual dissolution time `τ_diss` [s] for this medium's
    /// residual bubble; `None` (returned as `+∞` to callers) when the gas state
    /// does not dissolve (saturated/supersaturated).
    #[must_use]
    pub fn dissolution_time_s(&self) -> f64 {
        residual_dissolution_time_s(self.r0_m, tissue_gas_diffusion(self.saturation_fraction))
            .filter(|t| t.is_finite() && *t > 0.0)
            .unwrap_or(f64::INFINITY)
    }
}

/// Integration controls for the shielding balance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShieldingConfig {
    /// Total simulated exposure time [s].
    pub total_time_s: f64,
    /// Integration time step [s] (RK4); resolve the shorter of ON/OFF/`τ_diss`.
    pub dt_s: f64,
}

/// Per-sample trace plus scalar summaries of a shielding-controlled exposure.
#[derive(Debug, Clone)]
pub struct ShieldingTrace {
    /// Sample times [s].
    pub time_s: Vec<f64>,
    /// Focal void fraction `β(t)`.
    pub void_fraction: Vec<f64>,
    /// Delivered focal pressure `p_focus(t)` [Pa] (0 while the drive is OFF).
    pub delivered_pressure_pa: Vec<f64>,
    /// Delivered fraction `p_focus/p_drive` while ON (the instantaneous
    /// transmission through tissue + cloud).
    pub delivered_fraction: Vec<f64>,
    /// Peak void fraction over the exposure — the worst-case shielding layer.
    pub peak_void_fraction: f64,
    /// Time-averaged void fraction over the whole record — how shielded the
    /// focus is on average. Pulsing lowers this by letting β decay in the OFF
    /// intervals even when the per-pulse peak is fixed by the threshold balance.
    pub mean_void_fraction: f64,
    /// Time-averaged delivered transmission over the ON samples.
    pub mean_delivered_fraction_on: f64,
    /// Delivered acoustic energy proxy `∫ p_focus² dt` over ON samples [Pa²·s].
    pub delivered_energy: f64,
    /// Unshielded reference energy (same gating, `α_gas = 0`) [Pa²·s].
    pub unshielded_energy: f64,
    /// Shielding loss `1 − delivered/unshielded ∈ [0, 1]` — the fraction of
    /// otherwise-deliverable focal energy lost to the bubble cloud.
    pub shielding_loss_fraction: f64,
}

/// Scalar summary of one exposure (the comparison-table row).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShieldingSummary {
    /// Peak void fraction.
    pub peak_void_fraction: f64,
    /// Time-averaged void fraction over the whole record.
    pub mean_void_fraction: f64,
    /// Time-averaged delivered transmission over the ON samples.
    pub mean_delivered_fraction_on: f64,
    /// Delivered acoustic energy proxy [Pa²·s].
    pub delivered_energy: f64,
    /// Shielding loss fraction ∈ [0, 1].
    pub shielding_loss_fraction: f64,
}

impl From<&ShieldingTrace> for ShieldingSummary {
    fn from(t: &ShieldingTrace) -> Self {
        Self {
            peak_void_fraction: t.peak_void_fraction,
            mean_void_fraction: t.mean_void_fraction,
            mean_delivered_fraction_on: t.mean_delivered_fraction_on,
            delivered_energy: t.delivered_energy,
            shielding_loss_fraction: t.shielding_loss_fraction,
        }
    }
}

/// Delivered focal pressure through tissue + bubble-cloud shielding at void
/// fraction `beta`, drive frequency `freq_hz`, and surface drive `drive_pa`.
#[inline]
fn delivered_pressure(drive_pa: f64, freq_hz: f64, beta: f64, medium: &ShieldingMedium) -> f64 {
    let alpha_gas = commander_prosperetti_attenuation(
        freq_hz.max(1.0),
        beta.max(0.0),
        medium.r0_m,
        medium.c_liquid,
        medium.rho_liquid,
        medium.mu_liquid,
        medium.p0_pa,
        medium.polytropic,
    );
    let atten =
        (-(medium.alpha_tissue_np_m.max(0.0) + alpha_gas) * medium.path_len_m.max(0.0)).exp();
    drive_pa * atten
}

/// Void-fraction time derivative `dβ/dt` at `(t, beta)` under the drive.
#[allow(clippy::too_many_arguments)]
#[inline]
fn dbeta_dt(
    t_s: f64,
    beta: f64,
    drive_pa: f64,
    freq: &DriveFrequency,
    protocol: &PulseProtocol,
    prod: &CavitationProduction,
    medium: &ShieldingMedium,
    inv_tau_diss: f64,
) -> f64 {
    let decay = -beta.max(0.0) * inv_tau_diss;
    if !protocol.is_on(t_s) {
        return decay;
    }
    let p_focus = delivered_pressure(drive_pa, freq.at(t_s), beta, medium);
    let excess = ((p_focus - prod.p_threshold_pa) / prod.p_ref_pa.max(f64::EPSILON)).max(0.0);
    if excess <= 0.0 {
        return decay;
    }
    let headroom = (1.0 - beta / prod.beta_max.max(f64::EPSILON)).max(0.0);
    prod.k_prod_per_s.max(0.0) * excess.powf(prod.supralinearity.max(1.0)) * headroom + decay
}

/// Simulate the focal void-fraction balance over a pulsed/swept exposure and
/// report the per-sample trace plus shielding summaries.
///
/// The balance is `dβ/dt = production(p_focus, β)·[ON] − β/τ_diss`, integrated by
/// RK4 with `β` floored at zero. Production is gated by the pulse protocol and
/// driven by the *delivered* (post-shielding) focal pressure, so shielding
/// self-limits cavitation; the OFF interval relaxes `β` with the Epstein–Plesset
/// `τ_diss`, and the swept frequency lowers the instantaneous attenuation.
#[must_use]
pub fn simulate_shielding(
    drive_pressure_pa: f64,
    freq: &DriveFrequency,
    protocol: &PulseProtocol,
    prod: &CavitationProduction,
    medium: &ShieldingMedium,
    cfg: &ShieldingConfig,
) -> ShieldingTrace {
    let dt = cfg.dt_s;
    let n_steps =
        if dt.is_finite() && dt > 0.0 && cfg.total_time_s.is_finite() && cfg.total_time_s > 0.0 {
            (cfg.total_time_s / dt).round() as usize
        } else {
            0
        };

    let tau_diss = medium.dissolution_time_s();
    let inv_tau_diss = if tau_diss.is_finite() && tau_diss > 0.0 {
        1.0 / tau_diss
    } else {
        0.0
    };
    // Unshielded transmission (β = 0): tissue attenuation only — the reference
    // the cloud derates against.
    let unshielded_fraction = delivered_pressure(1.0, freq.at(0.0), 0.0, medium).max(0.0);

    let cap = n_steps + 1;
    let mut time_s = Vec::with_capacity(cap);
    let mut void_fraction = Vec::with_capacity(cap);
    let mut delivered_pressure_pa = Vec::with_capacity(cap);
    let mut delivered_fraction = Vec::with_capacity(cap);

    let mut beta = 0.0_f64;
    let mut peak_void_fraction = 0.0_f64;
    let mut sum_beta = 0.0_f64;
    let mut n_samples = 0usize;
    let mut sum_frac_on = 0.0_f64;
    let mut n_on = 0usize;
    let mut delivered_energy = 0.0_f64;
    let mut unshielded_energy = 0.0_f64;

    // β is bounded below by 0 (no negative gas) and above by the saturation
    // ceiling β_max; clamp every RK4 stage so the stiff logistic source cannot
    // numerically overshoot the physical ceiling.
    let beta_ceiling = prod.beta_max.max(0.0);
    let clamp_beta = |b: f64| b.clamp(0.0, beta_ceiling);
    let f = |t: f64, b: f64| {
        dbeta_dt(
            t,
            b,
            drive_pressure_pa,
            freq,
            protocol,
            prod,
            medium,
            inv_tau_diss,
        )
    };

    for i in 0..=n_steps {
        let t = i as f64 * dt;
        let on = protocol.is_on(t);
        // Unshielded reference uses the instantaneous frequency too, so the
        // shielding loss isolates the cloud term, not the tissue term.
        let unshielded_p = drive_pressure_pa * unshielded_fraction;
        let p_focus = if on {
            delivered_pressure(drive_pressure_pa, freq.at(t), beta, medium)
        } else {
            0.0
        };
        let frac = if drive_pressure_pa.abs() > 0.0 {
            p_focus / drive_pressure_pa
        } else {
            0.0
        };

        time_s.push(t);
        void_fraction.push(beta);
        delivered_pressure_pa.push(p_focus);
        delivered_fraction.push(frac);

        peak_void_fraction = peak_void_fraction.max(beta);
        sum_beta += beta;
        n_samples += 1;
        if on {
            sum_frac_on += frac;
            n_on += 1;
            delivered_energy += p_focus * p_focus * dt;
            unshielded_energy += unshielded_p * unshielded_p * dt;
        }

        if i == n_steps {
            break;
        }
        // RK4 step with β floored at zero each stage.
        let k1 = f(t, beta);
        let k2 = f(t + 0.5 * dt, clamp_beta(beta + 0.5 * dt * k1));
        let k3 = f(t + 0.5 * dt, clamp_beta(beta + 0.5 * dt * k2));
        let k4 = f(t + dt, clamp_beta(beta + dt * k3));
        beta = clamp_beta(beta + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4));
    }

    let mean_delivered_fraction_on = if n_on > 0 {
        sum_frac_on / n_on as f64
    } else {
        0.0
    };
    let mean_void_fraction = if n_samples > 0 {
        sum_beta / n_samples as f64
    } else {
        0.0
    };
    let shielding_loss_fraction = if unshielded_energy > 0.0 {
        (1.0 - delivered_energy / unshielded_energy).clamp(0.0, 1.0)
    } else {
        0.0
    };

    ShieldingTrace {
        time_s,
        void_fraction,
        delivered_pressure_pa,
        delivered_fraction,
        peak_void_fraction,
        mean_void_fraction,
        mean_delivered_fraction_on,
        delivered_energy,
        unshielded_energy,
        shielding_loss_fraction,
    }
}

/// 2×2 comparison of shielding control: {continuous, pulsed} × {fixed, swept},
/// at a common drive amplitude, medium, and production model. The fixed tone is
/// the sweep mean frequency, so the only differences are the exposure controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShieldingComparison {
    /// Continuous fixed-frequency drive (the conventional worst case).
    pub cw_fixed: ShieldingSummary,
    /// Continuous swept drive (sweep-only control).
    pub cw_swept: ShieldingSummary,
    /// Pulsed fixed-frequency drive (pulse-only control).
    pub pulsed_fixed: ShieldingSummary,
    /// Pulsed swept drive (both controls applied). Sweeping lowers the shielding
    /// at this duty; whether pulsing beats CW is regime-dependent (optimal PRF).
    pub pulsed_swept: ShieldingSummary,
}

/// Run the four {CW, pulsed} × {fixed, swept} exposures and summarise them.
///
/// `sweep` defines the swept drive; the fixed drive uses `sweep.mean_frequency_hz()`
/// so the comparison isolates the control strategy. `pulse` is the pulsed
/// protocol (its OFF interval drives the clearance); the CW rows use a continuous
/// drive of equal amplitude.
#[must_use]
pub fn compare_shielding_control(
    drive_pressure_pa: f64,
    sweep: &FrequencySweep,
    pulse: &PulseProtocol,
    prod: &CavitationProduction,
    medium: &ShieldingMedium,
    cfg: &ShieldingConfig,
) -> ShieldingComparison {
    let fixed = DriveFrequency::Fixed(sweep.mean_frequency_hz());
    let swept = DriveFrequency::Swept(*sweep);
    let cw = PulseProtocol::continuous();

    let run = |freq: &DriveFrequency, proto: &PulseProtocol| -> ShieldingSummary {
        (&simulate_shielding(drive_pressure_pa, freq, proto, prod, medium, cfg)).into()
    };

    ShieldingComparison {
        cw_fixed: run(&fixed, &cw),
        cw_swept: run(&swept, &cw),
        pulsed_fixed: run(&fixed, pulse),
        pulsed_swept: run(&swept, pulse),
    }
}

#[cfg(test)]
#[path = "shielding_tests.rs"]
mod tests;
