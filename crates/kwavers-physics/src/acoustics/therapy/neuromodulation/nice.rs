//! NICE coupling: intramembrane-cavitation drive of a Hodgkin–Huxley neuron.
//!
//! This is the *Neuronal Intramembrane Cavitation Excitation* pathway (Plaksin
//! et al. 2014): the acoustic carrier modulates the membrane capacitance
//! ([`CapacitanceModulation`]), and the resulting charge-redistribution
//! (displacement) current `−V·dC_m/dt` perturbs the Hodgkin–Huxley membrane
//! ([`super::hodgkin_huxley`]).
//!
//! # Net excitability shift (and its sign)
//!
//! On the fast carrier timescale (sub-µs) the membrane charge `Q = C_m·V` is
//! approximately conserved, so the voltage tracks `V ≈ Q/C_m(t)` and swings with
//! the carrier. Two competing effects then set the *cycle-averaged* membrane
//! potential:
//!
//! 1. **Geometric shift.** For a symmetric modulation
//!    `⟨1/(1 + ε·sinωt)⟩ = 1/√(1 − ε²) > 1`, so `⟨V⟩ ≈ V_rest/√(1 − ε²)`. Because
//!    `V_rest < 0` this moves the mean potential *further from zero*
//!    (hyperpolarising), with a magnitude that grows monotonically with `ε`.
//! 2. **Gating rectification.** The Na⁺ activation gate is a steep, nonlinear
//!    function of `V`, so it opens more on the depolarised half-cycle than it
//!    closes on the hyperpolarised half-cycle, injecting a *depolarising* net
//!    current that partially offsets (1).
//!
//! For the symmetric sinusoidal capacitance waveform used here, effect (1)
//! dominates and the net result is a depth-dependent **hyperpolarisation**
//! (a suppressive excitability shift); the per-cycle voltage excursions still
//! reach progressively more depolarised peaks as `ε` grows. Reproducing the net
//! *excitation* of the full NICE model requires the strongly asymmetric
//! bilayer-sonophore capacitance waveform (sharp capacitance excursions during
//! cavity expansion; Plaksin et al. 2014, Lemaire et al. 2019 SONIC), which is
//! recorded as future work. This module therefore models the validated, exact
//! displacement-current coupling and the resulting excitability modulation, not a
//! calibrated spike-rate prediction.
//!
//! # Numerical range
//!
//! The explicit RK4 integration is stable for moderate modulation depths
//! (`ε ≲ 0.6`); as `ε → 1` the trough capacitance `C_m0·(1 − ε) → 0` makes
//! `V = Q/C_m` stiff and the explicit scheme diverges. Use calibrated depths well
//! below 1 (the pressure→depth bridge clamps to 0.99 only as a positivity guard).
//!
//! # Integration
//!
//! The carrier (0.1–1 MHz) must be temporally resolved, so the integrator uses a
//! fixed small step (default: enough for ≥ ~50 samples per carrier cycle) and the
//! explicit RK4 right-hand side [`HhState::deriv`] evaluated with the
//! instantaneous `C_m(t)` and `dC_m/dt`. This is the direct (un-averaged)
//! formulation; it is exact up to the RK4 truncation error and is intended for
//! short windows (tens of ms). The cycle-averaged SONIC reduction (Lemaire et al.
//! 2019) is the route to second-scale simulations and is noted as future work.
//!
//! # References
//!
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). *Phys. Rev. X* 4, 011004.
//! - Lemaire, T. et al. (2019). *J. Neural Eng.* 16, 046007.

use super::hodgkin_huxley::{HhTrace, SPIKE_THRESHOLD_MV};
use super::intramembrane_cavitation::CapacitanceSource;
use super::membrane::{axpy_gates, rk4_combine_gates, Gates, Membrane};

/// Configuration for a NICE simulation, generic over the membrane model `M`
/// (e.g. [`super::hodgkin_huxley::HhParams`] or
/// [`super::cortical::CorticalNeuron`]) and the capacitance source `C` (e.g.
/// [`super::intramembrane_cavitation::CapacitanceModulation`] or
/// [`super::bls::BilayerSonophore`]). The generic bounds monomorphise the
/// integration per (membrane, source) pair with no dynamic dispatch.
#[derive(Debug, Clone)]
pub struct NiceConfig<M: Membrane, C: CapacitanceSource> {
    /// Membrane model. Its baseline capacitance must equal
    /// `source.baseline_capacitance()`.
    pub membrane: M,
    /// Resting membrane potential used for the self-consistent initial state [mV].
    pub v_rest_mv: f64,
    /// Time-varying membrane capacitance driven by the acoustic carrier.
    pub source: C,
    /// Constant external (e.g. synaptic/bias) current density [µA/cm²].
    pub i_bias_ua_cm2: f64,
    /// Integration step [ms].
    pub dt_ms: f64,
    /// Sonication onset [ms]: before this the capacitance is held at baseline.
    pub onset_ms: f64,
    /// Sonication offset [ms]: after this the capacitance returns to baseline.
    /// Action potentials evoked by the NICE mechanism typically appear *after*
    /// this time, when the accumulated membrane charge depolarises the membrane
    /// at the restored baseline capacitance — set `t_end_ms > offset_ms` to
    /// observe the post-stimulus response.
    pub offset_ms: f64,
    /// Total simulated duration [ms].
    pub t_end_ms: f64,
}

impl<M: Membrane, C: CapacitanceSource> NiceConfig<M, C> {
    /// Returns `true` if parameters are self-consistent: valid membrane and
    /// source, positive step, equal baseline capacitances, coherent time window.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.membrane.is_membrane_valid()
            && self.source.is_source_valid()
            && self.dt_ms > 0.0
            && self.t_end_ms > 0.0
            && self.onset_ms <= self.offset_ms
            && (self.membrane.cm0_uf_cm2() - self.source.baseline_capacitance()).abs()
                <= 1e-12 * self.membrane.cm0_uf_cm2().max(1.0)
    }

    /// Carrier cycles resolved per integration step (sampling adequacy check).
    /// Values ≥ ~50 indicate the carrier is well resolved.
    #[must_use]
    pub fn samples_per_cycle(&self) -> f64 {
        let period_ms = 2.0 * std::f64::consts::PI / self.source.carrier_omega_rad_ms();
        period_ms / self.dt_ms
    }
}

/// Run the NICE simulation, returning the membrane-voltage trace and spike times.
///
/// During `[onset_ms, offset_ms]` the membrane capacitance follows the source's
/// `C_m(t)`; outside that window it is held at the baseline `cm0` with zero rate
/// (no displacement current). Spikes are upward crossings of
/// [`SPIKE_THRESHOLD_MV`].
///
/// # Panics (debug)
///
/// Panics in debug builds if `cfg.is_valid()` is false.
#[must_use]
pub fn simulate_nice<M: Membrane, C: CapacitanceSource>(cfg: &NiceConfig<M, C>) -> HhTrace {
    debug_assert!(cfg.is_valid(), "NiceConfig failed validity check");
    let mem = &cfg.membrane;
    let n_steps = ((cfg.t_end_ms / cfg.dt_ms).ceil() as usize).max(1);
    let cm0 = cfg.source.baseline_capacitance();
    let mut v = cfg.v_rest_mv;
    let mut gates: Gates = mem.resting_gates(cfg.v_rest_mv);

    let mut time_ms = Vec::with_capacity(n_steps + 1);
    let mut voltage_mv = Vec::with_capacity(n_steps + 1);
    let mut charge_nc_cm2 = Vec::with_capacity(n_steps + 1);
    let mut spike_times_ms = Vec::new();
    time_ms.push(0.0);
    voltage_mv.push(v);
    charge_nc_cm2.push(cm0 * v);

    // Capacitance and its rate at absolute time t (gated by the sonication window).
    let cm_at = |t: f64| -> (f64, f64) {
        if t >= cfg.onset_ms && t <= cfg.offset_ms {
            (cfg.source.capacitance(t), cfg.source.capacitance_rate(t))
        } else {
            (cm0, 0.0)
        }
    };

    // dv/dt and gate derivatives at (v, gates) with capacitance (cm, dcm/dt).
    let deriv = |v: f64, g: &Gates, cm: f64, dcm: f64| -> (f64, Gates) {
        let i_ionic = mem.ionic_current(g, v);
        let dv = (cfg.i_bias_ua_cm2 - i_ionic - v * dcm) / cm;
        (dv, mem.gate_rates(g, v))
    };

    let dt = cfg.dt_ms;
    for k in 0..n_steps {
        let t = k as f64 * dt;
        let v_prev = v;

        let (cm1, dcm1) = cm_at(t);
        let (cm2, dcm2) = cm_at(t + 0.5 * dt);
        let (cm4, dcm4) = cm_at(t + dt);
        let (dv1, dg1) = deriv(v, &gates, cm1, dcm1);
        let (dv2, dg2) = deriv(v + 0.5 * dt * dv1, &axpy_gates(&gates, &dg1, 0.5 * dt), cm2, dcm2);
        let (dv3, dg3) = deriv(v + 0.5 * dt * dv2, &axpy_gates(&gates, &dg2, 0.5 * dt), cm2, dcm2);
        let (dv4, dg4) = deriv(v + dt * dv3, &axpy_gates(&gates, &dg3, dt), cm4, dcm4);
        v += dt / 6.0 * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4);
        gates = rk4_combine_gates(&gates, &dg1, &dg2, &dg3, &dg4, dt);

        let t_next = t + dt;
        if v_prev < SPIKE_THRESHOLD_MV && v >= SPIKE_THRESHOLD_MV {
            spike_times_ms.push(t_next);
        }
        let (cm_next, _) = cm_at(t_next);
        time_ms.push(t_next);
        voltage_mv.push(v);
        charge_nc_cm2.push(cm_next * v);
    }

    HhTrace {
        time_ms,
        voltage_mv,
        charge_nc_cm2,
        spike_times_ms,
    }
}
