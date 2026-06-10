//! SONIC: cycle-averaged (multi-scale) reduction of the NICE model
//! (Lemaire et al. 2019).
//!
//! The carrier-resolved integration ([`super::nice`]) must resolve the 0.1–1 MHz
//! acoustic carrier, which is infeasible over the seconds-to-minutes span of a
//! real neuromodulation protocol. The SONIC paradigm removes the stiffness by
//! recasting the membrane in terms of the **charge density** `Q = C_m·V_m` — a
//! slow (millisecond-scale) variable — and *cycle-averaging* the Hodgkin–Huxley
//! kinetics over one carrier period.
//!
//! # Method
//!
//! On the fast carrier timescale the charge `Q` is quasi-constant, so the
//! membrane potential tracks `V_m(t) = Q / C_m(t)` and swings with the carrier.
//! Because the gates are slow they are constant over a cycle, so the
//! cycle-averaged dynamics are:
//!
//! ```text
//! dQ/dt = I_ext − [ ḡ_Na m³h(⟨V_m⟩ − E_Na) + ḡ_K n⁴(⟨V_m⟩ − E_K)
//!                   + ḡ_L(⟨V_m⟩ − E_L) ]
//! dx/dt = ᾱ_x(Q)·(1 − x) − β̄_x(Q)·x        x ∈ {m, h, n}
//! ```
//!
//! with the cycle averages over one carrier period `T`:
//!
//! ```text
//! ⟨V_m⟩   = Q · ⟨1/C_m⟩
//! ᾱ_x(Q) = ⟨ α_x(Q / C_m(t)) ⟩,   β̄_x(Q) = ⟨ β_x(Q / C_m(t)) ⟩
//! ```
//!
//! The displacement current does not appear explicitly: it is absorbed into the
//! definition of `Q` (`dQ/dt = d(C_m V_m)/dt`). Because the ionic conductances
//! are linear in `V_m`, only the scalar `⟨1/C_m⟩` is needed for the driving
//! force, whereas the nonlinear rate functions require the full cycle average.
//! Here the averages are evaluated on the fly by sampling the bilayer-sonophore
//! capacitance cycle ([`super::bls`]); precomputing them into a `Q`-indexed
//! lookup table is the production optimisation for long protocols.
//!
//! # Validation
//!
//! For a single burst the SONIC reduction reproduces the carrier-resolved
//! [`super::nice::simulate_nice`] result (membrane hyperpolarisation during the
//! stimulus, charge accumulation, and the post-stimulus action potential) at a
//! fraction of the cost — verified by the differential test in the module's test
//! suite.
//!
//! # References
//!
//! - Lemaire, T. et al. (2019). Understanding ultrasound neuromodulation using a
//!   computationally efficient and interpretable model of intramembrane
//!   cavitation. *J. Neural Eng.* 16, 046007.
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). *Phys. Rev. X* 4, 011004.

use super::bls::{bls_capacitance, BilayerSonophore};
use super::hodgkin_huxley::{HhTrace, SPIKE_THRESHOLD_MV};
use super::intramembrane_cavitation::CapacitanceSource;
use super::membrane::{axpy_gates, rk4_combine_gates, Gates, Membrane};
use std::f64::consts::PI;

/// Configuration for a SONIC (cycle-averaged) simulation, generic over the
/// membrane model `M` ([`super::hodgkin_huxley::HhParams`] or
/// [`super::cortical::CorticalNeuron`]).
#[derive(Debug, Clone)]
pub struct SonicConfig<M: Membrane> {
    /// Membrane model (its baseline capacitance must equal the source's).
    pub membrane: M,
    /// Resting membrane potential / initial condition [mV].
    pub v_rest_mv: f64,
    /// Bilayer-sonophore source (geometry, peak deflection, carrier frequency).
    pub source: BilayerSonophore,
    /// Constant external (bias) current density [µA/cm²].
    pub i_bias_ua_cm2: f64,
    /// Slow integration step [ms] (millisecond-scale; need not resolve the carrier).
    pub dt_ms: f64,
    /// Number of samples used to average over one carrier cycle.
    pub cycle_samples: usize,
    /// Sonication onset [ms].
    pub onset_ms: f64,
    /// Sonication offset [ms].
    pub offset_ms: f64,
    /// Total simulated duration [ms].
    pub t_end_ms: f64,
}

impl<M: Membrane> SonicConfig<M> {
    /// Returns `true` if parameters are self-consistent.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.membrane.is_membrane_valid()
            && self.source.is_source_valid()
            && self.dt_ms > 0.0
            && self.t_end_ms > 0.0
            && self.cycle_samples >= 8
            && self.onset_ms <= self.offset_ms
            && (self.membrane.cm0_uf_cm2() - self.source.cm0_uf_cm2).abs()
                <= 1e-12 * self.membrane.cm0_uf_cm2().max(1.0)
    }
}

/// Precomputed per-cycle `1/C_m` samples for the sonicated (US-on) state.
struct CapacitanceCycle {
    inv_cm: Vec<f64>,
    inv_cm_mean: f64,
}

impl CapacitanceCycle {
    /// Sample `1/C_m(Z(t))` at `n` midpoints over one carrier period for the
    /// given bilayer sonophore. Independent of `Q`, so computed once.
    fn new(src: &BilayerSonophore, n: usize) -> Self {
        let period = 2.0 * PI / src.omega_rad_ms;
        let inv_cm: Vec<f64> = (0..n)
            .map(|i| {
                let t = (i as f64 + 0.5) / n as f64 * period;
                let z = 0.5 * src.deflection_amp_m * (1.0 - (src.omega_rad_ms * t).cos());
                1.0 / bls_capacitance(z, src.cm0_uf_cm2, src.radius_a_m, src.gap_delta_m)
            })
            .collect();
        let inv_cm_mean = inv_cm.iter().sum::<f64>() / n as f64;
        Self { inv_cm, inv_cm_mean }
    }
}

/// Run a SONIC (cycle-averaged) simulation, returning the membrane-potential
/// trace (`⟨V_m⟩ = Q·⟨1/C_m⟩`), the charge density, and spike times.
///
/// The cycle-averaged charge dynamics are `dQ/dt = I_ext − ⟨I_ionic⟩` (the ionic
/// current is linear in `V_m`, so `⟨I_ionic⟩ = I_ionic(Q·⟨1/C_m⟩)`), and the gate
/// derivatives are the cycle average of the nonlinear rate functions over the
/// per-cycle voltage samples `V_m,i = Q/C_m(t_i)`.
///
/// # Panics (debug)
///
/// Panics in debug builds if `cfg.is_valid()` is false.
#[must_use]
pub fn simulate_sonic<M: Membrane>(cfg: &SonicConfig<M>) -> HhTrace {
    debug_assert!(cfg.is_valid(), "SonicConfig failed validity check");
    let mem = &cfg.membrane;
    let cm0 = cfg.source.cm0_uf_cm2;
    let cycle = CapacitanceCycle::new(&cfg.source, cfg.cycle_samples);
    let n_steps = ((cfg.t_end_ms / cfg.dt_ms).ceil() as usize).max(1);
    let dt = cfg.dt_ms;

    let mut q = cm0 * cfg.v_rest_mv;
    let mut gates: Gates = mem.resting_gates(cfg.v_rest_mv);

    let mut time_ms = Vec::with_capacity(n_steps + 1);
    let mut voltage_mv = Vec::with_capacity(n_steps + 1);
    let mut charge_nc_cm2 = Vec::with_capacity(n_steps + 1);
    let mut spike_times_ms = Vec::new();
    let vmean_of = |q: f64, on: bool| q * if on { cycle.inv_cm_mean } else { 1.0 / cm0 };
    time_ms.push(0.0);
    voltage_mv.push(vmean_of(q, false));
    charge_nc_cm2.push(q);

    // (dQ, d[gates]) from cycle averages at charge `q`, gates `g`, on/off state.
    let deriv = |q: f64, g: &Gates, on: bool| -> (f64, Gates) {
        if on {
            // Ionic current is linear in V_m ⇒ ⟨I⟩ = I(Q·⟨1/C_m⟩); gate rates are
            // nonlinear ⇒ average over the per-cycle voltage samples.
            let i_ionic = mem.ionic_current(g, q * cycle.inv_cm_mean);
            let mut dg = [0.0_f64; 4];
            for &inv in &cycle.inv_cm {
                let r = mem.gate_rates(g, q * inv);
                for j in 0..4 {
                    dg[j] += r[j];
                }
            }
            let nrec = cycle.inv_cm.len() as f64;
            for v in &mut dg {
                *v /= nrec;
            }
            (cfg.i_bias_ua_cm2 - i_ionic, dg)
        } else {
            let v = q / cm0;
            (cfg.i_bias_ua_cm2 - mem.ionic_current(g, v), mem.gate_rates(g, v))
        }
    };

    for k in 0..n_steps {
        let t = k as f64 * dt;
        let on0 = t >= cfg.onset_ms && t <= cfg.offset_ms;
        let on_mid = (t + 0.5 * dt) >= cfg.onset_ms && (t + 0.5 * dt) <= cfg.offset_ms;
        let on_end = (t + dt) >= cfg.onset_ms && (t + dt) <= cfg.offset_ms;
        let v_prev = vmean_of(q, on0);

        let (dq1, dg1) = deriv(q, &gates, on0);
        let (dq2, dg2) = deriv(q + 0.5 * dt * dq1, &axpy_gates(&gates, &dg1, 0.5 * dt), on_mid);
        let (dq3, dg3) = deriv(q + 0.5 * dt * dq2, &axpy_gates(&gates, &dg2, 0.5 * dt), on_mid);
        let (dq4, dg4) = deriv(q + dt * dq3, &axpy_gates(&gates, &dg3, dt), on_end);
        q += dt / 6.0 * (dq1 + 2.0 * dq2 + 2.0 * dq3 + dq4);
        gates = rk4_combine_gates(&gates, &dg1, &dg2, &dg3, &dg4, dt);

        let t_next = t + dt;
        let v_next = vmean_of(q, on_end);
        if v_prev < SPIKE_THRESHOLD_MV && v_next >= SPIKE_THRESHOLD_MV {
            spike_times_ms.push(t_next);
        }
        time_ms.push(t_next);
        voltage_mv.push(v_next);
        charge_nc_cm2.push(q);
    }

    HhTrace {
        time_ms,
        voltage_mv,
        charge_nc_cm2,
        spike_times_ms,
    }
}
