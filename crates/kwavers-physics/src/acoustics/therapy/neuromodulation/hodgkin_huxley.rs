//! Hodgkin–Huxley conductance-based point neuron (squid giant axon, 1952).
//!
//! # Model
//!
//! The membrane is a parallel RC circuit with three voltage-gated ionic
//! conductances (fast Na⁺, delayed-rectifier K⁺, leak). Per unit membrane area:
//!
//! ```text
//! C_m · dV/dt = I_ext − I_Na − I_K − I_L
//! I_Na = g_Na · m³ · h · (V − E_Na)
//! I_K  = g_K  · n⁴      · (V − E_K)
//! I_L  = g_L           · (V − E_L)
//! dx/dt = α_x(V)·(1 − x) − β_x(V)·x      for x ∈ {m, h, n}
//! ```
//!
//! This module exposes the derivative (right-hand side) so that the
//! intramembrane-cavitation coupling ([`super::nice`]) can drive the same
//! membrane with a *time-varying* capacitance `C_m(t)`; the extra
//! charge-redistribution (displacement) current `−V·dC_m/dt` is added there.
//!
//! # Units
//!
//! SI-derived electrophysiology units, matching the original publication:
//! `V` [mV], `t` [ms], `C_m` [µF/cm²], `g` [mS/cm²], `I` [µA/cm²].
//!
//! # Rate constants
//!
//! Modern absolute-potential form (resting potential ≈ −65 mV), as tabulated in
//! Dayan & Abbott (2001) §5.5 and Gerstner et al. (2014) §2.2:
//!
//! ```text
//! α_m = 0.1·(V+40)/(1 − e^{−(V+40)/10})    β_m = 4·e^{−(V+65)/18}
//! α_h = 0.07·e^{−(V+65)/20}                β_h = 1/(1 + e^{−(V+35)/10})
//! α_n = 0.01·(V+55)/(1 − e^{−(V+55)/10})   β_n = 0.125·e^{−(V+65)/80}
//! ```
//!
//! The removable singularities of `α_m` (at V = −40) and `α_n` (at V = −55) are
//! evaluated by their analytic L'Hôpital limits (1.0 and 0.1 respectively).
//!
//! # References
//!
//! - Hodgkin, A.L. & Huxley, A.F. (1952). A quantitative description of membrane
//!   current and its application to conduction and excitation in nerve.
//!   *J. Physiol.* 117(4), 500-544.
//! - Dayan, P. & Abbott, L.F. (2001). *Theoretical Neuroscience*. MIT Press.
//! - Gerstner, W. et al. (2014). *Neuronal Dynamics*. Cambridge University Press.

/// Hodgkin–Huxley membrane parameters (electrophysiology units).
#[derive(Debug, Clone)]
pub struct HhParams {
    /// Specific membrane capacitance C_m [µF/cm²].
    pub cm_uf_cm2: f64,
    /// Maximal Na⁺ conductance ḡ_Na [mS/cm²].
    pub g_na_ms_cm2: f64,
    /// Maximal K⁺ conductance ḡ_K [mS/cm²].
    pub g_k_ms_cm2: f64,
    /// Leak conductance ḡ_L [mS/cm²].
    pub g_l_ms_cm2: f64,
    /// Na⁺ reversal potential E_Na [mV].
    pub e_na_mv: f64,
    /// K⁺ reversal potential E_K [mV].
    pub e_k_mv: f64,
    /// Leak reversal potential E_L [mV].
    pub e_l_mv: f64,
}

impl Default for HhParams {
    /// Canonical squid giant axon constants at 6.3 °C (Hodgkin & Huxley 1952),
    /// absolute-potential convention with E_L chosen so the resting potential is
    /// ≈ −65 mV.
    fn default() -> Self {
        Self {
            cm_uf_cm2: 1.0,
            g_na_ms_cm2: 120.0,
            g_k_ms_cm2: 36.0,
            g_l_ms_cm2: 0.3,
            e_na_mv: 50.0,
            e_k_mv: -77.0,
            e_l_mv: -54.387,
        }
    }
}

impl HhParams {
    /// Returns `true` if all conductances and capacitance are strictly positive
    /// and the ionic reversals bracket the leak reversal (E_K < E_L < E_Na).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.cm_uf_cm2 > 0.0
            && self.g_na_ms_cm2 > 0.0
            && self.g_k_ms_cm2 > 0.0
            && self.g_l_ms_cm2 > 0.0
            && self.e_k_mv < self.e_l_mv
            && self.e_l_mv < self.e_na_mv
    }
}

/// Instantaneous membrane state: voltage and three gating variables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HhState {
    /// Membrane potential V [mV].
    pub v_mv: f64,
    /// Na⁺ activation gate m ∈ [0, 1].
    pub m: f64,
    /// Na⁺ inactivation gate h ∈ [0, 1].
    pub h: f64,
    /// K⁺ activation gate n ∈ [0, 1].
    pub n: f64,
}

/// L'Hôpital-safe `x / (1 − e^{−x})`, the recurring Goldman-form factor in the
/// HH α-rates. The limit at `x → 0` is 1 (so `α = coeff` there).
#[inline]
fn exprel_inv(x: f64) -> f64 {
    if x.abs() < 1.0e-6 {
        // Series: x/(1 − e^{−x}) = 1 + x/2 + x²/12 + …
        1.0 + 0.5 * x
    } else {
        x / (1.0 - (-x).exp())
    }
}

/// Na⁺ activation opening rate α_m(V) [1/ms].
#[inline]
#[must_use]
pub fn alpha_m(v: f64) -> f64 {
    // 0.1·(V+40)/(1 − e^{−(V+40)/10}) = 1.0 · exprel_inv((V+40)/10)
    exprel_inv((v + 40.0) / 10.0)
}

/// Na⁺ activation closing rate β_m(V) [1/ms].
#[inline]
#[must_use]
pub fn beta_m(v: f64) -> f64 {
    4.0 * (-(v + 65.0) / 18.0).exp()
}

/// Na⁺ inactivation opening rate α_h(V) [1/ms].
#[inline]
#[must_use]
pub fn alpha_h(v: f64) -> f64 {
    0.07 * (-(v + 65.0) / 20.0).exp()
}

/// Na⁺ inactivation closing rate β_h(V) [1/ms].
#[inline]
#[must_use]
pub fn beta_h(v: f64) -> f64 {
    1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
}

/// K⁺ activation opening rate α_n(V) [1/ms].
#[inline]
#[must_use]
pub fn alpha_n(v: f64) -> f64 {
    // 0.01·(V+55)/(1 − e^{−(V+55)/10}) = 0.1 · exprel_inv((V+55)/10)
    0.1 * exprel_inv((v + 55.0) / 10.0)
}

/// K⁺ activation closing rate β_n(V) [1/ms].
#[inline]
#[must_use]
pub fn beta_n(v: f64) -> f64 {
    0.125 * (-(v + 65.0) / 80.0).exp()
}

/// Steady-state gating value x∞ = α/(α+β) for a rate pair.
#[inline]
fn x_inf(alpha: f64, beta: f64) -> f64 {
    let denom = alpha + beta;
    if denom > 0.0 {
        alpha / denom
    } else {
        0.0
    }
}

impl HhState {
    /// Resting state: voltage `v_rest_mv` with gates at their steady-state values
    /// for that voltage (`m∞`, `h∞`, `n∞`). This is the self-consistent quiescent
    /// initial condition for an unforced membrane.
    #[must_use]
    pub fn resting(v_rest_mv: f64) -> Self {
        let v = v_rest_mv;
        Self {
            v_mv: v,
            m: x_inf(alpha_m(v), beta_m(v)),
            h: x_inf(alpha_h(v), beta_h(v)),
            n: x_inf(alpha_n(v), beta_n(v)),
        }
    }

    /// Total ionic current density I_Na + I_K + I_L [µA/cm²] at this state.
    #[inline]
    #[must_use]
    pub fn ionic_current(&self, p: &HhParams) -> f64 {
        let i_na = p.g_na_ms_cm2 * self.m.powi(3) * self.h * (self.v_mv - p.e_na_mv);
        let i_k = p.g_k_ms_cm2 * self.n.powi(4) * (self.v_mv - p.e_k_mv);
        let i_l = p.g_l_ms_cm2 * (self.v_mv - p.e_l_mv);
        i_na + i_k + i_l
    }

    /// State time derivative `(dV, dm, dh, dn)` under external current density
    /// `i_ext` [µA/cm²], a (possibly time-varying) membrane capacitance
    /// `cm` [µF/cm²], and its rate of change `dcm_dt` [µF/cm²/ms].
    ///
    /// The voltage equation includes the charge-redistribution (displacement)
    /// current that a changing capacitance injects: starting from the conserved
    /// membrane charge `Q = C_m·V`,
    /// ```text
    /// dQ/dt = I_ext − I_ionic
    /// ⇒ C_m·dV/dt + V·dC_m/dt = I_ext − I_ionic
    /// ⇒ dV/dt = (I_ext − I_ionic − V·dC_m/dt) / C_m
    /// ```
    /// With `dcm_dt = 0` and `cm = p.cm_uf_cm2` this reduces to the standard HH
    /// membrane equation.
    #[must_use]
    pub fn deriv(&self, p: &HhParams, i_ext: f64, cm: f64, dcm_dt: f64) -> (f64, f64, f64, f64) {
        let v = self.v_mv;
        let i_ionic = self.ionic_current(p);
        let dv = (i_ext - i_ionic - v * dcm_dt) / cm;
        let dm = alpha_m(v) * (1.0 - self.m) - beta_m(v) * self.m;
        let dh = alpha_h(v) * (1.0 - self.h) - beta_h(v) * self.h;
        let dn = alpha_n(v) * (1.0 - self.n) - beta_n(v) * self.n;
        (dv, dm, dh, dn)
    }

    /// Advance one RK4 step of duration `dt` [ms] under a *constant* capacitance
    /// (`p.cm_uf_cm2`) and external current `i_ext` [µA/cm²].
    ///
    /// For the time-varying-capacitance integration used by the NICE coupling,
    /// see [`super::nice`], which calls [`Self::deriv`] directly with the
    /// instantaneous `C_m(t)` and `dC_m/dt`.
    #[must_use]
    pub fn rk4_step(&self, p: &HhParams, i_ext: f64, dt: f64) -> Self {
        let cm = p.cm_uf_cm2;
        let add = |s: &HhState, k: &(f64, f64, f64, f64), w: f64| HhState {
            v_mv: s.v_mv + w * k.0,
            m: s.m + w * k.1,
            h: s.h + w * k.2,
            n: s.n + w * k.3,
        };
        let k1 = self.deriv(p, i_ext, cm, 0.0);
        let k2 = add(self, &k1, 0.5 * dt).deriv(p, i_ext, cm, 0.0);
        let k3 = add(self, &k2, 0.5 * dt).deriv(p, i_ext, cm, 0.0);
        let k4 = add(self, &k3, dt).deriv(p, i_ext, cm, 0.0);
        HhState {
            v_mv: self.v_mv + dt / 6.0 * (k1.0 + 2.0 * k2.0 + 2.0 * k3.0 + k4.0),
            m: (self.m + dt / 6.0 * (k1.1 + 2.0 * k2.1 + 2.0 * k3.1 + k4.1)).clamp(0.0, 1.0),
            h: (self.h + dt / 6.0 * (k1.2 + 2.0 * k2.2 + 2.0 * k3.2 + k4.2)).clamp(0.0, 1.0),
            n: (self.n + dt / 6.0 * (k1.3 + 2.0 * k2.3 + 2.0 * k3.3 + k4.3)).clamp(0.0, 1.0),
        }
    }
}

impl super::membrane::Membrane for HhParams {
    /// Resting gates `[m∞, h∞, n∞, 0]` — the squid axon has no M-current, so the
    /// fourth gate is inert.
    #[inline]
    fn resting_gates(&self, v_rest_mv: f64) -> super::membrane::Gates {
        let v = v_rest_mv;
        [
            x_inf(alpha_m(v), beta_m(v)),
            x_inf(alpha_h(v), beta_h(v)),
            x_inf(alpha_n(v), beta_n(v)),
            0.0,
        ]
    }

    #[inline]
    fn ionic_current(&self, g: &super::membrane::Gates, v_mv: f64) -> f64 {
        let i_na = self.g_na_ms_cm2 * g[0].powi(3) * g[1] * (v_mv - self.e_na_mv);
        let i_k = self.g_k_ms_cm2 * g[2].powi(4) * (v_mv - self.e_k_mv);
        let i_l = self.g_l_ms_cm2 * (v_mv - self.e_l_mv);
        i_na + i_k + i_l
    }

    #[inline]
    fn gate_rates(&self, g: &super::membrane::Gates, v_mv: f64) -> super::membrane::Gates {
        [
            alpha_m(v_mv) * (1.0 - g[0]) - beta_m(v_mv) * g[0],
            alpha_h(v_mv) * (1.0 - g[1]) - beta_h(v_mv) * g[1],
            alpha_n(v_mv) * (1.0 - g[2]) - beta_n(v_mv) * g[2],
            0.0,
        ]
    }

    #[inline]
    fn cm0_uf_cm2(&self) -> f64 {
        self.cm_uf_cm2
    }

    #[inline]
    fn is_membrane_valid(&self) -> bool {
        self.is_valid()
    }
}

/// Result of a Hodgkin–Huxley simulation run.
#[derive(Debug, Clone)]
pub struct HhTrace {
    /// Time samples [ms].
    pub time_ms: Vec<f64>,
    /// Membrane potential samples [mV].
    pub voltage_mv: Vec<f64>,
    /// Membrane charge density samples `Q = C_m·V` [nC/cm²]
    /// (µF/cm²·mV = nC/cm²). For the NICE coupling this exposes the
    /// charge-accumulation that drives the post-stimulus depolarisation.
    pub charge_nc_cm2: Vec<f64>,
    /// Spike times [ms], detected by upward crossing of `spike_threshold_mv`.
    pub spike_times_ms: Vec<f64>,
}

impl HhTrace {
    /// Number of action potentials.
    #[inline]
    #[must_use]
    pub fn spike_count(&self) -> usize {
        self.spike_times_ms.len()
    }

    /// Mean firing rate [Hz] over the trace duration.
    #[must_use]
    pub fn mean_firing_rate_hz(&self) -> f64 {
        match (self.time_ms.first(), self.time_ms.last()) {
            (Some(&t0), Some(&t1)) if t1 > t0 => {
                self.spike_times_ms.len() as f64 / ((t1 - t0) * 1.0e-3)
            }
            _ => 0.0,
        }
    }
}

/// Spike-detection threshold [mV]: an action potential is counted on each upward
/// crossing of this level. 0 mV sits well above sub-threshold fluctuations and
/// below the ≈ +40 mV overshoot, so each AP is counted exactly once.
pub const SPIKE_THRESHOLD_MV: f64 = 0.0;

/// Simulate the Hodgkin–Huxley neuron under a time-dependent external current
/// `i_ext(t_ms)` [µA/cm²] with fixed step `dt_ms` over `[0, t_end_ms]`.
///
/// Returns the voltage trace and detected spike times. Spikes are upward
/// crossings of [`SPIKE_THRESHOLD_MV`].
///
/// # Examples
///
/// ```
/// use kwavers_physics::acoustics::therapy::neuromodulation::{simulate_hh, HhParams};
/// let p = HhParams::default();
/// // A sub-rheobase current (1 µA/cm²) evokes no action potentials.
/// assert_eq!(simulate_hh(&p, -65.0, |_| 1.0, 0.01, 20.0).spike_count(), 0);
/// // A supra-rheobase step (15 µA/cm²) drives repetitive firing.
/// assert!(simulate_hh(&p, -65.0, |_| 15.0, 0.01, 50.0).spike_count() >= 1);
/// ```
///
/// # Panics (debug)
///
/// Panics in debug builds if `p.is_valid()` is false or `dt_ms <= 0`.
#[must_use]
pub fn simulate_hh(
    p: &HhParams,
    v_rest_mv: f64,
    i_ext: impl Fn(f64) -> f64,
    dt_ms: f64,
    t_end_ms: f64,
) -> HhTrace {
    debug_assert!(p.is_valid(), "HhParams failed validity check");
    debug_assert!(dt_ms > 0.0, "dt_ms must be positive");
    let n_steps = ((t_end_ms / dt_ms).ceil() as usize).max(1);
    let cm = p.cm_uf_cm2;
    let mut state = HhState::resting(v_rest_mv);
    let mut time_ms = Vec::with_capacity(n_steps + 1);
    let mut voltage_mv = Vec::with_capacity(n_steps + 1);
    let mut charge_nc_cm2 = Vec::with_capacity(n_steps + 1);
    let mut spike_times_ms = Vec::new();
    time_ms.push(0.0);
    voltage_mv.push(state.v_mv);
    charge_nc_cm2.push(cm * state.v_mv);
    for k in 0..n_steps {
        let t = k as f64 * dt_ms;
        let v_prev = state.v_mv;
        state = state.rk4_step(p, i_ext(t), dt_ms);
        let t_next = t + dt_ms;
        if v_prev < SPIKE_THRESHOLD_MV && state.v_mv >= SPIKE_THRESHOLD_MV {
            spike_times_ms.push(t_next);
        }
        time_ms.push(t_next);
        voltage_mv.push(state.v_mv);
        charge_nc_cm2.push(cm * state.v_mv);
    }
    HhTrace {
        time_ms,
        voltage_mv,
        charge_nc_cm2,
        spike_times_ms,
    }
}
