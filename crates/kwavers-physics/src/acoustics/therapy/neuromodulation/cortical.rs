//! Pospischil et al. (2008) minimal cortical neuron — the membrane model the
//! NICE framework of Plaksin et al. (2014) actually uses.
//!
//! A single-compartment conductance neuron with a fast Na⁺ current, a
//! delayed-rectifier K⁺ current, a slow non-inactivating K⁺ (M-) current
//! responsible for spike-frequency adaptation, and a leak. Two electrophysiology
//! classes are provided as presets:
//!
//! - **Regular-spiking (RS)** — excitatory pyramidal-type neuron (the cell Plaksin
//!   2014 adapts for the NBLS model).
//! - **Fast-spiking (FS)** — inhibitory interneuron-type neuron.
//!
//! Because the two classes have different conductances and rate offsets, they
//! respond differently to the same intramembrane-cavitation drive — the basis of
//! the cell-type selectivity reported by Plaksin et al. (2016).
//!
//! # Kinetics (Pospischil et al. 2008; V in mV, rates in 1/ms)
//!
//! With `u = V − V_T` and `vtrap(x, y) = x / (exp(x/y) − 1)` (limit `y` at `x→0`):
//!
//! ```text
//! α_m = 0.32·vtrap(13 − u, 4)      β_m = 0.28·vtrap(u − 40, 5)
//! α_h = 0.128·exp(−(u − 17)/18)    β_h = 4/(1 + exp(−(u − 40)/5))
//! α_n = 0.032·vtrap(15 − u, 5)     β_n = 0.5·exp(−(u − 10)/40)
//! p∞  = 1/(1 + exp(−(V + 35)/10))  τ_p = τ_max/(3.3·e^{(V+35)/20} + e^{−(V+35)/20})
//! ```
//!
//! Currents: `I = ḡ_Na m³h(V−E_Na) + ḡ_Kd n⁴(V−E_K) + ḡ_M p(V−E_K) + ḡ_L(V−E_L)`.
//!
//! # Parameters (PySONIC reference implementation; Pospischil 2008)
//!
//! | Class | ḡ_Na | ḡ_Kd | ḡ_M  | ḡ_L   | E_L   | V_T   | τ_max | V_rest |
//! |-------|------|------|------|-------|-------|-------|-------|--------|
//! | RS    | 56.0 | 6.0  |0.075 |0.0205 |−70.3  |−56.2  |608 ms |−71.9   |
//! | FS    | 58.0 | 3.9  |0.0787|0.038  |−70.4  |−57.9  |502 ms |−71.4   |
//!
//! (conductances mS/cm², potentials mV; E_Na = 50, E_K = −90, C_m0 = 1 µF/cm²).
//!
//! # References
//!
//! - Pospischil, M. et al. (2008). *Biol. Cybern.* 99, 427-441.
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). *Phys. Rev. X* 4, 011004.
//! - Plaksin, M. et al. (2016). Cell-type-selective effects of intramembrane
//!   cavitation. *eNeuro* 3(3).

use super::membrane::{Gates, Membrane};

/// `vtrap(x, y) = x / (exp(x/y) − 1)`, with the removable `x→0` limit `y`.
#[inline]
fn vtrap(x: f64, y: f64) -> f64 {
    if (x / y).abs() < 1.0e-6 {
        y - 0.5 * x // series: y·(1 − x/2y)
    } else {
        x / ((x / y).exp() - 1.0)
    }
}

/// Steady-state gating value `x∞ = α/(α+β)`.
#[inline]
fn x_inf(a: f64, b: f64) -> f64 {
    let d = a + b;
    if d > 0.0 {
        a / d
    } else {
        0.0
    }
}

/// Pospischil cortical neuron parameters (electrophysiology units).
#[derive(Debug, Clone, Copy)]
pub struct CorticalNeuron {
    /// Specific membrane capacitance C_m0 [µF/cm²].
    pub cm0_uf_cm2: f64,
    /// Na⁺ conductance ḡ_Na [mS/cm²].
    pub g_na_ms_cm2: f64,
    /// Delayed-rectifier K⁺ conductance ḡ_Kd [mS/cm²].
    pub g_kd_ms_cm2: f64,
    /// M-current (slow K⁺) conductance ḡ_M [mS/cm²].
    pub g_m_ms_cm2: f64,
    /// Leak conductance ḡ_L [mS/cm²].
    pub g_l_ms_cm2: f64,
    /// Na⁺ reversal E_Na `mV`.
    pub e_na_mv: f64,
    /// K⁺ reversal E_K `mV` (shared by Kd and M).
    pub e_k_mv: f64,
    /// Leak reversal E_L `mV`.
    pub e_l_mv: f64,
    /// Spike-threshold adjustment V_T `mV`.
    pub v_t_mv: f64,
    /// Maximal M-current time constant τ_max `ms`.
    pub tau_max_ms: f64,
}

impl CorticalNeuron {
    /// Regular-spiking (excitatory pyramidal) preset — the NICE/NBLS neuron.
    #[must_use]
    pub fn regular_spiking() -> Self {
        Self {
            cm0_uf_cm2: 1.0,
            g_na_ms_cm2: 56.0,
            g_kd_ms_cm2: 6.0,
            g_m_ms_cm2: 0.075,
            g_l_ms_cm2: 0.0205,
            e_na_mv: 50.0,
            e_k_mv: -90.0,
            e_l_mv: -70.3,
            v_t_mv: -56.2,
            tau_max_ms: 608.0,
        }
    }

    /// Fast-spiking (inhibitory interneuron) preset.
    #[must_use]
    pub fn fast_spiking() -> Self {
        Self {
            cm0_uf_cm2: 1.0,
            g_na_ms_cm2: 58.0,
            g_kd_ms_cm2: 3.9,
            g_m_ms_cm2: 0.0787,
            g_l_ms_cm2: 0.038,
            e_na_mv: 50.0,
            e_k_mv: -90.0,
            e_l_mv: -70.4,
            v_t_mv: -57.9,
            tau_max_ms: 502.0,
        }
    }

    /// Canonical resting potential for the RS preset `mV`.
    pub const V_REST_RS_MV: f64 = -71.9;
    /// Canonical resting potential for the FS preset `mV`.
    pub const V_REST_FS_MV: f64 = -71.4;

    #[inline]
    fn alpha_m(&self, v: f64) -> f64 {
        0.32 * vtrap(13.0 - (v - self.v_t_mv), 4.0)
    }
    #[inline]
    fn beta_m(&self, v: f64) -> f64 {
        0.28 * vtrap((v - self.v_t_mv) - 40.0, 5.0)
    }
    #[inline]
    fn alpha_h(&self, v: f64) -> f64 {
        0.128 * (-((v - self.v_t_mv) - 17.0) / 18.0).exp()
    }
    #[inline]
    fn beta_h(&self, v: f64) -> f64 {
        4.0 / (1.0 + (-((v - self.v_t_mv) - 40.0) / 5.0).exp())
    }
    #[inline]
    fn alpha_n(&self, v: f64) -> f64 {
        0.032 * vtrap(15.0 - (v - self.v_t_mv), 5.0)
    }
    #[inline]
    fn beta_n(&self, v: f64) -> f64 {
        0.5 * (-((v - self.v_t_mv) - 10.0) / 40.0).exp()
    }
    #[inline]
    fn p_inf(v: f64) -> f64 {
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp())
    }
    #[inline]
    fn tau_p_ms(&self, v: f64) -> f64 {
        self.tau_max_ms / (3.3 * ((v + 35.0) / 20.0).exp() + (-(v + 35.0) / 20.0).exp())
    }
}

impl Membrane for CorticalNeuron {
    #[inline]
    fn resting_gates(&self, v_rest_mv: f64) -> Gates {
        let v = v_rest_mv;
        [
            x_inf(self.alpha_m(v), self.beta_m(v)),
            x_inf(self.alpha_h(v), self.beta_h(v)),
            x_inf(self.alpha_n(v), self.beta_n(v)),
            Self::p_inf(v),
        ]
    }

    #[inline]
    fn ionic_current(&self, g: &Gates, v_mv: f64) -> f64 {
        let i_na = self.g_na_ms_cm2 * g[0].powi(3) * g[1] * (v_mv - self.e_na_mv);
        let i_kd = self.g_kd_ms_cm2 * g[2].powi(4) * (v_mv - self.e_k_mv);
        let i_m = self.g_m_ms_cm2 * g[3] * (v_mv - self.e_k_mv);
        let i_l = self.g_l_ms_cm2 * (v_mv - self.e_l_mv);
        i_na + i_kd + i_m + i_l
    }

    #[inline]
    fn gate_rates(&self, g: &Gates, v_mv: f64) -> Gates {
        [
            self.alpha_m(v_mv) * (1.0 - g[0]) - self.beta_m(v_mv) * g[0],
            self.alpha_h(v_mv) * (1.0 - g[1]) - self.beta_h(v_mv) * g[1],
            self.alpha_n(v_mv) * (1.0 - g[2]) - self.beta_n(v_mv) * g[2],
            (Self::p_inf(v_mv) - g[3]) / self.tau_p_ms(v_mv),
        ]
    }

    #[inline]
    fn cm0_uf_cm2(&self) -> f64 {
        self.cm0_uf_cm2
    }

    #[inline]
    fn is_membrane_valid(&self) -> bool {
        self.cm0_uf_cm2 > 0.0
            && self.g_na_ms_cm2 > 0.0
            && self.g_kd_ms_cm2 > 0.0
            && self.g_m_ms_cm2 >= 0.0
            && self.g_l_ms_cm2 > 0.0
            && self.e_k_mv < self.e_l_mv
            && self.e_l_mv < self.e_na_mv
            && self.tau_max_ms > 0.0
    }
}
