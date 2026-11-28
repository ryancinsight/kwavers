//! Kreider et al. (2011) reduced-form bubble gas/vapour/thermal
//! model — the Eller–Flynn convolution gain law.
//!
//! Kreider, Hamilton, Hueter & Huang (2011), *J. Acoust. Soc. Am.*
//! 130, 3511–3530. Reduced form: Keller–Miksis (or Gilmore)
//! dynamics supply the radius history; the non-condensable gas
//! content follows the zeroth-order Eller–Flynn convolution,
//! Kreider Eq. (14),
//!
//! ```text
//! n_g(t) = n_g,0 − 4√(πD) ∫₀^{s(t)} F(s′) (s − s′)^(−1/2) ds′,
//! s(t) = ∫₀ᵗ R(t′)⁴ dt′,   F = C_{g,w} − C_{g,∞},
//! C_{g,w} = H · p_i · (n_g/n)          (Henry's law at the wall)
//! ```
//!
//! with the thermal pressure from the Eq. (12) first-law energy
//! balance (work term minus wall-gradient heat loss — **not** a
//! fixed polytrope), vapour from the Eq. (15) Hertz–Knudsen
//! kinetics with the Eq. (16) diffusion trapping cap, and the
//! wall temperature `T_w` from the SCL fallback (Kreider §III.B
//! substitutes SCL where the Eq. (22) Plesset–Zwick convolution
//! is unusable; the PZ convolution is **not** implemented here).
//! `p_sat` is always evaluated at `T_w` — the KW-PHYS-002/003
//! physics — via the crate's existing Wagner model
//! ([`crate::acoustics::bubble_dynamics::thermodynamics::ThermodynamicsCalculator`]).
//!
//! The history is carried in transformed time `s` across **all**
//! cycles — nothing resets at a cycle boundary. Because `s`
//! barely advances while `R⁴ → 0`, the collapse occupies ~zero
//! transformed time: the measured cut this law produces relative
//! to a fresh-layer-per-cycle law is boundary-layer
//! **non-renewal**, not collapse outflux (the flux is signed —
//! outflux occurs whenever `C_{g,w} > C_{g,∞}` — but the gross
//! collapse outflux is ~1 % of the gross influx at the trigger
//! operating points below).
//!
//! # Validation status (Build 12 of the triplet-pulse formation
//! work, Nov 21 2026 — prototype `formation/b12_impl.py`,
//! benchmarks `formation/out/b12_bench.json`)
//!
//! 1. **Veto gate — PASS.** At the Bader et al. (2021) agarose
//!    anchor (40 MPa, 2 cycles, 1 MHz, C\* = 0.2, G = 44 kPa,
//!    T = 293 K) the law predicts a residual diameter of
//!    **2.477 µm vs the measured 2.9 ± 0.1 µm (−14.6 %)**, inside
//!    the ±20 % gate. Note plainly: the Bader Eq. (7) influx
//!    ([`super::bader_influx`]) is *closer* at this same anchor
//!    (2.655 µm, −8.4 %). This law's claim is mechanistic
//!    defensibility, not a benchmark win there.
//! 2. **Differential gate — FAIL in the triplet-pulse formation
//!    context.** Relative to Eq. (7), the law cuts the trigger
//!    influx to **0.262× (Kuji, 25 cycles)** and **0.120×
//!    (Iwasaki, 125 cycles)**. The cut is a pure cycle-number
//!    effect (it lands on the derived, *unpublished* no-renewal
//!    bound √(2/N): 0.283 / 0.126), protocol-symmetric at fixed
//!    cycle count, so it breaks the Kuji match while fixing
//!    Iwasaki. **The law is NOT adopted as the formation model's
//!    trigger law**; Eq. (7) remains there, documented as
//!    regime-violating. This component exists for reuse where
//!    its regime and behaviour fit — not as the formation
//!    model's answer.
//! 3. **Burst-regime caveat — do NOT use as a burst law.** The
//!    convolution has **no periodic steady state** at burst
//!    conditions (5.4–10.5 MPa, 1–1.25 MHz): evaluated per cycle
//!    it sits **22.6–60.3× above** the Fyrillas–Szeri /
//!    Louisnard–Gomez per-cycle influx
//!    ([`super::louisnard_gomez`]) and is still decaying at
//!    cycle 100. F–S stands for the burst.
//!
//! # Regime
//!
//! Kreider's framework is validated for single spherical
//! bubbles: lithotripsy single collapses (p_sw = 50 MPa,
//! 83.3 kHz, saturated water, 20 °C), SBSL (4.5 µm, 26.5 kHz,
//! 1.2 bar), HIFU bubbles (2 µm, 2 MHz, 2 MPa). Multi-cycle
//! trigger application (1–1.25 MHz, 20 MPa, 25–125 cycles,
//! C\* = 0.2) is an extrapolation in cycle number and
//! saturation. The zeroth-order Eller–Flynn form is known to
//! **overestimate** monotonic transport and **underestimate**
//! rectification (Kreider §IV.B).
//!
//! # Equation verification status (from the Build 12 lit pass)
//!
//! - Eq. (14) prefactor `4√(πD)`: **verified by derivation** —
//!   the fixed-R, constant-F reduction of the product-integration
//!   form implemented here reproduces the Epstein–Plesset
//!   transient uptake `8R²|F|√(πDt)` exactly (prototype relative
//!   error 1.5×10⁻¹⁵; Rust test below, < 1 %). The printed
//!   prefactor was not sighted in a clean copy — flagged.
//! - Eq. (12): the exact printed bracket was garbled in the
//!   source text layer — **UNVERIFIED**. Implemented in the
//!   mathematically equivalent first-law form in (n_tot, θ̄)
//!   variables; the work-term-minus-heat-loss structure is
//!   unambiguous.
//! - Eq. (22) (PZ convolution for `T_w`): **not implemented**;
//!   SCL fallback used, as the lit specification permits.
//! - Eqs. (15)–(17): structure per the lit record; α = 0.4 as
//!   published. `A_m = 0.8` and `f_m = 0.998` are **fitted by
//!   Kreider** to lithotripsy rebounds (§IV.A) — used as
//!   published, never re-tuned here.
//!
//! # Numerical guards (both from Kreider, not invented)
//!
//! 1. **Coupled update, never frozen-m.** Within a cycle, `F` is
//!    evaluated with the cycle-start `n_g` (the lagged-derivative
//!    scheme of Kreider §III.A, Eq. (28)); the convolution then
//!    solves for the cycle-end `n_g` exactly from the full
//!    history. Freezing `n_g` across the whole exposure while
//!    evaluating the collapse flux is the Smith & Wang blow-up
//!    mechanism and is forbidden in this law.
//! 2. **Truncation at unphysical dynamics.** Steps where the
//!    wall Mach number exceeds 1 or `R` falls below the hard
//!    core `h = R_eq/8.76` advance `s` with `F = 0` (no flux
//!    credited through the unphysical part of the collapse) and
//!    are counted in the diagnostics — never extrapolated
//!    through.
//!
//! Units: SI throughout; gas contents are **moles** inside this
//! module (Kreider's convention). Callers in the Bader family
//! count molecules — convert with Avogadro's number explicitly
//! at the interface.

use crate::acoustics::bubble_dynamics::thermodynamics::ThermodynamicsCalculator;

/// Accommodation coefficient α (Kreider §II.B.2, as published).
pub const KREIDER_ALPHA_ACCOMMODATION: f64 = 0.4;
/// Trapping-layer scale `A_m` (Kreider Eq. (17)) — **fitted by
/// Kreider** to lithotripsy rebounds (§IV.A); used as published.
pub const KREIDER_A_M_FITTED: f64 = 0.8;
/// Critical vapour fraction `f_m` below which trapping applies
/// (Kreider Eq. (16)) — **fitted by Kreider** (§IV.A); used as
/// published.
pub const KREIDER_F_M_FITTED: f64 = 0.998;
/// Thermal boundary-layer scale `A_h` (Kreider §II.C.1).
pub const KREIDER_A_H: f64 = 0.5;
/// SCL liquid-layer scale `A_T` (Kreider §II.C.2).
pub const KREIDER_A_T: f64 = 1.3;
/// Critical temperature of water [K].
pub const WATER_T_CRIT_K: f64 = 647.096;
/// Critical pressure of water [Pa].
pub const WATER_P_CRIT_PA: f64 = 22.064e6;
/// Molar mass of water vapour [kg/mol].
pub const WATER_MOLAR_MASS: f64 = 0.018015;
/// Ideal-gas constant [J/(mol·K)].
pub const R_GAS: f64 = 8.314_462_618;
/// Gas thermal conductivity [W/(m·K)] (air, ~300 K).
pub const K_GAS_W_MK: f64 = 0.026;
/// Liquid thermal conductivity [W/(m·K)] (water).
pub const K_LIQ_W_MK: f64 = 0.60;
/// Liquid thermal diffusivity [m²/s] (water).
pub const ALPHA_LIQ: f64 = 1.43e-7;
/// Gas thermal diffusivity at ~STP [m²/s] (air).
pub const ALPHA_GAS_STP: f64 = 2.2e-5;
/// Vapour-in-air diffusivity at 298 K [m²/s].
pub const D12_298: f64 = 2.6e-5;
/// Ratio of specific heats of the gas mixture.
pub const GAMMA_GAS: f64 = 1.4;
/// Molar heat capacity at constant pressure [J/(mol·K)] (air).
pub const CP_MOL: f64 = 29.1;

/// Saturation vapour pressure of water [Pa] at `t_k` [K], from
/// the crate's Wagner model (the same model family the
/// prototype's IAPWS implementation used; anchors: 6.28 kPa at
/// 37 °C, 101.3 kPa at 100 °C). At/above `T_c` returns `P_c`
/// (phase change is off there anyway, Kreider §II.B.2).
pub fn saturation_vapour_pressure(t_k: f64) -> f64 {
    if t_k >= WATER_T_CRIT_K {
        return WATER_P_CRIT_PA;
    }
    ThermodynamicsCalculator::default().vapor_pressure(t_k)
}

/// Molar latent heat of vaporisation [J/mol] at `t_k` [K]
/// (Watson form; → 0 at `T_c` — the KW-PHYS-003 clamp, evaluated
/// at the wall temperature, never the bulk gas temperature).
pub fn latent_heat_molar(t_k: f64) -> f64 {
    if t_k >= WATER_T_CRIT_K {
        return 0.0;
    }
    const L_REF: f64 = 40_660.0; // J/mol at 373.15 K
    L_REF * ((WATER_T_CRIT_K - t_k) / (WATER_T_CRIT_K - 373.15)).powf(0.38)
}

/// Eller–Flynn convolution state: segments `(s_lo, s_hi, F)` in
/// transformed time `s = ∫R⁴dt` with product integration,
/// `J = ∫ F(s′) (s_now − s′)^(−1/2) ds′` evaluated analytically
/// per segment. Old segments are compressed into s-weighted
/// blocks so the sum stays O(cycles + steps in current cycle).
#[derive(Debug, Clone)]
pub struct EllerFlynnHistory {
    diffusivity: f64,
    s_lo: Vec<f64>,
    s_hi: Vec<f64>,
    f_seg: Vec<f64>,
    /// Current transformed time `s_now` [m⁴·s].
    pub s_now: f64,
}

impl EllerFlynnHistory {
    /// New empty history for gas diffusivity `d` [m²/s].
    pub fn new(diffusivity: f64) -> Self {
        Self {
            diffusivity,
            s_lo: Vec::new(),
            s_hi: Vec::new(),
            f_seg: Vec::new(),
            s_now: 0.0,
        }
    }

    /// Append one integrator step: radius goes `r → r_next` over
    /// `dt` [s] with wall-minus-far-field concentration
    /// difference `f_val` [mol/m³] held constant over the step.
    pub fn add_step(&mut self, r: f64, r_next: f64, dt: f64, f_val: f64) {
        let mut ds = 0.5 * (r.powi(4) + r_next.powi(4)) * dt;
        if ds <= 0.0 {
            ds = r.powi(4) * dt;
        }
        self.s_lo.push(self.s_now);
        self.s_now += ds;
        self.s_hi.push(self.s_now);
        self.f_seg.push(f_val);
    }

    /// Compress all but the last `keep_last_steps` segments into
    /// s-weighted blocks of ~200 (kernel is smooth for old
    /// segments).
    pub fn compress(&mut self, keep_last_steps: usize) {
        let n = self.f_seg.len();
        if n <= keep_last_steps + 200 {
            return;
        }
        let cut = n - keep_last_steps;
        let (mut lo, mut hi, mut f) = (Vec::new(), Vec::new(), Vec::new());
        let mut j = 0;
        while j < cut {
            let k = (j + 200).min(cut);
            let mut w_sum = 0.0;
            let mut fw_sum = 0.0;
            for m in j..k {
                let w = self.s_hi[m] - self.s_lo[m];
                w_sum += w;
                fw_sum += self.f_seg[m] * w;
            }
            lo.push(self.s_lo[j]);
            hi.push(self.s_hi[k - 1]);
            f.push(fw_sum / w_sum.max(1e-300));
            j = k;
        }
        lo.extend_from_slice(&self.s_lo[cut..]);
        hi.extend_from_slice(&self.s_hi[cut..]);
        f.extend_from_slice(&self.f_seg[cut..]);
        self.s_lo = lo;
        self.s_hi = hi;
        self.f_seg = f;
    }

    /// The convolution integral `J` [mol/m³ · m²·s^½] at `s_now`.
    pub fn integral_j(&self) -> f64 {
        let mut j = 0.0;
        for m in 0..self.f_seg.len() {
            let w = 2.0
                * ((self.s_now - self.s_lo[m]).max(0.0).sqrt()
                    - (self.s_now - self.s_hi[m]).max(0.0).sqrt());
            j += self.f_seg[m] * w;
        }
        j
    }

    /// `(J_pos, J_neg)` restricted to segments starting at or
    /// after `s_from` — gross outflux/influx bookkeeping for the
    /// current cycle (diagnostic; older segments' kernel weights
    /// also evolve, so this is approximate by construction).
    pub fn integral_j_split(&self, s_from: f64) -> (f64, f64) {
        let (mut jp, mut jn) = (0.0, 0.0);
        for m in 0..self.f_seg.len() {
            if self.s_lo[m] < s_from - 1e-12 {
                continue;
            }
            let w = 2.0
                * ((self.s_now - self.s_lo[m]).max(0.0).sqrt()
                    - (self.s_now - self.s_hi[m]).max(0.0).sqrt());
            let c = self.f_seg[m] * w;
            if c > 0.0 {
                jp += c;
            } else {
                jn += c;
            }
        }
        (jp, jn)
    }

    /// Gas content [mol] given the initial content `n_g0` [mol]:
    /// `n_g = n_g0 − 4√(πD)·J` (Kreider Eq. (14)).
    pub fn gas_content(&self, n_g0: f64) -> f64 {
        n_g0 - 4.0 * (std::f64::consts::PI * self.diffusivity).sqrt() * self.integral_j()
    }
}

/// SCL-fallback wall temperature [K] (Kreider §III.B): solves
/// the interface balance gas-conduction + latent load +
/// liquid-conduction = 0 for `T_w` by fixed-point iteration on
/// `p_sat(T_w)` inside the vapour flux. `ndot_v_area` is the
/// vapour molar flux per unit area [mol/(m²·s)] from the
/// previous estimate. Returns `(T_w, delta_h, delta_T)`.
/// The Eq. (22) PZ convolution is **not** implemented.
pub fn wall_temperature_scl(
    theta_bar: f64,
    r: f64,
    r_dot: f64,
    p_i: f64,
    ndot_v_area: f64,
    t_inf: f64,
    p_0: f64,
) -> (f64, f64, f64) {
    let v = r_dot.abs().max(1.0);
    let tau_dyn = r / v;
    let alpha_g = ALPHA_GAS_STP * (p_0 / p_i.max(p_0)) * (theta_bar / 273.0);
    let delta_h = (r)
        .min(KREIDER_A_H * (alpha_g * tau_dyn).max(0.0).sqrt())
        .max(1e-10);
    let delta_t = (r)
        .min(KREIDER_A_T * (ALPHA_LIQ * tau_dyn).max(0.0).sqrt())
        .max(1e-10);
    let mut t_w = t_inf;
    for _ in 0..4 {
        let q_lat = latent_heat_molar(t_w) * ndot_v_area;
        let num = K_GAS_W_MK * theta_bar / delta_h + K_LIQ_W_MK * t_inf / delta_t - q_lat;
        let den = K_GAS_W_MK / delta_h + K_LIQ_W_MK / delta_t;
        let t_new = (num / den).clamp(t_inf - 5.0, WATER_T_CRIT_K - 1.0);
        if (t_new - t_w).abs() < 0.01 * t_w.max(1.0) {
            t_w = t_new;
            break;
        }
        t_w = t_new;
    }
    (t_w, delta_h, delta_t)
}

/// Hertz–Knudsen vapour rate [mol/s] (Kreider Eq. (15)) with
/// the Eq. (16) trapping cap on condensation. Zero when either
/// temperature reaches `T_c` (Kreider §II.B.2, following
/// Akhatov et al.).
pub fn vapour_rate_hertz_knudsen(
    r: f64,
    r_dot: f64,
    t_w: f64,
    theta_bar: f64,
    vapour_fraction: f64,
    p_i: f64,
    n_v: f64,
) -> f64 {
    if theta_bar >= WATER_T_CRIT_K || t_w >= WATER_T_CRIT_K {
        return 0.0;
    }
    let kin = KREIDER_ALPHA_ACCOMMODATION
        * (WATER_MOLAR_MASS / (2.0 * std::f64::consts::PI * R_GAS * t_w)).sqrt();
    let mut ndot = 4.0
        * std::f64::consts::PI
        * r
        * r
        * kin
        * (saturation_vapour_pressure(t_w) - vapour_fraction * p_i);
    if ndot < 0.0 {
        // Trapping cap (Eq. (16)): condensation limited by
        // diffusion past the non-condensable gas near the wall.
        let d12 = D12_298 * (t_w / 298.0).powf(1.75);
        let tau_v = r / (3.0 * r_dot.abs().max(1.0));
        let delta_m = r.min(KREIDER_A_M_FITTED * (d12 * tau_v).sqrt()).max(1e-12);
        let volume = 4.0 / 3.0 * std::f64::consts::PI * r.powi(3);
        let c_v = n_v / volume;
        let c_vsat = saturation_vapour_pressure(t_w) / (R_GAS * t_w);
        let cap = 4.0 * std::f64::consts::PI * r * r * d12 * (c_v - c_vsat).max(0.0) / delta_m;
        ndot = -(-ndot).min(cap);
    }
    ndot
}

/// Parameters for the Kreider reduced-form integrator.
#[derive(Debug, Clone)]
pub struct KreiderGainParams {
    /// Ambient pressure `p0` [Pa].
    pub ambient_pressure: f64,
    /// Far-field temperature `T_inf` [K].
    pub temperature: f64,
    /// Saturation concentration `C0` [mol/m³] at `p0`
    /// (0.822 mol/m³, Bader 2018 §2).
    pub saturation_concentration: f64,
    /// Saturation ratio `C*` = C∞/C0.
    pub saturation_ratio: f64,
    /// Gas diffusivity in the liquid `D` [m²/s].
    pub diffusivity: f64,
    /// Surface tension σ [Pa·m].
    pub surface_tension: f64,
    /// Shear modulus `G` [Pa] of the medium (carried for the
    /// caller's dynamics; the convolution itself is G-free).
    pub shear_modulus: f64,
    /// Vapour kinetics on/off (sensitivity switch).
    pub vapour: bool,
    /// Sound speed [m/s] for the Mach truncation guard.
    pub sound_speed: f64,
}

impl Default for KreiderGainParams {
    /// Bader-family tissue conditions: 37 °C, C0 = 0.822 mol/m³,
    /// σ = 0.056 Pa·m, D = 2.42e-9 m²/s (water), G = 30 kPa.
    fn default() -> Self {
        Self {
            ambient_pressure: 101_325.0,
            temperature: 310.0,
            saturation_concentration: 0.822,
            saturation_ratio: 0.2,
            diffusivity: 2.42e-9,
            surface_tension: 0.056,
            shear_modulus: 30_000.0,
            vapour: true,
            sound_speed: 1481.0,
        }
    }
}

/// Per-cycle diagnostics from the Kreider integrator.
#[derive(Debug, Clone, Default)]
pub struct KreiderCycleDiag {
    /// Net gas change this cycle [mol] (signed).
    pub dn_gas_mol: f64,
    /// Gross influx this cycle [mol] (diagnostic, approximate —
    /// see [`EllerFlynnHistory::integral_j_split`]).
    pub gross_influx_mol: f64,
    /// Gross outflux this cycle [mol] (diagnostic, approximate).
    pub gross_outflux_mol: f64,
    /// Peak wall concentration `C_{g,w}` this cycle [mol/m³] —
    /// an **upper-bound diagnostic** when the dynamics history
    /// was integrated with a decoupled (vapour-free) closure:
    /// a vapour-loaded collapse arrests earlier than the history
    /// reaches.
    pub peak_wall_concentration: f64,
    /// Vapour fraction at minimum radius.
    pub vapour_fraction_at_rmin: f64,
    /// Steps truncated by the Mach / hard-core guard.
    pub truncated_steps: usize,
    /// Total steps this cycle.
    pub total_steps: usize,
}

/// Kreider reduced-form integrator state for one bubble. Feed
/// it radius histories cycle-block by cycle-block with
/// [`KreiderGainModel::integrate_cycles`]; the Eller–Flynn
/// history, gas/vapour contents and mean gas temperature carry
/// across blocks (guard 1: coupled update, never frozen-m).
#[derive(Debug, Clone)]
pub struct KreiderGainModel {
    params: KreiderGainParams,
    history: EllerFlynnHistory,
    n_gas_initial: f64,
    n_gas: f64,
    n_vapour: f64,
    theta_bar: f64,
    choke_events: usize,
    theta_clamps: usize,
}

impl KreiderGainModel {
    /// New model with initial non-condensable content `n_g0`
    /// [mol]. Vapour is initialised at saturation at `T_inf`
    /// (Kreider lit §1.2: at 37 °C ≈ 6.3 % of the initial
    /// moles) when the params enable vapour.
    pub fn new(n_g0: f64, params: KreiderGainParams) -> Self {
        let n_vapour = if params.vapour {
            let frac = saturation_vapour_pressure(params.temperature) / params.ambient_pressure;
            n_g0 / (1.0 - frac).max(1e-6) - n_g0
        } else {
            0.0
        };
        Self {
            history: EllerFlynnHistory::new(params.diffusivity),
            n_gas_initial: n_g0,
            n_gas: n_g0,
            n_vapour,
            theta_bar: params.temperature,
            params,
            choke_events: 0,
            theta_clamps: 0,
        }
    }

    /// Current non-condensable gas content [mol].
    pub fn gas_content_mol(&self) -> f64 {
        self.n_gas
    }

    /// Current vapour content [mol].
    pub fn vapour_content_mol(&self) -> f64 {
        self.n_vapour
    }

    /// Number of times the cycle-end solve drove `n_g` negative
    /// and it was floored at zero (choke events).
    pub fn choke_events(&self) -> usize {
        self.choke_events
    }

    /// Integrate `radius`/`radius_dot`/`time` samples spanning
    /// one or more whole cycles at equilibrium radius `r_eq`
    /// [m] (the dynamics were solved at `r_eq`; re-solve and
    /// call again as `n_g` — hence `r_eq` — evolves; blocks of
    /// ≤ 5 cycles, per the Build 11 finding that larger blocks
    /// silently skip the collapse branch in scaffold harnesses).
    /// Returns one [`KreiderCycleDiag`] per cycle, splitting the
    /// samples at `points_per_cycle` boundaries. Per-cycle `dn`
    /// is an **output** (cycle-end minus cycle-start `n_g`).
    ///
    /// # Panics
    ///
    /// Panics if `radius`, `radius_dot` and `time` do not have
    /// equal length, or if their length is not at least
    /// `points_per_cycle + 1` (whole cycles are consumed from
    /// the front; trailing partial-cycle samples are ignored).
    pub fn integrate_cycles(
        &mut self,
        radius: &[f64],
        radius_dot: &[f64],
        time: &[f64],
        points_per_cycle: usize,
        r_eq: f64,
    ) -> Vec<KreiderCycleDiag> {
        let p = self.params.clone();
        let h_inv = p.saturation_concentration / p.ambient_pressure; // H [mol/m³/Pa]
        let c_inf = p.saturation_ratio * p.saturation_concentration;
        let h_core = r_eq / 8.76;
        let cv_mol = CP_MOL / GAMMA_GAS;
        let n_cycles = (radius.len() - 1) / points_per_cycle;
        let mut diags = Vec::with_capacity(n_cycles);
        for c in 0..n_cycles {
            let lo = c * points_per_cycle;
            let hi = lo + points_per_cycle;
            let s_cyc_start = self.history.s_now;
            let ng_cyc_start = self.n_gas;
            // Pass 1: thermal + vapour stepping with the LAGGED
            // (cycle-start) n_g in the wall condition (Kreider
            // Eq. (28) scheme); record (R, R_next, dt, C_gw,
            // truncated) per step.
            let mut rec: Vec<(f64, f64, f64, f64, bool)> = Vec::with_capacity(points_per_cycle);
            let mut th = self.theta_bar;
            let mut nv = self.n_vapour;
            let mut peak_cgw = 0.0f64;
            let mut f_at_rmin = 0.0f64;
            let i_rmin = (lo..=hi)
                .min_by(|&a, &b| radius[a].total_cmp(&radius[b]))
                .unwrap()
                - lo;
            let mut truncated_steps = 0usize;
            for i in 0..points_per_cycle {
                let idx = lo + i;
                let r_raw = radius[idx];
                let r_m = r_raw.max(h_core);
                let r_dot = radius_dot[idx];
                let dt = time[idx + 1] - time[idx];
                if dt <= 0.0 {
                    continue;
                }
                let volume = 4.0 / 3.0 * std::f64::consts::PI * r_m.powi(3);
                let n_tot = (self.n_gas + nv).max(1e-300);
                let p_i = n_tot * R_GAS * th / volume;
                let mach = r_dot.abs() / p.sound_speed;
                let truncated = mach > 1.0 || r_raw < h_core;
                if truncated {
                    truncated_steps += 1;
                }
                let (mut t_w, mut delta_h, _) = wall_temperature_scl(
                    th,
                    r_m,
                    r_dot,
                    p_i,
                    0.0,
                    p.temperature,
                    p.ambient_pressure,
                );
                let mut ndot_v = 0.0;
                if p.vapour {
                    let f_vap = nv / n_tot;
                    ndot_v = vapour_rate_hertz_knudsen(r_m, r_dot, t_w, th, f_vap, p_i, nv);
                    let (t_w2, dh2, _) = wall_temperature_scl(
                        th,
                        r_m,
                        r_dot,
                        p_i,
                        ndot_v / (4.0 * std::f64::consts::PI * r_m * r_m).max(1e-300),
                        p.temperature,
                        p.ambient_pressure,
                    );
                    t_w = t_w2;
                    delta_h = dh2;
                }
                // Eq. (12) first-law step: n c_v dθ/dt =
                // −p_i dV/dt − Q_wall − L_v ṅ_v.
                let dv_dt = 4.0 * std::f64::consts::PI * r_m * r_m * r_dot;
                let q_wall =
                    4.0 * std::f64::consts::PI * r_m * r_m * K_GAS_W_MK * (th - t_w) / delta_h;
                let lv = latent_heat_molar(t_w);
                let mut dth =
                    (-p_i * dv_dt - q_wall - lv * ndot_v) / (n_tot * cv_mol).max(1e-300) * dt;
                let lim = 0.25 * th;
                if dth.abs() > lim {
                    dth = lim.copysign(dth);
                    self.theta_clamps += 1;
                }
                th = (th + dth).clamp(30.0, 8000.0);
                nv = (nv + ndot_v * dt).max(0.0);
                let p_g = (self.n_gas / n_tot) * p_i;
                let c_gw = h_inv * p_g;
                if i <= i_rmin {
                    f_at_rmin = nv / (self.n_gas + nv).max(1e-300);
                }
                peak_cgw = peak_cgw.max(c_gw);
                rec.push((r_raw, radius[idx + 1], dt, c_gw, truncated));
            }
            self.theta_bar = th;
            self.n_vapour = nv;
            // Pass 2: append the recorded F history (truncated
            // steps advance s with F = 0 — guard 2), then solve
            // for the cycle-end n_g exactly from the full
            // history (guard 1).
            for (ra, rb, dt, c_gw, truncated) in &rec {
                if *truncated {
                    self.history.add_step(*ra, *rb, *dt, 0.0);
                } else {
                    self.history.add_step(*ra, *rb, *dt, c_gw - c_inf);
                }
            }
            let pref = 4.0 * (std::f64::consts::PI * p.diffusivity).sqrt();
            let (jpos, jneg) = self.history.integral_j_split(s_cyc_start);
            let mut n_g_new = self.history.gas_content(self.n_gas_initial);
            if n_g_new < 0.0 {
                self.choke_events += 1;
                n_g_new = 0.0;
            }
            let dn = n_g_new - ng_cyc_start;
            self.n_gas = n_g_new;
            self.history.compress(points_per_cycle);
            diags.push(KreiderCycleDiag {
                dn_gas_mol: dn,
                gross_influx_mol: -pref * jneg,
                gross_outflux_mol: pref * jpos,
                peak_wall_concentration: peak_cgw,
                vapour_fraction_at_rmin: f_at_rmin,
                truncated_steps,
                total_steps: rec.len(),
            });
        }
        diags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_saturation_pressure_anchors() {
        // Prototype self-test anchors (IAPWS Wagner).
        let p37 = saturation_vapour_pressure(310.15);
        assert!((p37 - 6280.0).abs() / 6280.0 < 0.02, "p_sat(37 C) = {p37}");
        let p100 = saturation_vapour_pressure(373.15);
        assert!(
            (p100 - 101_325.0).abs() / 101_325.0 < 0.02,
            "p_sat(100 C) = {p100}"
        );
        assert_eq!(saturation_vapour_pressure(WATER_T_CRIT_K), WATER_P_CRIT_PA);
    }

    #[test]
    fn test_latent_heat_vanishes_at_critical() {
        assert_eq!(latent_heat_molar(WATER_T_CRIT_K), 0.0);
        assert!(latent_heat_molar(310.0) > 0.0);
    }

    #[test]
    fn test_convolution_reduces_to_epstein_plesset() {
        // Fixed R, F = −C∞ (wall concentration zero): the
        // product-integration form must reproduce the
        // Epstein–Plesset transient uptake 8R²C∞√(πDt) — the
        // derivation that verifies the Eq. (14) prefactor
        // (prototype relative error 1.5e-15; assert < 1 %).
        let d = 2.42e-9;
        let r_fix = 40e-6;
        let c_inf = 0.2 * 0.822;
        let f = 1.0e6;
        let dt = 1.0 / (f * 200.0);
        let n_steps = 200 * 25;
        let mut hist = EllerFlynnHistory::new(d);
        for _ in 0..n_steps {
            hist.add_step(r_fix, r_fix, dt, -c_inf);
        }
        let t_tot = n_steps as f64 * dt;
        let m_model = -4.0 * (std::f64::consts::PI * d).sqrt() * hist.integral_j();
        let m_ep = 8.0 * r_fix * r_fix * c_inf * (std::f64::consts::PI * d * t_tot).sqrt();
        let rel = (m_model - m_ep).abs() / m_ep;
        assert!(
            rel < 0.01,
            "EF vs EP: model {m_model:e}, EP {m_ep:e}, rel {rel:e}"
        );
        // No-renewal limit: cumulative ratio vs the flat
        // per-cycle law is exactly √(2/N) at N = 25.
        let q0 = 8.0 * c_inf * (std::f64::consts::PI * d * 0.5 / f).sqrt() * r_fix * r_fix;
        let rn = m_model / (25.0 * q0);
        assert!(
            (rn - (2.0f64 / 25.0).sqrt()).abs() < 0.01,
            "r(25) = {rn}, bound {}",
            (2.0f64 / 25.0).sqrt()
        );
    }

    #[test]
    fn test_vapour_rate_signs_and_cap() {
        // Undersaturated wall (p_sat > f·p_i): evaporation > 0.
        let evap = vapour_rate_hertz_knudsen(10e-6, -1.0, 310.0, 310.0, 0.0, 101_325.0, 0.0);
        assert!(evap > 0.0, "evaporation rate {evap}");
        // Above T_c: phase change off.
        let off =
            vapour_rate_hertz_knudsen(10e-6, -1.0, WATER_T_CRIT_K, 310.0, 0.0, 101_325.0, 0.0);
        assert_eq!(off, 0.0);
    }

    #[test]
    fn test_integrator_quiescent_history() {
        // A bubble held at its equilibrium radius with C* = 1
        // and vapour off: F ≈ C_{g,w} − C∞ with C_{g,w} ≈ C0 at
        // p_i ≈ p0 + 2σ/R — slightly supersaturated wall, so the
        // signed flux must be a small OUTFLOW (dissolution
        // direction), and nothing may blow up over 5 cycles.
        let params = KreiderGainParams {
            vapour: false,
            saturation_ratio: 1.0,
            ..KreiderGainParams::default()
        };
        // Equilibrium content at R_eq = 5 µm (implicit-map-free
        // estimate: p_g = p0 + 2σ/R, ideal gas at 310 K).
        let r_eq = 5e-6;
        let p_g = params.ambient_pressure + 2.0 * params.surface_tension / r_eq;
        let volume = 4.0 / 3.0 * std::f64::consts::PI * r_eq.powi(3);
        let n0 = p_g * volume / (R_GAS * params.temperature);
        let mut model = KreiderGainModel::new(n0, params);
        let ppc = 200;
        let f = 1.0e6;
        let dt = 1.0 / (f * ppc as f64);
        let radius = vec![r_eq; 5 * ppc + 1];
        let radius_dot = vec![0.0; 5 * ppc + 1];
        let time: Vec<f64> = (0..=5 * ppc).map(|i| i as f64 * dt).collect();
        let diags = model.integrate_cycles(&radius, &radius_dot, &time, ppc, r_eq);
        assert_eq!(diags.len(), 5);
        let dn = model.gas_content_mol() - n0;
        assert!(
            dn < 0.0 && dn.abs() / n0 < 0.05,
            "quiescent dn/n0 = {}",
            dn / n0
        );
        assert_eq!(model.choke_events(), 0);
    }
}
