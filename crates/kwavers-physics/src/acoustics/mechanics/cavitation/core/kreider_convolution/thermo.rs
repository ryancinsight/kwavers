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
