
use crate::physics::acoustics::bubble_dynamics::bubble_state::{
    viscous_bubble_wall_stress, young_laplace_pressure,
};
use crate::core::constants::numerical::{TWO_PI};

/// Minnaert resonance frequency of a spherical gas bubble.
///
/// ```text
/// f_M = 1/(2π·R₀) · √(3γP₀/ρ)   [Hz]
/// ```
///
/// # Arguments
/// * `r0_m` – equilibrium bubble radius [m]
/// * `gamma` – polytropic exponent of the gas (1.4 for air)
/// * `p0_pa` – ambient pressure [Pa]
/// * `rho` – liquid density [kg/m³]
///
/// # Reference
/// Minnaert (1933), *Philos. Mag.* 16, 235.
#[must_use]
#[inline]
pub fn minnaert_resonance_hz(r0_m: f64, gamma: f64, p0_pa: f64, rho: f64) -> f64 {
    if !(r0_m.is_finite()
        && gamma.is_finite()
        && p0_pa.is_finite()
        && rho.is_finite()
        && r0_m > 0.0
        && gamma > 0.0
        && p0_pa > 0.0
        && rho > 0.0)
    {
        return 0.0;
    }
    1.0 / (TWO_PI * r0_m) * (3.0 * gamma * p0_pa / rho).sqrt()
}

/// Blake threshold pressure (inertial cavitation onset).
///
/// Approximate closed-form expression derived from the static equilibrium of a
/// bubble under surface tension σ and ambient pressure P₀:
/// ```text
/// P_B = P₀ + (2σ)/(3R₀) · √(3/(P₀·R₀/(2σ) + 1))   (Blake 1949)
/// ```
///
/// # Arguments
/// * `r0_m` – equilibrium bubble radius [m]
/// * `p0_pa` – ambient pressure [Pa]
/// * `sigma_n_m` – surface tension coefficient [N/m] (water ≈ 0.0725)
///
/// # Reference
/// Blake (1949) *Appendix to: Cavitation*, HMSO Report.
#[must_use]
#[inline]
pub fn blake_threshold_pa(r0_m: f64, p0_pa: f64, sigma_n_m: f64) -> f64 {
    if !(r0_m.is_finite()
        && p0_pa.is_finite()
        && sigma_n_m.is_finite()
        && r0_m > 0.0
        && p0_pa > 0.0
        && sigma_n_m >= 0.0)
    {
        return 0.0;
    }
    let ratio = sigma_n_m / (r0_m * p0_pa);
    let inner = (1.0 + 2.0 * ratio / 3.0) * (1.0 + 1.0 / (1.5 * ratio + 1.0)).sqrt();
    p0_pa * inner
}

/// Rayleigh collapse time for a spherical cavity.
///
/// ```text
/// t_c = 0.9147 · R_max · √(ρ/P_∞)   [s]
/// ```
///
/// Coefficient = B(5/6, 1/2)·√(3/2)/3 where B is the beta function;
/// evaluated as Γ(5/6)Γ(1/2)/Γ(4/3) ≈ 2.241, giving 2.241·√(3/2)/3 ≈ 0.9147.
///
/// # Arguments
/// * `rmax_m` – maximum bubble radius [m]
/// * `p_inf_pa` – driving pressure at infinity (usually ambient) [Pa]
/// * `rho` – liquid density [kg/m³]
///
/// # Reference
/// Rayleigh (1917), *Philos. Mag.* 34, 94.
#[must_use]
#[inline]
pub fn rayleigh_collapse_time_s(rmax_m: f64, p_inf_pa: f64, rho: f64) -> f64 {
    if !(rmax_m.is_finite()
        && p_inf_pa.is_finite()
        && rho.is_finite()
        && rmax_m > 0.0
        && p_inf_pa > 0.0
        && rho > 0.0)
    {
        return 0.0;
    }
    0.9147 * rmax_m * (rho / p_inf_pa).sqrt()
}

/// Integrate the Rayleigh–Plesset equation using RK4.
///
/// The RP equation in state form, with state `y = [R, Ṙ]`:
/// ```text
/// R·R̈ + (3/2)·Ṙ² = [P_gas(R) − P₀ − P_ac(t) − 2σ/R − 4μ·Ṙ/R] / ρ
/// P_gas(R) = (P₀ + 2σ/R₀ − P_v)·(R₀/R)^(3κ) + P_v       [Brennen 1995 §2.4]
/// ```
/// where the non-condensable gas partial pressure (P₀ + 2σ/R₀ − P_v)
/// undergoes polytropic compression and P_v is added back as the
/// isothermal vapor-pressure floor.
///
/// # Arguments
/// * `r0_m` – equilibrium radius [m]
/// * `rdot0` – initial radial velocity [m/s]
/// * `p_ac_pa` – acoustic pressure amplitude [Pa]
/// * `freq_hz` – acoustic driving frequency [Hz]
/// * `t_arr` – time array (must be uniformly spaced) [s]
/// * `p0_pa` – ambient pressure [Pa]
/// * `rho` – liquid density [kg/m³]
/// * `sigma` – surface tension [N/m]
/// * `mu` – dynamic viscosity [Pa·s]
/// * `kappa` – polytropic exponent
/// * `p_v_pa` – vapour pressure [Pa]
///
/// Returns `(R_arr [m], Ṙ_arr [m/s])`.
///
/// # Reference
/// Plesset & Prosperetti (1977), *Annu. Rev. Fluid Mech.* 9, 145.
#[must_use]
pub fn rayleigh_plesset_rk4(
    r0_m: f64,
    rdot0: f64,
    p_ac_pa: f64,
    freq_hz: f64,
    t_arr: &[f64],
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = t_arr.len();
    let mut r_out = vec![0.0_f64; n];
    let mut rdot_out = vec![0.0_f64; n];

    r_out[0] = r0_m;
    rdot_out[0] = rdot0;

    let omega = TWO_PI * freq_hz;
    // Canonical polytropic non-condensable-gas closure with vapor-pressure
    // separation (Brennen 1995 sec 2.4):
    //   p_gas(r) = (p0 + 2 sigma / r0 - p_v) * (r0 / r)^(3 kappa) + p_v
    // Only the non-condensable gas partial pressure (p0 + 2 sigma / r0 - p_v)
    // undergoes polytropic compression; the saturated vapor pressure p_v is
    // isothermal and re-added after scaling. Prior to 2026-05-21 this used
    //   p_gas = (p0 + 2 sigma / r0) * (r0 / r)^(3 kappa) - p_v
    // which gives p_gas -> -p_v (negative) as r -> infinity instead of the
    // physically correct +p_v vapor-pressure floor, and double-counts the
    // vapor term in the equilibrium balance at r = r0. The canonical form
    // is already used in physics/.../keller_miksis/equation.rs; this aligns
    // the analytical Rayleigh-Plesset integrator with it.
    let p_eq = p0_pa + young_laplace_pressure(sigma, r0_m) - p_v_pa;

    let rhs = |r: f64, rdot: f64, t: f64| -> (f64, f64) {
        let r_clamped = r.max(1e-15);
        let p_gas = p_eq * (r0_m / r_clamped).powf(3.0 * kappa) + p_v_pa;
        let p_ac = p_ac_pa * (omega * t).sin();
        let rddot = (p_gas
            - p0_pa
            - p_ac
            - young_laplace_pressure(sigma, r_clamped)
            - viscous_bubble_wall_stress(mu, rdot, r_clamped))
            / (rho * r_clamped)
            - 1.5 * rdot * rdot / r_clamped;
        (rdot, rddot)
    };

    for i in 1..n {
        let dt = t_arr[i] - t_arr[i - 1];
        let t = t_arr[i - 1];
        let r = r_out[i - 1];
        let v = rdot_out[i - 1];

        let (k1r, k1v) = rhs(r, v, t);
        let (k2r, k2v) = rhs(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v, t + 0.5 * dt);
        let (k3r, k3v) = rhs(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v, t + 0.5 * dt);
        let (k4r, k4v) = rhs(r + dt * k3r, v + dt * k3v, t + dt);

        r_out[i] = r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r);
        rdot_out[i] = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
        r_out[i] = r_out[i].max(1e-15);
    }

    (r_out, rdot_out)
}

/// Integrate the Keller–Miksis equation (compressible-liquid correction) using RK4.
///
/// ```text
/// (1 − Ṙ/c_L)·R·R̈ + (3/2)·Ṙ²·(1 − Ṙ/(3c_L))
///   = (1 + Ṙ/c_L)/ρ · (P_L(R) − P_∞(t + R/c_L))
///     + R/(ρ·c_L) · d/dt[P_L(R) − P_∞(t + R/c_L)]
/// ```
/// where `P_L(R) = P_gas(R) − 2σ/R − 4μṘ/R` is the *full* liquid-side
/// wall pressure and `P_∞ = P₀ + P_ac(t + R/c_L)` is the far-field
/// pressure at retarded time.  The (1 + Ṙ/c_L) compressibility prefactor
/// applies to the entire `P_L − P_∞` — including the surface-tension
/// and viscous contributions — not just the gas / acoustic parts.
/// (Keller & Miksis 1980 Eq 2.3; Brennen 2014 Eq 4.5.)
///
/// # Arguments
/// Same as [`rayleigh_plesset_rk4`] with the addition of:
/// * `c_liquid` – sound speed in the liquid [m/s]
///
/// # Reference
/// Keller & Miksis (1980), *J. Acoust. Soc. Am.* 68, 628.
#[must_use]
pub fn keller_miksis_rk4(
    r0_m: f64,
    rdot0: f64,
    p_ac_pa: f64,
    freq_hz: f64,
    t_arr: &[f64],
    p0_pa: f64,
    rho: f64,
    sigma: f64,
    mu: f64,
    kappa: f64,
    p_v_pa: f64,
    c_liquid: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = t_arr.len();
    let mut r_out = vec![0.0_f64; n];
    let mut rdot_out = vec![0.0_f64; n];

    r_out[0] = r0_m;
    rdot_out[0] = rdot0;

    let omega = TWO_PI * freq_hz;
    // Canonical polytropic gas closure with vapor-pressure separation
    // (Brennen 1995 sec 2.4):
    //   p_gas_total(r) = p_nc(r) + p_v,
    //   p_nc(r)        = (p0 + 2 sigma / r0 - p_v) * (r0 / r)^(3 kappa)
    // Only the non-condensable partial pressure p_nc compresses
    // polytropically; the saturated vapor pressure p_v is isothermal.
    let p_eq = p0_pa + young_laplace_pressure(sigma, r0_m) - p_v_pa;

    // Keller-Miksis equation in its canonical form (Brennen 2014 Eq 4.5;
    // Keller & Miksis 1980 Eq 2.3):
    //
    //   (1 - Rdot/c) R Rddot + (3/2)(1 - Rdot/(3c)) Rdot^2
    //     = (1 + Rdot/c)/rho * (p_L(R) - p_inf(t + R/c))
    //       + R/(rho c) * d/dt [p_L(R) - p_inf(t + R/c)]
    //
    // p_L(R) = p_gas(R) - 2 sigma / R - 4 mu Rdot / R is the liquid-side bubble
    // wall pressure, so the (1 + Rdot/c) compressibility factor must multiply
    // the *entire* p_L - p_inf — including the surface-tension and viscous
    // contributions, not just the gas-pressure / acoustic-forcing parts.
    // The d/dt term likewise differentiates the full p_L(R) - p_inf, not just
    // the gas pressure.
    //
    // Prior to 2026-05-21 the surface-tension and viscous terms were added
    // outside the (1 + Rdot/c) factor and the radiation-damping derivative
    // included only dp_gas/dt.  At low wall-Mach number Rdot/c ~ 0.01 (typical
    // tissue-bubble drive) the error was sub-percent; near collapse where
    // Rdot/c → 0.1+, the error grew to multiple percent and biased the
    // radiation-damping balance.
    let rhs = |r: f64, rdot: f64, t: f64| -> (f64, f64) {
        let r_c = r.max(1e-15);
        // Non-condensable partial pressure and full internal pressure.
        let p_nc = p_eq * (r0_m / r_c).powf(3.0 * kappa);
        let p_gas = p_nc + p_v_pa;
        let t_ret = t + r_c / c_liquid;
        let p_ac_ret = p_ac_pa * (omega * t_ret).sin();

        // Liquid-side bubble wall pressure and far-field pressure.
        let p_wall = p_gas
            - young_laplace_pressure(sigma, r_c)
            - viscous_bubble_wall_stress(mu, rdot, r_c);
        let p_inf = p0_pa + p_ac_ret;

        // Time derivatives at the wall.  Polytropic non-condensable gas:
        //   dp_nc/dt = -3 kappa p_nc * Rdot / R   (p_v is constant in time)
        // Young-Laplace surface tension term:
        //   d(-2 sigma/R)/dt = 2 sigma Rdot / R^2
        // Viscous wall stress term (first-order, neglecting R-ddot back-reaction):
        //   d(-4 mu Rdot/R)/dt ≈ 4 mu Rdot^2 / R^2
        // Retarded far-field acoustic pressure derivative:
        //   dp_inf/dt = p_ac' (t_ret) * (1 + Rdot/c)
        //             = p_ac * omega * cos(omega t_ret) * (1 + Rdot/c)
        let dp_nc_dt = -3.0 * kappa * p_nc * rdot / r_c;
        let dp_surface_dt = young_laplace_pressure(sigma, r_c) * rdot / r_c;
        let dp_viscous_dt = 4.0 * mu * rdot * rdot / (r_c * r_c);
        let dp_wall_dt = dp_nc_dt + dp_surface_dt + dp_viscous_dt;
        let dp_inf_dt = p_ac_pa * omega * (omega * t_ret).cos() * (1.0 + rdot / c_liquid);

        let rdot_cl = rdot / c_liquid;
        let lhs_coeff = (1.0 - rdot_cl) * r_c;
        if lhs_coeff.abs() < 1e-20 {
            return (rdot, 0.0);
        }

        let pressure_term = (1.0 + rdot_cl) * (p_wall - p_inf) / rho;
        let radiation_term = r_c / (rho * c_liquid) * (dp_wall_dt - dp_inf_dt);
        let nonlinear_term = 1.5 * (1.0 - rdot_cl / 3.0) * rdot * rdot;

        let rddot = (pressure_term + radiation_term - nonlinear_term) / lhs_coeff;
        (rdot, rddot)
    };

    for i in 1..n {
        let dt = t_arr[i] - t_arr[i - 1];
        let t = t_arr[i - 1];
        let r = r_out[i - 1];
        let v = rdot_out[i - 1];

        let (k1r, k1v) = rhs(r, v, t);
        let (k2r, k2v) = rhs(r + 0.5 * dt * k1r, v + 0.5 * dt * k1v, t + 0.5 * dt);
        let (k3r, k3v) = rhs(r + 0.5 * dt * k2r, v + 0.5 * dt * k2v, t + 0.5 * dt);
        let (k4r, k4v) = rhs(r + dt * k3r, v + dt * k3v, t + dt);

        r_out[i] = (r + dt / 6.0 * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)).max(1e-15);
        rdot_out[i] = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v);
    }

    (r_out, rdot_out)
}
