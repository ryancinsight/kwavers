use std::f64::consts::PI;

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
#[inline]
pub fn minnaert_resonance_hz(r0_m: f64, gamma: f64, p0_pa: f64, rho: f64) -> f64 {
    1.0 / (2.0 * PI * r0_m) * (3.0 * gamma * p0_pa / rho).sqrt()
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
#[inline]
pub fn blake_threshold_pa(r0_m: f64, p0_pa: f64, sigma_n_m: f64) -> f64 {
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
#[inline]
pub fn rayleigh_collapse_time_s(rmax_m: f64, p_inf_pa: f64, rho: f64) -> f64 {
    0.9147 * rmax_m * (rho / p_inf_pa).sqrt()
}

/// Integrate the Rayleigh–Plesset equation using RK4.
///
/// The RP equation in state form, with state `y = [R, Ṙ]`:
/// ```text
/// R·R̈ + (3/2)·Ṙ² = [P_gas(R) − P₀ − P_ac(t) − 2σ/R − 4μ·Ṙ/R] / ρ
/// P_gas(R) = (P₀ + 2σ/R₀)·(R₀/R)^(3κ) − P_v
/// ```
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

    let omega = 2.0 * PI * freq_hz;
    let p_gas0 = p0_pa + 2.0 * sigma / r0_m;

    let rhs = |r: f64, rdot: f64, t: f64| -> (f64, f64) {
        let r_clamped = r.max(1e-15);
        let p_gas = p_gas0 * (r0_m / r_clamped).powf(3.0 * kappa) - p_v_pa;
        let p_ac = p_ac_pa * (omega * t).sin();
        let rddot = (p_gas - p0_pa - p_ac - 2.0 * sigma / r_clamped - 4.0 * mu * rdot / r_clamped)
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
///   = (1 + Ṙ/c_L)/ρ · (P_gas − P₀ − P_ac(t+R/c_L)) + Ṙ·dP_gas/dt/(ρ·c_L)
///   − 2σ/(ρ·R) − 4μ·Ṙ/(ρ·R)
/// ```
///
/// # Arguments
/// Same as [`rayleigh_plesset_rk4`] with the addition of:
/// * `c_liquid` – sound speed in the liquid [m/s]
///
/// # Reference
/// Keller & Miksis (1980), *J. Acoust. Soc. Am.* 68, 628.
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

    let omega = 2.0 * PI * freq_hz;
    let p_gas0 = p0_pa + 2.0 * sigma / r0_m;

    let rhs = |r: f64, rdot: f64, t: f64| -> (f64, f64) {
        let r_c = r.max(1e-15);
        let p_gas = p_gas0 * (r0_m / r_c).powf(3.0 * kappa) - p_v_pa;
        let t_ret = t + r_c / c_liquid;
        let p_ac_now = p_ac_pa * (omega * t).sin();
        let p_ac_ret = p_ac_pa * (omega * t_ret).sin();
        let dp_gas_dt = -3.0 * kappa * p_gas * rdot / r_c;

        let rdot_cl = rdot / c_liquid;
        let lhs_coeff = (1.0 - rdot_cl) * r_c;
        if lhs_coeff.abs() < 1e-20 {
            return (rdot, 0.0);
        }
        let rhs_val = (1.0 + rdot_cl) / rho * (p_gas - p0_pa - p_ac_ret)
            + r_c / (rho * c_liquid) * dp_gas_dt
            - 2.0 * sigma / (rho * r_c)
            - 4.0 * mu * rdot / (rho * r_c)
            - p_ac_now * rdot_cl / rho;
        let rddot = (rhs_val - 1.5 * rdot * rdot * (1.0 - rdot_cl / 3.0)) / lhs_coeff;
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
