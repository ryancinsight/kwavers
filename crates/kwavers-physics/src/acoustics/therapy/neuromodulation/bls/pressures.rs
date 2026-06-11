//! Bilayer-sonophore pressure physics and quasi-static leaflet deflection
//! (Krasovitski et al. 2011; Plaksin et al. 2014; PySONIC reference constants).
//!
//! This module implements the *force balance* on the bilayer leaflet — the
//! intermolecular (attraction–repulsion), electrical, elastic-tension and
//! intramembrane-gas pressures — and solves for the leaflet deflection `Z` that
//! balances them against the instantaneous acoustic pressure. It thereby replaces
//! the kinematic `(1−cos)` deflection surrogate of [`super::capacitance`] with a
//! deflection that is a genuine function of the acoustic drive and the membrane
//! biophysics.
//!
//! # Scope (honest)
//!
//! The deflection here is the **quasi-static** (inertia-free) equilibrium:
//! `P_total(Z) = 0` solved at each acoustic phase. This is exact in the
//! low-frequency / overdamped limit. The full *transient* dynamics — the leaflet
//! Rayleigh–Plesset ODE (Plaksin Eq. 2) with its sub-MHz resonance — additionally
//! require the inertial/viscous terms and a regularisation of the `Z=0` curvature
//! singularity (`R(Z)→∞`), and are left for a dynamic solver. The *pressure
//! terms* below are the exact Plaksin/PySONIC expressions and are reused
//! unchanged by such a solver.
//!
//! # Constants (PySONIC reference implementation; Krasovitski 2011)
//!
//! | Symbol | Value | Meaning |
//! |--------|-------|---------|
//! | a | 32 nm | sonophore radius |
//! | Δ* | 1.4 nm | reference (no-charge) inter-leaflet gap |
//! | p_Δ | 1.0×10⁵ Pa | intermolecular pressure coefficient |
//! | m, n | 5.0, 3.3 | repulsion / attraction exponents |
//! | k_A | 0.24 N/m | leaflet area-compression modulus |
//! | δ₀ | 2.0 nm | leaflet thickness |
//! | ρ_L | 1075 kg/m³ | surrounding fluid density |
//! | μ_S, μ_L | 0.035, 7×10⁻⁴ Pa·s | leaflet / fluid viscosity |
//! | ε₀ε_R | 8.854×10⁻¹² · 1 | permittivity between leaflets |
//! | P₀ | 1.0×10⁵ Pa | static ambient pressure |
//! | C₀, k_H | 0.62 mol/m³, 1.613×10⁵ Pa·m³/mol | gas conc. / Henry constant |
//! | D_gl, ξ | 3.68×10⁻⁹ m²/s, 0.5 nm | gas diffusivity / boundary layer |
//! | T | 309.15 K | temperature |
//!
//! # References
//!
//! - Krasovitski, B. et al. (2011). *PNAS* 108(8), 3258-3263.
//! - Plaksin, M., Shoham, S. & Kimmel, E. (2014). *Phys. Rev. X* 4, 011004
//!   (Eqs. 2–8).
//! - Lemaire, T. et al. (2019). *J. Neural Eng.* 16, 046007 (PySONIC).

use super::super::intramembrane_cavitation::{CapacitanceSource, PhaseCycle};
use super::capacitance::bls_capacitance;
use kwavers_core::constants::fundamental::GAS_CONSTANT;
use std::f64::consts::PI;

/// Sonophore radius a [m].
pub const A_RADIUS_M: f64 = 32.0e-9;
/// Reference (no-charge) inter-leaflet gap Δ* [m].
pub const DELTA_STAR_M: f64 = 1.4e-9;
/// Intermolecular pressure coefficient p_Δ [Pa].
pub const P_DELTA_PA: f64 = 1.0e5;
/// Repulsion exponent m.
pub const M_REPULSION: f64 = 5.0;
/// Attraction exponent n.
pub const N_ATTRACTION: f64 = 3.3;
/// Leaflet area-compression modulus k_A [N/m].
pub const KA_N_M: f64 = 0.24;
/// Leaflet thickness δ₀ [m].
pub const DELTA0_M: f64 = 2.0e-9;
/// Surrounding fluid density ρ_L [kg/m³].
pub const RHO_L_KG_M3: f64 = 1075.0;
/// Leaflet dynamic viscosity μ_S [Pa·s].
pub const MU_S_PA_S: f64 = 0.035;
/// Fluid dynamic viscosity μ_L [Pa·s].
pub const MU_L_PA_S: f64 = 7.0e-4;
/// Vacuum permittivity ε₀ [F/m].
pub const EPS0_F_M: f64 = 8.854e-12;
/// Relative permittivity between the leaflets ε_R [-].
pub const EPS_R: f64 = 1.0;
/// Static ambient pressure P₀ [Pa].
pub const P0_PA: f64 = 1.0e5;
/// Membrane temperature T [K].
pub const TEMP_K: f64 = 309.15;
/// Dissolved-gas molar concentration C₀ [mol/m³].
pub const C0_MOL_M3: f64 = 0.62;
/// Henry constant k_H [Pa·m³/mol].
pub const KH_PA_M3_MOL: f64 = 1.613e5;
/// Gas diffusivity in fluid D_gl [m²/s].
pub const DGL_M2_S: f64 = 3.68e-9;
/// Gas boundary-layer thickness ξ [m].
pub const XI_M: f64 = 0.5e-9;

/// Radius of curvature of the deflected leaflet `R(Z) = (a²+Z²)/(2Z)`
/// (`±∞` at `Z=0`, the flat membrane).
#[inline]
#[must_use]
pub fn curvature_radius(z: f64) -> f64 {
    if z.abs() < 1.0e-18 {
        f64::INFINITY
    } else {
        (A_RADIUS_M * A_RADIUS_M + z * z) / (2.0 * z)
    }
}

/// Local leaflet displacement at radius `r` for a dome of central deflection `z`
/// (and curvature radius `r_curv`): `z(r) = sign(Z)·(√(R²−r²) − |R| + |Z|)`.
#[inline]
fn local_displacement(r: f64, z: f64, r_curv: f64) -> f64 {
    if !r_curv.is_finite() {
        return 0.0; // flat membrane
    }
    let rc = r_curv.abs();
    z.signum() * ((rc * rc - r * r).max(0.0).sqrt() - rc + z.abs())
}

/// Local intermolecular (attraction–repulsion) pressure at a point whose local
/// inter-leaflet half-gap displacement is `zl`, for rest gap `delta`:
/// `p_Δ·[(Δ*/(2zl+Δ))^m − (Δ*/(2zl+Δ))^n]`.
#[inline]
fn molecular_local(zl: f64, delta: f64) -> f64 {
    // Compile-time invariant: the repulsion exponent is the integer 5, so the
    // hot-path power uses `powi(5)` (≈ an order of magnitude faster than `powf`).
    const _: () = assert!(M_REPULSION == 5.0);
    let gap = 2.0 * zl + delta;
    if gap <= 0.0 {
        return P_DELTA_PA * 1.0e6; // strong repulsion as gap → 0 (steric wall)
    }
    let ratio = DELTA_STAR_M / gap;
    P_DELTA_PA * (ratio.powi(5) - ratio.powf(N_ATTRACTION))
}

/// Area-averaged intermolecular pressure `P_M(Z)` over the leaflet disc
/// (midpoint quadrature in `r`): `(2/(a²+Z²))∫₀^a r·p_M,local(z(r)) dr`.
#[must_use]
pub fn molecular_pressure(z: f64, delta: f64) -> f64 {
    if z.abs() < 1.0e-18 {
        return molecular_local(0.0, delta);
    }
    let r_curv = curvature_radius(z);
    const NQ: usize = 64;
    let mut acc = 0.0;
    for i in 0..NQ {
        let r = (i as f64 + 0.5) / NQ as f64 * A_RADIUS_M;
        let zl = local_displacement(r, z, r_curv);
        acc += r * molecular_local(zl, delta);
    }
    let integral = acc * (A_RADIUS_M / NQ as f64);
    2.0 / (A_RADIUS_M * A_RADIUS_M + z * z) * integral
}

/// Electrical (Maxwell-stress) pressure `P_elec(Z,Q_m) = −(a²/(a²+Z²))·Q_m²/(2ε₀ε_R)`.
/// `q_m` is the membrane charge density [C/m²]; the result is attractive (≤ 0).
#[inline]
#[must_use]
pub fn electrical_pressure(z: f64, qm_c_m2: f64) -> f64 {
    -(A_RADIUS_M * A_RADIUS_M / (A_RADIUS_M * A_RADIUS_M + z * z)) * qm_c_m2 * qm_c_m2
        / (2.0 * EPS0_F_M * EPS_R)
}

/// Leaflet elastic tension `T_E(Z) = k_A·(Z/a)²` [N/m].
#[inline]
#[must_use]
pub fn elastic_tension(z: f64) -> f64 {
    KA_N_M * (z / A_RADIUS_M) * (z / A_RADIUS_M)
}

/// Elastic restoring pressure `−T_E(Z)/R(Z)` (0 at the flat membrane).
#[inline]
#[must_use]
pub fn elastic_pressure(z: f64) -> f64 {
    let r = curvature_radius(z);
    if r.is_finite() {
        -elastic_tension(z) / r
    } else {
        0.0
    }
}

/// Intramembrane-gas cavity volume `V_a(Z) = πa²Δ·[1 + (Z/3Δ)(Z²/a² + 3)]` [m³].
#[inline]
#[must_use]
pub fn cavity_volume(z: f64, delta: f64) -> f64 {
    PI * A_RADIUS_M
        * A_RADIUS_M
        * delta
        * (1.0 + (z / (3.0 * delta)) * (z * z / (A_RADIUS_M * A_RADIUS_M) + 3.0))
}

/// Initial intramembrane gas content `n_g0 = P₀·V_a(0)/(R_g·T)` [mol].
#[inline]
#[must_use]
pub fn initial_gas_mol(delta: f64) -> f64 {
    P0_PA * cavity_volume(0.0, delta) / (GAS_CONSTANT * TEMP_K)
}

/// Intramembrane gas pressure `P_in = n_g·R_g·T/V_a(Z)` [Pa].
#[inline]
#[must_use]
pub fn gas_pressure(ng_mol: f64, z: f64, delta: f64) -> f64 {
    ng_mol * GAS_CONSTANT * TEMP_K / cavity_volume(z, delta)
}

/// Solve for the rest inter-leaflet gap `Δ` that statically balances the leaflet
/// at `Z=0` for resting charge `qm0` (gas at ambient): find `Δ` such that
/// `P_M(0;Δ) + P_elec(0,qm0) = 0` (since `P_gas(0)=P₀`). Bisection on
/// `Δ ∈ (0.3, Δ*]` nm.
#[must_use]
pub fn rest_gap(qm0_c_m2: f64) -> f64 {
    // Need P_M(0;Δ) = −P_elec(0,qm0) = qm0²/(2ε₀ε_R) > 0.
    let target = qm0_c_m2 * qm0_c_m2 / (2.0 * EPS0_F_M * EPS_R);
    // f(Δ) = P_M(0;Δ) − target; P_M(0;Δ) decreases as Δ increases.
    let f = |d: f64| molecular_local(0.0, d) - target;
    let (mut lo, mut hi) = (0.3e-9, DELTA_STAR_M);
    // Ensure a sign change; if not, fall back to Δ* (no-charge gap).
    if f(lo) * f(hi) > 0.0 {
        return DELTA_STAR_M;
    }
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(lo) * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Static total pressure on the leaflet (inertia- and viscosity-free):
/// `P_M + P_gas − P₀ − P_ac + P_elastic + P_elec`.
#[must_use]
pub fn static_total_pressure(z: f64, ng_mol: f64, qm_c_m2: f64, pac_pa: f64, delta: f64) -> f64 {
    molecular_pressure(z, delta) + gas_pressure(ng_mol, z, delta) - P0_PA - pac_pa
        + elastic_pressure(z)
        + electrical_pressure(z, qm_c_m2)
}

/// Quasi-static leaflet deflection `Z ≥ 0` in equilibrium with acoustic pressure
/// `pac_pa` (sign convention: `pac > 0` compresses, `pac < 0` expands), at resting
/// charge `qm0` with the gas held at its initial content.
///
/// Returns the non-negative root of [`static_total_pressure`] (the cavity cannot
/// compress below the flat state, so the result is clamped at 0). Found by
/// bisection on `Z ∈ [0, 8a]`.
#[must_use]
pub fn quasistatic_deflection(pac_pa: f64, qm0_c_m2: f64, delta: f64) -> f64 {
    let ng0 = initial_gas_mol(delta);
    let f = |z: f64| static_total_pressure(z, ng0, qm0_c_m2, pac_pa, delta);
    let f0 = f(0.0);
    if f0 <= 0.0 {
        // Net inward (compressive) — leaflet stays flat.
        return 0.0;
    }
    // f(0) > 0 (net outward): find Z where it returns to 0 (elastic + curvature
    // restoring grow with Z). Expand the bracket until a sign change is found.
    let mut hi = 0.05e-9;
    let z_max = 8.0 * A_RADIUS_M;
    while f(hi) > 0.0 && hi < z_max {
        hi *= 1.5;
    }
    if f(hi) > 0.0 {
        return z_max; // no balance within range
    }
    let mut lo = 0.0;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        if f(lo) * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Pressure-driven bilayer-sonophore capacitance source: the leaflet deflection
/// is solved from the acoustic pressure via the quasi-static force balance
/// ([`quasistatic_deflection`]) and mapped to capacitance through the exact
/// curved-dome geometry ([`super::capacitance::bls_capacitance`]).
///
/// Unlike [`super::capacitance::BilayerSonophore`] (whose deflection *amplitude* is an
/// input), here the input is the **acoustic pressure amplitude** and the
/// deflection waveform `Z(t)` is a genuine, rectified function of pressure: the
/// cavity expands during the rarefactional half-cycle (`Z>0`) and stays flat
/// (`Z=0`, capacitance `C_m0`) during compression. The steady one-cycle
/// capacitance waveform is precomputed (the balance is phase-periodic), so the
/// per-step cost is an O(1) interpolation.
///
/// **Scope:** quasi-static (inertia-free) deflection — see the module note.
#[derive(Debug, Clone)]
pub struct BilayerSonophoreQuasistatic {
    cycle: PhaseCycle,
}

impl BilayerSonophoreQuasistatic {
    /// Number of phase samples used to precompute the steady cycle.
    const N_PHASE: usize = 360;

    /// Construct from rest capacitance [µF/cm²], carrier frequency [MHz], acoustic
    /// peak pressure [Pa], and resting potential [mV] (which sets the resting
    /// charge for the electrical pressure term). The rest gap `Δ` is solved from
    /// the resting-charge balance.
    #[must_use]
    pub fn new(cm0_uf_cm2: f64, freq_mhz: f64, pressure_amp_pa: f64, v_rest_mv: f64) -> Self {
        let omega_rad_ms = 2.0 * PI * 1.0e3 * freq_mhz;
        // Resting charge density [C/m²]: C_m0 [F/m²] · V_rest [V].
        let qm0 = (cm0_uf_cm2 * 1.0e-2) * (v_rest_mv * 1.0e-3);
        let delta = rest_gap(qm0);
        let n = Self::N_PHASE;
        // C_m(phase): pac = +A·sin(phase) (compression for sin>0 ⇒ Z=0).
        let cm_cycle: Vec<f64> = (0..n)
            .map(|i| {
                let phase = (i as f64) / n as f64 * 2.0 * PI;
                let pac = pressure_amp_pa * phase.sin();
                let z = quasistatic_deflection(pac, qm0, delta);
                bls_capacitance(z, cm0_uf_cm2, A_RADIUS_M, delta)
            })
            .collect();
        Self {
            cycle: PhaseCycle::new(cm0_uf_cm2, omega_rad_ms, cm_cycle),
        }
    }
}

impl CapacitanceSource for BilayerSonophoreQuasistatic {
    #[inline]
    fn capacitance(&self, t_ms: f64) -> f64 {
        self.cycle.capacitance(t_ms)
    }
    #[inline]
    fn capacitance_rate(&self, t_ms: f64) -> f64 {
        self.cycle.capacitance_rate(t_ms)
    }
    #[inline]
    fn baseline_capacitance(&self) -> f64 {
        self.cycle.baseline_capacitance()
    }
    #[inline]
    fn carrier_omega_rad_ms(&self) -> f64 {
        self.cycle.carrier_omega_rad_ms()
    }
    #[inline]
    fn is_source_valid(&self) -> bool {
        self.cycle.is_source_valid()
    }
}
