//! Shared helpers for symplectic bubble integrator tests.

use crate::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters;
use crate::physics::acoustics::bubble_dynamics::keller_miksis::KellerMiksisModel;

/// Build a minimal bubble setup for Hamiltonian conservation tests.
///
/// Inviscid (μ = 0) and effectively incompressible (c → ∞) so that K-M reduces
/// to the undamped Rayleigh-Plesset equation — which IS exactly conservative
/// (dH/dt = 0 analytically).  Both viscous and radiation damping are suppressed:
///   - μ = 0 → no viscous dissipation
///   - c = 1e12 m/s → radiation damping rate ~ ω₀²R₀/c ≈ 4.7e-5 s⁻¹, giving
///     < 0.01% energy loss over 1000 periods (vs. 99.9% loss with c = 1482 m/s)
pub(super) fn make_params(r0: f64) -> BubbleParameters {
    BubbleParameters {
        r0,
        mu_liquid: 0.0, // inviscid — no viscous energy dissipation
        c_liquid: 1e12, // effectively incompressible — suppresses radiation damping
        use_thermal_effects: false,
        use_mass_transfer: false,
        driving_frequency: 0.0, // no acoustic driving
        ..Default::default()
    }
}

pub(super) fn make_model(r0: f64) -> KellerMiksisModel {
    KellerMiksisModel::new(make_params(r0))
}

/// Compute bubble Hamiltonian H(R, Ṙ) under the isothermal polytropic model.
///
/// H = ½ ρ_L R³ Ṙ²  +  V_eff(R)
///
/// V_eff(R) = −∫_{R₀}^{R} [p_gas(R′) − p₀ − 2σ/R′] R′² dR′
/// ≈ (using polytropic p_gas = p_eq (R₀/R′)^{3γ}):
///   numerical integral via trapezoidal rule on [R₀·0.5, R·1.5]
pub(super) fn bubble_hamiltonian(
    r: f64,
    v: f64,
    params: &BubbleParameters,
    n_points: usize,
) -> f64 {
    let r0 = params.r0;
    let rho_l = params.rho_liquid;
    let p0 = params.p0;
    let sigma = params.sigma;
    let gamma = crate::physics::acoustics::bubble_dynamics::bubble_state::GasSpecies::Air.gamma();
    let p_eq = p0 + 2.0 * sigma / r0;

    // Kinetic energy: ½ ρ_L R³ Ṙ²
    let ke = 0.5 * rho_l * r * r * r * v * v;

    // Potential energy: ∫_{r_ref}^{R} p_net(R′) R′² dR′, p_net = p_gas − p₀ − 2σ/R′
    // We integrate from R₀ to R using the trapezoidal rule.
    // The integral is zero at R₀ (reference), sign follows from pressure balance.
    let r_lo = r0.min(r);
    let r_hi = r0.max(r);
    let sign = if r < r0 { -1.0 } else { 1.0 };

    let n = n_points.max(2);
    let dr = (r_hi - r_lo) / (n - 1) as f64;
    let mut pe = 0.0_f64;
    let mut prev = {
        let ri = r_lo;
        let p_gas = p_eq * (r0 / ri).powf(3.0 * gamma);
        let p_net = p_gas - p0 - 2.0 * sigma / ri;
        p_net * ri * ri
    };

    for k in 1..n {
        let ri = r_lo + k as f64 * dr;
        let p_gas = p_eq * (r0 / ri).powf(3.0 * gamma);
        let p_net = p_gas - p0 - 2.0 * sigma / ri;
        let cur = p_net * ri * ri;
        pe += 0.5 * (prev + cur) * dr;
        prev = cur;
    }

    // V_eff = -∫_{R₀}^{R} p_net R'² dR'
    // For R > R₀: p_net < 0 → pe < 0 → -sign*pe = -(+1)*pe > 0 ✓
    // For R < R₀: p_net > 0 → pe > 0 → -sign*pe = -(-1)*pe > 0 ✓
    ke - sign * pe
}
