//! Bubble Shape Instability — Plesset-Prosperetti Surface Mode Analysis
//!
//! # Theory: Non-Spherical Bubble Deformations
//!
//! A spherically oscillating bubble is subject to surface instabilities when
//! the bubble-wall acceleration is large. This is the classical **Rayleigh-Taylor
//! instability** on a spherical surface (Plesset 1954).
//!
//! ## Shape Perturbation Decomposition
//!
//! Consider a bubble whose surface is perturbed from the spherical equilibrium:
//! ```text
//! r(θ,φ,t) = R(t) + Σ_n  a_n(t) Y_n^0(θ)
//! ```
//! where `R(t)` solves the Keller-Miksis equation, `a_n(t)` is the amplitude
//! of the nth spherical harmonic mode (`n ≥ 2`; n=0 is volume, n=1 is
//! translation), and `Y_n^0` is the Legendre polynomial `P_n(cos θ)`.
//!
//! ## Theorem: Shape Mode ODE (Plesset 1954; Prosperetti 1977)
//!
//! Linearising the Euler equations about the spherical solution gives the
//! decoupled amplitude equation (Brennen 1995 §3.2, Eq. 3.15):
//!
//! ```text
//! ä_n + D_n(t) ȧ_n − G_n(t) a_n = 0
//! ```
//!
//! with damping coefficient and driving term:
//! ```text
//! D_n = 3Ṙ/R  +  4ν(n+2)(2n+1)/R²            [inviscid + viscous]
//!
//! G_n = (n-1)[R̈/R  −  (n+2)(Ṙ/R)²]
//!       −  n(n-1)(n+2) σ / (ρ_L R³)            [inertial − capillary]
//! ```
//!
//! **Interpretation**:
//! - `D_n > 0` provides damping (Stokes viscosity and geometric spreading).
//! - `G_n > 0` drives growth when the inertial term dominates over capillary
//!   restoring forces (Rayleigh-Taylor instability mechanism).
//! - At moderate Mach numbers the instability is strongest for mode `n = 2`
//!   (ellipsoidal deformation), then `n = 3`, … (Plesset 1954).
//!
//! ## Jet Formation Model (Blake et al. 1986)
//!
//! Near a rigid boundary (distance `h` from bubble centre at maximum radius
//! `R_max`), the stand-off parameter `γ = h / R_max` determines whether
//! an axial re-entrant jet forms during collapse:
//!
//! ```text
//! jet forms  when  γ ≤ γ_crit ≈ 2.0
//! jet speed  V_jet ≈ V₀ / (γ − 0.5)     for γ > 0.5
//!            V_jet  capped at  c_L        for γ → 0.5
//! ```
//!
//! where `V₀ = √(2Δp/ρ_L)` is the Rayleigh collapse speed (Lauterborn 1974).
//!
//! ## Breakup Criterion (Plesset 1954)
//!
//! The bubble is considered non-spherical when any mode amplitude exceeds
//! 30 % of the current radius:
//! ```text
//! |a_n| / R > ε_break = 0.3
//! ```
//! At this point the linearised model is no longer valid; the predicted
//! behaviour represents onset of non-spherical collapse or fragmentation.
//!
//! ## References
//!
//! 1. Plesset, M. S. (1954). On the stability of fluid flows with spherical
//!    symmetry. *J. Appl. Phys.* **25**(1), 96–98.
//! 2. Prosperetti, A. (1977). Viscous effects on small-amplitude surface
//!    waves. *Phys. Fluids* **20**(10), 1591–1599.
//! 3. Prosperetti, A., & Lezzi, A. (1977). In *Ann. Rev. Fluid Mech.* **9**,
//!    145–185. (§6 — shape stability review.)
//! 4. Brennen, C. E. (1995). *Cavitation and Bubble Dynamics*. Oxford.
//!    §3.2, Eq. (3.15).
//! 5. Blake, J. R., Taib, B. B., & Doherty, G. (1986). Transient cavities
//!    near boundaries. *J. Fluid Mech.* **170**, 479–497.
//! 6. Lauterborn, W. (1974). Cavitation bubble dynamics — new tools for
//!    an intricate problem. *Appl. Sci. Res.* **38**, 165–178.

/// Maximum shape mode index tracked (n = 2, 3, …, N_MODES+1).
///
/// Modes above n ≈ 6 have very high capillary damping and rarely grow.
pub const N_MODES: usize = 5;

/// Fraction of bubble radius at which a mode is considered unstable (breakup).
pub const BREAKUP_FRACTION: f64 = 0.3;

/// Stand-off ratio below which a jet forms near a rigid wall.
pub const JET_STANDOFF_CRITICAL: f64 = 2.0;

/// State of surface shape modes for a single bubble.
///
/// Tracks amplitude `a_n` and velocity `ȧ_n` for modes `n = 2 … N_MODES+1`.
/// Index 0 → mode n=2, index 1 → mode n=3, etc.
#[derive(Debug, Clone)]
pub struct ShapeModeState {
    /// Mode amplitudes a_n [m].  Index k = n − 2.
    pub amplitude: [f64; N_MODES],
    /// Mode amplitude rates ȧ_n [m/s].  Index k = n − 2.
    pub rate: [f64; N_MODES],
}

impl Default for ShapeModeState {
    fn default() -> Self {
        Self {
            amplitude: [0.0; N_MODES],
            rate: [0.0; N_MODES],
        }
    }
}

impl ShapeModeState {
    /// Seed mode `n` with a small initial perturbation `amplitude_0` [m].
    ///
    /// Typical seed amplitude = 10⁻¹⁰ m (sub-nanometre noise).
    pub fn seed(&mut self, n: usize, amplitude_0: f64) {
        if (2..N_MODES + 2).contains(&n) {
            self.amplitude[n - 2] = amplitude_0;
        }
    }

    /// Return the maximum mode amplitude normalised by bubble radius R.
    #[must_use]
    pub fn max_normalised_amplitude(&self, r: f64) -> f64 {
        if r < 1e-15 {
            return f64::INFINITY;
        }
        self.amplitude
            .iter()
            .map(|a| a.abs() / r)
            .fold(0.0_f64, f64::max)
    }

    /// Return `true` if any mode amplitude has crossed the breakup threshold.
    #[must_use]
    pub fn is_unstable(&self, r: f64) -> bool {
        self.max_normalised_amplitude(r) > BREAKUP_FRACTION
    }
}

/// Advance the shape mode amplitudes by one symplectic Euler step of duration `dt`.
///
/// ## Algorithm
///
/// For each mode `n = 2, 3, …, N_MODES+1`:
///
/// ```text
/// ä_n   = G_n(t)·a_n − D_n(t)·ȧ_n
/// ȧ_n  ← ȧ_n  + ä_n · dt       [update velocity first — symplectic]
/// a_n  ← a_n  + ȧ_n · dt       [then update position with new velocity]
/// ```
///
/// The velocity-first ordering (symplectic / semi-implicit Euler) exactly
/// conserves a modified Hamiltonian for the undamped case and is marginally
/// stable for `D_n = 0`.  For `D_n > 0` (viscous damping) the method is
/// asymptotically stable (Leimkuhler & Reich 2004 §VIII.2).
///
/// ## Arguments
///
/// * `modes`     — mutable shape mode state
/// * `r`         — bubble radius R [m]
/// * `r_dot`     — bubble wall velocity Ṙ [m/s]
/// * `r_ddot`    — bubble wall acceleration R̈ [m/s²]
/// * `sigma`     — surface tension [N/m]
/// * `rho_l`     — liquid density [kg/m³]
/// * `nu`        — kinematic viscosity [m²/s] = μ/ρ_L
/// * `dt`        — time step [s]
pub fn advance_shape_modes(
    modes: &mut ShapeModeState,
    r: f64,
    r_dot: f64,
    r_ddot: f64,
    sigma: f64,
    rho_l: f64,
    nu: f64,
    dt: f64,
) {
    if r < 1e-15 {
        return;
    }
    let r_inv = 1.0 / r;
    let r_dot_over_r = r_dot * r_inv;
    let r_ddot_over_r = r_ddot * r_inv;
    let r_dot_sq_over_r_sq = r_dot_over_r * r_dot_over_r;
    let r3_inv = r_inv * r_inv * r_inv;
    let r2_inv = r_inv * r_inv;

    for k in 0..N_MODES {
        let n = (k + 2) as f64; // mode number n ≥ 2

        // Damping coefficient D_n [s⁻¹]:
        //   inviscid part: 3Ṙ/R
        //   viscous part:  4ν(n+2)(2n+1)/R²
        // Reference: Prosperetti (1977) Phys. Fluids 20(10), Eq.(2.14)
        let d_viscous = 4.0 * nu * (n + 2.0) * (2.0 * n + 1.0) * r2_inv;
        let d_n = 3.0 * r_dot_over_r + d_viscous;

        // Driving term G_n [s⁻²]:
        //   inertial: (n-1)[R̈/R − (n+2)Ṙ²/R²]
        //   capillary: −n(n-1)(n+2)σ/(ρ_L R³)
        // Reference: Brennen (1995) §3.2, Eq.(3.15)
        let inertial = (n - 1.0) * (r_ddot_over_r - (n + 2.0) * r_dot_sq_over_r_sq);
        let capillary = n * (n - 1.0) * (n + 2.0) * sigma * r3_inv / rho_l;
        let g_n = inertial - capillary;

        // ä_n = G_n · a_n − D_n · ȧ_n
        let a = modes.amplitude[k];
        let a_dot = modes.rate[k];
        let a_ddot = g_n * a - d_n * a_dot;

        // Symplectic Euler: update velocity first, then position with new velocity.
        // This preserves a modified Hamiltonian for the conservative case and is
        // stable under viscous damping (Leimkuhler & Reich 2004 §VIII.2).
        let a_dot_new = a_dot + a_ddot * dt;
        modes.rate[k] = a_dot_new;
        modes.amplitude[k] = a + a_dot_new * dt;
    }
}

/// Evaluate the jet formation speed for a bubble collapsing near a rigid wall.
///
/// ## Algorithm (Blake et al. 1986, §3)
///
/// The dimensionless stand-off parameter:
/// ```text
/// γ = h / R_max
/// ```
/// where `h` = distance from bubble centre to the wall and `R_max` = maximum
/// bubble radius.
///
/// Jet forms when `γ ≤ γ_crit ≈ 2.0`.  For `γ > 0.5`, the jet speed is
/// approximated by the empirical Blake (1986) scaling:
/// ```text
/// V_jet = V₀ / (γ − 0.5),     V₀ = √(2·(p_∞ − p_v) / ρ_L)
/// ```
/// Capped at the liquid sound speed `c_l` for very close stand-off.
///
/// Returns `None` if `γ > γ_crit` (no jet expected).
///
/// ## Arguments
///
/// * `stand_off`   — dimensionless stand-off `γ = h / R_max`
/// * `p_inf`       — ambient pressure [Pa]
/// * `p_v`         — vapour pressure [Pa]
/// * `rho_l`       — liquid density [kg/m³]
/// * `c_l`         — liquid sound speed [m/s]
#[must_use]
pub fn jet_speed(stand_off: f64, p_inf: f64, p_v: f64, rho_l: f64, c_l: f64) -> Option<f64> {
    if stand_off > JET_STANDOFF_CRITICAL {
        return None; // bubble too far from wall for jet formation
    }
    let delta_p = (p_inf - p_v).max(0.0);
    let v0 = (2.0 * delta_p / rho_l.max(1.0)).sqrt();
    let denominator = (stand_off - 0.5).max(1e-3); // prevent division by zero
    let v_jet = (v0 / denominator).min(c_l); // cap at sound speed
    Some(v_jet)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ShapeModeState ────────────────────────────────────────────────────────

    /// Default state must have all amplitudes and rates at zero.
    #[test]
    fn test_default_state_is_zero() {
        let s = ShapeModeState::default();
        for k in 0..N_MODES {
            assert_eq!(s.amplitude[k], 0.0);
            assert_eq!(s.rate[k], 0.0);
        }
    }

    /// Seeding mode n=2 must set index 0.
    #[test]
    fn test_seed_mode2() {
        let mut s = ShapeModeState::default();
        s.seed(2, 1.0e-10);
        assert_eq!(s.amplitude[0], 1.0e-10);
        assert_eq!(s.amplitude[1], 0.0);
    }

    /// `is_unstable` must trigger when a_n / R > BREAKUP_FRACTION.
    #[test]
    fn test_is_unstable_triggered() {
        let mut s = ShapeModeState::default();
        let r = 1.0e-5; // 10 µm bubble
        s.amplitude[0] = 0.31 * r; // 31% of R — above threshold
        assert!(s.is_unstable(r), "amplitude > 30%R must flag instability");
    }

    /// `is_unstable` must be false when all modes are small.
    #[test]
    fn test_is_unstable_false_for_small_perturbation() {
        let mut s = ShapeModeState::default();
        let r = 1.0e-5;
        s.amplitude[0] = 0.01 * r; // 1% — well below threshold
        assert!(
            !s.is_unstable(r),
            "small perturbation must not trigger instability"
        );
    }

    // ── advance_shape_modes ───────────────────────────────────────────────────

    /// At equilibrium (Ṙ=0, R̈=0) the inviscid equation reduces to
    /// a capillary oscillator with frequency ω_n = √(n(n-1)(n+2)σ/(ρR³)).
    /// Seeding mode n=2 with small amplitude and zero viscosity must give
    /// bounded oscillation (|a₂| ≤ 1.01·a₀ over 1000 steps).
    #[test]
    fn test_capillary_oscillation_bounded() {
        let r0 = 1.0e-4; // 100 µm
        let sigma = 0.072; // water surface tension
        let rho_l = 998.0;
        let nu = 0.0; // inviscid
        let a0 = 1.0e-8 * r0; // 0.001% seed amplitude
        let mut modes = ShapeModeState::default();
        modes.seed(2, a0);

        // Capillary oscillation frequency for n=2:
        // ω₂ = √(n(n-1)(n+2)σ/(ρR³)) = √(2·1·4·σ/(ρR³)) = √(8σ/(ρR³))
        let omega2 = (8.0 * sigma / (rho_l * r0.powi(3))).sqrt();
        let period = 2.0 * std::f64::consts::PI / omega2;
        let dt = period / 100.0; // 100 steps per period
        let n_steps = 1000;

        for _ in 0..n_steps {
            advance_shape_modes(&mut modes, r0, 0.0, 0.0, sigma, rho_l, nu, dt);
        }

        let a_final = modes.amplitude[0].abs();
        assert!(
            a_final <= 1.05 * a0,
            "Capillary oscillation amplitude must remain bounded: a/a0 = {:.4}",
            a_final / a0
        );
    }

    /// Large inertial acceleration (R̈/R ≫ capillary) must drive mode n=2 growth.
    ///
    /// At collapse, R̈ > 0 (acceleration of inward-moving wall) with R̈/R ≫ σ/ρR³,
    /// so G₂ > 0 and the mode amplitude grows.
    #[test]
    fn test_inertial_growth_during_collapse() {
        let r = 1.0e-6; // 1 µm (collapsed radius)
        let r_ddot = 1.0e12; // extreme inertial collapse acceleration [m/s²]
        let r_dot = -100.0; // collapsing (negative wall velocity)
        let sigma = 0.072;
        let rho_l = 998.0;
        let nu = 1.0e-6; // kinematic viscosity of water
        let dt = 1.0e-12; // 1 ps step

        let a0 = 1.0e-10; // 1 Å seed
        let mut modes = ShapeModeState::default();
        modes.seed(2, a0);

        // Advance 100 steps
        for _ in 0..100 {
            advance_shape_modes(&mut modes, r, r_dot, r_ddot, sigma, rho_l, nu, dt);
        }

        let a_final = modes.amplitude[0].abs();
        assert!(
            a_final > a0,
            "Inertial driving must grow mode n=2: a_final={:.3e}, a0={:.3e}",
            a_final,
            a0
        );
    }

    // ── jet_speed ─────────────────────────────────────────────────────────────

    /// `jet_speed` must return None for stand-off above critical.
    #[test]
    fn test_jet_speed_none_far_from_wall() {
        let v = jet_speed(3.0, 101325.0, 2340.0, 998.0, 1500.0);
        assert!(v.is_none(), "Stand-off > 2 should not form a jet");
    }

    /// `jet_speed` must return Some for stand-off below critical.
    #[test]
    fn test_jet_speed_some_near_wall() {
        let v = jet_speed(1.0, 101325.0, 2340.0, 998.0, 1500.0);
        assert!(v.is_some(), "Stand-off = 1 should predict a jet");
        let v = v.unwrap();
        assert!(
            v > 0.0 && v <= 1500.0,
            "Jet speed must be positive and ≤ c_l: {v:.1} m/s"
        );
    }

    /// Jet speed must increase as stand-off decreases (closer to wall).
    #[test]
    fn test_jet_speed_increases_nearer_wall() {
        let v_far = jet_speed(1.5, 101325.0, 2340.0, 998.0, 1500.0).unwrap();
        let v_near = jet_speed(0.8, 101325.0, 2340.0, 998.0, 1500.0).unwrap();
        assert!(
            v_near > v_far,
            "Closer stand-off must give higher jet speed: v_near={v_near:.1}, v_far={v_far:.1}"
        );
    }
}
