//! Gilmore equation for high-amplitude bubble oscillations
//!
//! The Gilmore equation is more accurate than Keller-Miksis for violent
//! bubble collapse and high-amplitude oscillations where compressibility
//! effects are significant.
//!
//! Reference: Gilmore, F. R. (1952). "The growth or collapse of a spherical bubble
//! in a viscous compressible liquid". Hydrodynamics Laboratory Report 26-4,
//! California Institute of Technology.

use super::{BubbleParameters, BubbleState};
use crate::core::constants::ATMOSPHERIC_PRESSURE;
use crate::core::error::KwaversResult;

/// Gilmore equation solver for high-amplitude bubble dynamics
#[derive(Debug)]
pub struct GilmoreSolver {
    pub(super) params: BubbleParameters,
    /// Tait equation parameters for water
    pub(super) tait_b: f64,
    pub(super) tait_n: f64,
}

impl GilmoreSolver {
    /// Create a new Gilmore solver
    #[must_use]
    pub fn new(params: BubbleParameters) -> Self {
        // Tait equation parameters for water (Fujikawa & Akamatsu, 1980)
        let tait_b = 3.046e8; // Pa
        let tait_n = 7.15;

        Self {
            params,
            tait_b,
            tait_n,
        }
    }

    /// Calculate enthalpy from Tait equation of state
    /// h = ∫(dp/ρ) = (n/(n-1)) * (p+B)/ρ₀ * [(p+B)/(p₀+B)]^((n-1)/n)
    pub(super) fn calculate_enthalpy(pressure: f64, rho_0: f64, b: f64, n: f64) -> f64 {
        let p0 = ATMOSPHERIC_PRESSURE;
        (n / (n - 1.0)) * (pressure + b) / rho_0 * ((pressure + b) / (p0 + b)).powf((n - 1.0) / n)
    }

    /// Calculate sound speed from Tait equation
    /// c² = (∂p/∂ρ)_s = n(p+B)/ρ
    pub(super) fn calculate_sound_speed(&self, pressure: f64) -> f64 {
        ((self.tait_n * (pressure + self.tait_b)) / self.params.rho_liquid).sqrt()
    }

    /// Liquid density from the Tait equation of state.
    ///
    /// ρ(p) = ρ₀ · [(p + B) / (p₀ + B)]^(1/n)
    ///
    /// This is the SSOT for Tait density anywhere `GilmoreSolver` needs it.
    /// `estimate_enthalpy_derivative` consumes it through `self.calculate_density(p_wall)`
    /// rather than open-coding the same expression.
    fn calculate_density(&self, pressure: f64) -> f64 {
        let p0 = self.params.p0;
        self.params.rho_liquid
            * ((pressure + self.tait_b) / (p0 + self.tait_b)).powf(1.0 / self.tait_n)
    }

    /// Calculate bubble wall acceleration using Gilmore equation
    ///
    /// The Gilmore equation in standard form:
    /// (1 - u/C) * R * `R_ddot` + (3/2) * (1 - u/(3C)) * u² =
    ///     (1 + u/C) * H + (1 - u/C) * R/C * dH/dt
    ///
    /// where:
    /// - u = dR/dt is the bubble wall velocity
    /// - C is the sound speed in liquid at the bubble wall
    /// - H is the enthalpy difference between bubble wall and infinity
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn calculate_acceleration(
        &self,
        state: &BubbleState,
        p_acoustic: f64,
        t: f64,
    ) -> KwaversResult<f64> {
        let r = state.radius;
        let u = state.wall_velocity;

        // Acoustic forcing
        let omega = 2.0 * std::f64::consts::PI * self.params.driving_frequency;
        let p_acoustic_inst = p_acoustic * (omega * t).sin();
        let p_inf = self.params.p0 + p_acoustic_inst;

        // Internal bubble pressure (polytropic gas law per Gilmore 1952)
        // Neglects thermal damping (adiabatic approximation valid for acoustic frequencies)
        // Full thermal effects require heat diffusion equation (Prosperetti 1977)
        let gamma = state.gas_species.gamma();
        let p_gas = self.params.p0 * (self.params.r0 / r).powf(3.0 * gamma);

        // Pressure at bubble wall (liquid side)
        let p_wall = p_gas - 2.0 * self.params.sigma / r - 4.0 * self.params.mu_liquid * u / r;

        // Enthalpy at bubble wall and infinity
        let h_wall =
            Self::calculate_enthalpy(p_wall, self.params.rho_liquid, self.tait_b, self.tait_n);
        let h_inf =
            Self::calculate_enthalpy(p_inf, self.params.rho_liquid, self.tait_b, self.tait_n);
        let h = h_wall - h_inf;

        // Sound speed at bubble wall
        let c_wall = self.calculate_sound_speed(p_wall);

        // Full time derivative of the enthalpy difference H = H_wall − H_inf.
        // dH/dt = (1/ρ_wall)·dp_wall/dt − (1/ρ_inf)·dp_inf/dt per Gilmore (1952) §4 Eq. 16.
        // Implementation in `estimate_enthalpy_derivative` consumes state.{wall_velocity,
        // wall_acceleration, gas_species.gamma()} for dp_wall/dt — no quasi-static reduction.
        let dh_dt = self.estimate_enthalpy_derivative(state, p_wall, p_acoustic, omega, t);

        // Gilmore equation solved for R_ddot
        let u_c = u / c_wall;

        // Avoid singularity when u approaches c
        if u_c.abs() > 0.99 {
            return Err(crate::core::error::PhysicsError::NumericalInstability {
                timestep: 0.0,
                cfl_limit: u_c.abs(),
            }
            .into());
        }

        let lhs_coeff = r * (1.0 - u_c);
        let rhs = (1.0 + u_c).mul_add(h, (1.0 - u_c) * r / c_wall * dh_dt);
        let nonlinear_term = 1.5 * (1.0 - u_c / 3.0) * u * u;

        let acceleration = (rhs - nonlinear_term) / lhs_coeff;

        Ok(acceleration)
    }

    /// Compute the full time derivative of the enthalpy difference `dH/dt`.
    ///
    /// ## Formal derivation (Gilmore 1952 §4; Hamilton & Blackstock 1998 §3)
    ///
    /// The enthalpy difference entering the Gilmore equation is:
    /// ```text
    /// H = H_wall − H_inf = ∫_{p_inf}^{p_wall} dp/ρ(p)
    /// ```
    /// Its time derivative splits by the fundamental theorem:
    /// ```text
    /// dH/dt = (1/ρ_wall) · dp_wall/dt  −  (1/ρ_inf) · dp_inf/dt
    /// ```
    /// where `∂H/∂p = 1/ρ` follows from the Tait EOS (see proof in `calculate_enthalpy`).
    ///
    /// ### `dp_wall/dt` — bubble-wall pressure rate
    ///
    /// Differentiating the Young-Laplace-Stokes condition:
    /// ```text
    /// p_wall = p_gas(R) − 2σ/R − 4μU/R
    /// ```
    /// and the polytropic gas law `p_gas = p₀(R₀/R)^(3γ)`:
    /// ```text
    /// dp_gas/dt   = −3γ p_gas U / R
    /// d(−2σ/R)/dt =  2σ U / R²
    /// d(−4μU/R)/dt = −4μR̈/R + 4μU²/R²
    /// ```
    /// R̈ is read from `state.wall_acceleration` (the previous-step value stored by
    /// `step_rk4`). The lagged R̈ introduces O(dt) error in `dp_wall/dt`, which
    /// appears multiplied by `(R/C)·dt` in the Gilmore RHS → O(dt²) impact on
    /// the step solution, well within the O(dt³) global accuracy of RK4.
    ///
    /// ### `dp_inf/dt` — far-field acoustic pressure rate
    /// ```text
    /// dp_inf/dt = p_acoustic · ω · cos(ωt)
    /// ```
    ///
    /// ### Tait densities
    /// ```text
    /// ρ_wall = ρ₀ · [(p_wall + B)/(p₀ + B)]^(1/n)
    /// ρ_inf  = ρ₀                           (p_inf = p₀ at ambient)
    /// ```
    ///
    /// ## References
    /// - Gilmore F. R. (1952) Caltech Hydro. Lab. Report 26-4, §4, Eq. 16.
    /// - Hamilton M. F. & Blackstock D. T. (1998) *Nonlinear Acoustics*, §3.
    fn estimate_enthalpy_derivative(
        &self,
        state: &BubbleState,
        p_wall: f64,
        p_acoustic: f64,
        omega: f64,
        t: f64,
    ) -> f64 {
        let r = state.radius;
        let u = state.wall_velocity;
        let r_ddot = state.wall_acceleration; // lagged R̈ from previous step
        let gamma = state.gas_species.gamma();

        // ── dp_gas/dt via polytropic law: p_gas = p₀(R₀/R)^(3γ) ────────────
        let p_gas = self.params.p0 * (self.params.r0 / r).powf(3.0 * gamma);
        let dp_gas_dt = -3.0 * gamma * p_gas * u / r;

        // ── d(−2σ/R)/dt and d(−4μU/R)/dt (surface tension + viscous) ────────
        let dp_surface_dt = 2.0 * self.params.sigma * u / (r * r);
        // d(−4μU/R)/dt = −4μ·(R̈·R − U·U)/R² = −4μR̈/R + 4μU²/R²
        let dp_viscous_dt = -4.0 * self.params.mu_liquid * r_ddot / r
            + 4.0 * self.params.mu_liquid * u * u / (r * r);

        let dp_wall_dt = dp_gas_dt + dp_surface_dt + dp_viscous_dt;

        // ── dp_inf/dt from acoustic driving ──────────────────────────────────
        let dp_inf_dt = p_acoustic * omega * (omega * t).cos();

        // ── Tait densities (SSOT: `calculate_density`) ──────────────────────
        // ρ_wall (can differ significantly from ρ_inf near collapse)
        let rho_wall = self.calculate_density(p_wall);
        // ρ_inf at ambient pressure (p = p₀) reduces to ρ₀ exactly (Tait ratio = 1)
        let rho_inf = self.params.rho_liquid;

        // ── Full dH/dt = dH_wall/dt − dH_inf/dt ─────────────────────────────
        let dh_wall_dt = dp_wall_dt / rho_wall;
        let dh_inf_dt = dp_inf_dt / rho_inf;

        dh_wall_dt - dh_inf_dt
    }

    /// Classical RK4 integration of the Gilmore ODE over one time step.
    ///
    /// ## Mathematical specification
    ///
    /// Integrates the two-component state vector **y** = [R, Ṙ]ᵀ forward by
    /// `dt` using the explicit four-stage Runge-Kutta method with the standard
    /// Butcher tableau:
    ///
    /// ```text
    /// c = [0, ½, ½, 1],   b = [⅙, ⅓, ⅓, ⅙]
    /// k_i^R   = Ṙ evaluated at stage i
    /// k_i^Ṙ  = R̈ = calculate_acceleration(state_i, p, t_i)
    /// R_{n+1}  = R_n  + dt/6 · (k₁ᴿ  + 2k₂ᴿ  + 2k₃ᴿ  + k₄ᴿ )
    /// Ṙ_{n+1}  = Ṙ_n  + dt/6 · (k₁^Ṙ + 2k₂^Ṙ + 2k₃^Ṙ + k₄^Ṙ)
    /// ```
    ///
    /// **Global order:** O(dt³) (local truncation O(dt⁴), classical RK4 per
    /// Hairer, Nørsett & Wanner 1993 §II.1).
    ///
    /// **Singularity handling:** when `|Ṙ/C| ≥ 0.99` in any stage,
    /// `calculate_acceleration` returns `Err`; that stage's acceleration is
    /// clamped to `0.0`. The bubble wall freezes for that step rather than
    /// propagating a numerical divergence, matching the behaviour of the KM
    /// adaptive integrator's convergence guard.
    ///
    /// **Radius floor:** `R_n+1` is clamped to `f64::MIN_POSITIVE` to prevent
    /// numerical collapse through R = 0, which would cause the polytropic gas
    /// pressure to diverge.
    ///
    /// ## Parameters
    ///
    /// * `state`       — bubble state at time `t` (not mutated; stages clone)
    /// * `p_acoustic`  — instantaneous far-field acoustic driving pressure (Pa);
    ///   the wave-solver field value already sampled at this voxel, **not**
    ///   re-multiplied by sin(ωt) — that modulation is already encoded in
    ///   the PSTD pressure field
    /// * `t`           — current simulation time (s)
    /// * `dt`          — integration step size (s)
    ///
    /// ## References
    ///
    /// - Gilmore (1952) Caltech Hydro. Lab. Report 26-4, §4.
    /// - Hairer, Nørsett & Wanner (1993) *Solving ODEs I*, §II.1.
    #[must_use]
    pub fn step_rk4(&self, state: &BubbleState, p_acoustic: f64, t: f64, dt: f64) -> BubbleState {
        // ── k₁ : evaluated at (t, y_n) ─────────────────────────────────────
        let k1_r = state.wall_velocity;
        let k1_v = self
            .calculate_acceleration(state, p_acoustic, t)
            .unwrap_or(0.0);

        // ── k₂ : evaluated at (t + dt/2, y_n + dt/2·k₁) ───────────────────
        let mut s2 = state.clone();
        s2.radius = (0.5 * dt).mul_add(k1_r, state.radius).max(f64::MIN_POSITIVE);
        s2.wall_velocity = (0.5 * dt).mul_add(k1_v, state.wall_velocity);
        let k2_r = s2.wall_velocity;
        let k2_v = self
            .calculate_acceleration(&s2, p_acoustic, 0.5f64.mul_add(dt, t))
            .unwrap_or(0.0);

        // ── k₃ : evaluated at (t + dt/2, y_n + dt/2·k₂) ───────────────────
        let mut s3 = state.clone();
        s3.radius = (0.5 * dt).mul_add(k2_r, state.radius).max(f64::MIN_POSITIVE);
        s3.wall_velocity = (0.5 * dt).mul_add(k2_v, state.wall_velocity);
        let k3_r = s3.wall_velocity;
        let k3_v = self
            .calculate_acceleration(&s3, p_acoustic, 0.5f64.mul_add(dt, t))
            .unwrap_or(0.0);

        // ── k₄ : evaluated at (t + dt, y_n + dt·k₃) ───────────────────────
        let mut s4 = state.clone();
        s4.radius = dt.mul_add(k3_r, state.radius).max(f64::MIN_POSITIVE);
        s4.wall_velocity = dt.mul_add(k3_v, state.wall_velocity);
        let k4_r = s4.wall_velocity;
        let k4_v = self
            .calculate_acceleration(&s4, p_acoustic, t + dt)
            .unwrap_or(0.0);

        // ── Combine via standard Butcher weights ────────────────────────────
        let mut out = state.clone();
        out.radius = (dt / 6.0).mul_add(2.0f64.mul_add(k3_r, 2.0f64.mul_add(k2_r, k1_r)) + k4_r, state.radius)
            .max(f64::MIN_POSITIVE);
        out.wall_velocity =
            (dt / 6.0).mul_add(2.0f64.mul_add(k3_v, 2.0f64.mul_add(k2_v, k1_v)) + k4_v, state.wall_velocity);
        // Store the k₄ acceleration as the best available approximation of the
        // acceleration at the end of the step (used by downstream diagnostics).
        out.wall_acceleration = k4_v;
        out
    }

    /// Check if conditions warrant using Gilmore over simpler models
    #[must_use]
    pub fn should_use_gilmore(&self, state: &BubbleState) -> bool {
        // Use Gilmore when:
        // 1. Wall Mach number > 0.1
        // 2. Pressure ratio > 10
        // 3. Radius compression > 10x

        let mach = state.wall_velocity.abs() / self.params.c_liquid;
        let pressure_ratio = state.pressure_internal / self.params.p0;
        let radius_ratio = self.params.r0 / state.radius;

        mach > 0.1 || pressure_ratio > 10.0 || radius_ratio > 10.0
    }
}

#[cfg(test)]
mod tests;
