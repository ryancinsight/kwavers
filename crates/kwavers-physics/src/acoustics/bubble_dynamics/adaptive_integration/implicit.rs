//! Implicit SDIRK2 integration for stiff bubble dynamics
//! (KW-PHYS-009 item 5).
//!
//! The explicit RK4 step-doubling path in [`super::integrator`] cannot
//! traverse an inertial collapse: through the ~ns collapse spike its
//! full-step and half-step error estimates are *correlated* (both
//! under-resolve the same rebound and agree on a wrong state), and the
//! accepted overshoot below the true Rmin multiplies the gas pressure
//! by ~10³–10⁴, injecting energy that the rebound converts into
//! spurious Rmax growth. Measured at the Louisnard & Gomez (2003)
//! Fig. 4 anchor (R0 = 5 µm, p = 5 bar): the explicit driver returns a
//! window growth rate of 8.81×10⁶ µm/s against the paper's ~300 µm/s,
//! while the library path deadlocked in `ConvergenceFailure` at the
//! stability monitor's old |Ṙ| > 0.1·c bound.
//!
//! This module provides the stiff-capable path: a two-stage,
//! singly diagonally implicit Runge–Kutta method (SDIRK2,
//! Alexander 1977), L-stable and stiffly accurate, with Newton
//! iteration on the 2×2 (R, Ṙ) system (finite-difference Jacobian),
//! wrapped in the same step-doubling error controller shape as the
//! explicit path (full step vs two half steps, half-step result
//! accepted, no Richardson extrapolation — the extrapolated value is
//! part of the explicit path's energy-injection mechanism). L-stability
//! is the operative property: unresolved stiff components are damped,
//! not amplified, so the integrator can step over the collapse spike
//! at the accuracy the *expansion* phase requires — which is all the
//! cycle-averaged laws read (Louisnard & Gomez's R⁴ weighting
//! suppresses the collapse phase by construction).
//!
//! ## Validation status
//!
//! Validated in Build 22 Part 3 against the prototype's LSODA
//! reference integration of the Keller–Miksis ODE at the
//! Louisnard & Gomez Fig. 4 conditions (26.5 kHz), with the
//! reference's dense output sampled finely enough to resolve the
//! collapse spike (the diagnosis's earlier spike extrema —
//! Rmin = 0.023 µm, max |Ṙ| = 6044 m/s at (5 µm, 5 bar) — were
//! themselves sampling artefacts of the spike; properly sampled,
//! LSODA gives Rmin = 0.016 µm and max |Ṙ| ≈ 5.3×10⁴ m/s there;
//! the 2-bar anchor values from that sampling are superseded —
//! see below). This integrator reproduces
//! the Fig. 4 growth-rate anchors at all six points within the
//! paper's factor-of-2 read bands — matching the LSODA law values
//! to ≲0.5%. Trajectory extremes: as originally measured against a
//! uniformly-sampled LSODA dense-output reference, Rmin agreed to
//! 0–15% and peak |Ṙ| to a factor ≲2 at the 2-bar anchors; the
//! physics review of 2026-11-27 showed that reference was itself
//! under-sampled at the ~ps-wide spike, restored the Keller–Miksis
//! LHS +4μ/(ρc) term the equation had been missing, and re-measured
//! the reference spike-resolved (solver internal nodes, fs-scale
//! windows): the library now matches it to printed precision at
//! all three measured anchors (0.0149 µm / 29,479 m/s at
//! (3 µm, 2 bar); 0.0297 µm / 20,842 m/s at (5 µm, 2 bar);
//! 0.0160 µm / 54,157 m/s at (5 µm, 5 bar)). Benchmark in
//! `acoustics::mechanics::cavitation::core::louisnard_gomez`.
//!
//! ## Validity regime
//!
//! Any single-bubble model implementing [`AdaptiveBubbleModel`]
//! whose acceleration is a smooth function of (R, Ṙ) between collapse
//! spikes. The integrator imposes no wall-Mach bound: states with
//! Mach ≳ 1 are traversed and flagged via `BubbleState::mach_number`
//! (the underlying Keller–Miksis equation is a first-order-in-Mach
//! theory; Mach ≳ 1 states are the equation-as-written's solution,
//! outside its asymptotic validity — see
//! [`super::super::keller_miksis::equation`]). Thermal / mass-transfer
//! updates use the same operator splitting as the explicit path
//! (mechanical step, then the model's update hooks over the step).

use super::config::AdaptiveBubbleConfig;
use super::integrator::bubble_state_is_stable;
use super::statistics::IntegrationStatistics;
use super::AdaptiveBubbleModel;
use crate::acoustics::bubble_dynamics::{acoustic_pressure::InstantaneousPressure, BubbleState};
use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::constants::numerical::{HALF_STEP_FACTOR, INITIAL_TIME_STEP_FRACTION};
use kwavers_core::error::{KwaversResult, PhysicsError};

/// SDIRK2 diagonal coefficient γ = 1 − 1/√2 (Alexander 1977): the
/// value that makes the two-stage SDIRK method second-order and
/// L-stable.
const SDIRK_GAMMA: f64 = 1.0 - std::f64::consts::FRAC_1_SQRT_2;

/// Newton iteration cap per implicit stage solve. Through the
/// collapse spike the stage equations are severely stiff; the cap is
/// generous because a failed solve is a soft step rejection (the
/// controller retries at smaller h, where the implicit system is
/// better conditioned), never a silently accepted state.
const MAX_NEWTON_ITERATIONS: usize = 50;

/// Newton convergence threshold on the scaled increment norm (scales
/// from the config tolerances, as in the step error norm). Kept far
/// below the step controller's unit threshold so the step-doubling
/// error estimate stays truncation-dominated rather than
/// solver-noise-dominated: at the collapse turnaround the stiff
/// coupling ∂R̈/∂R ≈ −6×10²⁵ m⁻¹s⁻² amplifies any residual radius
/// increment of a stage solve into the velocity component. Newton
/// contracts quadratically in smooth phases, so the tight bound
/// costs about one extra iteration per stage there; the f64
/// rounding floor of the scaled increment is ≲10⁻⁴ even at the
/// spike, so 10⁻³ remains reachable.
const NEWTON_TOLERANCE: f64 = 1e-3;

/// Step-doubling error-control exponent for a second-order method,
/// 1/(p + 1) with p = 2. (The explicit path's
/// `ERROR_CONTROL_EXPONENT` = 0.2 is the p = 4 value.)
const IMPLICIT_ERROR_EXPONENT: f64 = 1.0 / 3.0;

/// Adaptive implicit (SDIRK2) integrator for bubble dynamics.
///
/// Same outer contract as
/// [`AdaptiveBubbleIntegrator`](super::AdaptiveBubbleIntegrator):
/// the drive (`p_acoustic`, `dp_dt`) is held constant across the main
/// step (zero-order hold) while the integrator sub-cycles adaptively
/// inside it. Use this path whenever the trajectory may collapse
/// inertially (wall Mach approaching or exceeding 1); use the
/// explicit path for weakly driven, non-stiff regimes.
#[derive(Debug)]
pub struct ImplicitBubbleIntegrator<'a, Model: AdaptiveBubbleModel> {
    solver: &'a Model,
    config: AdaptiveBubbleConfig,
    /// Current adaptive time step
    dt_adaptive: f64,
    /// Statistics
    total_substeps: usize,
    rejected_steps: usize,
    min_dt_used: f64,
    max_dt_used: f64,
    /// Trajectory extremes over accepted sub-steps (diagnostics for
    /// the collapse spike, which the caller's output grid undersamples
    /// by construction — the spike lasts ~ns).
    min_radius_seen: f64,
    max_wall_speed_seen: f64,
}

impl<'a, Model> ImplicitBubbleIntegrator<'a, Model>
where
    Model: AdaptiveBubbleModel,
{
    /// Create a new implicit integrator
    #[must_use]
    pub fn new(solver: &'a Model, config: AdaptiveBubbleConfig) -> Self {
        let dt_adaptive = config.dt_max * INITIAL_TIME_STEP_FRACTION;

        Self {
            solver,
            config,
            dt_adaptive,
            total_substeps: 0,
            rejected_steps: 0,
            min_dt_used: f64::MAX,
            max_dt_used: 0.0,
            min_radius_seen: f64::INFINITY,
            max_wall_speed_seen: 0.0,
        }
    }

    /// Integrate bubble dynamics with implicit adaptive sub-cycling.
    ///
    /// Mirrors
    /// [`AdaptiveBubbleIntegrator::integrate_adaptive`](super::AdaptiveBubbleIntegrator::integrate_adaptive),
    /// including its convergence guard: `max_substeps` bounds accepted
    /// steps and `max_substeps × 100` bounds total attempts, after
    /// which the call returns `ConvergenceFailure` rather than
    /// looping on a step size that never succeeds.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn integrate_adaptive(
        &mut self,
        state: &mut BubbleState,
        p_acoustic: InstantaneousPressure,
        dp_dt: PressureRate,
        dt_main: f64,
        t_start: f64,
    ) -> KwaversResult<()> {
        let mut t = t_start;
        let t_end = t_start + dt_main;
        let mut substeps = 0;
        let mut total_attempts: usize = 0;
        let max_total_attempts = self.config.max_substeps.saturating_mul(100);

        while t < t_end && substeps < self.config.max_substeps {
            total_attempts += 1;
            if total_attempts > max_total_attempts {
                return Err(PhysicsError::ConvergenceFailure {
                    solver: "ImplicitBubbleIntegrator".to_owned(),
                    iterations: total_attempts,
                    residual: (t_end - t).abs(),
                }
                .into());
            }

            let dt = self.dt_adaptive.min(t_end - t);

            let (success, dt_next) = self.try_step(state, p_acoustic, dp_dt, dt, t)?;

            if success {
                t += dt;
                substeps += 1;
                self.total_substeps += 1;
                self.min_dt_used = self.min_dt_used.min(dt);
                self.max_dt_used = self.max_dt_used.max(dt);
            } else {
                self.rejected_steps += 1;
            }

            self.dt_adaptive = dt_next.max(self.config.dt_min).min(self.config.dt_max);
        }

        if substeps >= self.config.max_substeps {
            return Err(PhysicsError::ConvergenceFailure {
                solver: "ImplicitBubbleIntegrator".to_owned(),
                iterations: substeps,
                residual: (t_end - t).abs(),
            }
            .into());
        }

        Ok(())
    }

    /// Try a single step with step-doubling error estimation.
    ///
    /// The full-vs-half comparison at the endpoint is supplemented
    /// by a half-vs-quarter comparison at the midpoint: the midpoint
    /// is part of the accepted trajectory, and in this stiff,
    /// strongly restoring system an intermediate excursion is
    /// invisible to endpoint agreement alone.
    ///
    /// As in the explicit path, errors during a trial step (including
    /// Newton non-convergence inside a stage solve) are *soft*: the
    /// step is rejected and retried at a smaller dt. Only the accepted
    /// half-step pair commits state and trajectory extremes.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn try_step(
        &mut self,
        state: &mut BubbleState,
        p_acoustic: InstantaneousPressure,
        dp_dt: PressureRate,
        dt: f64,
        t: f64,
    ) -> KwaversResult<(bool, f64)> {
        let state_orig = *state;

        // Full step (auxiliary; Richardson error estimation only).
        let mut state_full = state_orig;
        let full_ok = self
            .step_sdirk2(&mut state_full, p_acoustic, dp_dt, dt, t)
            .is_ok();

        // Two half steps (the accepted state if the step passes).
        // The midpoint (first half step's endpoint) is kept: it is
        // part of the accepted trajectory, so it is certified
        // separately below — endpoint agreement alone cannot see an
        // intermediate excursion in this stiff, strongly restoring
        // system (measured in the Build 22 Part 3 build: an accepted
        // pair whose midpoint reached |Ṙ| ≈ 54 km/s against the
        // LSODA reference's 6 km/s, with endpoints in agreement).
        let mut state_half = state_orig;
        let dt_half = dt * HALF_STEP_FACTOR;
        let mut min_r_half = f64::INFINITY;
        let mut max_v_half = 0.0_f64;
        let mut state_mid = state_orig;
        let mut half_ok = self
            .step_sdirk2(&mut state_half, p_acoustic, dp_dt, dt_half, t)
            .is_ok();
        if half_ok {
            min_r_half = min_r_half.min(state_half.radius);
            max_v_half = max_v_half.max(state_half.wall_velocity.abs());
            state_mid = state_half;
            half_ok = self
                .step_sdirk2(&mut state_half, p_acoustic, dp_dt, dt_half, t + dt_half)
                .is_ok();
        }
        min_r_half = min_r_half.min(state_half.radius);
        max_v_half = max_v_half.max(state_half.wall_velocity.abs());

        // Two quarter steps over the first half, certifying the
        // midpoint against a finer computation of the same point.
        let dt_quarter = dt_half * HALF_STEP_FACTOR;
        let mut state_quarter = state_orig;
        let quarter_ok = self
            .step_sdirk2(&mut state_quarter, p_acoustic, dp_dt, dt_quarter, t)
            .is_ok()
            && self
                .step_sdirk2(
                    &mut state_quarter,
                    p_acoustic,
                    dp_dt,
                    dt_quarter,
                    t + dt_quarter,
                )
                .is_ok();

        if !full_ok || !half_ok || !quarter_ok {
            return Ok((false, dt * self.config.dt_decrease_max));
        }

        // Step-doubling error estimates (raw differences, as in the
        // explicit path — conservative for a second-order method):
        // full-vs-half at the endpoint, half-vs-quarter at the
        // midpoint. The step is accepted only if both pass.
        let scale_r = self
            .config
            .rtol
            .mul_add(state_half.radius.abs(), self.config.atol);
        let scale_v = self
            .config
            .rtol
            .mul_add(state_half.wall_velocity.abs(), self.config.atol);
        let error_r = (state_full.radius - state_half.radius).abs();
        let error_v = (state_full.wall_velocity - state_half.wall_velocity).abs();
        let end_norm = (error_r / scale_r).hypot(error_v / scale_v) / 2.0_f64.sqrt();

        let scale_rm = self
            .config
            .rtol
            .mul_add(state_mid.radius.abs(), self.config.atol);
        let scale_vm = self
            .config
            .rtol
            .mul_add(state_mid.wall_velocity.abs(), self.config.atol);
        let error_rm = (state_quarter.radius - state_mid.radius).abs();
        let error_vm = (state_quarter.wall_velocity - state_mid.wall_velocity).abs();
        let mid_norm = (error_rm / scale_rm).hypot(error_vm / scale_vm) / 2.0_f64.sqrt();

        let error_norm = end_norm.max(mid_norm);

        let dt_next = if error_norm > 0.0 {
            let factor =
                self.config.safety_factor * (1.0 / error_norm).powf(IMPLICIT_ERROR_EXPONENT);
            let factor = factor
                .min(self.config.dt_increase_max)
                .max(self.config.dt_decrease_max);
            dt * factor
        } else {
            dt * self.config.dt_increase_max
        };

        let accept = error_norm <= 1.0 && bubble_state_is_stable(&self.config, &state_half);

        if accept {
            *state = state_half;
            self.min_radius_seen = self.min_radius_seen.min(min_r_half);
            self.max_wall_speed_seen = self.max_wall_speed_seen.max(max_v_half);
        }

        Ok((accept, dt_next))
    }

    /// One SDIRK2 step of size `h` from the state at time `t`.
    ///
    /// Stage 1 solves z₁ = y + hγ·f(t + γh, z₁); stage 2 solves
    /// z₂ = y + h(1−γ)·f(t + γh, z₁) + hγ·f(t + h, z₂) and, the
    /// method being stiffly accurate, z₂ is the step result. After
    /// the mechanical step, the model's temperature and mass-transfer
    /// hooks run over `h` (operator splitting, as in the explicit
    /// path's `step_rk4`).
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    fn step_sdirk2(
        &self,
        state: &mut BubbleState,
        p_acoustic: InstantaneousPressure,
        dp_dt: PressureRate,
        h: f64,
        t: f64,
    ) -> KwaversResult<()> {
        let y = *state;

        // Stage 1: implicit solve about y, explicit-predictor guess.
        let f_y = self.accel(&y, y.radius, y.wall_velocity, p_acoustic, dp_dt, t)?;
        let guess1 = (
            (h * SDIRK_GAMMA).mul_add(f_y.0, y.radius),
            (h * SDIRK_GAMMA).mul_add(f_y.1, y.wall_velocity),
        );
        let z1 = self.implicit_stage_solve(
            &y,
            y.radius,
            y.wall_velocity,
            h,
            t + SDIRK_GAMMA * h,
            p_acoustic,
            dp_dt,
            guess1,
        )?;
        let f_z1 = self.accel(&y, z1.0, z1.1, p_acoustic, dp_dt, t + SDIRK_GAMMA * h)?;

        // Stage 2: implicit solve about w = y + h(1−γ)·f(z₁).
        let w = (
            (h * (1.0 - SDIRK_GAMMA)).mul_add(f_z1.0, y.radius),
            (h * (1.0 - SDIRK_GAMMA)).mul_add(f_z1.1, y.wall_velocity),
        );
        let guess2 = (
            (h * SDIRK_GAMMA).mul_add(f_z1.0, w.0),
            (h * SDIRK_GAMMA).mul_add(f_z1.1, w.1),
        );
        let z2 = self.implicit_stage_solve(&y, w.0, w.1, h, t + h, p_acoustic, dp_dt, guess2)?;

        // Commit the stiffly accurate stage-2 state, then refresh the
        // derived quantities (wall acceleration, pressures, Mach) at
        // the committed state — the same end-of-step state the
        // explicit path leaves after its k4 evaluation.
        state.radius = z2.0;
        state.wall_velocity = z2.1;
        let _ = self.accel(state, z2.0, z2.1, p_acoustic, dp_dt, t + h)?;
        state.update_compression(self.solver.params().r0);
        state.update_collapse_state();

        // Operator splitting: thermal / chemical step over h.
        self.solver.update_temperature(state, h)?;
        self.solver.update_mass_transfer(state, h)?;

        Ok(())
    }

    /// Solve one SDIRK stage equation z = w + hγ·f(t_stage, z) by
    /// Newton iteration with a finite-difference Jacobian on the
    /// 2×2 (R, Ṙ) system.
    ///
    /// `template` supplies the state's non-mechanical fields
    /// (temperature, gas content, species) held fixed across the
    /// step, exactly as the explicit path's stage states do.
    /// # Errors
    /// - [`PhysicsError::ConvergenceFailure`] when Newton does not
    ///   converge in [`MAX_NEWTON_ITERATIONS`] iterations (a soft
    ///   failure: callers reject the step and retry smaller).
    ///
    #[allow(clippy::too_many_arguments)]
    fn implicit_stage_solve(
        &self,
        template: &BubbleState,
        w_r: f64,
        w_v: f64,
        h: f64,
        t_stage: f64,
        p_acoustic: InstantaneousPressure,
        dp_dt: PressureRate,
        guess: (f64, f64),
    ) -> KwaversResult<(f64, f64)> {
        let hg = h * SDIRK_GAMMA;
        let (mut zr, mut zv) = guess;
        if !zr.is_finite() || !zv.is_finite() {
            (zr, zv) = (w_r, w_v);
        }
        let mut prev_norm = f64::INFINITY;

        for iteration in 0..MAX_NEWTON_ITERATIONS {
            let f = self.accel(template, zr, zv, p_acoustic, dp_dt, t_stage)?;
            let g_r = hg.mul_add(-f.0, zr) - w_r;
            let g_v = hg.mul_add(-f.1, zv) - w_v;

            // Finite-difference Jacobian of f wrt (R, Ṙ).
            let dr = 1e-6 * zr.abs().max(1e-12);
            let dv = 1e-6 * zv.abs().max(1.0);
            let f_r = self.accel(template, zr + dr, zv, p_acoustic, dp_dt, t_stage)?;
            let f_v = self.accel(template, zr, zv + dv, p_acoustic, dp_dt, t_stage)?;

            // J = I − hγ·∂f/∂z, with ∂f/∂z = [[∂Ṙ/∂R, ∂Ṙ/∂Ṙ], [∂a/∂R, ∂a/∂Ṙ]]
            // (the ∂Ṙ/∂R = 0, ∂Ṙ/∂Ṙ = 1 row comes out of the differences
            // like everything else — no analytic structure is assumed).
            let j00 = 1.0 - hg * (f_r.0 - f.0) / dr;
            let j01 = -hg * (f_v.0 - f.0) / dv;
            let j10 = -hg * (f_r.1 - f.1) / dr;
            let j11 = 1.0 - hg * (f_v.1 - f.1) / dv;
            let det = j00.mul_add(j11, -(j01 * j10));
            if !det.is_finite() || det == 0.0 {
                break;
            }

            // Solve J·Δ = −G by Cramer's rule:
            // Δ_r = (j01·G_v − j11·G_r)/det, Δ_v = (j10·G_r − j00·G_v)/det.
            let d_r = j01.mul_add(g_v, -(j11 * g_r)) / det;
            let d_v = j10.mul_add(g_r, -(j00 * g_v)) / det;
            zr += d_r;
            zv += d_v;
            if !zr.is_finite() || !zv.is_finite() {
                break;
            }

            let norm = self.scaled_increment_norm(d_r, d_v, zr, zv);
            if norm < NEWTON_TOLERANCE {
                return Ok((zr, zv));
            }
            // Divergence guard: a growing increment after the first
            // iterations means the guess is outside Newton's basin at
            // this h — fail soft and let the controller shrink h.
            if iteration > 3 && norm > prev_norm {
                break;
            }
            prev_norm = norm;
        }

        Err(PhysicsError::ConvergenceFailure {
            solver: "ImplicitBubbleIntegrator::implicit_stage_solve".to_owned(),
            iterations: MAX_NEWTON_ITERATIONS,
            residual: prev_norm,
        }
        .into())
    }

    /// Evaluate the ODE right-hand side (Ṙ, R̈) at an arbitrary
    /// (R, Ṙ) on a copy of `template` (whose non-mechanical fields
    /// are held fixed), returning (Ṙ, R̈).
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    fn accel(
        &self,
        template: &BubbleState,
        r: f64,
        v: f64,
        p_acoustic: InstantaneousPressure,
        dp_dt: PressureRate,
        t: f64,
    ) -> KwaversResult<(f64, f64)> {
        let mut scratch = *template;
        scratch.radius = r;
        scratch.wall_velocity = v;
        let a = self
            .solver
            .calculate_acceleration(&mut scratch, p_acoustic, dp_dt, t)?;
        Ok((v, a))
    }

    /// Scaled norm of a Newton increment, using the config tolerance
    /// scales (same scales as the step-doubling error norm).
    fn scaled_increment_norm(&self, d_r: f64, d_v: f64, r: f64, v: f64) -> f64 {
        let scale_r = self.config.rtol.mul_add(r.abs(), self.config.atol);
        let scale_v = self.config.rtol.mul_add(v.abs(), self.config.atol);
        (d_r / scale_r).hypot(d_v / scale_v)
    }

    /// Smallest radius over accepted sub-steps since construction
    /// (or the last [`Self::reset_statistics`]). The collapse spike
    /// lasts ~ns, so caller output grids undersample it; this tracks
    /// the accepted trajectory's own resolution of it.
    #[must_use]
    pub fn min_radius_seen(&self) -> f64 {
        self.min_radius_seen
    }

    /// Largest |Ṙ| over accepted sub-steps since construction (or
    /// the last [`Self::reset_statistics`]).
    #[must_use]
    pub fn max_wall_speed_seen(&self) -> f64 {
        self.max_wall_speed_seen
    }

    /// Get integration statistics
    #[must_use]
    pub fn statistics(&self) -> IntegrationStatistics {
        IntegrationStatistics {
            total_substeps: self.total_substeps,
            rejected_steps: self.rejected_steps,
            min_dt_used: self.min_dt_used,
            max_dt_used: self.max_dt_used,
            rejection_rate: self.rejected_steps as f64 / self.total_substeps.max(1) as f64,
        }
    }

    /// Reset statistics (including the trajectory extremes)
    pub fn reset_statistics(&mut self) {
        self.total_substeps = 0;
        self.rejected_steps = 0;
        self.min_dt_used = f64::MAX;
        self.max_dt_used = 0.0;
        self.min_radius_seen = f64::INFINITY;
        self.max_wall_speed_seen = 0.0;
    }
}
