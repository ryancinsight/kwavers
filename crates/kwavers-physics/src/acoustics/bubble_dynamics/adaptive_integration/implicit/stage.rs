use super::super::AdaptiveBubbleModel;
use super::{ImplicitBubbleIntegrator, SDIRK_GAMMA};
use crate::acoustics::bubble_dynamics::{acoustic_pressure::InstantaneousPressure, BubbleState};
use aequitas::systems::si::quantities::PressureRate;
use kwavers_core::error::{KwaversResult, PhysicsError};

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

/// One SDIRK stage equation `z = w + hγ·f(t_stage, z)` with its Newton
/// starting guess.
#[derive(Debug, Clone, Copy)]
pub(super) struct StageProblem<'a> {
    /// State whose non-mechanical fields are held fixed.
    pub(super) template: &'a BubbleState,
    /// Radius component of the explicit part `w` [m].
    pub(super) w_r: f64,
    /// Wall-velocity component of the explicit part `w` [m/s].
    pub(super) w_v: f64,
    /// Step size `h` [s].
    pub(super) h: f64,
    /// Time of the stage [s].
    pub(super) t_stage: f64,
    /// Instantaneous drive pressure, held over the step.
    pub(super) p_acoustic: InstantaneousPressure,
    /// Drive-pressure rate, held over the step.
    pub(super) dp_dt: PressureRate,
    /// Newton starting point (R, Ṙ).
    pub(super) guess: (f64, f64),
}

impl<Model> ImplicitBubbleIntegrator<'_, Model>
where
    Model: AdaptiveBubbleModel,
{
    /// Solve one SDIRK stage equation z = w + hγ·f(t_stage, z) by
    /// Newton iteration with a finite-difference Jacobian on the
    /// 2×2 (R, Ṙ) system.
    ///
    /// The problem's `template` supplies the state's non-mechanical
    /// fields (temperature, gas content, species) held fixed across
    /// the step, exactly as the explicit path's stage states do.
    /// # Errors
    /// - [`PhysicsError::ConvergenceFailure`] when Newton does not
    ///   converge in [`MAX_NEWTON_ITERATIONS`] iterations (a soft
    ///   failure: callers reject the step and retry smaller).
    ///
    pub(super) fn implicit_stage_solve(
        &self,
        problem: StageProblem<'_>,
    ) -> KwaversResult<(f64, f64)> {
        let StageProblem {
            template,
            w_r,
            w_v,
            h,
            t_stage,
            p_acoustic,
            dp_dt,
            guess,
        } = problem;
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
    pub(super) fn accel(
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
}
