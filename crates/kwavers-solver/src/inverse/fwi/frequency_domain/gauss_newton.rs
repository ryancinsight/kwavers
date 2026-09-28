//! Matrix-free Gauss-Newton (truncated Newton-CG) inversion.
//!
//! The nonlinear-conjugate-gradient loop in [`super::inversion`] scales the
//! steepest-descent direction by a fixed slowness step and backtracks; when the
//! model is already close to the truth the gradient is small, the trial steps
//! fall below the objective's numerical-decrease threshold, and no step is
//! accepted (a *differential* monitor starting from a known background recovers
//! nothing). A Newton step solves the normal equations `H p = -g` and lands a
//! correctly-sized step in one shot, independent of the gradient magnitude.
//!
//! This is matrix-free: the Gauss-Newton/Hessian action `H v` is obtained by a
//! finite difference of the exact adjoint gradient,
//! `H v ≈ [g(m + ε v) − g(m)] / ε`, so it works for **any**
//! [`super::operator::HelmholtzForwardOperator`] (single-scatter Born or CBS)
//! without assembling a Jacobian. The inner solve is Steihaug-truncated conjugate
//! gradients with Levenberg-Marquardt damping; the outer step uses backtracking
//! from the full Newton step.
//!
//! References: Nocedal & Wright (2006) *Numerical Optimization* §7.1 (Newton-CG,
//! Steihaug); Métivier et al. (2013) truncated-Newton FWI.

use super::acquisition::TransmissionAcquisition;
use super::gradient::{hessian_vector, max_abs, objective_and_gradient};
use super::inversion::clamp_slowness;
use super::types::{
    Config, FrequencyObservation, InversionResult, FREQUENCY_DOMAIN_FWI_SOLVER_MODEL,
};
use crate::krylov::{policy, solve_cg, CpuBackend};
use athena_core::{Identity, KrylovBackend, LinearOperator, Termination};
use athena_leto::LetoBackendError;
use core::cell::RefCell;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    slowness_to_sound_speed, sound_speed_to_slowness,
};
use leto::{Array1, Array3};

/// Gauss-Newton / Newton-CG tuning.
#[derive(Clone, Copy, Debug)]
pub struct GaussNewtonConfig {
    /// Inner conjugate-gradient iterations per Newton step.
    pub cg_iterations: usize,
    /// Initial Levenberg-Marquardt damping `λ` in `(H + λI) p = -g`.
    pub lm_damping: f64,
    /// Relative slowness perturbation for the finite-difference Hessian action,
    /// as a fraction of the reference slowness.
    pub fd_epsilon: f64,
    /// Maximum LM damping increases per outer Newton step before giving up.
    pub max_lm_tries: usize,
}

impl Default for GaussNewtonConfig {
    fn default() -> Self {
        Self {
            cg_iterations: 8,
            lm_damping: 1.0e-3,
            fd_epsilon: 1.0e-3,
            max_lm_tries: 12,
        }
    }
}

/// LM damping increase factor when a trial step fails to reduce the objective.
const LM_INCREASE: f64 = 4.0;
/// LM damping decrease factor after a successful step (toward Gauss-Newton).
const LM_DECREASE: f64 = 0.5;
/// Lower bound on LM damping (keeps the operator nonsingular).
const LM_MIN: f64 = 1.0e-12;
/// Convergence tolerance of the inner Newton-CG solve, as a fraction of
/// `‖r₀‖ = ‖g‖`.
///
/// The truncation-only loop this replaced stopped on `rs_new <= 1e-12 ·
/// rs_initial` where `rs = ⟨r, r⟩`, that is on `‖r‖² ≤ 1e-12‖r₀‖²`. Athena's
/// convergence policy is `‖r‖ ≤ max(absolute, relative · ‖b‖₂)` with `b = −g`,
/// so `relative = 1e-6` and `absolute = 0` state the same rule: squaring both
/// sides gives back `1e-12`.
const CG_RELATIVE_TOLERANCE: f64 = 1.0e-6;

/// Gauss-Newton inversion. Same contract as [`super::inversion::invert`] but with
/// Newton-CG steps that engage near-truth residuals.
///
/// `config.iterations` bounds the outer Newton steps.
///
/// # Errors
/// Propagates forward/adjoint evaluation errors from `objective_and_gradient`.
pub fn invert_gauss_newton(
    observations: &[FrequencyObservation],
    acquisition: &dyn TransmissionAcquisition,
    initial_sound_speed_m_s: &Array3<f64>,
    config: &Config,
    gn: &GaussNewtonConfig,
) -> KwaversResult<InversionResult> {
    let mut slowness = sound_speed_to_slowness(initial_sound_speed_m_s)?;
    let (mut objective, mut gradient) =
        objective_and_gradient(&slowness, observations, acquisition, config)?;
    let mut history = vec![objective];
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let mut lambda = gn.lm_damping.max(LM_MIN);

    for _outer in 0..config.iterations {
        if max_abs(&gradient) <= f64::EPSILON {
            break;
        }

        // Levenberg-Marquardt: solve (H + λI) p = -g, increasing λ until the full
        // step reduces the objective. Large λ → small, well-scaled steepest-descent
        // step (engages near-truth residuals); small λ → Gauss-Newton step (fast
        // far from truth). This adapts the step scale without a separate line
        // search, fixing the negative-curvature / tiny-gradient stall.
        let mut accepted = None;
        for _ in 0..gn.max_lm_tries {
            let step = newton_cg(
                &slowness,
                &gradient,
                observations,
                acquisition,
                config,
                gn,
                reference_slowness,
                lambda,
            )?;
            if max_abs(&step) <= f64::EPSILON {
                lambda *= LM_INCREASE;
                continue;
            }
            let mut candidate = slowness.clone();
            for (s, &p) in candidate.iter_mut().zip(step.iter()) {
                *s += p;
            }
            clamp_slowness(&mut candidate, config);
            let (candidate_objective, candidate_gradient) =
                objective_and_gradient(&candidate, observations, acquisition, config)?;
            if candidate_objective < objective {
                accepted = Some((candidate, candidate_objective, candidate_gradient));
                lambda = (lambda * LM_DECREASE).max(LM_MIN);
                break;
            }
            lambda *= LM_INCREASE;
        }

        let Some((candidate, candidate_objective, candidate_gradient)) = accepted else {
            break;
        };
        slowness = candidate;
        objective = candidate_objective;
        gradient = candidate_gradient;
        history.push(objective);
    }

    Ok(InversionResult {
        sound_speed_m_s: slowness_to_sound_speed(&slowness)?,
        objective_history: history,
        frequencies_used: (observations.len()),
        transmissions_used: observations
            .first()
            .map(|obs| obs.observed_pressure.shape()[0])
            .unwrap_or(0),
        receivers_used: acquisition.receiver_count(),
        model_family: FREQUENCY_DOMAIN_FWI_SOLVER_MODEL,
    })
}

/// The matrix-free `(H + λI)` action Athena's conjugate gradients sees.
///
/// Athena's backend error vocabulary is Leto's and cannot carry a kwavers
/// error, so a failed forward/adjoint evaluation rides out as a Leto error
/// naming it, and the real reason is stashed in `failure` for
/// [`newton_cg`] to return. The stash is authoritative: the placeholder never
/// reaches a caller.
struct HessianAction<'a> {
    slowness: &'a Array3<f64>,
    gradient: &'a Array3<f64>,
    observations: &'a [FrequencyObservation],
    acquisition: &'a dyn TransmissionAcquisition,
    config: &'a Config,
    reference_slowness: f64,
    fd_epsilon: f64,
    lambda: f64,
    /// The probe direction, held so the operator can hand `hessian_vector` the
    /// owned `Array3` it takes without allocating per application.
    direction: RefCell<Array3<f64>>,
    failure: RefCell<Option<KwaversError>>,
}

impl HessianAction<'_> {
    fn take_failure(&self) -> Option<KwaversError> {
        self.failure.borrow_mut().take()
    }
}

impl LinearOperator<CpuBackend> for HessianAction<'_> {
    fn dimension(&self) -> usize {
        self.slowness.len()
    }

    fn apply(
        &self,
        _backend: &CpuBackend,
        input: <CpuBackend as KrylovBackend>::View<'_>,
        mut output: <CpuBackend as KrylovBackend>::ViewMut<'_>,
    ) -> Result<(), LetoBackendError> {
        let input = input
            .as_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let output = output
            .as_mut_slice()
            .ok_or(LetoBackendError::NonContiguousVector)?;
        let mut direction = self.direction.borrow_mut();
        {
            let scratch = direction
                .as_slice_mut()
                .ok_or(LetoBackendError::NonContiguousVector)?;
            scratch.copy_from_slice(input);
        }

        let mut image = match hessian_vector(
            self.slowness,
            self.gradient,
            &direction,
            self.observations,
            self.acquisition,
            self.config,
            self.reference_slowness,
            self.fd_epsilon,
        ) {
            Ok(image) => image,
            Err(error) => {
                let reason = error.to_string();
                *self.failure.borrow_mut() = Some(error);
                return Err(LetoBackendError::Leto(leto::LetoError::InvalidInput(
                    format!("the Gauss-Newton Hessian action could not be evaluated: {reason}"),
                )));
            }
        };
        if self.lambda > 0.0 {
            for (h, &d) in image.iter_mut().zip(direction.iter()) {
                *h += self.lambda * d;
            }
        }

        output.copy_from_slice(
            image
                .as_slice()
                .ok_or(LetoBackendError::NonContiguousVector)?,
        );
        Ok(())
    }
}

/// Steihaug-truncated conjugate gradients solving `(H + λI) p = -g`.
///
/// On negative curvature it truncates: the CG iterate so far, or **zeros** on the
/// first iteration to signal the caller to increase `λ` (rather than returning an
/// unscaled steepest-descent step that the line search would reject near truth).
///
/// The recurrence itself is Athena's (`crate::krylov::solve_cg`); this function
/// supplies only the operator, the identity preconditioner, and the policy.
/// Athena's `NonPositiveCurvature` termination *is* Steihaug truncation: it
/// reports at iteration `k − 1` with the solution buffer holding the iterate
/// before the step that broke the curvature test, which is exactly the early
/// `return Ok(p)` this loop used to perform.
// allow(too_many_arguments): distinct Newton-CG inputs (model, gradient, the
// forward-problem triple, GN/regularisation parameters) — see hessian_vector.
#[allow(clippy::too_many_arguments)]
fn newton_cg(
    slowness: &Array3<f64>,
    gradient: &Array3<f64>,
    observations: &[FrequencyObservation],
    acquisition: &dyn TransmissionAcquisition,
    config: &Config,
    gn: &GaussNewtonConfig,
    reference_slowness: f64,
    lambda: f64,
) -> KwaversResult<Array3<f64>> {
    let shape = slowness.shape();
    let mut solution = Array3::<f64>::zeros([shape[0], shape[1], shape[2]]);

    // Residual of `(H + λI) p = −g` at `p = 0` is `−g`. A gradient whose norm
    // is at or below `f64::EPSILON` is the zero step, evaluated before the
    // solver rather than by its initial-residual test: the two agree only when
    // the gradient is exactly zero.
    let right_hand_side: Vec<f64> = gradient.iter().map(|value| -value).collect();
    let squared_norm = right_hand_side
        .iter()
        .map(|value| value * value)
        .fold(0.0_f64, |acc, value| acc + value);
    if squared_norm <= f64::EPSILON * f64::EPSILON || gn.cg_iterations == 0 {
        return Ok(solution);
    }
    let dimension = right_hand_side.len();
    let right_hand_side = Array1::from_shape_vec([dimension], right_hand_side)
        .expect("invariant: the right-hand side has the operator's dimension");
    let mut iterate = Array1::<f64>::zeros([dimension]);

    let action = HessianAction {
        slowness,
        gradient,
        observations,
        acquisition,
        config,
        reference_slowness,
        fd_epsilon: gn.fd_epsilon,
        lambda,
        direction: RefCell::new(Array3::<f64>::zeros([shape[0], shape[1], shape[2]])),
        failure: RefCell::new(None),
    };
    let outcome = solve_cg(
        &action,
        &Identity,
        &right_hand_side,
        &mut iterate,
        policy(0.0, CG_RELATIVE_TOLERANCE, gn.cg_iterations)?,
    );
    if let Some(error) = action.take_failure() {
        return Err(error);
    }
    let report = outcome?;
    debug_assert!(
        !matches!(
            report.termination,
            Termination::NonFinite | Termination::Breakdown
        ),
        "the inner solve reached {:?}: the Hessian action is not well posed here",
        report.termination
    );

    solution
        .as_slice_mut()
        .expect("invariant: a freshly allocated array is contiguous")
        .copy_from_slice(
            iterate
                .as_slice()
                .expect("invariant: a freshly allocated array is contiguous"),
        );
    Ok(solution)
}

#[cfg(test)]
mod tests {
    use super::super::acquisition::RingAcquisition;
    use super::*;
    use crate::inverse::fwi::frequency_domain::{simulate_frequency_observation, Config};
    use aequitas::systems::si::quantities::Length;
    use aequitas::systems::si::units::Meter;
    use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
    use leto::Array3;

    fn ring(n_elem: usize, diameter: f64) -> MultiRowRingArray {
        MultiRowRingArray::new(
            n_elem,
            1,
            Length::from_unit::<Meter>(diameter),
            Length::from_unit::<Meter>(0.0),
        )
        .unwrap()
    }

    /// From the EXACT background (where NLCG accepts no step), Gauss-Newton must
    /// reduce the objective and recover a positive Δc at the inclusion — the
    /// near-truth engagement the monitor needs.
    #[test]
    fn gauss_newton_engages_near_truth_where_nlcg_stalls() {
        let n = 10;
        let centre = n / 2;
        let spacing = 1.0e-3;
        let array = ring(16, 0.024);
        let config = Config {
            reference_sound_speed_m_s: 1500.0,
            spacing_m: spacing,
            iterations: 6,
            min_sound_speed_m_s: 1400.0,
            max_sound_speed_m_s: 1700.0,
            estimate_source_scaling: false,
            ..Config::default()
        };

        // Background homogeneous; perturbed has a +60 m/s inclusion at the centre.
        let background = Array3::from_elem([n, n, 1], 1500.0);
        let mut perturbed = background.clone();
        for i in centre - 1..=centre + 1 {
            for j in centre - 1..=centre + 1 {
                perturbed[[i, j, 0]] = 1560.0;
            }
        }
        let observations: Vec<FrequencyObservation> = [3.0e5, 5.0e5]
            .iter()
            .map(|&f| {
                FrequencyObservation::new(
                    f,
                    simulate_frequency_observation(
                        &perturbed,
                        &RingAcquisition::new(&array),
                        f,
                        &config,
                    )
                    .unwrap(),
                )
            })
            .collect();

        // Objective at the exact background start.
        let start_slowness = sound_speed_to_slowness(&background).unwrap();
        let (obj_start, _) = objective_and_gradient(
            &start_slowness,
            &observations,
            &RingAcquisition::new(&array),
            &config,
        )
        .unwrap();

        let gn = GaussNewtonConfig::default();
        let result = invert_gauss_newton(
            &observations,
            &RingAcquisition::new(&array),
            &background,
            &config,
            &gn,
        )
        .unwrap();

        let obj_end = *result.objective_history.last().unwrap();
        eprintln!(
            "GN objective {obj_start:.4e} -> {obj_end:.4e} ({} steps); centre Δc {:+.2}",
            (result.objective_history.len()),
            result.sound_speed_m_s[[centre, centre, 0]] - 1500.0
        );
        assert!(
            obj_end < obj_start,
            "Gauss-Newton must reduce the objective from the exact background: {obj_start} -> {obj_end}"
        );
        let centre_dc = result.sound_speed_m_s[[centre, centre, 0]] - 1500.0;
        assert!(
            centre_dc > 0.0,
            "Gauss-Newton must recover a positive Δc at the inclusion, got {centre_dc}"
        );
    }
}
