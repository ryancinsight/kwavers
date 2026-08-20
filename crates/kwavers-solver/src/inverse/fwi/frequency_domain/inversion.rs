//! Nonlinear conjugate-gradient inversion loop.
//!
//! # Step length
//!
//! The step is the exact minimizer of the objective's quadratic model along the
//! search direction `d`:
//!
//! ```text
//! α = −⟨g, d⟩ / ⟨d, H d⟩
//! ```
//!
//! This is scale-free: it is invariant to the magnitude of `d` and adapts to the
//! local curvature, whereas a step seeded from a fixed slowness increment is not.
//! The distinction is not cosmetic — with a fixed seed, a model already close to
//! the truth has a small gradient, every trial step overshoots into a region of
//! no numerical decrease, none is accepted, and the loop stalls having recovered
//! nothing (see [`super::gauss_newton`], which was introduced to work around
//! exactly that stall).
//!
//! `⟨d, H d⟩` reuses the matrix-free
//! [`super::gradient::hessian_vector`] action, so the curvature costs one extra
//! gradient evaluation per iteration and no Jacobian is assembled. Where the
//! quadratic model does not bound the step — non-positive curvature `⟨d,Hd⟩ ≤ 0`,
//! or a non-finite α — there is no model minimizer along `d`, and the step falls
//! back to `config.initial_step_s_per_m`. Backtracking is retained below either
//! seed as the safeguard against the model being locally inaccurate.
//!
//! Reference: Nocedal & Wright (2006) *Numerical Optimization* §3.1 (the exact
//! minimizer of a quadratic model along a direction).

use super::acquisition::TransmissionAcquisition;
use super::gauss_newton::GaussNewtonConfig;
use super::gradient::{dot, hessian_vector, max_abs, objective_and_gradient};
use super::types::{
    Config, FrequencyObservation, InversionResult, FREQUENCY_DOMAIN_FWI_SOLVER_MODEL,
};
use kwavers_core::error::KwaversResult;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    slowness_to_sound_speed, sound_speed_to_slowness, MultiRowRingArray,
};
use leto::Array3;

/// Reconstruct a sound-speed volume from ring-array frequency-domain data.
///
/// # Errors
/// Returns an error when observations, geometry, config, or model values are
/// outside the solver contract.
pub fn invert(
    observations: &[FrequencyObservation],
    acquisition: &dyn TransmissionAcquisition,
    initial_sound_speed_m_s: &Array3<f64>,
    config: &Config,
) -> KwaversResult<InversionResult> {
    let mut slowness = sound_speed_to_slowness(initial_sound_speed_m_s)?;
    let (mut objective, mut gradient) =
        objective_and_gradient(&slowness, observations, acquisition, config)?;
    let mut history = vec![objective];
    let mut direction = gradient.mapv(|value| -value);
    let mut previous_gradient = gradient.clone();

    for iteration in 0..config.iterations {
        if iteration > 0 {
            let mut diff = gradient.clone();
            for (value, &previous) in diff.iter_mut().zip(previous_gradient.iter()) {
                *value -= previous;
            }
            // Gilbert-Nocedal hybrid: beta = min(max(beta_PR, 0), beta_FR).
            //
            // beta_PR alone restarts well after a poor step but is unbounded,
            // so a near-orthogonal gradient pair can inflate it and throw the
            // search direction far from the descent cone. beta_FR is bounded
            // but stalls. Capping PR by FR keeps PR's restart behaviour while
            // retaining the global convergence guarantee FR carries under an
            // inexact line search, which is what this loop performs.
            let previous_energy = dot(&previous_gradient, &previous_gradient).max(f64::EPSILON);
            let beta_polak_ribiere = (dot(&gradient, &diff) / previous_energy).max(0.0);
            let beta_fletcher_reeves = dot(&gradient, &gradient) / previous_energy;
            let beta = beta_polak_ribiere.min(beta_fletcher_reeves);
            for (dir, &grad) in direction.iter_mut().zip(gradient.iter()) {
                *dir = -grad + beta * *dir;
            }
            if dot(&direction, &gradient) >= 0.0 {
                direction.assign(&gradient.mapv(|value| -value));
            }
        }

        let direction_scale = max_abs(&direction);
        if direction_scale <= f64::EPSILON {
            break;
        }

        let seed_step = model_minimizer_step(
            &slowness,
            &gradient,
            &direction,
            observations,
            acquisition,
            config,
        )?
        .unwrap_or(config.initial_step_s_per_m / direction_scale);

        let mut accepted = None;
        for search_step in 0..8 {
            let step = seed_step * 0.5_f64.powi(search_step);
            let mut candidate = slowness.clone();
            for (value, &dir) in candidate.iter_mut().zip(direction.iter()) {
                *value += step * dir;
            }
            clamp_slowness(&mut candidate, config);
            let (candidate_objective, candidate_gradient) =
                objective_and_gradient(&candidate, observations, acquisition, config)?;
            // (config is &Config; no move)
            if candidate_objective < objective {
                accepted = Some((candidate, candidate_objective, candidate_gradient));
                break;
            }
        }

        let Some((candidate, candidate_objective, candidate_gradient)) = accepted else {
            break;
        };

        previous_gradient = gradient;
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

/// Exact minimizer of the quadratic model along `direction`:
/// `α = −⟨g, d⟩ / ⟨d, H d⟩`.
///
/// Returns `None` when the model does not bound the step along `d` — non-positive
/// curvature (`⟨d, H d⟩ ≤ 0`, where the model decreases without limit and has no
/// minimizer), a vanishing directional derivative, or a non-finite result. The
/// caller then falls back to the configured fixed seed.
///
/// # Errors
/// Propagates forward/adjoint evaluation errors from the Hessian action.
fn model_minimizer_step(
    slowness: &Array3<f64>,
    gradient: &Array3<f64>,
    direction: &Array3<f64>,
    observations: &[FrequencyObservation],
    acquisition: &dyn TransmissionAcquisition,
    config: &Config,
) -> KwaversResult<Option<f64>> {
    let directional_derivative = dot(gradient, direction);
    // A non-descent direction is the caller's safeguard to handle, not ours.
    if directional_derivative >= 0.0 {
        return Ok(None);
    }

    let fd_epsilon = GaussNewtonConfig::default().fd_epsilon;
    let reference_slowness = 1.0 / config.reference_sound_speed_m_s;
    let hd = hessian_vector(
        slowness,
        gradient,
        direction,
        observations,
        acquisition,
        config,
        reference_slowness,
        fd_epsilon,
    )?;

    let curvature = dot(direction, &hd);
    if curvature <= 0.0 {
        return Ok(None);
    }

    let alpha = -directional_derivative / curvature;
    if alpha.is_finite() && alpha > 0.0 {
        Ok(Some(alpha))
    } else {
        Ok(None)
    }
}

pub(super) fn clamp_slowness(slowness: &mut Array3<f64>, config: &Config) {
    let min_slowness = 1.0 / config.max_sound_speed_m_s;
    let max_slowness = 1.0 / config.min_sound_speed_m_s;
    for value in slowness.iter_mut() {
        *value = value.clamp(min_slowness, max_slowness);
    }
}
