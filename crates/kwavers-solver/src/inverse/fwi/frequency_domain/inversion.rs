//! Nonlinear conjugate-gradient inversion loop.

use super::gradient::{dot, max_abs, objective_and_gradient};
use super::types::{
    Config, FrequencyObservation, InversionResult, FREQUENCY_DOMAIN_FWI_SOLVER_MODEL,
};
use kwavers_core::error::KwaversResult;
use kwavers_physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::{
    slowness_to_sound_speed, sound_speed_to_slowness, MultiRowRingArray,
};
use ndarray::{Array3, Zip};

/// Reconstruct a sound-speed volume from ring-array frequency-domain data.
///
/// # Errors
/// Returns an error when observations, geometry, config, or model values are
/// outside the solver contract.
pub fn invert(
    observations: &[FrequencyObservation],
    array: &MultiRowRingArray,
    initial_sound_speed_m_s: &Array3<f64>,
    config: &Config,
) -> KwaversResult<InversionResult> {
    let mut slowness = sound_speed_to_slowness(initial_sound_speed_m_s)?;
    let (mut objective, mut gradient) =
        objective_and_gradient(&slowness, observations, array, config)?;
    let mut history = vec![objective];
    let mut direction = gradient.mapv(|value| -value);
    let mut previous_gradient = gradient.clone();

    for iteration in 0..config.iterations {
        if iteration > 0 {
            let mut diff = gradient.clone();
            Zip::from(&mut diff)
                .and(&previous_gradient)
                .for_each(|value, &previous| *value -= previous);
            let beta = (dot(&gradient, &diff)
                / dot(&previous_gradient, &previous_gradient).max(f64::EPSILON))
            .max(0.0);
            Zip::from(&mut direction)
                .and(&gradient)
                .for_each(|dir, &grad| *dir = -grad + beta * *dir);
            if dot(&direction, &gradient) >= 0.0 {
                direction.assign(&gradient.mapv(|value| -value));
            }
        }

        let direction_scale = max_abs(&direction);
        if direction_scale <= f64::EPSILON {
            break;
        }

        let mut accepted = None;
        for search_step in 0..8 {
            let step = config.initial_step_s_per_m * 0.5_f64.powi(search_step) / direction_scale;
            let mut candidate = slowness.clone();
            Zip::from(&mut candidate)
                .and(&direction)
                .for_each(|value, &dir| *value += step * dir);
            clamp_slowness(&mut candidate, config);
            let (candidate_objective, candidate_gradient) =
                objective_and_gradient(&candidate, observations, array, config)?;
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
        frequencies_used: observations.len(),
        transmissions_used: observations
            .first()
            .map(|obs| obs.observed_pressure.nrows())
            .unwrap_or(0),
        receivers_used: array.element_count(),
        model_family: FREQUENCY_DOMAIN_FWI_SOLVER_MODEL,
    })
}

pub(super) fn clamp_slowness(slowness: &mut Array3<f64>, config: &Config) {
    let min_slowness = 1.0 / config.max_sound_speed_m_s;
    let max_slowness = 1.0 / config.min_sound_speed_m_s;
    slowness.mapv_inplace(|value| value.clamp(min_slowness, max_slowness));
}
