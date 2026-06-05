//! FWI data-misfit and regularization objective function.

use crate::therapy::theranostic_guidance::nonlinear3d::{
    forward::{forward_with_schedule, ForwardInput},
    regularization::h1_penalty,
};

use super::types::ObjectiveInput;

pub(in crate::therapy::theranostic_guidance::nonlinear3d) fn objective_for_model(
    speed: &[f64],
    beta: &[f64],
    input: ObjectiveInput<'_>,
) -> f64 {
    let data = input
        .observed
        .iter()
        .map(|shot| {
            let predicted = forward_with_schedule(ForwardInput {
                speed,
                density: input.density,
                beta,
                attenuation_np_per_m_mhz: Some(input.attenuation_np_per_m_mhz),
                attenuation_power_law_y: Some(input.attenuation_power_law_y),
                source_body_mask: Some(input.source_body_mask),
                n: input.n,
                spacing_m: input.spacing_m,
                aperture: input.aperture,
                config: input.config,
                schedule: input.schedule,
                encoding: shot.encoding,
                source_scale: input.source_scale,
                retain_history: false,
            });
            predicted
                .traces
                .iter()
                .zip(shot.traces.iter())
                .map(|(p, o)| (p - o).powi(2))
                .sum::<f64>()
        })
        .sum::<f64>();
    0.5 * data / input.observed_energy
        + h1_penalty(
            speed,
            input.background_speed,
            input.body,
            input.n,
            input.config.sound_speed_regularization,
            input.config.lesion_delta_c_m_s.abs(),
        )
        + h1_penalty(
            beta,
            input.background_beta,
            input.body,
            input.n,
            input.config.nonlinearity_regularization,
            input.config.lesion_delta_beta.abs(),
        )
}

/// Wrapper that plumbs LineSearchInput fields into ObjectiveInput and calls objective_for_model.
pub(super) fn line_search_objective(
    speed: &[f64],
    beta: &[f64],
    input: &super::types::LineSearchInput<'_>,
) -> f64 {
    objective_for_model(
        speed,
        beta,
        ObjectiveInput {
            observed: input.observed,
            observed_energy: input.observed_energy,
            density: input.density,
            attenuation_np_per_m_mhz: input.attenuation_np_per_m_mhz,
            attenuation_power_law_y: input.attenuation_power_law_y,
            source_body_mask: input.source_body_mask,
            background_speed: input.background_speed,
            background_beta: input.background_beta,
            body: input.body,
            n: input.n,
            spacing_m: input.spacing_m,
            aperture: input.aperture,
            config: input.config,
            schedule: input.schedule,
            source_scale: input.source_scale,
        },
    )
}
