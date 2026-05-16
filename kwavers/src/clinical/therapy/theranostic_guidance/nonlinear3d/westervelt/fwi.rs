use ndarray::Array3;

use super::super::types::Nonlinear3dVolume;
use super::calibration::{calibrated_source_scale, SourceCalibrationInput};
use super::types::WesterveltFwiResult;
use super::{
    add_h1_gradient, apply_line_search, forward_with_schedule, gradient, h1_penalty, index,
    metrics_from_score, multiparameter_score, objective_for_model, smooth_gradient, time_schedule,
    EncodedTrace, ForwardInput, GradientInput, LineSearchInput, LineSearchWorkspace,
    Nonlinear3dAperture, Nonlinear3dConfig, ObjectiveInput, ParameterGradient, SourceEncoding,
};

pub fn run_fwi(
    volume: &Nonlinear3dVolume,
    aperture: &Nonlinear3dAperture,
    config: &Nonlinear3dConfig,
) -> WesterveltFwiResult {
    let n = volume.body_mask.dim().0;
    let true_speed = flatten(&volume.true_sound_speed_m_s);
    let background = flatten(&volume.background_sound_speed_m_s);
    let density = flatten(&volume.density_kg_m3);
    let true_beta = flatten(&volume.true_beta);
    let background_beta = flatten(&volume.background_beta);
    let attenuation_alpha0 = flatten(&volume.attenuation_np_per_m_mhz);
    let attenuation_y = flatten(&volume.attenuation_power_law_y);
    let body = volume.body_mask.iter().copied().collect::<Vec<_>>();
    let inversion = volume.inversion_mask.iter().copied().collect::<Vec<_>>();
    let target = volume.target_mask.iter().copied().collect::<Vec<_>>();
    let schedule = time_schedule(&true_speed, n, volume.spacing_m, config);
    let encodings = SourceEncoding::all(config.source_encoding_count);
    let source_scale = calibrated_source_scale(SourceCalibrationInput {
        background_speed: &background,
        density: &density,
        attenuation_alpha0: &attenuation_alpha0,
        attenuation_y: &attenuation_y,
        target: &target,
        n,
        spacing_m: volume.spacing_m,
        aperture,
        config,
        schedule,
    });
    let mut therapy_peak = vec![0.0; n * n * n];
    let observed = encodings
        .iter()
        .copied()
        .map(|encoding| {
            let forward = forward_with_schedule(ForwardInput {
                speed: &true_speed,
                density: &density,
                beta: &true_beta,
                attenuation_np_per_m_mhz: Some(&attenuation_alpha0),
                attenuation_power_law_y: Some(&attenuation_y),
                n,
                spacing_m: volume.spacing_m,
                aperture,
                config,
                schedule,
                encoding,
                source_scale,
                retain_history: false,
            });
            if encoding.index == 0 {
                therapy_peak.copy_from_slice(&forward.peak_pressure);
            }
            EncodedTrace {
                encoding,
                traces: forward.traces,
            }
        })
        .collect::<Vec<_>>();
    let observed_energy = observed
        .iter()
        .flat_map(|shot| shot.traces.iter())
        .map(|value| value * value)
        .sum::<f64>()
        .max(1.0e-24);
    let mut current = background.clone();
    let mut current_beta = background_beta.clone();
    let mut objective_history = Vec::with_capacity(config.iterations + 1);
    let cells = n * n * n;
    let mut residual = vec![0.0; schedule.time_steps * aperture.receivers.len()];
    let mut line_search_workspace = LineSearchWorkspace::new(cells);
    for _ in 0..config.iterations {
        let mut data_objective = 0.0;
        let mut grad = ParameterGradient {
            sound_speed: vec![0.0; cells],
            beta: vec![0.0; cells],
        };
        for shot in &observed {
            let predicted = forward_with_schedule(ForwardInput {
                speed: &current,
                density: &density,
                beta: &current_beta,
                attenuation_np_per_m_mhz: Some(&attenuation_alpha0),
                attenuation_power_law_y: Some(&attenuation_y),
                n,
                spacing_m: volume.spacing_m,
                aperture,
                config,
                schedule,
                encoding: shot.encoding,
                source_scale,
                retain_history: true,
            });
            debug_assert_eq!(predicted.traces.len(), residual.len());
            residual
                .iter_mut()
                .zip(predicted.traces.iter())
                .zip(shot.traces.iter())
                .for_each(|((dst, p), o)| *dst = p - o);
            data_objective += residual.iter().map(|value| value * value).sum::<f64>();
            let history = predicted
                .history
                .as_ref()
                .expect("FWI forward pass must retain pressure history");
            let shot_grad = gradient(GradientInput {
                history,
                cells,
                residual: &residual,
                speed: &current,
                density: &density,
                beta: &current_beta,
                attenuation_np_per_m_mhz: Some(&attenuation_alpha0),
                attenuation_power_law_y: Some(&attenuation_y),
                body: &inversion,
                n,
                spacing_m: volume.spacing_m,
                aperture,
                config,
                schedule,
                encoding: shot.encoding,
                source_scale,
                dt: predicted.dt_s,
                observed_energy,
            });
            accumulate_gradient(&mut grad, &shot_grad);
        }
        let objective = 0.5 * data_objective / observed_energy
            + h1_penalty(
                &current,
                &background,
                &inversion,
                n,
                config.sound_speed_regularization,
                config.lesion_delta_c_m_s.abs(),
            )
            + h1_penalty(
                &current_beta,
                &background_beta,
                &inversion,
                n,
                config.nonlinearity_regularization,
                config.lesion_delta_beta.abs(),
            );
        objective_history.push(objective);
        add_h1_gradient(
            &mut grad.sound_speed,
            &current,
            &background,
            &inversion,
            n,
            config.sound_speed_regularization,
            config.lesion_delta_c_m_s.abs(),
        );
        add_h1_gradient(
            &mut grad.beta,
            &current_beta,
            &background_beta,
            &inversion,
            n,
            config.nonlinearity_regularization,
            config.lesion_delta_beta.abs(),
        );
        smooth_gradient(
            &mut grad.sound_speed,
            &inversion,
            n,
            config.gradient_smoothing_steps,
        );
        smooth_gradient(
            &mut grad.beta,
            &inversion,
            n,
            config.gradient_smoothing_steps,
        );
        if !apply_line_search(LineSearchInput {
            current_speed: &mut current,
            current_beta: &mut current_beta,
            workspace: &mut line_search_workspace,
            background_speed: &background,
            background_beta: &background_beta,
            body: &inversion,
            grad_speed: &grad.sound_speed,
            grad_beta: &grad.beta,
            objective,
            observed: &observed,
            observed_energy,
            density: &density,
            attenuation_np_per_m_mhz: &attenuation_alpha0,
            attenuation_power_law_y: &attenuation_y,
            n,
            spacing_m: volume.spacing_m,
            aperture,
            config,
            schedule,
            source_scale,
        }) {
            break;
        }
    }
    objective_history.push(objective_for_model(
        &current,
        &current_beta,
        ObjectiveInput {
            observed: &observed,
            observed_energy,
            density: &density,
            attenuation_np_per_m_mhz: &attenuation_alpha0,
            attenuation_power_law_y: &attenuation_y,
            background_speed: &background,
            background_beta: &background_beta,
            body: &inversion,
            n,
            spacing_m: volume.spacing_m,
            aperture,
            config,
            schedule,
            source_scale,
        },
    ));
    let delta = current
        .iter()
        .zip(background.iter())
        .map(|(c, b)| c - b)
        .collect::<Vec<_>>();
    let delta_beta = current_beta
        .iter()
        .zip(background_beta.iter())
        .map(|(b, b0)| b - b0)
        .collect::<Vec<_>>();
    let score = multiparameter_score(&delta, &delta_beta, &inversion, config, n);
    let score_vec = score.iter().copied().collect::<Vec<_>>();
    WesterveltFwiResult {
        reconstructed_sound_speed_m_s: unflatten(&current, n),
        reconstructed_delta_c_m_s: unflatten(&delta, n),
        reconstructed_beta: unflatten(&current_beta, n),
        reconstructed_delta_beta: unflatten(&delta_beta, n),
        multiparameter_fwi_score: score,
        peak_pressure_pa: unflatten(&therapy_peak, n),
        objective_history,
        metrics: metrics_from_score(&score_vec, &target, &body),
        dt_s: schedule.dt_s,
        time_steps: schedule.time_steps,
    }
}

fn accumulate_gradient(total: &mut ParameterGradient, shot: &ParameterGradient) {
    for (dst, src) in total.sound_speed.iter_mut().zip(shot.sound_speed.iter()) {
        *dst += src;
    }
    for (dst, src) in total.beta.iter_mut().zip(shot.beta.iter()) {
        *dst += src;
    }
}

fn flatten(values: &Array3<f64>) -> Vec<f64> {
    values.iter().copied().collect()
}

fn unflatten(values: &[f64], n: usize) -> Array3<f64> {
    Array3::from_shape_fn((n, n, n), |(x, y, z)| values[index(x, y, z, n)])
}
