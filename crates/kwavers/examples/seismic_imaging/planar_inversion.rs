//! Multi-scale 2-D skull full-waveform inversion workflow.

use super::seismic_imaging::medium::SkullModel;
use super::{
    seismic_acquisition, seismic_initial_model, seismic_metrics, Array2, Array3, FwiGeometry,
    FwiParameters, FwiProcessor, Grid, KwaversResult, RegularizationParameters, DX,
    SOUND_SPEED_WATER_SIM, STEP_SIZE,
};
use std::time::Instant;

/// Value-semantic outputs produced by the planar skull inversion stage.
pub(super) struct PlanarInversionResult {
    pub(super) true_model: Array3<f64>,
    pub(super) initial_model: Array3<f64>,
    pub(super) reconstructed: Array3<f64>,
    pub(super) shots_fine: Vec<(FwiGeometry, Array2<f64>)>,
}

fn fwi_parameters(
    max_iterations: usize,
    frequency: f64,
    nt: usize,
    dt: f64,
    source_mute_radius: usize,
) -> FwiParameters {
    FwiParameters {
        max_iterations,
        frequency,
        nt,
        dt,
        n_trace: seismic_acquisition::N_RECEIVERS,
        n_depth: 1,
        step_size: STEP_SIZE,
        tolerance: 1e-12,
        regularization: RegularizationParameters {
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            directional_tv_weight: 0.0,
            directional_tv_adaptive: false,
            smoothness_weight: 0.0,
        },
        source_mute_radius,
        ..FwiParameters::default()
    }
}

/// Run multi-scale planar skull FWI and return its typed workflow artifacts.
pub(super) fn run_skull_inversion(
    phantom: &SkullModel,
    grid: &Grid,
    dt: f64,
    t_transit: f64,
    scales: &[(f64, usize)],
) -> KwaversResult<PlanarInversionResult> {
    let true_model = phantom.acoustic().sound_speed.clone();

    // Gaussian-blurred CT provides a geometrically correct, cycle-skip-safe prior.
    let initial_model = seismic_initial_model::gaussian_blur_xz(&true_model, 3.0);
    let mut current_model = initial_model.clone();

    let nt_fine = ((t_transit * 1.2 + 3.0 / seismic_acquisition::F0_HZ) / dt).ceil() as usize;
    let mut shots_fine: Vec<(FwiGeometry, Array2<f64>)> =
        Vec::with_capacity(seismic_acquisition::N_SHOTS);
    {
        let tmp_fwi = FwiProcessor::new(fwi_parameters(
            1,
            seismic_acquisition::F0_HZ,
            nt_fine,
            dt,
            4,
        ));
        let observed_at = Instant::now();
        for &element_index in &seismic_acquisition::TRANSMIT_ELEMENT_INDICES {
            let geometry = seismic_acquisition::build_shot(
                element_index,
                seismic_acquisition::F0_HZ,
                nt_fine,
                dt,
            )?;
            let observed = tmp_fwi.generate_synthetic_data(&true_model, &geometry, grid)?;
            shots_fine.push((geometry, observed));
        }
        println!(
            "  {} observed gathers at {} kHz ({:.1} s)",
            seismic_acquisition::N_SHOTS,
            seismic_acquisition::F0_HZ * 1e-3,
            observed_at.elapsed().as_secs_f32()
        );
    }

    let j_initial = {
        let fwi_tmp = FwiProcessor::new(fwi_parameters(
            1,
            seismic_acquisition::F0_HZ,
            nt_fine,
            dt,
            4,
        ));
        shots_fine
            .iter()
            .map(|(geometry, observed)| {
                let synthetic = fwi_tmp.generate_synthetic_data(&initial_model, geometry, grid)?;
                Ok::<f64, super::KwaversError>(
                    synthetic
                        .iter()
                        .zip(observed.iter())
                        .map(|(&synthetic, &observed)| (synthetic - observed).powi(2))
                        .sum::<f64>()
                        * 0.5
                        * dt,
                )
            })
            .try_fold(0.0, |total, value| value.map(|value| total + value))?
    };

    println!("\n  Quality before inversion:");
    seismic_metrics::print_quality_report(&true_model, &initial_model);
    println!(
        "  Joint J₀ (150 kHz) : {j_initial:.6e} Pa²·s  ({} shots)",
        seismic_acquisition::N_SHOTS
    );

    let inversion_at = Instant::now();
    for (scale_index, &(frequency, iterations)) in scales.iter().enumerate() {
        let nt_scale = ((t_transit * 1.2 + 3.0 / frequency) / dt).ceil() as usize;
        let mute_radius = ((SOUND_SPEED_WATER_SIM / (2.0 * frequency)) / DX).floor() as usize;
        let mute_radius = mute_radius.clamp(2, 12);
        let mut scale_shots: Vec<(FwiGeometry, Array2<f64>)> =
            Vec::with_capacity(seismic_acquisition::N_SHOTS);
        let fwi_scale = FwiProcessor::new(fwi_parameters(
            iterations,
            frequency,
            nt_scale,
            dt,
            mute_radius,
        ));

        let scale_started_at = Instant::now();
        for &element_index in &seismic_acquisition::TRANSMIT_ELEMENT_INDICES {
            let geometry = seismic_acquisition::build_shot(element_index, frequency, nt_scale, dt)?;
            let observed = fwi_scale.generate_synthetic_data(&true_model, &geometry, grid)?;
            scale_shots.push((geometry, observed));
        }
        println!(
            "\n  ── Scale {} / {} : f₀ = {:.0} kHz, {} iter, nt = {}, mute_r = {} ──",
            scale_index + 1,
            scales.len(),
            frequency * 1e-3,
            iterations,
            nt_scale,
            mute_radius
        );

        current_model = fwi_scale.invert_multi_source(&scale_shots, &current_model, grid)?;
        current_model = current_model.mapv(|velocity| velocity.max(SOUND_SPEED_WATER_SIM));
        let c_now_max = current_model
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let c_now_min = current_model.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "    Scale {} done ({:.1} s): c ∈ [{:.0}, {:.0}] m/s",
            scale_index + 1,
            scale_started_at.elapsed().as_secs_f32(),
            c_now_min,
            c_now_max
        );
    }

    let reconstructed = current_model;
    println!(
        "\n  FWI completed in {:.1} s",
        inversion_at.elapsed().as_secs_f32()
    );

    let j_final = {
        let fwi_tmp = FwiProcessor::new(fwi_parameters(
            1,
            seismic_acquisition::F0_HZ,
            nt_fine,
            dt,
            4,
        ));
        shots_fine
            .iter()
            .map(|(geometry, observed)| {
                let synthetic = fwi_tmp.generate_synthetic_data(&reconstructed, geometry, grid)?;
                Ok::<f64, super::KwaversError>(
                    synthetic
                        .iter()
                        .zip(observed.iter())
                        .map(|(&synthetic, &observed)| (synthetic - observed).powi(2))
                        .sum::<f64>()
                        * 0.5
                        * dt,
                )
            })
            .try_fold(0.0, |total, value| value.map(|value| total + value))?
    };
    let j_reduction_pct = (1.0 - j_final / j_initial) * 100.0;
    println!("\n  Quality after inversion:");
    seismic_metrics::print_quality_report(&true_model, &reconstructed);
    println!("  Joint J (150 kHz) : {j_final:.6e} Pa²·s");
    println!("  J reduction       : {j_reduction_pct:7.1} %  (150 kHz joint L2)");

    Ok(PlanarInversionResult {
        true_model,
        initial_model,
        reconstructed,
        shots_fine,
    })
}
