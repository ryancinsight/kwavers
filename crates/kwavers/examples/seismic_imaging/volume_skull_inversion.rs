//! Multi-scale 3-D skull full-waveform inversion workflow.

use super::seismic_imaging::medium::SkullModel;
use super::{
    Array2, Array3, F0_HZ, FwiGeometry, FwiParameters, FwiProcessor, Grid, KwaversResult,
    N_RECEIVERS_3D, RegularizationParameters, SOUND_SPEED_WATER_SIM, STEP_SIZE, seismic_metrics,
    seismic_volume_acquisition,
};
use std::time::Instant;

/// Run the multi-scale skull inversion and return the reconstructed velocity.
pub(super) fn run_skull_inversion(
    phantom: &SkullModel,
    grid: &Grid,
    all_elements: &[[usize; 3]],
    transmit_indices: &[usize],
    dt: f64,
    t_transit: f64,
    scales: &[(f64, usize)],
) -> KwaversResult<Array3<f64>> {
    let true_model = phantom.acoustic().sound_speed.clone();
    let initial_model = super::seismic_volume_initial_model::gaussian_blur_3d(&true_model, 3.0);
    let mut current_model = initial_model.clone();

    let nt_fine = ((t_transit * 1.2 + 3.0 / F0_HZ) / dt).ceil() as usize;
    let mut shots_fine: Vec<(FwiGeometry, Array2<f64>)> =
        Vec::with_capacity(transmit_indices.len());
    {
        let tmp_fwi = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS_3D,
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
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let t0 = Instant::now();
        for &elem_idx in transmit_indices {
            let geom = seismic_volume_acquisition::build_shot_3d(
                all_elements[elem_idx],
                all_elements,
                elem_idx,
                F0_HZ,
                nt_fine,
                dt,
            )?;
            let obs = tmp_fwi.generate_synthetic_data(&true_model, &geom, grid)?;
            shots_fine.push((geom, obs));
        }
        println!(
            "  {} observed gathers at {} kHz ({:.1} s)",
            transmit_indices.len(),
            F0_HZ * 1e-3,
            t0.elapsed().as_secs_f32()
        );
    }

    let j_initial = {
        let fwi_tmp = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS_3D,
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
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let mut j = 0.0_f64;
        for (geom, obs) in &shots_fine {
            let d_syn = fwi_tmp.generate_synthetic_data(&initial_model, geom, grid)?;
            j += d_syn
                .iter()
                .zip(obs.iter())
                .map(|(&s, &o)| (s - o).powi(2))
                .sum::<f64>()
                * 0.5
                * dt;
        }
        j
    };

    println!("\n  Quality before inversion (all voxels):");
    seismic_metrics::print_quality_report(&true_model, &initial_model);
    println!(
        "  J₀ (150 kHz)    : {j_initial:.6e} Pa²·s  ({} shots)",
        transmit_indices.len()
    );

    let t_inv = Instant::now();
    for (scale_idx, &(f0, n_iter)) in scales.iter().enumerate() {
        let nt_scale = ((t_transit * 1.2 + 3.0 / f0) / dt).ceil() as usize;
        let mute_r =
            (((SOUND_SPEED_WATER_SIM / (2.0 * f0)) / super::DX).floor() as usize).clamp(2, 12);
        let mut scale_shots: Vec<(FwiGeometry, Array2<f64>)> =
            Vec::with_capacity(transmit_indices.len());
        let fwi_scale = FwiProcessor::new(FwiParameters {
            max_iterations: n_iter,
            frequency: f0,
            nt: nt_scale,
            dt,
            n_trace: N_RECEIVERS_3D,
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
            source_mute_radius: mute_r,
            ..FwiParameters::default()
        });

        let t_scale = Instant::now();
        for &elem_idx in transmit_indices {
            let geom = seismic_volume_acquisition::build_shot_3d(
                all_elements[elem_idx],
                all_elements,
                elem_idx,
                f0,
                nt_scale,
                dt,
            )?;
            let obs = fwi_scale.generate_synthetic_data(&true_model, &geom, grid)?;
            scale_shots.push((geom, obs));
        }
        println!(
            "\n  ── Scale {} / {} : f₀ = {:.0} kHz, {} iter, nt = {}, mute_r = {} ──",
            scale_idx + 1,
            scales.len(),
            f0 * 1e-3,
            n_iter,
            nt_scale,
            mute_r
        );

        current_model = fwi_scale.invert_multi_source(&scale_shots, &current_model, grid)?;
        current_model = current_model.mapv(|c| c.max(SOUND_SPEED_WATER_SIM));
        let c_now_max = current_model
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let c_now_min = current_model.iter().copied().fold(f64::INFINITY, f64::min);
        println!(
            "    Scale {} done ({:.1} s): c ∈ [{:.0}, {:.0}] m/s",
            scale_idx + 1,
            t_scale.elapsed().as_secs_f32(),
            c_now_min,
            c_now_max
        );
    }

    let reconstructed = current_model;
    println!(
        "\n  FWI completed in {:.1} s",
        t_inv.elapsed().as_secs_f32()
    );

    let j_final = {
        let fwi_tmp = FwiProcessor::new(FwiParameters {
            max_iterations: 1,
            frequency: F0_HZ,
            nt: nt_fine,
            dt,
            n_trace: N_RECEIVERS_3D,
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
            source_mute_radius: 4,
            ..FwiParameters::default()
        });
        let mut j = 0.0_f64;
        for (geom, obs) in &shots_fine {
            let d_syn = fwi_tmp.generate_synthetic_data(&reconstructed, geom, grid)?;
            j += d_syn
                .iter()
                .zip(obs.iter())
                .map(|(&s, &o)| (s - o).powi(2))
                .sum::<f64>()
                * 0.5
                * dt;
        }
        j
    };

    println!("\n  Quality after inversion (all voxels):");
    seismic_metrics::print_quality_report(&true_model, &reconstructed);
    println!("  J₀              : {j_initial:.6e} Pa²·s");
    println!(
        "  J_final         : {j_final:.6e} Pa²·s  (reduction: {:.1}×)",
        if j_final > 0.0 {
            j_initial / j_final
        } else {
            f64::INFINITY
        }
    );

    Ok(reconstructed)
}
