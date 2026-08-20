//! Stage-two brain-tissue FWI execution for the planar workflow.

use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::{FwiGeometry, FwiProcessor};
use kwavers_solver::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use leto::{Array2, Array3};
use std::time::Instant;

use super::brain_prior::BrainPriorMode;
use super::seismic_acquisition;
use super::seismic_brain_model;
use super::seismic_imaging::medium::SkullModel;
use super::{BRAIN_C_MAX, BRAIN_C_MIN, DX, F0_BRAIN_HZ, N_BRAIN_ITER, NX, STEP_SIZE_BRAIN};

/// Value-semantic outputs produced by the successful brain inversion stage.
pub(super) struct BrainInversionResult {
    pub(super) true_model: Array3<f64>,
    pub(super) reconstructed: Array3<f64>,
}

/// Run the masked brain-tissue FWI stage from the selected prior.
pub(super) fn run_brain_fwi(
    phantom: &SkullModel,
    prior_mode: &BrainPriorMode,
    grid: &Grid,
    dt: f64,
) -> KwaversResult<BrainInversionResult> {
    let brain_true =
        seismic_brain_model::build_brain_prior(phantom, prior_mode).map_err(|error| {
            KwaversError::InvalidInput(format!("selected brain prior failed: {error:#}"))
        })?;

    let skull_mask = seismic_brain_model::build_skull_mask(&phantom.acoustic().sound_speed);
    let n_frozen = skull_mask.iter().filter(|&&frozen| frozen).count();
    let n_free = skull_mask.len() - n_frozen;
    println!("  Skull mask        : {n_frozen} frozen bone voxels, {n_free} free brain voxels");

    let (brain_min, brain_max) = skull_mask
        .indexed_iter()
        .filter(|(_, &frozen)| !frozen)
        .map(|([ix, iy, iz], _)| brain_true[[ix, iy, iz]])
        .fold(
            (f64::INFINITY, f64::NEG_INFINITY),
            |(min, max), velocity| (min.min(velocity), max.max(velocity)),
        );
    println!("  True brain c      : [{brain_min:.1}, {brain_max:.1}] m/s");

    let nt_brain = {
        let domain_transit_s = (NX as f64 * DX) / SOUND_SPEED_WATER_SIM;
        let source_duration_s = 3.0 / F0_BRAIN_HZ;
        ((domain_transit_s + source_duration_s) / dt).ceil() as usize
    };
    let fwi_brain = FwiProcessor::new(FwiParameters {
        max_iterations: N_BRAIN_ITER,
        frequency: F0_BRAIN_HZ,
        nt: nt_brain,
        dt,
        n_trace: seismic_acquisition::N_RECEIVERS,
        n_depth: 1,
        step_size: STEP_SIZE_BRAIN,
        tolerance: 1e-14,
        regularization: RegularizationParameters {
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            directional_tv_weight: 0.0,
            directional_tv_adaptive: false,
            smoothness_weight: 0.0,
        },
        source_mute_radius: 2,
        ..FwiParameters::default()
    });

    let mut brain_shots: Vec<(FwiGeometry, Array2<f64>)> =
        Vec::with_capacity(seismic_acquisition::N_SHOTS);
    let observed_at = Instant::now();
    for &element_index in &seismic_acquisition::TRANSMIT_ELEMENT_INDICES {
        let geom = seismic_acquisition::build_shot(element_index, F0_BRAIN_HZ, nt_brain, dt)?;
        match fwi_brain.generate_synthetic_data(&brain_true, &geom, grid) {
            Ok(observed) => brain_shots.push((geom, observed)),
            Err(error) => {
                eprintln!("  Brain gather failed for element {element_index}: {error:#}");
            }
        }
    }
    println!(
        "  {} brain gathers at {:.0} kHz ({:.1} s)",
        seismic_acquisition::N_SHOTS,
        F0_BRAIN_HZ * 1e-3,
        observed_at.elapsed().as_secs_f32()
    );
    if brain_shots.is_empty() {
        return Err(KwaversError::InvalidInput(format!(
            "brain FWI produced no successful gathers from {} shots",
            seismic_acquisition::N_SHOTS
        )));
    }

    let mut brain_initial =
        skull_mask.mapv(|frozen| if frozen { 0.0 } else { SOUND_SPEED_WATER_SIM });
    let [nx, ny, nz] = brain_initial.shape();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                if skull_mask[[i, j, k]] {
                    brain_initial[[i, j, k]] = phantom.acoustic().sound_speed[[i, j, k]];
                }
            }
        }
    }

    println!(
        "  Running {N_BRAIN_ITER} iterations at {:.0} kHz (nt={nt_brain}) …",
        F0_BRAIN_HZ * 1e-3
    );
    let inversion_at = Instant::now();
    let brain_reconstructed = fwi_brain
        .invert_multi_source_masked(
            &brain_shots,
            &brain_initial,
            &phantom.acoustic().sound_speed,
            &skull_mask,
            BRAIN_C_MIN,
            BRAIN_C_MAX,
            grid,
        )
        .map_err(|error| {
            KwaversError::InvalidInput(format!("brain FWI inversion failed: {error:#}"))
        })?;
    println!(
        "  Brain FWI done ({:.1} s)",
        inversion_at.elapsed().as_secs_f32()
    );

    Ok(BrainInversionResult {
        true_model: brain_true,
        reconstructed: brain_reconstructed,
    })
}
