//! Stage-two 3-D brain-tissue full-waveform inversion workflow.

use super::seismic_imaging::medium::SkullModel;
use super::{
    Array2, Array3, BONE_VELOCITY_THRESHOLD, BRAIN_C_MAX, BRAIN_C_MIN, BrainPriorMode, DX,
    FwiGeometry, FwiParameters, FwiProcessor, Grid, KwaversError, KwaversResult, N_RECEIVERS_3D,
    NX, NY, NZ, R_SKULL_IN, SOUND_SPEED_WATER_SIM, seismic_metrics, seismic_volume_acquisition,
    seismic_volume_brain_model,
};
use anyhow::Context as _;
use kwavers_solver::inverse::seismic::parameters::RegularizationParameters;
use std::time::Instant;

const F0_BRAIN_HZ: f64 = 400_000.0;
const N_BRAIN_ITER: usize = 15;
const STEP_SIZE_BRAIN: f64 = 30.0;

/// Value-semantic outputs produced by the successful 3-D brain inversion stage.
pub(super) struct BrainInversionResult {
    pub(super) reconstructed: Option<Array3<f64>>,
    pub(super) t1_model: Option<Array3<f64>>,
}

/// Run the masked 3-D brain-tissue inversion from the selected prior.
pub(super) fn run_brain_inversion(
    phantom: &SkullModel,
    prior: &BrainPriorMode,
    grid: &Grid,
    dt: f64,
    all_elements: &[[usize; 3]],
    transmit_indices: &[usize],
    t1_result: Option<&(Array3<f64>, [f64; 3])>,
) -> KwaversResult<BrainInversionResult> {
    let brain_true =
        seismic_volume_brain_model::build_brain_prior_3d(phantom, prior).map_err(|error| {
            KwaversError::InvalidInput(format!("selected brain prior failed: {error:#}"))
        })?;

    let skull_mask = phantom
        .acoustic()
        .sound_speed
        .mapv(|velocity| velocity > BONE_VELOCITY_THRESHOLD);
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

    let t1_brain = match (prior, t1_result) {
        (_, Some((t1_volume, spacing))) => Some(
            seismic_volume_brain_model::build_brain_velocity_from_t1(phantom, t1_volume, *spacing),
        ),
        (BrainPriorMode::T1(path) | BrainPriorMode::MniT1 { t1: path, .. }, None) => {
            let (t1_volume, spacing) =
                seismic_volume_brain_model::load_t1_mri(path).with_context(|| {
                    format!("explicit T1 prior could not be loaded: {}", path.display())
                })?;
            Some(seismic_volume_brain_model::build_brain_velocity_from_t1(
                phantom, &t1_volume, spacing,
            ))
        }
        _ => None,
    };

    let mut brain_initial = match &t1_brain {
        Some(model) => model.clone(),
        None => skull_mask.mapv(|frozen| if frozen { 0.0 } else { SOUND_SPEED_WATER_SIM }),
    };
    let [brain_nx, brain_ny, brain_nz] = brain_initial.shape();
    for ix in 0..brain_nx {
        for iy in 0..brain_ny {
            for iz in 0..brain_nz {
                if skull_mask[[ix, iy, iz]] {
                    brain_initial[[ix, iy, iz]] = phantom.acoustic().sound_speed[[ix, iy, iz]];
                }
            }
        }
    }

    let nt_brain = {
        let domain_transit = (NX as f64 * DX) / SOUND_SPEED_WATER_SIM;
        let source_duration = 3.0 / F0_BRAIN_HZ;
        ((domain_transit + source_duration) / dt).ceil() as usize
    };
    let fwi_brain = FwiProcessor::new(FwiParameters {
        max_iterations: N_BRAIN_ITER,
        frequency: F0_BRAIN_HZ,
        nt: nt_brain,
        dt,
        n_trace: N_RECEIVERS_3D,
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
        Vec::with_capacity(transmit_indices.len());
    let observed_at = Instant::now();
    for &element_index in transmit_indices {
        let geometry = seismic_volume_acquisition::build_shot_3d(
            all_elements[element_index],
            all_elements,
            element_index,
            F0_BRAIN_HZ,
            nt_brain,
            dt,
        )?;
        match fwi_brain.generate_synthetic_data(&brain_true, &geometry, grid) {
            Ok(observed) => brain_shots.push((geometry, observed)),
            Err(error) => {
                eprintln!("  Brain gather failed for element {element_index}: {error:#}");
            }
        }
    }
    println!(
        "  {} brain gathers at {:.0} kHz ({:.1} s)",
        transmit_indices.len(),
        F0_BRAIN_HZ * 1e-3,
        observed_at.elapsed().as_secs_f32()
    );
    if brain_shots.is_empty() {
        return Err(KwaversError::InvalidInput(format!(
            "brain FWI produced no successful gathers from {} shots",
            transmit_indices.len()
        )));
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
    println!("  Quality (brain voxels only, r_3d < R_SKULL_IN):");
    print_quality_report(&brain_true, &brain_reconstructed);

    Ok(BrainInversionResult {
        reconstructed: Some(brain_reconstructed),
        t1_model: t1_brain,
    })
}

fn print_quality_report(true_model: &Array3<f64>, reconstructed: &Array3<f64>) {
    let cx = (NX / 2) as f64;
    let cy = (NY / 2) as f64;
    let cz = (NZ / 2) as f64;
    let free_pairs: Vec<(f64, f64)> = true_model
        .indexed_iter()
        .filter(|([ix, iy, iz], _)| {
            let dx = *ix as f64 - cx;
            let dy = *iy as f64 - cy;
            let dz = *iz as f64 - cz;
            (dx * dx + dy * dy + dz * dz).sqrt() < R_SKULL_IN
        })
        .map(|([ix, iy, iz], &truth)| (truth, reconstructed[[ix, iy, iz]]))
        .collect();
    seismic_metrics::print_quality_pairs(&free_pairs);
}
