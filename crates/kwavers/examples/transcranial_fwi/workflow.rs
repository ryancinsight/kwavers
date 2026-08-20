//! Typed execution workflow for the transcranial FWI example.

use super::acquisition::{build_shot, SOURCE_POSITIONS};
use super::config::GridSpec;
use super::metrics::print_quality_report;
use super::phantom::{build_skull_phantom, load_ct_slice};
use super::seismic_input::SeismicInputMode;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_solver::inverse::fwi::time_domain::FwiProcessor;
use kwavers_solver::inverse::seismic::parameters::{FwiParameters, RegularizationParameters};
use leto::{Array2, Array3};
use std::time::Instant;

/// Execute the complete synthetic-or-CT transcranial FWI workflow.
pub(crate) fn run() -> KwaversResult<()> {
    env_logger::init_from_env(env_logger::Env::default().default_filter_or("warn"));

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║   Transcranial Full-Wave Inversion (FWI) — kwavers       ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");

    println!("[ 1 / 5 ]  Building skull phantom …");
    let input_mode = SeismicInputMode::from_env("KWAVERS_SEISMIC_INPUT_MODE")
        .map_err(KwaversError::InvalidInput)?;
    let phantom = match input_mode {
        SeismicInputMode::Synthetic => {
            println!("  Input mode       : synthetic analytical skull");
            build_skull_phantom()?
        }
        SeismicInputMode::Ct(path) => {
            println!("  Input mode       : CT {}", path.display());
            let path = path.to_str().ok_or_else(|| {
                KwaversError::InvalidInput("CT input path is not valid UTF-8".to_owned())
            })?;
            load_ct_slice(path, "", 0)?
        }
        SeismicInputMode::CtMri { .. } => {
            return Err(KwaversError::InvalidInput(
                "transcranial_fwi accepts synthetic or ct:<path> input only".to_owned(),
            ));
        }
    };

    let sound_speed = &phantom.acoustic().sound_speed;
    let c_min = sound_speed.iter().copied().fold(f64::INFINITY, f64::min);
    let c_max = sound_speed
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let hu_min = phantom.hu().iter().copied().fold(f64::INFINITY, f64::min);
    let hu_max = phantom
        .hu()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  Grid            : {}×{}×{} voxels @ {:.0} mm",
        GridSpec::NX,
        GridSpec::NY,
        GridSpec::NZ,
        GridSpec::DX * 1e3
    );
    println!(
        "  Domain          : {:.0}×{:.0} mm",
        GridSpec::NX as f64 * GridSpec::DX * 1e3,
        GridSpec::NZ as f64 * GridSpec::DX * 1e3
    );
    println!("  HU range        : [{hu_min:.0}, {hu_max:.0}]");
    println!("  Sound-speed     : [{c_min:.0}, {c_max:.0}] m/s");

    println!("\n[ 2 / 5 ]  Constructing computational grid …");
    let grid = Grid::new(
        GridSpec::NX,
        GridSpec::NY,
        GridSpec::NZ,
        GridSpec::DX,
        GridSpec::DX,
        GridSpec::DX,
    )?;
    println!("  Grid OK");

    println!("\n[ 3 / 5 ]  Configuring FWI parameters …");
    let dt = 0.3 * GridSpec::DX / (c_max * 3.0_f64.sqrt());
    let frequency = 150_000.0;
    let transit_time = GridSpec::NX as f64 * GridSpec::DX / SOUND_SPEED_WATER_SIM;
    let nt = ((transit_time * 1.2) / dt).ceil() as usize;
    let fwi_params = FwiParameters {
        max_iterations: 5,
        tolerance: 1e-12,
        step_size: 50.0,
        frequency,
        nt,
        dt,
        n_trace: 8,
        n_depth: 1,
        regularization: RegularizationParameters {
            tikhonov_weight: 0.0,
            tv_weight: 0.0,
            directional_tv_weight: 0.0,
            directional_tv_adaptive: false,
            smoothness_weight: 0.0,
        },
        source_mute_radius: 0,
        ..FwiParameters::default()
    };
    println!("  dt              : {:.2} ns", dt * 1e9);
    println!("  f₀              : {:.0} kHz", frequency * 1e-3);
    println!(
        "  nt              : {nt} steps  ({:.1} μs)",
        nt as f64 * dt * 1e6
    );
    println!("  step_size       : {:.1} m/s", fwi_params.step_size);
    println!("  FWI iterations  : {}", fwi_params.max_iterations);

    println!("\n[ 4 / 5 ]  Building hemispherical acquisition geometry …");
    let n_shots = SOURCE_POSITIONS.len();
    println!(
        "  {} sources on left-hemisphere arc (θ = ±20°, ±60°)",
        n_shots
    );
    println!(
        "  Receivers : 8-element array at x={} (right bath, z={}-{})",
        GridSpec::NX - 5,
        GridSpec::NZ / 2 - 3,
        GridSpec::NZ / 2 + 4
    );
    for (shot, &(source_x, source_z)) in SOURCE_POSITIONS.iter().enumerate() {
        println!("  Shot {shot:1}: source at (x={source_x:2}, y=0, z={source_z:2})");
    }

    println!("\n[ 5 / 5 ]  Running transcranial FWI …");
    let fwi = FwiProcessor::new(fwi_params.clone());
    let true_model = phantom.acoustic().sound_speed.clone();
    println!("\n  ── Forward models (true skull, {n_shots} shots) ──");
    let start = Instant::now();
    let mut shots = Vec::with_capacity(n_shots);
    for &(source_x, source_z) in &SOURCE_POSITIONS {
        let geometry = build_shot(source_x, source_z, nt, dt, frequency)?;
        let observed = fwi.generate_synthetic_data(&true_model, &geometry, &grid)?;
        shots.push((geometry, observed));
    }
    println!(
        "  {n_shots} observed data arrays ({:.1} s total)",
        start.elapsed().as_secs_f32()
    );

    let initial_model = Array3::from_elem(
        (GridSpec::NX, GridSpec::NY, GridSpec::NZ),
        SOUND_SPEED_WATER_SIM,
    );
    println!(
        "\n  ── FWI inversion ({} iterations, {n_shots} shots) ──",
        fwi_params.max_iterations
    );
    println!("  Initial model: homogeneous water, c = {SOUND_SPEED_WATER_SIM} m/s");

    let initial_objective = objective(&fwi, &initial_model, &shots, &grid, dt)?;
    println!("\n  Quality before inversion:");
    print_quality_report(&true_model, &initial_model);
    println!("  Joint J₀        : {initial_objective:.6e} Pa²·s  ({n_shots} shots)");

    let inversion_start = Instant::now();
    let reconstructed = fwi.invert_multi_source(&shots, &initial_model, &grid)?;
    println!(
        "\n  FWI completed in {:.1} s",
        inversion_start.elapsed().as_secs_f32()
    );

    let final_objective = objective(&fwi, &reconstructed, &shots, &grid, dt)?;
    let reduction = (1.0 - final_objective / initial_objective) * 100.0;
    println!("\n  Quality after inversion:");
    print_quality_report(&true_model, &reconstructed);
    println!("  Joint J         : {final_objective:.6e} Pa²·s  ({n_shots} shots)");
    println!("  J reduction     : {reduction:7.1} %  (joint data-space L2 — the FWI objective)");
    println!("\n═══════════════════════════════════════════════════════════");
    println!(
        "  Reconstructed velocity range: [{:.0}, {:.0}] m/s",
        reconstructed.iter().copied().fold(f64::INFINITY, f64::min),
        reconstructed
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    );
    println!("  True velocity range         : [{c_min:.0}, {c_max:.0}] m/s");
    println!("\n  Physics verified against:");
    println!("    Aubry 2003  — HU→c,ρ BVF mixing model");
    println!("    Tarantola 1984 — adjoint-state FWI gradient");
    println!("    Virieux & Operto 2009 — FWI objective and chain rule");
    println!("\n  To use an explicit BabelBrain CT input:");
    println!("    set KWAVERS_SEISMIC_INPUT_MODE=ct:sub-001_CT.nii.gz");
    println!("    cargo run --example transcranial_fwi");
    println!("═══════════════════════════════════════════════════════════");
    Ok(())
}

fn objective(
    fwi: &FwiProcessor,
    model: &Array3<f64>,
    shots: &[(
        kwavers_solver::inverse::fwi::time_domain::FwiGeometry,
        Array2<f64>,
    )],
    grid: &Grid,
    dt: f64,
) -> KwaversResult<f64> {
    let mut total = 0.0;
    for (geometry, observed) in shots {
        let synthetic = fwi.generate_synthetic_data(model, geometry, grid)?;
        total += synthetic
            .iter()
            .zip(observed.iter())
            .map(|(&synthetic, &observed)| (synthetic - observed).powi(2))
            .sum::<f64>()
            * 0.5
            * dt;
    }
    Ok(total)
}
