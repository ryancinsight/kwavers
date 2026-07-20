//! Imaging modality phases for the liver-assessment workflow.

use super::{CEUSResult, LiverAssessmentWorkflow, SWEResult};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_imaging::ultrasound::elastography::{InversionMethod, NonlinearInversionMethod};
use kwavers_medium::Medium;
use kwavers_physics::acoustics::imaging::modalities::ceus::CeusPerfusionModel;
use kwavers_physics::acoustics::imaging::modalities::elastography::radiation_force::PushPulseParameters;
use kwavers_physics::acoustics::imaging::modalities::elastography::{
    AcousticRadiationForce, DisplacementField, HarmonicDetectionConfig, HarmonicDetector,
};
use kwavers_solver::forward::elastic::{ElasticWaveConfig, ElasticWaveSolver};
use kwavers_solver::inverse::elastography::{
    NonlinearInversion, NonlinearInversionConfig, ShearWaveInversion, ShearWaveInversionConfig,
};
use leto::{Array3, Array4, SliceArg};

fn estimate_elastic_time_step<M: Medium>(
    grid: &Grid,
    medium: &M,
    config: &ElasticWaveConfig,
) -> f64 {
    if config.time_step > 0.0 {
        return config.time_step;
    }

    let min_dx = grid.dx.min(grid.dy).min(grid.dz);
    let mut max_c: f64 = 0.0;

    for k in 0..grid.nz {
        for j in 0..grid.ny {
            for i in 0..grid.nx {
                let rho = medium.density(i, j, k);
                if rho <= 0.0 {
                    continue;
                }

                let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                let lambda = medium.lame_lambda(x, y, z, grid);
                let mu = medium.lame_mu(x, y, z, grid);

                let cs = (mu / rho).sqrt();
                let cp = ((lambda + 2.0 * mu) / rho).sqrt();
                max_c = max_c.max(cs.max(cp));
            }
        }
    }

    if max_c <= 0.0 {
        return 0.0;
    }

    let cfl_dt = min_dx / (3.0_f64.sqrt() * max_c);
    cfl_dt * config.cfl_factor
}

pub(super) fn perform_shear_wave_elastography(
    workflow: &mut LiverAssessmentWorkflow,
) -> KwaversResult<SWEResult> {
    let push_location = [0.025, 0.025, 0.025];
    let mut arfi = AcousticRadiationForce::new(&workflow.liver_grid, &workflow.liver_tissue)?;
    arfi.set_parameters(PushPulseParameters::new(
        2.0e6,
        0.005,
        450.0,
        push_location[2],
        2.0,
    )?);
    let body_force = arfi.push_pulse_body_force(push_location)?;

    let solver_config = ElasticWaveConfig::default();
    let dt =
        estimate_elastic_time_step(&workflow.liver_grid, &workflow.liver_tissue, &solver_config);
    let solver =
        ElasticWaveSolver::new(&workflow.liver_grid, &workflow.liver_tissue, solver_config)?;
    let displacement_history =
        solver.propagate_waves_with_body_force_only_override(Some(&body_force))?;

    let last = displacement_history.last().ok_or_else(|| {
        KwaversError::InvalidInput(
            "elastic propagation must return at least one displacement field".to_string(),
        )
    })?;
    let mut displacement = DisplacementField::zeros(
        workflow.liver_grid.nx,
        workflow.liver_grid.ny,
        workflow.liver_grid.nz,
    );
    displacement.ux.assign(&last.ux);
    displacement.uy.assign(&last.uy);
    displacement.uz.assign(&last.uz);

    let config = ShearWaveInversionConfig::new(InversionMethod::TimeOfFlight);
    let inversion = ShearWaveInversion::new(config);
    let elasticity = inversion.reconstruct(&displacement, &workflow.liver_grid)?;
    let stiffness_map = elasticity.youngs_modulus.mapv(|value| (value / 1e3) as f32);

    let (nx, ny, nz) = workflow.liver_grid.dimensions();
    let mut displacement_series = Array4::<f64>::zeros((nx, ny, nz, displacement_history.len()));
    for (time_index, field) in displacement_history.iter().enumerate() {
        displacement_series
            .slice_with_mut::<3>(&[
                SliceArg::All,
                SliceArg::All,
                SliceArg::All,
                SliceArg::Index(time_index as isize),
            ])
            .map_err(|error| {
                KwaversError::InvalidInput(format!(
                    "displacement time slice {time_index} must exist: {error}"
                ))
            })?
            .assign(&field.uz);
    }

    let detector = HarmonicDetector::new(HarmonicDetectionConfig::default());
    let harmonic_field = detector.analyze_harmonics(&displacement_series, 1.0 / dt)?;
    let nonlinear_inversion = NonlinearInversion::new(NonlinearInversionConfig::new(
        NonlinearInversionMethod::HarmonicRatio,
    ));
    let nonlinear_analysis =
        nonlinear_inversion.reconstruct(&harmonic_field, &workflow.liver_grid)?;
    let fibrosis_metrics =
        workflow.calculate_fibrosis_metrics(&stiffness_map, &nonlinear_analysis)?;

    println!(
        "SWE completed: Mean stiffness = {:.1} kPa",
        fibrosis_metrics.mean_stiffness
    );
    println!("Fibrosis stage: {}", fibrosis_metrics.fibrosis_stage);

    Ok(SWEResult {
        stiffness_map,
        displacement_history,
        nonlinear_analysis,
        fibrosis_metrics,
    })
}

pub(super) fn perform_contrast_enhanced_ultrasound(
    workflow: &mut LiverAssessmentWorkflow,
) -> KwaversResult<CEUSResult> {
    let injection_profile = workflow.ceus_system.simulate_bolus_injection(5e6)?;
    let contrast_signal = workflow
        .ceus_system
        .simulate_contrast_signal(&injection_profile, 30.0)?;

    let perfusion_model = CeusPerfusionModel::gamma_variate_model();
    let perfusion_map_nd = workflow
        .ceus_system
        .estimate_perfusion(&contrast_signal, &perfusion_model)?;
    let [nx, ny, nz] = perfusion_map_nd.shape();
    let perfusion_map =
        Array3::from_shape_vec([nx, ny, nz], perfusion_map_nd.iter().copied().collect()).map_err(
            |error| {
                KwaversError::InvalidInput(format!(
                    "CEUS perfusion map must convert to Leto Array3: {error}"
                ))
            },
        )?;
    let perfusion_metrics = workflow.calculate_perfusion_metrics(&perfusion_map)?;

    println!(
        "CEUS completed: Peak enhancement = {:.1} dB",
        perfusion_metrics.peak_enhancement
    );
    println!(
        "Perfusion rate: {:.2} mL/min/100g",
        perfusion_metrics.perfusion_rate
    );

    Ok(CEUSResult {
        contrast_signal,
        perfusion_map,
        perfusion_metrics,
    })
}
