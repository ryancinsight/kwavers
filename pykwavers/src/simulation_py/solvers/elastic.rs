use kwavers::core::error::{KwaversError, KwaversResult};
use kwavers::domain::grid::Grid as KwaversGrid;

use crate::medium_py::MediumInner;
use crate::sensor_py::Sensor;
use crate::simulation_result_py::SimulationRunResult;

use super::super::Simulation;

impl Simulation {
    /// Run the elastic-wave solver.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_elastic_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        grid_source: kwavers::domain::source::GridSource,
        sensor: Option<&Sensor>,
        pml_size: Option<usize>,
        _pml_inside: bool,
        elastic_ivp_axis: Option<&str>,
        elastic_velocity_source: super::super::ElasticVelocitySource,
    ) -> KwaversResult<SimulationRunResult> {
        
        use kwavers::solver::forward::elastic::swe::{
            ElasticWaveConfig, ElasticWaveField, ElasticWaveSolver,
        };

        let grid_source_axis_suffix: Option<String> = elastic_ivp_axis.map(|s| s.to_string());
        let (nx, ny, nz) = grid.dimensions();

        let has_ivp = grid_source.p0.is_some();
        let has_vel_source = elastic_velocity_source.is_some();
        if !has_ivp && !has_vel_source {
            return Err(KwaversError::InvalidInput(
                "SolverType.Elastic requires either Source.from_initial_displacement(...) \
                 (initial-value problem) or Source.from_elastic_velocity_source(...) \
                 (driven velocity source). No source was supplied."
                    .to_string(),
            ));
        }

        let u0_opt = match grid_source.p0 {
            Some(u0) => {
                if u0.dim() != (nx, ny, nz) {
                    return Err(KwaversError::InvalidInput(format!(
                        "Elastic initial displacement shape {:?} must equal grid ({}, {}, {})",
                        u0.dim(), nx, ny, nz
                    )));
                }
                Some(u0)
            }
            None => None,
        };

        let sensor_mask: Option<ndarray::Array3<bool>> = sensor.and_then(|s| s.mask.clone());

        let elastic_vsrc_kw: Option<kwavers::solver::forward::elastic::swe::ElasticVelocitySource> =
            if let Some((mask, ux_sig, uy_sig, uz_sig, mode_str)) = elastic_velocity_source {
                if mask.dim() != (nx, ny, nz) {
                    return Err(KwaversError::InvalidInput(format!(
                        "Elastic velocity-source mask shape {:?} must equal grid ({}, {}, {})",
                        mask.dim(), nx, ny, nz
                    )));
                }
                let validate_signal = |sig: &Option<ndarray::Array1<f64>>, name: &str| -> KwaversResult<()> {
                    if let Some(s) = sig {
                        if s.len() != time_steps {
                            return Err(KwaversError::InvalidInput(format!(
                                "Elastic velocity-source {} signal length {} must equal time_steps {}",
                                name, s.len(), time_steps
                            )));
                        }
                    }
                    Ok(())
                };
                validate_signal(&ux_sig, "ux")?;
                validate_signal(&uy_sig, "uy")?;
                validate_signal(&uz_sig, "uz")?;
                let kw_mode = match mode_str.as_str() {
                    "dirichlet" => kwavers::solver::forward::elastic::swe::ElasticVelocitySourceMode::Dirichlet,
                    _ => kwavers::solver::forward::elastic::swe::ElasticVelocitySourceMode::Additive,
                };
                Some(kwavers::solver::forward::elastic::swe::ElasticVelocitySource {
                    mask, ux_signal: ux_sig, uy_signal: uy_sig, uz_signal: uz_sig, mode: kw_mode,
                })
            } else {
                None
            };

        let pml_thickness = pml_size.unwrap_or(10);
        let config = ElasticWaveConfig {
            time_step: dt,
            simulation_time: dt * (time_steps as f64),
            pml_thickness,
            save_every: 1,
            sensor_mask,
            velocity_source: elastic_vsrc_kw,
            ..ElasticWaveConfig::default()
        };

        let medium_ref: &dyn kwavers::domain::medium::traits::Medium = medium.as_medium();
        let mut solver = ElasticWaveSolver::new(grid, medium_ref, config)?;

        let mut initial_field = ElasticWaveField::new(nx, ny, nz);
        if let Some(u0) = u0_opt {
            let axis_suffix = grid_source_axis_suffix.as_deref().unwrap_or("z");
            match axis_suffix {
                "x" => initial_field.ux.assign(&u0),
                "y" => initial_field.uy.assign(&u0),
                _ => initial_field.uz.assign(&u0),
            }
        }

        let duration = dt * (time_steps as f64);
        let _final_field = solver.propagate(&initial_field, duration, None)?;

        let recorded_p = solver.extract_recorded_data();
        let sensor_data = recorded_p.unwrap_or_else(|| ndarray::Array2::zeros((1, 0)));
        let (ux_data, uy_data, uz_data) = solver.extract_recorded_velocity_components();

        Ok(SimulationRunResult {
            sensor_data, stats: None, ux_data, uy_data, uz_data,
            ix_data: None, iy_data: None, iz_data: None,
            i_avg_x: None, i_avg_y: None, i_avg_z: None,
            velocity_stats: None, full_grid_stats: None,
        })
    }

    /// Run the pseudospectral elastic path (`SolverType::ElasticPSTD`).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_elastic_pstd_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        sensor: Option<&Sensor>,
        elastic_velocity_source: super::super::ElasticVelocitySource,
    ) -> KwaversResult<SimulationRunResult> {

        use kwavers::solver::forward::pstd::extensions::{
            ElasticPstdMedium, ElasticPstdOrchestrator, ElasticPstdSourceMode,
            ElasticPstdVelocitySource,
        };

        let medium_ref: &dyn kwavers::domain::medium::traits::Medium = medium.as_medium();
        let lame_lambda = medium_ref.lame_lambda_array();
        let lame_mu = medium_ref.lame_mu_array();
        let density = medium_ref.density_array().to_owned();

        let pstd_medium = ElasticPstdMedium { lame_lambda, lame_mu, density };
        let mut orch = ElasticPstdOrchestrator::new(grid, pstd_medium, dt)?;

        let source = elastic_velocity_source.map(|(mask, ux, uy, uz, mode_str)| {
            let mode = match mode_str.as_str() {
                "dirichlet" => ElasticPstdSourceMode::Dirichlet,
                _ => ElasticPstdSourceMode::Additive,
            };
            ElasticPstdVelocitySource { mask, ux, uy, uz, mode }
        });

        let sensor_mask: Option<ndarray::Array3<bool>> = sensor.and_then(|s| s.mask.clone());
        let recorded = orch.propagate(time_steps, source.as_ref(), sensor_mask.as_ref())?;

        let sensor_data = recorded
            .vz
            .clone()
            .unwrap_or_else(|| ndarray::Array2::zeros((1, 0)));

        Ok(SimulationRunResult {
            sensor_data, stats: None,
            ux_data: recorded.vx, uy_data: recorded.vy, uz_data: recorded.vz,
            ix_data: None, iy_data: None, iz_data: None,
            i_avg_x: None, i_avg_y: None, i_avg_z: None,
            velocity_stats: None, full_grid_stats: None,
        })
    }
}
