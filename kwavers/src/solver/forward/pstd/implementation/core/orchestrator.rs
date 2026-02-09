//! PSTD Solver Orchestrator

use crate::core::error::KwaversResult;
use crate::domain::boundary::Boundary;
use crate::domain::boundary::{CPMLBoundary, PMLBoundary};
use crate::domain::field::wave::WaveFields;
use crate::domain::grid::Grid;
use crate::domain::medium::MaterialFields;
use crate::domain::medium::Medium;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::GridSource;
use crate::domain::source::Source;
use crate::domain::source::SourceInjectionMode;
use crate::math::fft::{Complex64, ProcessorFft3d};
use crate::solver::fdtd::SourceHandler;
use crate::solver::forward::pstd::config::{
    BoundaryConfig, CompatibilityMode, KSpaceMethod, PSTDConfig,
};
use crate::solver::forward::pstd::implementation::k_space::{PSTDKSGrid, PSTDKSOperators};
use crate::solver::forward::pstd::numerics::operators::initialize_spectral_operators;
use crate::solver::forward::pstd::numerics::spectral_correction::CorrectionMethod;
use crate::solver::forward::pstd::physics::absorption::initialize_absorption_operators;
use crate::solver::forward::pstd::utils::compute_k_magnitude;
use ndarray::{Array2, Array3, Zip};
use std::sync::Arc;
use tracing::debug;

/// Core PSTD solver implementing the pseudospectral method
pub struct PSTDSolver {
    pub(crate) config: PSTDConfig,
    pub(crate) grid: Arc<Grid>,
    pub(crate) sensor_recorder: SensorRecorder,
    pub(crate) source_handler: SourceHandler,
    pub(crate) dynamic_sources: Vec<(Arc<dyn Source>, Array3<f64>)>,
    pub(crate) source_injection_modes: Vec<SourceInjectionMode>,
    pub(crate) time_step_index: usize,
    pub(crate) fft: Arc<ProcessorFft3d>,
    pub(crate) kappa: Array3<f64>,
    pub(crate) source_kappa: Array3<f64>,
    pub(crate) k_vec: (Array3<f64>, Array3<f64>, Array3<f64>),
    pub(crate) filter: Option<Array3<f64>>,
    pub(crate) c_ref: f64,
    pub(crate) k_max: f64,
    pub(crate) boundary: Option<Box<dyn Boundary>>,
    pub fields: WaveFields,
    pub rhox: Array3<f64>,
    pub rhoy: Array3<f64>,
    pub rhoz: Array3<f64>,
    pub(crate) p_k: Array3<Complex64>,
    pub(crate) ux_k: Array3<Complex64>,
    pub(crate) uy_k: Array3<Complex64>,
    pub(crate) uz_k: Array3<Complex64>,
    pub(crate) grad_x_k: Array3<Complex64>,
    pub(crate) grad_y_k: Array3<Complex64>,
    pub(crate) grad_z_k: Array3<Complex64>,
    pub(crate) materials: MaterialFields,
    pub(crate) bon: Array3<f64>,
    pub(crate) grad_rho0_x: Array3<f64>,
    pub(crate) grad_rho0_y: Array3<f64>,
    pub(crate) grad_rho0_z: Array3<f64>,
    pub(crate) absorb_tau: Array3<f64>,
    pub(crate) absorb_eta: Array3<f64>,
    pub(crate) absorb_y: Array3<f64>, // Spatially-varying absorption exponent
    pub(crate) kspace_operators: Option<PSTDKSOperators>,
    // Temporary scratch arrays for gradient/divergence computations
    pub(crate) dpx: Array3<f64>,
    pub(crate) dpy: Array3<f64>,
    pub(crate) dpz: Array3<f64>,
    pub(crate) div_u: Array3<f64>,
}

impl PSTDSolver {
    /// Compute total density (rhox + rhoy + rhoz) into the provided buffer
    pub fn fill_rho_sum(&self, dest: &mut Array3<f64>) {
        Zip::from(dest)
            .and(&self.rhox)
            .and(&self.rhoy)
            .and(&self.rhoz)
            .for_each(|rho_sum, &rx, &ry, &rz| {
                *rho_sum = rx + ry + rz;
            });
    }

    pub fn new(
        mut config: PSTDConfig,
        grid: Grid,
        medium: &dyn Medium,
        source: GridSource,
    ) -> KwaversResult<Self> {
        let grid = Arc::new(grid);
        if config.compatibility_mode == CompatibilityMode::Reference {
            config.spectral_correction.method = CorrectionMethod::Treeby2010;
        }

        let (k_ops, kappa, k_max, c_ref) = initialize_spectral_operators(&config, &grid, medium)?;
        let k_mag = compute_k_magnitude(&k_ops.kx, &k_ops.ky, &k_ops.kz);
        let source_kappa = k_mag.mapv(|k| (0.5 * c_ref * config.dt * k).cos());

        let boundary: Option<Box<dyn Boundary>> = match &config.boundary {
            BoundaryConfig::PML(pml_config) => {
                Some(Box::new(PMLBoundary::new(pml_config.clone())?))
            }
            BoundaryConfig::CPML(cpml_config) => Some(Box::new(CPMLBoundary::new(
                cpml_config.clone(),
                &grid,
                c_ref,
            )?)),
            BoundaryConfig::None => None,
        };

        let (absorb_tau, absorb_eta, absorb_y) =
            initialize_absorption_operators(&config, &grid, medium, k_max, c_ref)?;
        let field_arrays =
            crate::solver::forward::pstd::data::initialize_field_arrays(&grid, medium)?;
        let k_vec = (k_ops.kx, k_ops.ky, k_ops.kz);

        let mut rho0 = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut c0 = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut bon = Array3::zeros((grid.nx, grid.ny, grid.nz));

        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    rho0[[i, j, k]] = crate::domain::medium::density_at(medium, x, y, z, &grid);
                    c0[[i, j, k]] = crate::domain::medium::sound_speed_at(medium, x, y, z, &grid);
                    bon[[i, j, k]] = crate::domain::medium::nonlinearity_at(medium, x, y, z, &grid);
                }
            }
        }

        let shape = (grid.nx, grid.ny, grid.nz);
        let sensor_recorder = SensorRecorder::new(config.sensor_mask.as_ref(), shape, config.nt)?;
        let source_handler = SourceHandler::new(source, &grid)?;

        let mut solver = Self {
            config,
            grid: grid.clone(),
            sensor_recorder,
            source_handler,
            dynamic_sources: Vec::new(),
            source_injection_modes: Vec::new(),
            time_step_index: 0,
            fft: crate::math::fft::get_fft_for_grid(grid.nx, grid.ny, grid.nz),
            kappa,
            source_kappa,
            k_vec,
            filter: k_ops.filter,
            k_max,
            c_ref,
            boundary,
            fields: WaveFields {
                p: field_arrays.p,
                ux: field_arrays.ux,
                uy: field_arrays.uy,
                uz: field_arrays.uz,
            },
            rhox: Array3::zeros(shape),
            rhoy: Array3::zeros(shape),
            rhoz: Array3::zeros(shape),
            p_k: field_arrays.p_k,
            ux_k: Array3::zeros(shape),
            uy_k: Array3::zeros(shape),
            uz_k: Array3::zeros(shape),
            grad_x_k: Array3::zeros(shape),
            grad_y_k: Array3::zeros(shape),
            grad_z_k: Array3::zeros(shape),
            materials: MaterialFields { rho0, c0 },
            bon,
            grad_rho0_x: Array3::zeros(shape),
            grad_rho0_y: Array3::zeros(shape),
            grad_rho0_z: Array3::zeros(shape),
            absorb_tau,
            absorb_eta,
            absorb_y,
            kspace_operators: None,
            dpx: Array3::zeros(shape),
            dpy: Array3::zeros(shape),
            dpz: Array3::zeros(shape),
            div_u: Array3::zeros(shape),
        };

        if solver.config.kspace_method == KSpaceMethod::FullKSpace {
            let k_grid = PSTDKSGrid::from_spatial_grid(&grid)?;
            solver.kspace_operators = Some(PSTDKSOperators::new(k_grid));
        }

        solver.source_handler.prepare_pressure_source_scaling(
            &grid,
            &solver.materials.c0,
            solver.config.dt,
        );

        let mut rho_init = Array3::zeros(shape);
        solver.source_handler.apply_initial_conditions(
            &mut solver.fields.p,
            &mut rho_init,
            &solver.materials.c0,
            &mut solver.fields.ux,
            &mut solver.fields.uy,
            &mut solver.fields.uz,
        );

        // Split initial density across components (k-wave-python compatibility)
        // rho = rhox + rhoy + rhoz
        Zip::from(&mut solver.rhox)
            .and(&mut solver.rhoy)
            .and(&mut solver.rhoz)
            .and(&rho_init)
            .for_each(|rx, ry, rz, &rho| {
                let split = rho / 3.0;
                *rx = split;
                *ry = split;
                *rz = split;
            });
        solver.compute_rho0_gradients()?;
        Ok(solver)
    }

    pub fn sensor_indices(&self) -> &[(usize, usize, usize)] {
        self.sensor_recorder.sensor_indices()
    }
    pub fn extract_pressure_data(&self) -> Option<Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }
    pub fn get_timestep(&self) -> f64 {
        self.config.dt
    }

    pub(super) fn apply_boundary(&mut self, time_index: usize) -> KwaversResult<()> {
        let Some(boundary) = &mut self.boundary else {
            return Ok(());
        };
        // Only apply PML to pressure here; velocity (ux, uy, uz) and split-density
        // (rhox, rhoy, rhoz) are already PML-damped inside update_velocity() and
        // update_density() respectively. Applying PML again would double the decay
        // rate, causing excessive absorption near boundaries.
        boundary.apply_acoustic(self.fields.p.view_mut(), &self.grid, time_index)?;
        Ok(())
    }

    pub(super) fn compute_rho0_gradients(&mut self) -> KwaversResult<()> {
        self.fft.forward_into(&self.materials.rho0, &mut self.p_k);
        let i_img = Complex64::new(0.0, 1.0);
        Zip::from(&mut self.grad_x_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });
        Zip::from(&mut self.grad_y_k)
            .and(&self.p_k)
            .and(&self.k_vec.1)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });
        Zip::from(&mut self.grad_z_k)
            .and(&self.p_k)
            .and(&self.k_vec.2)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.grad_rho0_x, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.grad_rho0_y, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.grad_rho0_z, &mut self.ux_k);
        Ok(())
    }

    pub fn pressure_field(&self) -> &Array3<f64> {
        &self.fields.p
    }
    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.fields.ux, &self.fields.uy, &self.fields.uz)
    }

    pub fn run_orchestrated(&mut self, steps: usize) -> KwaversResult<Option<Array2<f64>>> {
        for _ in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }

    pub(crate) fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);
        let mode = Self::determine_injection_mode(&mask);
        self.dynamic_sources.push((source, mask));
        self.source_injection_modes.push(mode);
        Ok(())
    }

    /// Determine source injection mode based on mask characteristics
    ///
    /// For PSTD, we always use additive mode due to the periodic nature of FFT-based methods.
    /// Dirichlet boundary conditions cannot be properly enforced with spectral methods.
    fn determine_injection_mode(mask: &Array3<f64>) -> SourceInjectionMode {
        let shape = mask.dim();
        let mut num_active = 0;
        let mut first_i = None;
        let mut first_j = None;
        let mut first_k = None;
        let mut all_same_i = true;
        let mut all_same_j = true;
        let mut all_same_k = true;
        let mut mask_sum = 0.0;
        let mut mask_min = f64::MAX;
        let mut mask_max = f64::MIN;

        for ((i, j, k), &m) in mask.indexed_iter() {
            if m.abs() > 1e-12 {
                num_active += 1;
                mask_sum += m;
                mask_min = mask_min.min(m);
                mask_max = mask_max.max(m);

                if let Some(fi) = first_i {
                    if fi != i {
                        all_same_i = false;
                    }
                } else {
                    first_i = Some(i);
                }

                if let Some(fj) = first_j {
                    if fj != j {
                        all_same_j = false;
                    }
                } else {
                    first_j = Some(j);
                }

                if let Some(fk) = first_k {
                    if fk != k {
                        all_same_k = false;
                    }
                } else {
                    first_k = Some(k);
                }
            }
        }

        // Check if it's a boundary plane
        let is_boundary_plane = (all_same_i
            && (first_i == Some(0) || first_i == Some(shape.0 - 1)))
            || (all_same_j && (first_j == Some(0) || first_j == Some(shape.1 - 1)))
            || (all_same_k && (first_k == Some(0) || first_k == Some(shape.2 - 1)));

        // For boundary planes (plane waves), use scale=1.0 (no normalization)
        // For point/volume sources, normalize by number of active points
        let scale = if is_boundary_plane {
            1.0 // Plane wave: each boundary point gets full amplitude
        } else if num_active > 0 {
            1.0 / (num_active as f64) // Point/volume source: normalize
        } else {
            1.0
        };

        debug!(
            num_active,
            mask_sum,
            mask_min,
            mask_max,
            is_boundary_plane,
            scale,
            "PSTD source injection mode determined"
        );
        debug!(
            "  Mask geometry: all_same_i={}, all_same_j={}, all_same_k={}, first_i={:?}, first_j={:?}, first_k={:?}",
            all_same_i, all_same_j, all_same_k, first_i, first_j, first_k
        );

        SourceInjectionMode::Additive { scale }
    }
}

impl std::fmt::Debug for PSTDSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PSTDSolver")
            .field("config", &self.config)
            .field("grid", &"Grid { ... }")
            .field("k_max", &self.k_max)
            .finish()
    }
}

impl crate::solver::interface::Solver for PSTDSolver {
    fn name(&self) -> &str {
        "PSTD"
    }
    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        Ok(())
    }
    fn add_source(&mut self, source: Box<dyn crate::domain::source::Source>) -> KwaversResult<()> {
        self.add_source_arc(Arc::from(source))
    }
    fn add_sensor(&mut self, _sensor: &crate::domain::sensor::GridSensorSet) -> KwaversResult<()> {
        Ok(())
    }
    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        self.run_orchestrated(num_steps).map(|_| ())
    }
    fn pressure_field(&self) -> &ndarray::Array3<f64> {
        &self.fields.p
    }
    fn velocity_fields(
        &self,
    ) -> (
        &ndarray::Array3<f64>,
        &ndarray::Array3<f64>,
        &ndarray::Array3<f64>,
    ) {
        (&self.fields.ux, &self.fields.uy, &self.fields.uz)
    }
    fn statistics(&self) -> crate::solver::interface::SolverStatistics {
        let max_p = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let max_v = self
            .fields
            .ux
            .iter()
            .chain(self.fields.uy.iter())
            .chain(self.fields.uz.iter())
            .fold(0.0f64, |m, &v| m.max(v.abs()));
        crate::solver::interface::SolverStatistics {
            total_steps: self.time_step_index,
            current_step: self.time_step_index,
            computation_time: std::time::Duration::default(),
            memory_usage: 0,
            max_pressure: max_p,
            max_velocity: max_v,
        }
    }
    fn supports_feature(&self, _feature: crate::solver::feature::SolverFeature) -> bool {
        true
    }
    fn enable_feature(
        &mut self,
        _feature: crate::solver::feature::SolverFeature,
        _enable: bool,
    ) -> KwaversResult<()> {
        Ok(())
    }
}
