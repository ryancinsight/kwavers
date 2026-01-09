//! Generalized Spectral Solver Implementation
//!
//! Main solver implementation following GRASP principles.
//! This module focuses solely on the core solving logic.

use crate::core::error::KwaversResult;
use crate::domain::boundary::{Boundary, CPMLBoundary, PMLBoundary};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::math::fft::Complex64;
use ndarray::{Array2, Array3, Zip};

use super::config::{BoundaryConfig, CompatibilityMode, PSTDConfig};
use super::numerics::operators::initialize_spectral_operators;
use super::numerics::spectral_correction::CorrectionMethod;
use super::physics::absorption::initialize_absorption_operators;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::{GridSource, SourceMode};
use crate::domain::source::{Source, SourceField};
use crate::math::fft::ProcessorFft3d;
use crate::solver::fdtd::SourceHandler;
use std::sync::Arc;

use crate::domain::field::wave::WaveFields;
use crate::domain::medium::MaterialFields;

/// Core PSTD solver implementing the pseudospectral method
pub struct PSTDSolver {
    pub(super) config: PSTDConfig,
    #[allow(dead_code)]
    pub(super) grid: Arc<Grid>,

    pub(super) sensor_recorder: SensorRecorder,

    // Sources
    pub(super) source_handler: SourceHandler,
    dynamic_sources: Vec<(Arc<dyn Source>, Array3<f64>)>,
    pub(super) time_step_index: usize,

    pub(super) fft: Arc<ProcessorFft3d>,

    // Spectral operators
    pub(super) kappa: Array3<f64>, // k-space correction
    pub(super) k_vec: (Array3<f64>, Array3<f64>, Array3<f64>), // k-vectors
    pub(super) filter: Option<Array3<f64>>, // anti-aliasing filter
    pub(super) c_ref: f64,
    pub(super) k_max: f64,

    // Boundary conditions
    pub(super) boundary: Option<Box<dyn Boundary>>,

    // Field variables (Physical Space)
    pub fields: WaveFields,
    pub rho: Array3<f64>, // Density perturbation (kept separate for PSTD specific logic)

    // Field variables (Spectral Space - Scratch)
    // Pre-allocated to avoid allocation in time loop
    pub(super) p_k: Array3<Complex64>,
    pub(super) ux_k: Array3<Complex64>,
    pub(super) uy_k: Array3<Complex64>,
    pub(super) uz_k: Array3<Complex64>,

    // Gradient Scratch Spaces
    pub(super) grad_x_k: Array3<Complex64>,
    pub(super) grad_y_k: Array3<Complex64>,
    pub(super) grad_z_k: Array3<Complex64>,

    // Physical Space Scratch (Pre-allocated)
    pub(super) dpx: Array3<f64>,
    pub(super) dpy: Array3<f64>,
    pub(super) dpz: Array3<f64>,
    pub(super) div_u: Array3<f64>,

    // Material Properties (Pre-computed)
    pub(super) materials: MaterialFields,
    pub(super) bon: Array3<f64>, // Nonlinearity parameter B/A

    // Material Gradients
    pub(super) grad_rho0_x: Array3<f64>,
    pub(super) grad_rho0_y: Array3<f64>,
    pub(super) grad_rho0_z: Array3<f64>,

    // Absorption variables
    pub(super) absorb_tau: Array3<f64>,
    pub(super) absorb_eta: Array3<f64>,
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

impl PSTDSolver {
    pub fn sensor_indices(&self) -> &[(usize, usize, usize)] {
        self.sensor_recorder.sensor_indices()
    }

    pub fn extract_pressure_data(&self) -> Option<Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    /// Get the time step size
    pub fn get_timestep(&self) -> f64 {
        self.config.dt
    }

    /// Compute the divergence of the particle velocity field in k-space
    pub fn compute_divergence(&mut self) -> Array3<f64> {
        let (kx, ky, kz) = &self.k_vec;

        // FFT each component to k-space
        self.fft.forward_into(&self.fields.ux, &mut self.ux_k);
        self.fft.forward_into(&self.fields.uy, &mut self.uy_k);
        self.fft.forward_into(&self.fields.uz, &mut self.uz_k);

        // Compute divergence in k-space: ik_x*ux + ik_y*uy + ik_z*uz
        // Split Zip because ndarray::Zip only supports up to 6 arrays
        Zip::from(&mut self.p_k)
            .and(&self.ux_k)
            .and(&self.uy_k)
            .and(&self.uz_k)
            .and(kx)
            .and(ky)
            .for_each(|div_k, &ux_k, &uy_k, &_uz_k, &kx, &ky| {
                let ikx = Complex64::new(0.0, kx);
                let iky = Complex64::new(0.0, ky);
                *div_k = ikx * ux_k + iky * uy_k;
            });

        Zip::from(&mut self.p_k)
            .and(&self.uz_k)
            .and(kz)
            .and(&self.kappa)
            .for_each(|div_k, &uz_k, &kz, &kappa| {
                let ikz = Complex64::new(0.0, kz);
                *div_k = (*div_k + ikz * uz_k) * kappa;
            });

        // Apply anti-aliasing if enabled
        if let Some(filter) = &self.filter {
            Zip::from(&mut self.p_k)
                .and(filter)
                .for_each(|p, &f| *p *= f);
        }

        // IFFT back to physical space
        let mut div = Array3::zeros(self.fields.p.dim());
        // Use ux_k as scratch for IFFT
        self.fft.inverse_into(&self.p_k, &mut div, &mut self.ux_k);
        div
    }

    /// Compute the gradient of the pressure field in k-space
    pub fn compute_gradient(&mut self) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        let (kx, ky, kz) = &self.k_vec;

        // FFT to k-space
        self.fft.forward_into(&self.fields.p, &mut self.p_k);

        // Apply k-space correction and anti-aliasing
        if let Some(filter) = &self.filter {
            Zip::from(&mut self.p_k)
                .and(filter)
                .for_each(|p, &f| *p *= f);
        }

        // Compute gradients in k-space
        Zip::from(&mut self.grad_x_k)
            .and(&self.p_k)
            .and(kx)
            .and(&self.kappa)
            .for_each(|gx, &pk, &kx, &kappa| {
                *gx = pk * Complex64::new(0.0, kx * kappa);
            });

        Zip::from(&mut self.grad_y_k)
            .and(&self.p_k)
            .and(ky)
            .and(&self.kappa)
            .for_each(|gy, &pk, &ky, &kappa| {
                *gy = pk * Complex64::new(0.0, ky * kappa);
            });

        Zip::from(&mut self.grad_z_k)
            .and(&self.p_k)
            .and(kz)
            .and(&self.kappa)
            .for_each(|gz, &pk, &kz, &kappa| {
                *gz = pk * Complex64::new(0.0, kz * kappa);
            });

        // IFFT back to physical space
        let mut gx = Array3::zeros(self.fields.p.dim());
        let mut gy = Array3::zeros(self.fields.p.dim());
        let mut gz = Array3::zeros(self.fields.p.dim());

        // Use ux_k as scratch
        self.fft
            .inverse_into(&self.grad_x_k, &mut gx, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_y_k, &mut gy, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_z_k, &mut gz, &mut self.ux_k);

        Ok((gx, gy, gz))
    }

    /// Create a new PSTD solver
    pub fn new(
        mut config: PSTDConfig,
        grid: Grid,
        medium: &dyn Medium,
        source: GridSource,
    ) -> KwaversResult<Self> {
        let grid = Arc::new(grid);

        // Adjust spectral correction based on compatibility mode if not explicitly set
        if config.compatibility_mode == CompatibilityMode::Reference {
            config.spectral_correction.method = CorrectionMethod::Treeby2010;
        }

        let (k_ops, kappa, k_max, c_ref) = initialize_spectral_operators(&config, &grid, medium)?;

        // Initialize Boundary Condition
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

        let (absorb_tau, absorb_eta) =
            initialize_absorption_operators(&config, &grid, medium, k_max, c_ref)?;

        let field_arrays = super::data::initialize_field_arrays(&grid, medium)?;

        let k_vec = (k_ops.kx, k_ops.ky, k_ops.kz);

        // Pre-compute material properties
        let mut rho0 = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut c0 = Array3::zeros((grid.nx, grid.ny, grid.nz));
        let mut bon = Array3::zeros((grid.nx, grid.ny, grid.nz));

        // Use parallel iteration if possible, otherwise standard
        // For now, simple loop is fine during init
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

        // Initialize Source Handler
        let source_handler = SourceHandler::new(source, &grid)?;

        // Initialize solver partially to compute gradients of rho0
        let mut solver_partial = Self {
            config: config.clone(),
            grid: grid.clone(),
            sensor_recorder,
            source_handler,
            dynamic_sources: Vec::new(),
            time_step_index: 0,
            fft: crate::math::fft::get_fft_for_grid(grid.nx, grid.ny, grid.nz),
            kappa,
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
            rho: Array3::zeros(shape),
            p_k: field_arrays.p_k, // Re-used as scratch
            ux_k: Array3::zeros(shape),
            uy_k: Array3::zeros(shape),
            uz_k: Array3::zeros(shape),
            grad_x_k: Array3::zeros(shape),
            grad_y_k: Array3::zeros(shape),
            grad_z_k: Array3::zeros(shape),
            dpx: Array3::zeros(shape),
            dpy: Array3::zeros(shape),
            dpz: Array3::zeros(shape),
            div_u: Array3::zeros(shape),
            materials: MaterialFields { rho0, c0 },
            bon,
            grad_rho0_x: Array3::zeros(shape),
            grad_rho0_y: Array3::zeros(shape),
            grad_rho0_z: Array3::zeros(shape),
            absorb_tau,
            absorb_eta,
        };

        // Apply initial conditions
        solver_partial.source_handler.apply_initial_conditions(
            &mut solver_partial.fields.p,
            &mut solver_partial.rho,
            &solver_partial.materials.c0,
            &mut solver_partial.fields.ux,
            &mut solver_partial.fields.uy,
            &mut solver_partial.fields.uz,
        );

        // Compute gradients of rho0
        solver_partial.compute_rho0_gradients()?;

        Ok(solver_partial)
    }

    fn apply_boundary(&mut self, time_index: usize) -> KwaversResult<()> {
        let Some(boundary) = &mut self.boundary else {
            return Ok(());
        };

        boundary.apply_acoustic(self.fields.p.view_mut(), &self.grid, time_index)?;
        boundary.apply_acoustic(self.rho.view_mut(), &self.grid, time_index)?;
        boundary.apply_acoustic(self.fields.ux.view_mut(), &self.grid, time_index)?;
        boundary.apply_acoustic(self.fields.uy.view_mut(), &self.grid, time_index)?;
        boundary.apply_acoustic(self.fields.uz.view_mut(), &self.grid, time_index)?;

        Ok(())
    }

    /// Compute gradients of rho0 using spectral method
    fn compute_rho0_gradients(&mut self) -> KwaversResult<()> {
        // Use p_k as scratch for rho0_k
        self.fft.forward_into(&self.materials.rho0, &mut self.p_k);

        let i_img = Complex64::new(0.0, 1.0);

        // Compute gradients in k-space
        // grad_rho0_k = i * k * kappa * rho0_k

        // x
        Zip::from(&mut self.grad_x_k)
            .and(&self.p_k)
            .and(&self.k_vec.0)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });

        // y
        Zip::from(&mut self.grad_y_k)
            .and(&self.p_k)
            .and(&self.k_vec.1)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });

        // z
        Zip::from(&mut self.grad_z_k)
            .and(&self.p_k)
            .and(&self.k_vec.2)
            .and(&self.kappa)
            .for_each(|grad, &rho_k, &k, &kap| {
                *grad = i_img * k * kap * rho_k;
            });

        // IFFT to get real gradients
        // Reuse ux_k as scratch
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.grad_rho0_x, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.grad_rho0_y, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.grad_rho0_z, &mut self.ux_k);

        Ok(())
    }

    /// Get reference to pressure field
    pub fn pressure_field(&self) -> &Array3<f64> {
        &self.fields.p
    }

    /// Get reference to velocity fields (ux, uy, uz)
    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.fields.ux, &self.fields.uy, &self.fields.uz)
    }

    pub fn run(&mut self, steps: usize) -> KwaversResult<Option<Array2<f64>>> {
        for _step in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }

    // Methods moved to propagator/pressure.rs and propagator/velocity.rs

    /// Perform a single time step using k-space pseudospectral method
    /// Implements:
    /// 1. Mass conservation: d(rho)/dt = -rho0 * div(u) - u.grad(rho0)
    /// 2. Momentum conservation: d(u)/dt = -1/rho0 * grad(p)
    /// 3. Equation of State: p = c0^2 * (rho + B/2A * rho^2/rho0 + ...)
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let dt = self.config.dt;
        let time_index = self.time_step_index;

        if self.source_handler.has_pressure_source() {
            self.source_handler
                .inject_mass_source(time_index, &mut self.rho, &self.materials.c0);
        }

        self.update_pressure();

        self.update_velocity(dt)?;

        if self.source_handler.has_velocity_source() {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.fields.ux,
                &mut self.fields.uy,
                &mut self.fields.uz,
            );
        }

        self.update_density(dt)?;

        self.apply_absorption(dt)?;

        if self.source_handler.pressure_mode() == SourceMode::Dirichlet {
            self.source_handler.inject_mass_source(
                time_index + 1,
                &mut self.rho,
                &self.materials.c0,
            );
        }

        self.update_pressure();

        if self.filter.is_some() {
            self.apply_anti_aliasing_filter()?;
        }

        self.apply_boundary(time_index)?;

        // Record sensor data
        self.sensor_recorder.record_step(&self.fields.p)?;

        // Advance time step
        // Advance time step
        self.apply_dynamic_sources(dt);
        self.time_step_index += 1;

        Ok(())
    }

    fn apply_dynamic_sources(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        for (source, mask) in &self.dynamic_sources {
            let amp = source.amplitude(t);
            // PSTD injection:
            // Pressure sources are mass sources: rho += val / c0^2
            // Velocity sources are force sources: u += val
            // We need to be consistent with PSTD philosophy (updates rho and u).
            // BUT SourceField::Pressure usually means "add to pressure".
            // In PSTD, p is derived from rho. So we add to rho.
            // rho += (mask * amp) / c0^2

            match source.source_type() {
                SourceField::Pressure => {
                    // Add to rho
                    Zip::from(&mut self.rho)
                        .and(mask)
                        .and(&self.materials.c0)
                        .for_each(|rho, &m, &c| *rho += (m * amp) / (c * c));
                }
                SourceField::VelocityX => {
                    Zip::from(&mut self.fields.ux)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
                SourceField::VelocityY => {
                    Zip::from(&mut self.fields.uy)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
                SourceField::VelocityZ => {
                    Zip::from(&mut self.fields.uz)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
            }
        }
    }

    /// Apply anti-aliasing filter to field variables in k-space
    fn apply_anti_aliasing_filter(&mut self) -> KwaversResult<()> {
        if let Some(ref filter) = self.filter {
            // Filter Pressure
            self.fft.forward_into(&self.fields.p, &mut self.p_k);
            Zip::from(&mut self.p_k).and(filter).for_each(|p_k, &f| {
                *p_k *= f;
            });
            self.fft
                .inverse_into(&self.p_k, &mut self.fields.p, &mut self.ux_k);

            // Filter Velocity X
            self.fft.forward_into(&self.fields.ux, &mut self.ux_k);
            Zip::from(&mut self.ux_k).and(filter).for_each(|u_k, &f| {
                *u_k *= f;
            });
            self.fft
                .inverse_into(&self.ux_k, &mut self.fields.ux, &mut self.uy_k);

            // Filter Velocity Y
            self.fft.forward_into(&self.fields.uy, &mut self.uy_k);
            Zip::from(&mut self.uy_k).and(filter).for_each(|u_k, &f| {
                *u_k *= f;
            });
            self.fft
                .inverse_into(&self.uy_k, &mut self.fields.uy, &mut self.ux_k);

            // Filter Velocity Z
            self.fft.forward_into(&self.fields.uz, &mut self.uz_k);
            Zip::from(&mut self.uz_k).and(filter).for_each(|u_k, &f| {
                *u_k *= f;
            });
            self.fft
                .inverse_into(&self.uz_k, &mut self.fields.uz, &mut self.ux_k);
        }
        Ok(())
    }

    pub(crate) fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);
        self.dynamic_sources.push((source, mask));
        Ok(())
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
        for _ in 0..num_steps {
            self.step_forward()?;
        }
        Ok(())
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
        // Compute max pressure and velocity on the fly
        let max_pressure = self.fields.p.iter().fold(0.0f64, |m, &v| m.max(v.abs()));
        let max_velocity = self
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
            max_pressure,
            max_velocity,
        }
    }

    fn supports_feature(&self, _feature: crate::solver::interface::feature::SolverFeature) -> bool {
        // PSTD features check
        true
    }

    fn enable_feature(
        &mut self,
        _feature: crate::solver::interface::feature::SolverFeature,
        _enable: bool,
    ) -> KwaversResult<()> {
        Ok(())
    }
}
