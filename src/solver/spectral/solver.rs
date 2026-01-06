//! Generalized Spectral Solver Implementation
//!
//! Main solver implementation following GRASP principles.
//! This module focuses solely on the core solving logic.

use crate::error::{KwaversError, KwaversResult, ValidationError};
use crate::fft::Complex64;
use crate::grid::Grid;
use crate::medium::{ArrayAccess, CoreMedium, Medium};
use ndarray::{Array3, Zip};

use super::absorption::initialize_absorption_operators;
use super::config::{BoundaryConfig, CompatibilityMode, SpectralConfig};
use super::operators::initialize_spectral_operators;
use super::sources::{SourceHandler, SourceMode, SpectralSource};
use crate::boundary::{Boundary, CPMLBoundary, PMLBoundary};
use crate::fft::ProcessorFft3d;
use crate::recorder::{SensorData, SensorHandler};
use crate::solver::spectral_correction::CorrectionMethod;
use std::sync::Arc;

/// Core spectral solver implementing the pseudospectral method
pub struct SpectralSolver {
    pub(super) config: SpectralConfig,
    #[allow(dead_code)]
    pub(super) grid: Arc<Grid>,

    // Sensors
    pub(super) sensor_handler: SensorHandler,

    // Sources
    pub(super) source_handler: SourceHandler,
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
    pub p: Array3<f64>,   // Pressure
    pub rho: Array3<f64>, // Density perturbation
    pub ux: Array3<f64>,  // Velocity x
    pub uy: Array3<f64>,  // Velocity y
    pub uz: Array3<f64>,  // Velocity z

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
    pub(super) rho0: Array3<f64>,
    pub(super) c0: Array3<f64>,
    pub(super) bon: Array3<f64>, // Nonlinearity parameter B/A

    // Material Gradients
    pub(super) grad_rho0_x: Array3<f64>,
    pub(super) grad_rho0_y: Array3<f64>,
    pub(super) grad_rho0_z: Array3<f64>,

    // Absorption variables
    pub(super) absorb_tau: Array3<f64>,
    pub(super) absorb_eta: Array3<f64>,
}

impl std::fmt::Debug for SpectralSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpectralSolver")
            .field("config", &self.config)
            .field("grid", &"Grid { ... }")
            .field("k_max", &self.k_max)
            .finish()
    }
}

impl SpectralSolver {
    /// Get access to the sensor handler
    pub fn sensor_handler(&self) -> &SensorHandler {
        &self.sensor_handler
    }

    /// Get the time step size
    pub fn get_timestep(&self) -> f64 {
        self.config.dt
    }

    /// Compute the divergence of the particle velocity field in k-space
    pub fn compute_divergence(&mut self) -> Array3<f64> {
        let (kx, ky, kz) = &self.k_vec;

        // FFT each component to k-space
        self.fft.forward_into(&self.ux, &mut self.ux_k);
        self.fft.forward_into(&self.uy, &mut self.uy_k);
        self.fft.forward_into(&self.uz, &mut self.uz_k);

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
        let mut div = Array3::zeros(self.p.dim());
        // Use ux_k as scratch for IFFT
        self.fft.inverse_into(&self.p_k, &mut div, &mut self.ux_k);
        div
    }

    /// Compute the gradient of the pressure field in k-space
    pub fn compute_gradient(&mut self) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        let (kx, ky, kz) = &self.k_vec;

        // FFT to k-space
        self.fft.forward_into(&self.p, &mut self.p_k);

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
        let mut gx = Array3::zeros(self.p.dim());
        let mut gy = Array3::zeros(self.p.dim());
        let mut gz = Array3::zeros(self.p.dim());

        // Use ux_k as scratch
        self.fft
            .inverse_into(&self.grad_x_k, &mut gx, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_y_k, &mut gy, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_z_k, &mut gz, &mut self.ux_k);

        Ok((gx, gy, gz))
    }

    /// Create a new spectral solver
    pub fn new(
        mut config: SpectralConfig,
        grid: Grid,
        medium: &dyn Medium,
        source: SpectralSource,
    ) -> KwaversResult<Self> {
        let grid = Arc::new(grid);

        // Adjust spectral correction based on compatibility mode if not explicitly set
        if config.compatibility_mode == CompatibilityMode::KWave {
            config.spectral_correction.method = CorrectionMethod::KWave;
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
                    rho0[[i, j, k]] = crate::medium::density_at(medium, x, y, z, &grid);
                    c0[[i, j, k]] = crate::medium::sound_speed_at(medium, x, y, z, &grid);
                    bon[[i, j, k]] = crate::medium::nonlinearity_at(medium, x, y, z, &grid);
                }
            }
        }

        let shape = (grid.nx, grid.ny, grid.nz);

        let mut sensor_config = config.sensor_config.clone();
        if let Some(mask) = &config.sensor_mask {
            sensor_config.mask = mask.clone();
        }

        let mask_dim = sensor_config.mask.dim();
        if mask_dim != shape {
            if mask_dim == (1, 1, 1) && !sensor_config.mask[[0, 0, 0]] {
                sensor_config.mask = Array3::from_elem(shape, false);
            } else {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Sensor mask shape mismatch: expected {:?}, got {:?}",
                            shape, mask_dim
                        ),
                    },
                ));
            }
        }

        let sensor_handler = SensorHandler::with_expected_steps(sensor_config, shape, config.nt);

        // Initialize Source Handler
        let source_handler = SourceHandler::new(source, &grid)?;

        // Initialize solver partially to compute gradients of rho0
        let mut solver_partial = Self {
            config: config.clone(),
            grid: grid.clone(),
            sensor_handler,
            source_handler,
            time_step_index: 0,
            fft: crate::fft::get_fft_for_grid(grid.as_ref()),
            kappa,
            k_vec,
            filter: k_ops.filter,
            k_max,
            c_ref,
            boundary,
            p: field_arrays.p,
            rho: Array3::zeros(shape),
            p_k: field_arrays.p_k, // Re-used as scratch
            ux: field_arrays.ux,
            uy: field_arrays.uy,
            uz: field_arrays.uz,
            dpx: Array3::zeros(shape),
            dpy: Array3::zeros(shape),
            dpz: Array3::zeros(shape),
            div_u: Array3::zeros(shape),
            ux_k: Array3::zeros(shape),
            uy_k: Array3::zeros(shape),
            uz_k: Array3::zeros(shape),
            grad_x_k: Array3::zeros(shape),
            grad_y_k: Array3::zeros(shape),
            grad_z_k: Array3::zeros(shape),
            rho0,
            c0,
            bon,
            grad_rho0_x: Array3::zeros(shape),
            grad_rho0_y: Array3::zeros(shape),
            grad_rho0_z: Array3::zeros(shape),
            absorb_tau,
            absorb_eta,
        };

        // Apply initial conditions
        solver_partial.source_handler.apply_initial_conditions(
            &mut solver_partial.p,
            &mut solver_partial.rho,
            &solver_partial.c0,
            &mut solver_partial.ux,
            &mut solver_partial.uy,
            &mut solver_partial.uz,
        );

        // Compute gradients of rho0
        solver_partial.compute_rho0_gradients()?;

        Ok(solver_partial)
    }

    fn apply_boundary(&mut self, time_index: usize) -> KwaversResult<()> {
        let Some(boundary) = &mut self.boundary else {
            return Ok(());
        };

        boundary.apply_acoustic(self.p.view_mut(), &self.grid, time_index)?;
        boundary.apply_acoustic(self.rho.view_mut(), &self.grid, time_index)?;
        boundary.apply_acoustic(self.ux.view_mut(), &self.grid, time_index)?;
        boundary.apply_acoustic(self.uy.view_mut(), &self.grid, time_index)?;
        boundary.apply_acoustic(self.uz.view_mut(), &self.grid, time_index)?;

        Ok(())
    }

    /// Compute gradients of rho0 using spectral method
    fn compute_rho0_gradients(&mut self) -> KwaversResult<()> {
        // Use p_k as scratch for rho0_k
        self.fft.forward_into(&self.rho0, &mut self.p_k);

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
        &self.p
    }

    /// Get reference to velocity fields (ux, uy, uz)
    pub fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.ux, &self.uy, &self.uz)
    }

    /// Run the k-Wave simulation
    pub fn run(&mut self, steps: usize) -> KwaversResult<SensorData> {
        for _step in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_handler.extract_data())
    }

    /// Add a pressure source term to the mass equation (density perturbation)
    pub fn add_pressure_source(&mut self, source_term: &Array3<f64>) {
        // Source term in mass equation: S_m = source_p / c0^2
        Zip::from(&mut self.rho)
            .and(source_term)
            .and(&self.c0)
            .for_each(|rho, &s, &c| {
                *rho += s / (c * c);
            });
    }

    /// Update pressure from density using Equation of State
    pub fn update_pressure(&mut self) {
        Zip::from(&mut self.p)
            .and(&self.c0)
            .and(&self.rho)
            .and(&self.rho0)
            .and(&self.bon)
            .for_each(|p, &c, &rho, &rho0, &bon| {
                let linear = rho;
                let nonlinear = if rho0 > 0.0 {
                    0.5 * bon * rho * rho / rho0
                } else {
                    0.0
                };
                *p = c * c * (linear + nonlinear);
            });
    }

    /// Update velocity using momentum conservation
    /// d(u)/dt = -1/rho0 * grad(p)
    pub fn update_velocity(&mut self, dt: f64) -> KwaversResult<()> {
        // Compute grad(p)
        self.fft.forward_into(&self.p, &mut self.p_k);

        // Compute derivatives in k-space
        // grad_x_k = i * kx * p_k
        let (kx, ky, kz) = &self.k_vec;

        // dpx/dx
        Zip::from(&mut self.grad_x_k)
            .and(&self.p_k)
            .and(kx)
            .and(&self.kappa)
            .for_each(|g, &p, &k, &kappa| {
                *g = p * Complex64::new(0.0, k * kappa);
            });

        // dpy/dy
        Zip::from(&mut self.grad_y_k)
            .and(&self.p_k)
            .and(ky)
            .and(&self.kappa)
            .for_each(|g, &p, &k, &kappa| {
                *g = p * Complex64::new(0.0, k * kappa);
            });

        // dpz/dz
        Zip::from(&mut self.grad_z_k)
            .and(&self.p_k)
            .and(kz)
            .and(&self.kappa)
            .for_each(|g, &p, &k, &kappa| {
                *g = p * Complex64::new(0.0, k * kappa);
            });

        // Inverse transform to get gradients in physical space
        // Reuse buffers: dpx <- grad_x_k, dpy <- grad_y_k, dpz <- grad_z_k
        // We need to be careful with scratch space for IFFT.
        // We can use ux_k as scratch since we haven't computed it yet for this step?
        // No, ux_k will be used in update_density.
        // But here we are in update_velocity, ux_k is free to be overwritten.

        self.fft
            .inverse_into(&self.grad_x_k, &mut self.dpx, &mut self.ux_k);
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.dpy, &mut self.uy_k);
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.dpz, &mut self.uz_k);

        // Update velocity
        // u = u - dt/rho0 * grad(p)
        Zip::from(&mut self.ux)
            .and(&self.rho0)
            .and(&self.dpx)
            .for_each(|u, &rho0, &dp| {
                *u -= dt / rho0 * dp;
            });

        Zip::from(&mut self.uy)
            .and(&self.rho0)
            .and(&self.dpy)
            .for_each(|u, &rho0, &dp| {
                *u -= dt / rho0 * dp;
            });

        Zip::from(&mut self.uz)
            .and(&self.rho0)
            .and(&self.dpz)
            .for_each(|u, &rho0, &dp| {
                *u -= dt / rho0 * dp;
            });

        Ok(())
    }

    /// Update density using mass conservation
    /// d(rho)/dt = -rho0 * div(u) - u.grad(rho0)
    pub fn update_density(&mut self, dt: f64) -> KwaversResult<()> {
        // Compute div(u)
        self.fft.forward_into(&self.ux, &mut self.ux_k);
        self.fft.forward_into(&self.uy, &mut self.uy_k);
        self.fft.forward_into(&self.uz, &mut self.uz_k);

        let (kx, ky, kz) = &self.k_vec;

        // div_u_k = i*kx*ux_k + i*ky*uy_k + i*kz*uz_k
        // We can accumulate into grad_x_k (reused as div_u_k accumulator)

        // 1. i*kx*ux_k
        Zip::from(&mut self.grad_x_k)
            .and(&self.ux_k)
            .and(kx)
            .and(&self.kappa)
            .for_each(|res, &u, &k, &kappa| {
                *res = u * Complex64::new(0.0, k * kappa);
            });

        // 2. Add i*ky*uy_k
        Zip::from(&mut self.grad_x_k)
            .and(&self.uy_k)
            .and(ky)
            .and(&self.kappa)
            .for_each(|res, &u, &k, &kappa| {
                *res += u * Complex64::new(0.0, k * kappa);
            });

        // 3. Add i*kz*uz_k
        Zip::from(&mut self.grad_x_k)
            .and(&self.uz_k)
            .and(kz)
            .and(&self.kappa)
            .for_each(|res, &u, &k, &kappa| {
                *res += u * Complex64::new(0.0, k * kappa);
            });

        // Inverse transform div(u)
        // Reuse div_u buffer
        // Scratch: p_k (we used p_k in update_velocity, now we can reuse it)
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.div_u, &mut self.p_k);

        // Update density
        // rho = rho - dt * (rho0 * div_u + u.grad(rho0))

        // Calculate u.grad(rho0) in parts to avoid Zip limit (max 6 inputs)
        // Store in dpx (scratch)

        // 1. ux * grad_x
        Zip::from(&mut self.dpx)
            .and(&self.ux)
            .and(&self.grad_rho0_x)
            .for_each(|res, &u, &g| *res = u * g);

        // 2. Add uy * grad_y
        Zip::from(&mut self.dpx)
            .and(&self.uy)
            .and(&self.grad_rho0_y)
            .for_each(|res, &u, &g| *res += u * g);

        // 3. Add uz * grad_z
        Zip::from(&mut self.dpx)
            .and(&self.uz)
            .and(&self.grad_rho0_z)
            .for_each(|res, &u, &g| *res += u * g);

        // Final update
        Zip::from(&mut self.rho)
            .and(&self.rho0)
            .and(&self.div_u)
            .and(&self.dpx)
            .for_each(|rho, &rho0, &div, &u_dot_grad| {
                *rho -= dt * (rho0 * div + u_dot_grad);
            });

        Ok(())
    }

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
                .inject_mass_source(time_index, &mut self.rho, &self.c0);
        }

        self.update_pressure();

        self.update_velocity(dt)?;

        if self.source_handler.has_velocity_source() {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.ux,
                &mut self.uy,
                &mut self.uz,
            );
        }

        self.update_density(dt)?;

        self.apply_absorption(dt)?;

        if self.source_handler.pressure_mode() == SourceMode::Dirichlet {
            self.source_handler
                .inject_mass_source(time_index + 1, &mut self.rho, &self.c0);
        }

        self.update_pressure();

        if self.filter.is_some() {
            self.apply_anti_aliasing_filter()?;
        }

        self.apply_boundary(time_index)?;

        // Record sensor data
        self.sensor_handler
            .record_step(&self.p, &self.ux, &self.uy, &self.uz);

        // Advance time step
        self.time_step_index += 1;

        Ok(())
    }

    /// Apply anti-aliasing filter to field variables in k-space
    fn apply_anti_aliasing_filter(&mut self) -> KwaversResult<()> {
        if let Some(ref filter) = self.filter {
            // Filter Pressure
            self.fft.forward_into(&self.p, &mut self.p_k);
            Zip::from(&mut self.p_k).and(filter).for_each(|p_k, &f| {
                *p_k *= f;
            });
            self.fft
                .inverse_into(&self.p_k, &mut self.p, &mut self.ux_k);

            // Filter Velocity X
            self.fft.forward_into(&self.ux, &mut self.ux_k);
            Zip::from(&mut self.ux_k).and(filter).for_each(|u_k, &f| {
                *u_k *= f;
            });
            self.fft
                .inverse_into(&self.ux_k, &mut self.ux, &mut self.uy_k);

            // Filter Velocity Y
            self.fft.forward_into(&self.uy, &mut self.uy_k);
            Zip::from(&mut self.uy_k).and(filter).for_each(|u_k, &f| {
                *u_k *= f;
            });
            self.fft
                .inverse_into(&self.uy_k, &mut self.uy, &mut self.ux_k);

            // Filter Velocity Z
            self.fft.forward_into(&self.uz, &mut self.uz_k);
            Zip::from(&mut self.uz_k).and(filter).for_each(|u_k, &f| {
                *u_k *= f;
            });
            self.fft
                .inverse_into(&self.uz_k, &mut self.uz, &mut self.ux_k);
        }
        Ok(())
    }
}
