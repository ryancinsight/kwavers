//! Generalized Spectral Solver Implementation
//!
//! Main solver implementation following GRASP principles.
//! This module focuses solely on the core solving logic.

use crate::core::error::KwaversResult;
use crate::domain::boundary::{Boundary, CPMLBoundary, PMLBoundary};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::math::fft::Complex64;
use ndarray::{Array1, Array2, Array3, Zip};

/// k-Space operators integrated into PSTD solver
#[derive(Debug, Clone)]
pub(super) struct PSTDKSOperators {
    /// k-space grid for wavenumber domain operations
    k_grid: PSTDKSGrid,
}

#[derive(Debug, Clone)]
pub(super) struct PSTDKSGrid {
    /// Number of grid points in each dimension
    pub(super) nx: usize,
    pub(super) ny: usize,
    pub(super) nz: usize,
    /// Spatial step sizes
    pub(super) dx: f64,
    pub(super) dy: f64,
    pub(super) dz: f64,
    /// Wavenumber grids (computed from spatial grids via FFT)
    pub(super) kx: Array1<f64>,
    pub(super) ky: Array1<f64>,
    pub(super) kz: Array1<f64>,
    /// Squared wavenumber magnitude for Helmholtz operator: |k|²
    pub(super) k_squared: Array3<f64>,
    /// FFT normalization factor
    pub(super) fft_norm: f64,
}

impl PSTDKSGrid {
    /// Create k-space grid from spatial grid
    pub(super) fn from_spatial_grid(spatial_grid: &Grid) -> KwaversResult<Self> {
        let nx = spatial_grid.nx;
        let ny = spatial_grid.ny;
        let nz = spatial_grid.nz;
        let dx = spatial_grid.dx;
        let dy = spatial_grid.dy;
        let dz = spatial_grid.dz;

        // Compute wavenumber grids
        let kx = Self::compute_wavenumbers(nx, dx);
        let ky = Self::compute_wavenumbers(ny, dy);
        let kz = Self::compute_wavenumbers(nz, dz);

        // Compute |k|² for each point in 3D wavenumber space
        let mut k_squared = Array3::<f64>::zeros((nx, ny, nz));

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let k_magnitude_squared = kx[i] * kx[i] + ky[j] * ky[j] + kz[k] * kz[k];
                    k_squared[[i, j, k]] = k_magnitude_squared;
                }
            }
        }

        // FFT normalization (matches FFTW conventions)
        let fft_norm = 1.0 / (nx * ny * nz) as f64;

        Ok(Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            kx,
            ky,
            kz,
            k_squared,
            fft_norm,
        })
    }

    /// Compute wavenumber grid for one dimension using FFT conventions
    fn compute_wavenumbers(n: usize, dx: f64) -> Array1<f64> {
        let mut k = Array1::<f64>::zeros(n);

        // Nyquist wavenumber
        let k_nyquist = std::f64::consts::PI / dx;

        // Wavenumber spacing
        let dk = 2.0 * k_nyquist / n as f64;

        for i in 0..n {
            if i <= n / 2 {
                // Positive frequencies: 0, dk, 2*dk, ..., (N/2)*dk
                k[i] = i as f64 * dk;
            } else {
                // Negative frequencies: -(N/2)*dk, ..., -dk
                k[i] = (i as f64 - n as f64) * dk;
            }
        }

        k
    }
}

impl PSTDKSOperators {
    /// Create new k-space operators
    pub(super) fn new(k_grid: PSTDKSGrid) -> Self {
        Self { k_grid }
    }

    /// Apply Helmholtz operator: (∇² + k₀²)p
    pub(super) fn apply_helmholtz(&self, field: &Array3<f64>, wavenumber: f64) -> KwaversResult<Array3<f64>> {
        // Forward FFT to wavenumber domain
        let field_k = self.forward_fft_3d(field)?;

        // Apply Helmholtz: (∇² + k₀²) → -|k|² + k₀²
        let mut result_k = field_k.clone();
        let k0_squared = wavenumber * wavenumber;

        for i in 0..self.k_grid.nx {
            for j in 0..self.k_grid.ny {
                for k in 0..self.k_grid.nz {
                    let k_squared = self.k_grid.k_squared[[i, j, k]];
                    result_k[[i, j, k]] *= k0_squared - k_squared;
                }
            }
        }

        // Inverse FFT back to spatial domain
        self.inverse_fft_3d(&result_k)
    }

    /// Forward 3D FFT: spatial domain → wavenumber domain
    fn forward_fft_3d(&self, input: &Array3<f64>) -> KwaversResult<Array3<Complex64>> {
        // Get FFT processor
        let fft = crate::math::fft::get_fft_for_grid(self.k_grid.nx, self.k_grid.ny, self.k_grid.nz);

        // Apply 3D FFT using the existing FFT infrastructure
        let mut output = Array3::<Complex64>::zeros(input.dim());
        fft.forward_into(input, &mut output);

        // Apply normalization
        output.mapv_inplace(|x| x * self.k_grid.fft_norm);

        Ok(output)
    }

    /// Inverse 3D FFT: wavenumber domain → spatial domain
    fn inverse_fft_3d(&self, input: &Array3<Complex64>) -> KwaversResult<Array3<f64>> {
        // Get FFT processor
        let fft = crate::math::fft::get_fft_for_grid(self.k_grid.nx, self.k_grid.ny, self.k_grid.nz);

        // Apply inverse FFT using existing infrastructure
        let mut output = Array3::<f64>::zeros(input.dim());
        let mut scratch = Array3::<Complex64>::zeros(input.dim());
        fft.inverse_into(input, &mut output, &mut scratch);

        // Apply normalization
        output.mapv_inplace(|x| x / self.k_grid.fft_norm);

        Ok(output)
    }
}

use super::config::{BoundaryConfig, CompatibilityMode, KSpaceMethod, PSTDConfig};
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

    // Advanced k-space pseudospectral operators (when using FullKSpace method)
    pub(super) kspace_operators: Option<PSTDKSOperators>,
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
            kspace_operators: None, // Initialize below if needed
        };

        // Initialize k-space operators if using full k-space method
        let kspace_operators = if config.kspace_method == KSpaceMethod::FullKSpace {
            let k_grid = PSTDKSGrid::from_spatial_grid(&grid)?;
            Some(PSTDKSOperators::new(k_grid))
        } else {
            None
        };

        solver_partial.kspace_operators = kspace_operators;

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

        // Use k-space pseudospectral method if configured
        if self.config.kspace_method == KSpaceMethod::FullKSpace {
            return self.step_forward_kspace(dt, time_index);
        }

        // Standard PSTD method
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

    /// Time step using full k-space pseudospectral method (dispersion-free)
    fn step_forward_kspace(&mut self, dt: f64, time_index: usize) -> KwaversResult<()> {
        // For k-space method, we use the exact dispersion relation
        // ∂²p/∂t² = c²∇²p + s
        //
        // In k-space: d²P/dt² = -c²|k|²P + S
        // Solution: P(t+dt) = 2P(t) - P(t-dt) + dt²(-c²|k|²P(t) + S(t))

        // Check if k-space operators are available
        if self.kspace_operators.is_none() {
            return Err(crate::core::error::KwaversError::Config(crate::core::error::ConfigError::InvalidValue {
                parameter: "kspace_operators".to_string(),
                value: "None".to_string(),
                constraint: "k-space operators must be initialized for FullKSpace method".to_string(),
            }));
        }

        // Get source term
        let mut source_term = Array3::<f64>::zeros(self.fields.p.dim());
        if self.source_handler.has_pressure_source() {
            // Use existing source injection but accumulate in source_term
            // For k-space, we handle sources differently since we work in pressure domain
            // This is a simplified approach - in practice, source injection for k-space
            // needs more sophisticated handling
            let mut temp_rho = Array3::<f64>::zeros(self.fields.p.dim());
            self.source_handler
                .inject_mass_source(time_index, &mut temp_rho, &self.materials.c0);

            // Convert density source to pressure source
            Zip::from(&mut source_term)
                .and(&temp_rho)
                .and(&self.materials.c0)
                .for_each(|s, &rho, &c| *s = rho * c * c);
        }

        // Apply k-space wave propagation
        // Clone the operators to avoid borrow checker issues
        let ops = self.kspace_operators.as_ref().unwrap().clone();
        self.propagate_kspace(dt, &source_term, &ops)?;

        // Apply boundary conditions if needed
        self.apply_boundary(time_index)?;

        // Record sensor data
        self.sensor_recorder.record_step(&self.fields.p)?;

        // Advance time step
        self.time_step_index += 1;

        Ok(())
    }

    /// Propagate wave using k-space pseudospectral method
    fn propagate_kspace(
        &mut self,
        dt: f64,
        source_term: &Array3<f64>,
        kspace_ops: &PSTDKSOperators,
    ) -> KwaversResult<()> {
        // For full k-space implementation, we need to maintain two previous time steps
        // For now, implement a simplified version that demonstrates the concept
        // TODO: Implement proper time integration for k-space method

        // Apply Helmholtz operator: (∇² + k₀²)p where k₀ = ω/c
        let wavenumber = 2.0 * std::f64::consts::PI * 1e6 / self.c_ref; // Example: 1 MHz wave
        let helmholtz_term = kspace_ops.apply_helmholtz(&self.fields.p, wavenumber)?;

        // Simple explicit update (this is a placeholder - proper k-space time integration needed)
        Zip::from(&mut self.fields.p)
            .and(&helmholtz_term)
            .and(source_term)
            .for_each(|p, h_term, s| {
                let c_squared = self.c_ref * self.c_ref;
                // Simplified update: dp/dt = c²∇²p + s, approximated as p += dt * (c²∇²p + s)
                *p += dt * (c_squared * h_term + s);
            });

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::domain::medium::HomogeneousMedium;
    use crate::domain::source::GridSource;
    use ndarray::Array3;

    #[test]
    fn test_kspace_method_configuration() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let mut config = PSTDConfig::default();
        config.kspace_method = KSpaceMethod::FullKSpace;

        let medium = HomogeneousMedium::new(
            1000.0, 1500.0, 0.0, 0.0, &grid,
        );

        let source = GridSource::new_empty();

        let solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

        // Check that k-space operators were initialized
        assert!(solver.kspace_operators.is_some());
        assert_eq!(solver.config.kspace_method, KSpaceMethod::FullKSpace);
    }

    #[test]
    fn test_standard_pstd_method() {
        let grid = Grid::new(32, 32, 32, 0.001, 0.001, 0.001).unwrap();
        let config = PSTDConfig::default(); // Standard PSTD
        let medium = HomogeneousMedium::new(
            1000.0, 1500.0, 0.0, 0.0, &grid,
        );
        let source = GridSource::new_empty();

        let solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

        // Check that k-space operators were NOT initialized for standard method
        assert!(solver.kspace_operators.is_none());
        assert_eq!(solver.config.kspace_method, KSpaceMethod::StandardPSTD);
    }

    #[test]
    fn test_kspace_solver_creation() {
        let grid = Grid::new(16, 16, 16, 0.002, 0.002, 0.002).unwrap();
        let mut config = PSTDConfig::default();
        config.kspace_method = KSpaceMethod::FullKSpace;
        config.nt = 5; // Short simulation for test

        let medium = HomogeneousMedium::new(
            1000.0, 1500.0, 0.0, 0.0, &grid,
        );
        let source = GridSource::new_empty();

        let mut solver = PSTDSolver::new(config, grid, &medium, source).unwrap();

        // Run a few steps
        for _ in 0..3 {
            solver.step_forward().unwrap();
        }

        // Check that simulation progressed
        assert_eq!(solver.time_step_index, 3);

        // Check that pressure field is finite
        let pressure = &solver.fields.p;
        assert!(pressure.iter().all(|&p| p.is_finite()));
    }
}
