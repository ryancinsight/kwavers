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
use ndarray::{Array1, Array2, Array3, Zip};
use std::f64::consts::PI;
use std::sync::Arc;

/// Core PSTD solver implementing the pseudospectral method
pub struct PSTDSolver {
    pub(crate) config: PSTDConfig,
    pub(crate) grid: Arc<Grid>,
    pub sensor_recorder: SensorRecorder,
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
    /// Precomputed additive mass-source scale: 2·Δt / (N·c₀·Δx_min).
    /// Converts a pressure source amplitude [Pa] to a per-component density
    /// injection rate [kg/m³/step] for the dynamic-source additive path.
    /// Constant throughout the simulation; computed once at construction.
    pub(crate) mass_source_scale: f64,
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
    /// Spectral absorption operator ∇^(y−2): precomputed |k|^(y−2) in FFT order.
    /// Applied to per-axis velocity divergence in the density update step.
    /// Treeby & Cox (2010) Eq. 10; k-Wave: nabla1 = kgrid.k .^ (y - 2).
    pub(crate) absorb_nabla1: Array3<f64>,
    /// Spectral dispersion operator ∇^(y−1): precomputed |k|^(y−1) in FFT order.
    /// Applied to per-axis velocity divergence in the density update step.
    /// Treeby & Cox (2010) Eq. 10; k-Wave: nabla2 = kgrid.k .^ (y - 1).
    pub(crate) absorb_nabla2: Array3<f64>,
    pub(crate) kspace_operators: Option<PSTDKSOperators>,
    // Staggered grid shift operators (1D, per axis, complex)
    // ddx_k_shift_pos[x] = i*kx * exp(+i*kx*dx/2) — for pressure gradient → velocity
    // ddx_k_shift_neg[x] = i*kx * exp(-i*kx*dx/2) — for velocity gradient → density
    pub(crate) ddx_k_shift_pos: Array1<Complex64>,
    pub(crate) ddy_k_shift_pos: Array1<Complex64>,
    pub(crate) ddz_k_shift_pos: Array1<Complex64>,
    pub(crate) ddx_k_shift_neg: Array1<Complex64>,
    pub(crate) ddy_k_shift_neg: Array1<Complex64>,
    pub(crate) ddz_k_shift_neg: Array1<Complex64>,
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
        
        let _x_max: f64 = 0.0;
        let source_kappa = k_mag.mapv(|k| (0.5 * c_ref * config.dt * k).cos());
        let boundary: Option<Box<dyn Boundary>> = match &config.boundary {
            BoundaryConfig::PML(pml_config) => {
                Some(Box::new(PMLBoundary::new(pml_config.clone())?))
            }
            BoundaryConfig::CPML(cpml_config) => Some(Box::new(CPMLBoundary::new_with_time_step(
                cpml_config.clone(),
                &grid,
                c_ref,
                Some(config.dt),
            )?)),
            BoundaryConfig::None => None,
        };

        let (absorb_tau, absorb_eta, absorb_y, absorb_nabla1, absorb_nabla2) =
            initialize_absorption_operators(&config, &grid, medium, &k_mag, k_max, c_ref)?;
        let field_arrays =
            crate::solver::forward::pstd::data::initialize_field_arrays(&grid, medium)?;
        let k_vec = (k_ops.kx, k_ops.ky, k_ops.kz);

        // Generate staggered grid shift operators matching the C++ k-wave binary.
        // These implement the half-grid-point spatial shift for staggered grids:
        //   ddx_k_shift_pos[x] = i·kx · exp(+i·kx·dx/2)  (pressure → velocity)
        //   ddx_k_shift_neg[x] = i·kx · exp(-i·kx·dx/2)  (velocity → density)
        let generate_shift_1d =
            |n: usize, dk: f64, ds: f64| -> (Array1<Complex64>, Array1<Complex64>) {
                let i_unit = Complex64::new(0.0, 1.0);
                let mut shift_pos = Array1::zeros(n);
                let mut shift_neg = Array1::zeros(n);
                for idx in 0..n {
                    // Wavenumber in FFT order: [0, 1, ..., n/2, -(n/2-1), ..., -1]*dk
                    // For even n, the Nyquist bin (idx=n/2) uses k = n/2 * dk = +pi/ds.
                    // k-Wave uses the negative Nyquist (-pi/ds) which gives the SAME
                    // shift operator value since i*k*exp(+i*k*ds/2) evaluates to the
                    // same real number at both +pi/ds and -pi/ds. Do NOT zero this bin:
                    // k-Wave C++ includes the Nyquist in propagation, and zeroing it
                    // removes ~18% of k-space energy, causing a 1.64x amplitude error.
                    let shifted = if idx <= n / 2 {
                        idx as isize
                    } else {
                        idx as isize - n as isize
                    };
                    let k_val = dk * shifted as f64;
                    let exponent = k_val * ds * 0.5;
                    shift_pos[idx] = i_unit
                        * Complex64::new(k_val, 0.0)
                        * Complex64::new(exponent.cos(), exponent.sin());
                    shift_neg[idx] = i_unit
                        * Complex64::new(k_val, 0.0)
                        * Complex64::new(exponent.cos(), -exponent.sin());
                }
                (shift_pos, shift_neg)
            };

        let dk_x = 2.0 * PI / (grid.nx as f64 * grid.dx);
        let dk_y = 2.0 * PI / (grid.ny as f64 * grid.dy);
        let dk_z = 2.0 * PI / (grid.nz as f64 * grid.dz);
        let (ddx_k_shift_pos, ddx_k_shift_neg) = generate_shift_1d(grid.nx, dk_x, grid.dx);
        let (ddy_k_shift_pos, ddy_k_shift_neg) = generate_shift_1d(grid.ny, dk_y, grid.dy);
        let (ddz_k_shift_pos, ddz_k_shift_neg) = generate_shift_1d(grid.nz, dk_z, grid.dz);

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
        // Allocate space for Nt+1 steps to include the t=0 initial state, matching k-Wave
        let sensor_recorder = SensorRecorder::new(config.sensor_mask.as_ref(), shape, config.nt + 1)?;
        let source_handler = SourceHandler::new(source, &grid)?;

        // Capture config.dt before the struct literal moves `config`.
        let config_dt = config.dt;

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
            mass_source_scale: {
                let n_dim = [grid.nx > 1, grid.ny > 1, grid.nz > 1]
                    .iter()
                    .filter(|&&d| d)
                    .count()
                    .max(1) as f64;
                let dx_min = grid.dx.min(grid.dy).min(grid.dz);
                // Match k-Wave's scale_pressure_source_uniform_grid:
                //   source_p *= 2*dt / (N * c0 * dx)
                2.0 * config_dt / (n_dim * c_ref * dx_min)
            },
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
            absorb_nabla1,
            absorb_nabla2,
            kspace_operators: None,
            ddx_k_shift_pos,
            ddy_k_shift_pos,
            ddz_k_shift_pos,
            ddx_k_shift_neg,
            ddy_k_shift_neg,
            ddz_k_shift_neg,
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
        // Exact IVP velocity initialization: if p0 was provided and no explicit u0,
        // compute ux/uy/uz at t=−dt/2 using the traveling-wave relation.
        // This matches k-Wave's C++ initialization and eliminates the ~1-step phase lag.
        if solver.source_handler.has_initial_pressure()
            && !solver.source_handler.has_initial_velocity()
        {
            solver.initialize_ivp_velocity()?;
        }

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
        // Apply PML to pressure field.
        // Note: velocity and split-density are already PML-damped inside
        // update_velocity() and update_density() respectively.
        boundary.apply_acoustic(self.fields.p.view_mut(), &self.grid, time_index)?;
        Ok(())
    }

    /// Initialize velocity fields at t = −dt/2 for exact IVP staggered leapfrog start.
    ///
    /// The exact traveling-wave solution gives, for each Fourier mode:
    ///   ux_sgx_k[i,j,k] = ddx_k_shift_pos[i] / |k| · sin(c_ref·dt·|k|/2) / (ρ₀·c_ref) · p0_k[i,j,k]
    ///
    /// This is derived from u(x,t) = IFFT(−i · k̂ · sin(c₀|k|t)/(ρ₀c₀) · FFT(p0)) at t = −dt/2.
    /// Without this init, velocity starts from zero and the wave arrives ≈1 step later than k-Wave.
    fn initialize_ivp_velocity(&mut self) -> KwaversResult<()> {
        let c_ref = self.c_ref;
        let dt = self.config.dt;
        let rho0_ref = self.materials.rho0.mean().unwrap_or(1000.0);

        // k-magnitude array — computed from k_vec without borrowing conflicts
        let k_mag: Array3<f64> = compute_k_magnitude(&self.k_vec.0, &self.k_vec.1, &self.k_vec.2);

        // sin(c_ref·dt·|k|/2) / (|k|·ρ₀·c_ref) — the spectral IVP scaling factor per mode
        // DC component (k=0) → 0 (no net velocity from uniform pressure)
        let sin_scale: Array3<f64> = k_mag.mapv(|km| {
            if km < 1e-30 {
                0.0
            } else {
                (c_ref * dt * km / 2.0).sin() / (km * rho0_ref * c_ref)
            }
        });

        // FFT the initial pressure p0 into the p_k scratch buffer
        self.fft.forward_into(&self.fields.p, &mut self.p_k);

        // --- X component: ux_sgx_k = ddx_k_shift_pos[i] * sin_scale[i,j,k] * p0_k[i,j,k] ---
        {
            let ddx = self.ddx_k_shift_pos.view();
            let sin_s = sin_scale.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_x_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(i, _j, _k), gx, &ss, &p| {
                    *gx = ddx[i] * ss * p;
                });
        }
        self.fft
            .inverse_into(&self.grad_x_k, &mut self.fields.ux, &mut self.ux_k);

        // --- Y component ---
        {
            let ddy = self.ddy_k_shift_pos.view();
            let sin_s = sin_scale.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_y_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(_i, j, _k), gy, &ss, &p| {
                    *gy = ddy[j] * ss * p;
                });
        }
        self.fft
            .inverse_into(&self.grad_y_k, &mut self.fields.uy, &mut self.uy_k);

        // --- Z component ---
        {
            let ddz = self.ddz_k_shift_pos.view();
            let sin_s = sin_scale.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_z_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(_i, _j, k_idx), gz, &ss, &p| {
                    *gz = ddz[k_idx] * ss * p;
                });
        }
        self.fft
            .inverse_into(&self.grad_z_k, &mut self.fields.uz, &mut self.uz_k);

        Ok(())
    }

    pub(super) fn compute_rho0_gradients(&mut self) -> KwaversResult<()> {
        self.fft.forward_into(&self.materials.rho0, &mut self.p_k);
        let (nx, ny, nz) = self.p_k.dim();
        // Use positive shift for rho0 gradient (same as pressure gradient direction)
        for i in 0..nx {
            let shift_x = self.ddx_k_shift_pos[i];
            for j in 0..ny {
                let shift_y = self.ddy_k_shift_pos[j];
                for k in 0..nz {
                    let shift_z = self.ddz_k_shift_pos[k];
                    let kap = Complex64::new(self.kappa[[i, j, k]], 0.0);
                    let rho_k = self.p_k[[i, j, k]];
                    let e_kappa = kap * rho_k;
                    self.grad_x_k[[i, j, k]] = shift_x * e_kappa;
                    self.grad_y_k[[i, j, k]] = shift_y * e_kappa;
                    self.grad_z_k[[i, j, k]] = shift_z * e_kappa;
                }
            }
        }
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
        // Record initial state t=0 to match k-Wave's convention (returning Nt+1 points)
        if self.time_step_index == 0 {
            self.sensor_recorder.record_step(&self.fields.p)?;
        }
        for _ in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }

    pub(crate) fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);
        let mode = super::source_injection::determine_injection_mode(&mask);
        self.dynamic_sources.push((source, mask));
        self.source_injection_modes.push(mode);
        Ok(())
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
