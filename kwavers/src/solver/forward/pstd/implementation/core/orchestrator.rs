//! PSTD Solver Orchestrator

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::Boundary;
use crate::domain::boundary::{CPMLBoundary, PMLBoundary};
use crate::domain::field::wave::WaveFields;
use crate::domain::grid::Grid;
use crate::domain::medium::MaterialFields;
use crate::domain::medium::Medium;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::{GridSource, Source, SourceField, SourceInjectionMode};
use crate::math::fft::shift_operators::generate_shift_1d;
use crate::math::fft::{Complex64, ProcessorFft3d};
use crate::solver::fdtd::SourceHandler;
use crate::solver::forward::pstd::config::{
    BoundaryConfig, CompatibilityMode, KSpaceMethod, PSTDConfig,
};
use crate::solver::forward::pstd::implementation::k_space::{PSTDKSGrid, PSTDKSOperators};
use crate::solver::forward::pstd::numerics::operators::initialize_spectral_operators;
use crate::solver::forward::pstd::numerics::spectral_correction::CorrectionMethod;
use crate::solver::forward::pstd::physics::absorption::{
    initialize_absorption_operators, AbsorptionKernel,
};
use crate::solver::forward::pstd::propagator::axisymmetric::AsContext;
use crate::solver::forward::pstd::utils::compute_k_magnitude;
use crate::solver::geometry::Geometry;
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
    /// Spectral gradient masks for velocity sources, indexed parallel to `dynamic_sources`.
    ///
    /// Entry `i` is `Some(∂mask/∂α)` when `dynamic_sources[i]` is a `VelocityX/Y/Z` source
    /// **and** `FullKSpace` operators were available at registration time; `None` otherwise.
    /// Used in `step_forward_kspace` to inject the pressure-equivalent `−c²·amp·∂mask/∂α`.
    pub(crate) velocity_source_grad_masks: Vec<Option<Array3<f64>>>,
    pub(crate) time_step_index: usize,
    pub(crate) fft: Arc<ProcessorFft3d>,
    pub(crate) kappa: Array3<f64>,
    pub(crate) source_kappa: Array3<f64>,
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
    /// Single k-space gradient scratch, reused for all three spatial axes sequentially.
    /// Replaces the former grad_x_k / grad_y_k / grad_z_k trio, saving 2 × N³ × 16 bytes.
    pub(crate) grad_k: Array3<Complex64>,
    pub(crate) materials: MaterialFields,
    pub(crate) bon: Array3<f64>,
    /// Precomputed absorption kernel — `None` for lossless simulations (saves 4 × N³ × 8 bytes).
    pub(crate) absorption: Option<AbsorptionKernel>,
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
    /// Axisymmetric WSWA-FFT context — `Some` when `config.geometry == CylindricalAS`.
    pub(crate) as_ctx: Option<AsContext>,
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
        let mut k_mag = compute_k_magnitude(&k_ops.kx, &k_ops.ky, &k_ops.kz);

        let _x_max: f64 = 0.0;
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

        let absorption =
            initialize_absorption_operators(&config, &grid, medium, &k_mag, k_max, c_ref)?;
        // source_kappa = cos(c_ref·dt·k/2).
        // Matches k-Wave Python kspaceFirstOrder3D.py line 302:
        //   source_kappa = ifftshift(cos(c_ref * k * dt / 2))
        // This is the half-step phase factor from the leapfrog staggered time-stepping
        // scheme — distinct from kappa (propagation), which uses np.sinc.
        // `k_mag` has no live use after absorption initialization, so transform it
        // in place to avoid one full-volume Array3 allocation during PSTD setup.
        k_mag.mapv_inplace(|k| (0.5 * c_ref * config.dt * k).cos());
        let source_kappa = k_mag;
        let field_arrays =
            crate::solver::forward::pstd::data::initialize_field_arrays(&grid, medium)?;
        // Generate staggered grid shift operators using the shared canonical utility.
        // See `math::fft::shift_operators::generate_shift_1d` for the full derivation.
        // Both PSTD and k-space FDTD call the same function, ensuring bit-identical
        // operators and a single point of truth (SSOT, DRY).
        let dk_x = 2.0 * PI / (grid.nx as f64 * grid.dx);
        let dk_y = 2.0 * PI / (grid.ny as f64 * grid.dy);
        let dk_z = 2.0 * PI / (grid.nz as f64 * grid.dz);
        let (ddx_k_shift_pos, ddx_k_shift_neg) = generate_shift_1d(grid.nx, dk_x, grid.dx);
        let (ddy_k_shift_pos, ddy_k_shift_neg) = generate_shift_1d(grid.ny, dk_y, grid.dy);
        let (ddz_k_shift_pos, ddz_k_shift_neg) = generate_shift_1d(grid.nz, dk_z, grid.dz);

        let shape = (grid.nx, grid.ny, grid.nz);
        let (rho0, c0, bon) = if medium.is_homogeneous() {
            (
                Array3::from_elem(shape, medium.density(0, 0, 0)),
                Array3::from_elem(shape, medium.sound_speed(0, 0, 0)),
                Array3::from_elem(shape, medium.nonlinearity(0, 0, 0)),
            )
        } else {
            let mut rho0 = Array3::zeros(shape);
            let mut c0 = Array3::zeros(shape);
            let mut bon = Array3::zeros(shape);

            for k in 0..grid.nz {
                for j in 0..grid.ny {
                    for i in 0..grid.nx {
                        let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                        rho0[[i, j, k]] =
                            crate::domain::medium::density_at(medium, x, y, z, &grid);
                        c0[[i, j, k]] =
                            crate::domain::medium::sound_speed_at(medium, x, y, z, &grid);
                        bon[[i, j, k]] =
                            crate::domain::medium::nonlinearity_at(medium, x, y, z, &grid);
                    }
                }
            }

            (rho0, c0, bon)
        };
        // Allocate space for Nt+1 steps to include the t=0 initial state, matching k-Wave
        let sensor_recorder =
            SensorRecorder::new(config.sensor_mask.as_ref(), shape, config.nt + 1)?;
        let mut source_handler = SourceHandler::new(source, &grid)?;
        // Precompute per-voxel source_kappa for velocity sources in additive mode.
        // k-Wave applies ifftshift(cos(c_ref·|k|·dt/2)) at each source voxel position
        // for additive velocity sources (NotATransducer / u_mode="additive").
        if source_handler.has_velocity_source() {
            source_handler.set_velocity_source_kappa(&source_kappa, grid.nx, grid.ny, grid.nz);
        }

        // Capture config.dt before the struct literal moves `config`.
        let config_dt = config.dt;

        let mut solver = Self {
            config,
            grid: grid.clone(),
            sensor_recorder,
            source_handler,
            dynamic_sources: Vec::new(),
            source_injection_modes: Vec::new(),
            velocity_source_grad_masks: Vec::new(),
            time_step_index: 0,
            fft: crate::math::fft::get_fft_for_grid(grid.nx, grid.ny, grid.nz),
            kappa,
            source_kappa,
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
            grad_k: Array3::zeros(shape),
            materials: MaterialFields { rho0, c0 },
            bon,
            absorption,
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
            as_ctx: None,
        };

        if solver.config.kspace_method == KSpaceMethod::FullKSpace {
            let k_grid = PSTDKSGrid::from_spatial_grid(&grid)?;
            solver.kspace_operators = Some(PSTDKSOperators::new(k_grid));
        }

        if solver.config.geometry == Geometry::CylindricalAS {
            if grid.ny != 1 {
                return Err(KwaversError::InvalidInput(
                    "CylindricalAS requires ny = 1".into(),
                ));
            }
            solver.as_ctx = Some(AsContext::new(
                grid.nx,
                grid.nz, // nr = nz (radial dimension)
                grid.dx,
                grid.dz, // dr = dz
                c_ref,
                config_dt,
                solver.ddx_k_shift_pos.clone(),
                solver.ddx_k_shift_neg.clone(),
            )?);
        }

        solver.source_handler.prepare_pressure_source_scaling(
            &grid,
            &solver.materials.c0,
            solver.config.dt,
        );

        solver.source_handler.apply_initial_conditions(
            &mut solver.fields.p,
            &mut solver.div_u,
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
            .and(&solver.div_u)
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
        let dt = self.config.dt;
        let rho0_ref = self.materials.rho0.mean().unwrap_or(1000.0);
        let has_y = self.grid.ny > 1;
        let has_z = self.grid.nz > 1;

        // The spectral IVP scale is expressed through the source-injection phase
        // factor:
        //   sin(c0|k|dt/2)/(ρ0 c0 |k|) = (dt / (2ρ0)) * sinc(arccos(κ_src))
        // This avoids materializing a separate |k| array.
        if !dt.is_finite() || dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "dt must be finite and positive, got {dt}"
            )));
        }
        if !rho0_ref.is_finite() || rho0_ref <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "rho0_ref must be finite and positive, got {rho0_ref}"
            )));
        }
        let scale_prefactor = dt / (2.0 * rho0_ref);
        Zip::from(&mut self.div_u)
            .and(&self.source_kappa)
            .for_each(|scale, &kap| {
                let theta = kap.clamp(-1.0, 1.0).acos();
                *scale = if theta < 1e-30 {
                    scale_prefactor
                } else {
                    scale_prefactor * theta.sin() / theta
                };
            });

        // FFT the initial pressure p0 into the p_k scratch buffer
        self.fft.forward_into(&self.fields.p, &mut self.p_k);

        // --- X component: ux_sgx_k = ddx_k_shift_pos[i] * sin_scale[i,j,k] * p0_k[i,j,k] ---
        // grad_k is reused for each axis (p_k is only read, not modified).
        {
            let ddx = self.ddx_k_shift_pos.view();
            let sin_s = self.div_u.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(i, _j, _k), gk, &ss, &p| {
                    *gk = ddx[i] * ss * p;
                });
        }
        self.fft
            .inverse_into(&self.grad_k, &mut self.fields.ux, &mut self.ux_k);

        // --- Y component ---
        // A singleton embedding axis has only k_y = 0, therefore the exact
        // IVP velocity component along that axis is identically zero.
        if has_y {
            let ddy = self.ddy_k_shift_pos.view();
            let sin_s = self.div_u.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(_i, j, _k), gk, &ss, &p| {
                    *gk = ddy[j] * ss * p;
                });
            self.fft
                .inverse_into(&self.grad_k, &mut self.fields.uy, &mut self.uy_k);
        } else {
            self.fields.uy.fill(0.0);
        }

        // --- Z component ---
        // A singleton z-axis has k_z = 0, so the exact z velocity is zero.
        if has_z {
            let ddz = self.ddz_k_shift_pos.view();
            let sin_s = self.div_u.view();
            let p_k = self.p_k.view();
            Zip::indexed(self.grad_k.view_mut())
                .and(sin_s)
                .and(p_k)
                .for_each(|(_i, _j, k_idx), gk, &ss, &p| {
                    *gk = ddz[k_idx] * ss * p;
                });
            self.fft
                .inverse_into(&self.grad_k, &mut self.fields.uz, &mut self.uz_k);
        } else {
            self.fields.uz.fill(0.0);
        }

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

    /// Run `checkpoint_steps` steps then persist full solver state to `path`.
    ///
    /// Records the initial field (step 0) if this is the first call, runs
    /// `checkpoint_steps` time steps, then serialises the solver state using
    /// the binary KWCP format (see [`checkpoint`][crate::solver::forward::pstd::checkpoint]).
    /// Call [`run_from_checkpoint`] to resume bit-exactly.
    ///
    /// # Invariants
    /// - `checkpoint_steps ≤ config.nt`
    /// - Grid dims and `dt` in the checkpoint match those of the resuming solver
    pub fn run_to_checkpoint(
        &mut self,
        checkpoint_steps: usize,
        path: &std::path::Path,
    ) -> KwaversResult<()> {
        use crate::solver::forward::pstd::checkpoint::PSTDCheckpoint;

        if self.time_step_index == 0 {
            self.sensor_recorder.record_step(&self.fields.p)?;
        }
        for _ in 0..checkpoint_steps {
            self.step_forward()?;
        }

        let (sensor_data, sensor_next_step, sensor_expected_steps) = self
            .sensor_recorder
            .checkpoint_state()
            .map(|(d, ns, es)| (Some(d), ns, es))
            .unwrap_or((None, 0, 0));

        let ckpt = PSTDCheckpoint {
            nx: self.grid.nx,
            ny: self.grid.ny,
            nz: self.grid.nz,
            time_step_index: self.time_step_index,
            total_steps: self.config.nt,
            dt: self.config.dt,
            p: self.fields.p.clone(),
            ux: self.fields.ux.clone(),
            uy: self.fields.uy.clone(),
            uz: self.fields.uz.clone(),
            rhox: self.rhox.clone(),
            rhoy: self.rhoy.clone(),
            rhoz: self.rhoz.clone(),
            sensor_data,
            sensor_next_step,
            sensor_expected_steps,
        };
        ckpt.save(path)
    }

    /// Restore state from `path` and run `remaining_steps` steps to completion.
    ///
    /// Loads the KWCP checkpoint, validates grid dims and `dt`, restores all
    /// seven field arrays, `time_step_index`, and the sensor recorder, then
    /// continues the time loop for exactly `remaining_steps` steps.  Deletes
    /// the checkpoint file after a successful restore (matching k-Wave convention).
    ///
    /// Returns the full sensor pressure matrix (`n_sensors × expected_steps`), or
    /// `None` if no sensor mask was configured.
    ///
    /// # Errors
    /// Returns `KwaversError::InvalidInput` / `DimensionMismatch` if the checkpoint
    /// does not match this solver's grid or `dt`.
    pub fn run_from_checkpoint(
        &mut self,
        path: &std::path::Path,
        remaining_steps: usize,
    ) -> KwaversResult<Option<Array2<f64>>> {
        use crate::solver::forward::pstd::checkpoint::PSTDCheckpoint;

        let ckpt = PSTDCheckpoint::load(path)?;
        self.run_from_checkpoint_loaded(ckpt, path, remaining_steps)
    }

    /// Restore state from an already loaded checkpoint and continue the run.
    ///
    /// This helper avoids a second file parse when the caller has already read
    /// the checkpoint metadata to determine the number of remaining steps.
    pub fn run_from_checkpoint_loaded(
        &mut self,
        ckpt: crate::solver::forward::pstd::checkpoint::PSTDCheckpoint,
        path: &std::path::Path,
        remaining_steps: usize,
    ) -> KwaversResult<Option<Array2<f64>>> {
        ckpt.validate_restore_contract(
            self.grid.nx,
            self.grid.ny,
            self.grid.nz,
            self.config.nt,
            self.config.dt,
        )?;

        let expected_remaining = self
            .config
            .nt
            .checked_sub(ckpt.time_step_index)
            .ok_or_else(|| {
                KwaversError::InvalidInput(format!(
                    "checkpoint time_step_index {} exceeds solver total_steps {}",
                    ckpt.time_step_index, self.config.nt
                ))
            })?;
        if remaining_steps != expected_remaining {
            return Err(KwaversError::InvalidInput(format!(
                "checkpoint remaining_steps {} ≠ expected {}",
                remaining_steps, expected_remaining
            )));
        }

        self.fields.p.assign(&ckpt.p);
        self.fields.ux.assign(&ckpt.ux);
        self.fields.uy.assign(&ckpt.uy);
        self.fields.uz.assign(&ckpt.uz);
        self.rhox.assign(&ckpt.rhox);
        self.rhoy.assign(&ckpt.rhoy);
        self.rhoz.assign(&ckpt.rhoz);
        self.time_step_index = ckpt.time_step_index;

        if let Some(sensor_data) = ckpt.sensor_data {
            self.sensor_recorder
                .restore_from_checkpoint(sensor_data, ckpt.sensor_next_step)?;
        }

        // Delete checkpoint file after successful restore (k-Wave convention).
        let _ = std::fs::remove_file(path);

        for _ in 0..remaining_steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }

    pub(crate) fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);
        let mode = super::source_injection::determine_injection_mode(&mask);

        // Pre-compute ∂mask/∂α for velocity sources when FullKSpace operators are available.
        //
        // Mathematical basis: a velocity source f_u = amp·mask·ê_α contributes to the
        // pressure wave equation as −c²·∇·f_u = −c²·amp·∂mask/∂α.  The spectral
        // derivative is computed once here and reused on every time step.
        let grad_mask: Option<Array3<f64>> = match source.source_type() {
            SourceField::VelocityX => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_x(&mask)?)
                } else {
                    None
                }
            }
            SourceField::VelocityY => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_y(&mask)?)
                } else {
                    None
                }
            }
            SourceField::VelocityZ => {
                if let Some(ops) = &self.kspace_operators {
                    Some(ops.spectral_grad_z(&mask)?)
                } else {
                    None
                }
            }
            _ => None,
        };

        self.dynamic_sources.push((source, mask));
        self.source_injection_modes.push(mode);
        self.velocity_source_grad_masks.push(grad_mask);
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
