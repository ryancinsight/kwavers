//! Core FDTD solver implementation
//!
//! # Theorem: Yee (1966) Staggered-Grid FDTD for Acoustic Waves
//!
//! The linearized Euler equations for acoustic wave propagation are:
//! ```text
//!   ρ₀ ∂u/∂t = −∇p                       (momentum conservation)
//!   ∂p/∂t    = −ρ₀c₀² ∇·u                (mass conservation + EOS)
//! ```
//!
//! Yee's staggered-grid FDTD discretizes these in a leapfrog (velocity–pressure)
//! update order. Pressure lives at integer time steps `t^n = nΔt`; velocity
//! at half-integer steps `t^{n+½} = (n+½)Δt`:
//!
//! ```text
//!   u^{n+½} = u^{n−½} − (Δt/ρ₀) · ∇p^n             [velocity update]
//!   p^{n+1} = p^n     − ρ₀c₀²Δt · ∇·u^{n+½}        [pressure update]
//! ```
//!
//! Spatial derivatives use centered finite differences on a staggered Cartesian grid:
//! pressure at cell centers `(i, j, k)`, velocity components at half-shifted faces
//! `(i+½, j, k)`, `(i, j+½, k)`, `(i, j, k+½)`.
//!
//! ## Stability: CFL Condition
//!
//! The FDTD scheme is stable only when the Courant-Friedrichs-Lewy (CFL) condition is met:
//! ```text
//!   c₀ · Δt · √(1/Δx² + 1/Δy² + 1/Δz²) ≤ 1
//! ```
//! In 3D with uniform spacing Δx = Δy = Δz:
//! ```text
//!   Δt_max = Δx / (c₀ · √3) ≈ 0.577 · Δx / c₀
//! ```
//! CFL safety factor 0.95 is applied by default.
//!
//! ## Spatial Accuracy
//!
//! | Stencil order | Accuracy | PPW required |
//! |---------------|----------|--------------|
//! | 2nd (default) | O(Δx²)   | ~10          |
//! | 4th           | O(Δx⁴)   | ~5           |
//! | 6th           | O(Δx⁶)   | ~4           |
//!
//! PPW = points per wavelength at the maximum frequency of interest.
//!
//! ## Boundary Conditions
//!
//! Absorbing boundaries use CPML (Convolutional PML, Roden & Gedney 2000).
//! See `domain/boundary/cpml/` for the recursive-convolution memory update.
//!
//! ## References
//! - Yee, K.S. (1966). Numerical solution of initial boundary value problems
//!   involving Maxwell's equations in isotropic media.
//!   IEEE Trans. Antennas Propag. 14(3), 302–307.
//! - Taflove, A. & Hagness, S.C. (2005). Computational Electrodynamics:
//!   The Finite-Difference Time-Domain Method, 3rd ed. Artech House.
//! - Virieux, J. (1986). P-SV wave propagation in heterogeneous media:
//!   Velocity-stress finite-difference method. Geophysics 51(4), 889–901.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::boundary::cpml::CPMLBoundary;
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::{Source, SourceField, SourceInjectionMode};
use crate::math::numerics::operators::{
    CentralDifference2, CentralDifference4, CentralDifference6, DifferentialOperator,
    StaggeredGridOperator,
};
use crate::physics::acoustics::mechanics::acoustic_wave::SpatialOrder;
use log::info;
use ndarray::{s, Array3, ArrayView3, Zip};
use std::sync::Arc;

use super::config::FdtdConfig;
use super::metrics::FdtdMetrics;
use super::source_handler::SourceHandler;
use crate::domain::sensor::recorder::simple::SensorRecorder;
use crate::domain::source::grid_source::GridSource;

use crate::domain::field::wave::WaveFields;
use crate::domain::medium::MaterialFields;

#[derive(Debug, Clone)]
pub(crate) enum CentralDifferenceOperator {
    Order2(CentralDifference2),
    Order4(CentralDifference4),
    Order6(CentralDifference6),
}

impl CentralDifferenceOperator {
    fn new(order: usize, dx: f64, dy: f64, dz: f64) -> KwaversResult<Self> {
        match order {
            2 => Ok(Self::Order2(CentralDifference2::new(dx, dy, dz)?)),
            4 => Ok(Self::Order4(CentralDifference4::new(dx, dy, dz)?)),
            6 => Ok(Self::Order6(CentralDifference6::new(dx, dy, dz)?)),
            _ => Err(KwaversError::InvalidInput(format!(
                "spatial_order must be 2, 4, or 6, got {order}"
            ))),
        }
    }

    pub(crate) fn apply_x(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        match self {
            Self::Order2(op) => op.apply_x(field),
            Self::Order4(op) => op.apply_x(field),
            Self::Order6(op) => op.apply_x(field),
        }
    }

    pub(crate) fn apply_y(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        match self {
            Self::Order2(op) => op.apply_y(field),
            Self::Order4(op) => op.apply_y(field),
            Self::Order6(op) => op.apply_y(field),
        }
    }

    pub(crate) fn apply_z(&self, field: ArrayView3<f64>) -> KwaversResult<Array3<f64>> {
        match self {
            Self::Order2(op) => op.apply_z(field),
            Self::Order4(op) => op.apply_z(field),
            Self::Order6(op) => op.apply_z(field),
        }
    }

    pub(crate) fn gradient(
        &self,
        field: ArrayView3<f64>,
    ) -> KwaversResult<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        Ok((
            self.apply_x(field)?,
            self.apply_y(field)?,
            self.apply_z(field)?,
        ))
    }

    #[allow(dead_code)]
    fn divergence(
        &self,
        vx: ArrayView3<f64>,
        vy: ArrayView3<f64>,
        vz: ArrayView3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let dvx_dx = self.apply_x(vx)?;
        let dvy_dy = self.apply_y(vy)?;
        let dvz_dz = self.apply_z(vz)?;
        Ok(&dvx_dx + &dvy_dy + &dvz_dz)
    }
}

/// FDTD solver for acoustic wave propagation
///
/// Supports both linear and nonlinear (Westervelt) wave propagation, CPML absorbing
/// boundaries, staggered-grid spatial operators (2nd / 4th / 6th order), and optional
/// AVX-512 SIMD pressure update kernels.
///
/// Westervelt nonlinear term `(β/ρ₀c₀⁴) ∂²p²/∂t²` is enabled via
/// [`FdtdConfig::enable_nonlinear`]. When enabled the solver allocates
/// `p_prev`, `p_prev2`, `nl_scratch`, `beta_arr`, and `c0_fourth` arrays and
/// calls [`FdtdSolver::apply_westervelt_nonlinear_correction`] each step.
///
/// Reference: Westervelt (1963), Hamilton & Blackstock (1998) Ch. 4.
pub struct FdtdSolver {
    /// Configuration
    pub(crate) config: FdtdConfig,
    /// Grid reference
    pub(crate) grid: Grid,
    pub(crate) central_operator: CentralDifferenceOperator,
    pub(crate) staggered_operator: StaggeredGridOperator,
    /// Performance metrics
    metrics: FdtdMetrics,
    /// C-PML boundary (if enabled)
    pub(crate) cpml_boundary: Option<CPMLBoundary>,
    /// Spatial order enum (validated at construction)
    spatial_order: SpatialOrder,
    pub(crate) gpu_accelerator: Option<Arc<dyn FdtdGpuAccelerator>>,

    // Shared components for source handling and sensor recording
    pub(crate) source_handler: SourceHandler,
    dynamic_sources: Vec<(Arc<dyn Source>, Array3<f64>)>,
    source_injection_modes: Vec<SourceInjectionMode>,
    pub sensor_recorder: SensorRecorder,

    // State
    pub(crate) time_step_index: usize,

    // Wave Fields (p, ux, uy, uz)
    pub fields: WaveFields,

    // Material Properties (rho0, c0)
    pub(crate) materials: MaterialFields,

    // Precomputed fields
    pub(crate) rho_c_squared: Array3<f64>,

    // Nonlinear Westervelt fields (populated only when config.enable_nonlinear = true)
    // ---------------------------------------------------------------------------------
    // Algorithm: Westervelt (1963) explicit FDTD — Eq. A.5 of Hamilton & Blackstock (1998).
    //
    // Nonlinear source term added to the pressure update:
    //   Δp_nl = dt * (β_arr / (ρ₀ · c₀⁴)) · ∂²(p²)/∂t²
    //   ∂²(p²)/∂t² = 2p * (p - 2p' + p'') / dt² + 2 * ((p - p') / dt)²
    //
    // where p' = p^{n-1} and p'' = p^{n-2}. For the first two steps,
    // the incomplete history terms are zero-initialized (Aanonsen et al. 1984).
    //
    // References:
    //   Westervelt, P. J. (1963). J. Acoust. Soc. Am. 35(4), 535–537.
    //   Hamilton, M. F. & Blackstock, D. T. (1998). Nonlinear Acoustics. Academic Press.
    //   Aanonsen, S. I. et al. (1984). J. Acoust. Soc. Am. 75(3), 749–768.
    /// Previous pressure field p^{n-1} for Westervelt nonlinear term
    pub(crate) p_prev: Option<Array3<f64>>,
    /// Two steps back p^{n-2} for Westervelt nonlinear term
    pub(crate) p_prev2: Option<Array3<f64>>,
    /// Pre-allocated scratch for nonlinear term to avoid per-step allocation
    pub(crate) nl_scratch: Option<Array3<f64>>,
    /// Nonlinearity coefficient β = 1 + B/(2A) at each grid point
    pub(crate) beta_arr: Option<Array3<f64>>,
    /// c₀⁴ at each grid point (precomputed to avoid recomputing in hot path)
    pub(crate) c0_fourth: Option<Array3<f64>>,
}

impl FdtdSolver {
    /// Create a new FDTD solver
    pub fn new(
        config: FdtdConfig,
        grid: &Grid,
        medium: &dyn Medium,
        source: GridSource,
    ) -> KwaversResult<Self> {
        info!("Initializing FDTD solver with config: {:?}", config);

        // Validate spatial order by converting to enum
        let spatial_order = SpatialOrder::from_usize(config.spatial_order)?;

        let central_operator =
            CentralDifferenceOperator::new(config.spatial_order, grid.dx, grid.dy, grid.dz)?;
        let staggered_operator = StaggeredGridOperator::new(grid.dx, grid.dy, grid.dz)?;

        let source_handler = SourceHandler::new(source, grid)?;
        let sensor_recorder = SensorRecorder::new(
            config.sensor_mask.as_ref(),
            (grid.nx, grid.ny, grid.nz),
            config.nt + 1,
        )?;

        // Initialize fields
        let shape = (grid.nx, grid.ny, grid.nz);
        let mut fields = WaveFields::new(shape);
        let mut materials = MaterialFields::new(shape);

        // Pre-compute material properties
        for k in 0..grid.nz {
            for j in 0..grid.ny {
                for i in 0..grid.nx {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    materials.rho0[[i, j, k]] =
                        crate::domain::medium::density_at(medium, x, y, z, grid);
                    materials.c0[[i, j, k]] =
                        crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                }
            }
        }

        // Pre-compute rho * c^2 element-wise
        let mut rho_c_squared = Array3::<f64>::zeros(shape);
        Zip::from(&mut rho_c_squared)
            .and(&materials.rho0)
            .and(&materials.c0)
            .for_each(|rho_c_sq, &rho, &c| {
                *rho_c_sq = rho * c * c;
            });

        // Precompute k-Wave compatible pressure source scaling
        let mut source_handler = source_handler;
        source_handler.prepare_pressure_source_scaling(grid, &materials.c0, config.dt);

        // Apply initial conditions (p0, u0) — mirrors PSTD solver behaviour
        let mut rho_init = Array3::zeros(shape);
        source_handler.apply_initial_conditions(
            &mut fields.p,
            &mut rho_init,
            &materials.c0,
            &mut fields.ux,
            &mut fields.uy,
            &mut fields.uz,
        );
        // Note: FDTD uses a single rho field so no split needed (cf. PSTD rhox/rhoy/rhoz)

        // Precompute nonlinear medium property arrays (only when nonlinear mode is on)
        let (p_prev, p_prev2, nl_scratch, beta_arr, c0_fourth) =
            if config.enable_nonlinear {
                let mut beta = Array3::<f64>::zeros(shape);
                let mut c4 = Array3::<f64>::zeros(shape);
                for k in 0..grid.nz {
                    for j in 0..grid.ny {
                        for i in 0..grid.nx {
                            let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                            let bn = crate::domain::medium::nonlinearity_at(medium, x, y, z, grid);
                            let c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
                            // β = 1 + B/(2A) where B/A is returned by nonlinearity_at
                            beta[[i, j, k]] = 1.0 + bn * 0.5;
                            c4[[i, j, k]] = c.powi(4);
                        }
                    }
                }
                (
                    Some(Array3::<f64>::zeros(shape)),
                    Some(Array3::<f64>::zeros(shape)),
                    Some(Array3::<f64>::zeros(shape)),
                    Some(beta),
                    Some(c4),
                )
            } else {
                (None, None, None, None, None)
            };

        Ok(Self {
            config,
            grid: grid.clone(),
            central_operator,
            staggered_operator,
            metrics: FdtdMetrics::new(),
            cpml_boundary: None,
            spatial_order,
            gpu_accelerator: None,
            source_handler,
            dynamic_sources: Vec::new(),
            source_injection_modes: Vec::new(),
            sensor_recorder,
            time_step_index: 0,
            fields,
            materials,
            rho_c_squared,
            p_prev,
            p_prev2,
            nl_scratch,
            beta_arr,
            c0_fourth,
        })
    }

    pub fn set_gpu_accelerator(&mut self, accelerator: Arc<dyn FdtdGpuAccelerator>) {
        self.gpu_accelerator = Some(accelerator);
    }

    /// Enable C-PML boundary conditions
    pub fn enable_cpml(
        &mut self,
        config: crate::domain::boundary::cpml::CPMLConfig,
        dt: f64,
        max_sound_speed: f64,
    ) -> KwaversResult<()> {
        info!("Enabling C-PML boundary conditions");
        self.cpml_boundary = Some(CPMLBoundary::new_with_time_step(
            config,
            &self.grid,
            max_sound_speed,
            Some(dt),
        )?);
        Ok(())
    }

    /// Perform a single time step
    pub fn step_forward(&mut self) -> KwaversResult<()> {
        let time_index = self.time_step_index;
        let dt = self.config.dt;

        // 1. Update Velocity (from current pressure field)
        self.update_velocity(dt)?;
        if self.fields.ux.iter().any(|&x| x.is_nan()) {
            panic!("NaN in ux after update_velocity at step {}", time_index);
        }

        // 2. Inject Force Sources
        if self.source_handler.has_velocity_source() {
            self.source_handler.inject_force_source(
                time_index,
                &mut self.fields.ux,
                &mut self.fields.uy,
                &mut self.fields.uz,
            );
        }

        self.apply_dynamic_velocity_sources(dt);
        if self.fields.ux.iter().any(|&x| x.is_nan()) {
            panic!(
                "NaN in ux after dynamic velocity sources at step {}",
                time_index
            );
        }

        // 3. Update Pressure
        self.update_pressure(dt)?;
        if self.fields.p.iter().any(|&x| x.is_nan()) {
            panic!("NaN in p after update_pressure at step {}", time_index);
        }

        // 4. Apply pressure sources after update (additive + Dirichlet enforcement)
        if self.source_handler.has_pressure_source() {
            self.source_handler
                .inject_pressure_source(time_index, &mut self.fields.p);
        }
        self.apply_dynamic_pressure_sources(dt);
        self.source_handler
            .enforce_pressure_dirichlet(time_index, &mut self.fields.p);
        self.apply_dynamic_pressure_dirichlet(dt);

        if self.fields.p.iter().any(|&x| x.is_nan()) {
            panic!("NaN in p after pressure sources at step {}", time_index);
        }

        // 5. Apply Boundary (CPML is applied within updates via self.cpml_boundary)

        // 6. Record Sensors
        self.sensor_recorder.record_step(&self.fields.p)?;

        self.time_step_index += 1;

        Ok(())
    }

    fn apply_dynamic_pressure_sources(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        let FdtdSolver {
            ref dynamic_sources,
            ref mut fields,
            ref grid,
            ref materials,
            source_injection_modes: _,
            ..
        } = self;

        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let _c0_ref = materials.c0[[nx / 2, ny / 2, nz / 2]];
        let _dx = grid.dx;

        for (idx, (source, mask)) in dynamic_sources.iter().enumerate() {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            match source.source_type() {
                SourceField::Pressure => {
                    let mode = self.source_injection_modes[idx];
                    match mode {
                        SourceInjectionMode::Boundary => {
                            // Dirichlet: enforce p = amplitude at boundary
                            Zip::from(&mut fields.p).and(mask).for_each(|p, &m| {
                                if m > 0.0 {
                                    *p = amp;
                                }
                            });
                        }
                        SourceInjectionMode::Additive { .. } => {
                            // Additive: p += mask * amplitude
                            // For parity with k-Wave's additive mass sources, we do not normalize by mask sum
                            // and we expect the physical scaling to be handled by the caller or precomputed.
                            Zip::from(&mut fields.p)
                                .and(mask)
                                .for_each(|p, &m| *p += m * amp);
                        }
                    }
                }
                SourceField::VelocityX | SourceField::VelocityY | SourceField::VelocityZ => {}
            }
        }
    }

    fn apply_dynamic_pressure_dirichlet(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        let FdtdSolver {
            ref dynamic_sources,
            ref mut fields,
            ..
        } = self;

        for (idx, (source, mask)) in dynamic_sources.iter().enumerate() {
            if source.source_type() != SourceField::Pressure {
                continue;
            }
            if self.source_injection_modes[idx] != SourceInjectionMode::Boundary {
                continue;
            }
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }
            Zip::from(&mut fields.p).and(mask).for_each(|p, &m| {
                if m > 0.0 {
                    *p = amp;
                }
            });
        }
    }

    fn apply_dynamic_velocity_sources(&mut self, dt: f64) {
        let t = self.time_step_index as f64 * dt;
        let FdtdSolver {
            ref dynamic_sources,
            ref mut fields,
            ..
        } = self;

        for (source, mask) in dynamic_sources {
            let amp = source.amplitude(t);
            if amp.abs() < 1e-12 {
                continue;
            }

            match source.source_type() {
                SourceField::Pressure => {}
                SourceField::VelocityX => {
                    Zip::from(&mut fields.ux)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
                SourceField::VelocityY => {
                    Zip::from(&mut fields.uy)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
                SourceField::VelocityZ => {
                    Zip::from(&mut fields.uz)
                        .and(mask)
                        .for_each(|u, &m| *u += m * amp);
                }
            }
        }
    }

    /// Calculate maximum stable time step based on CFL condition
    pub fn max_stable_dt(&self, max_sound_speed: f64) -> f64 {
        let min_dx = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let cfl_limit = self.spatial_order.cfl_limit();
        self.config.cfl_factor * cfl_limit * min_dx / max_sound_speed
    }

    /// Check if given timestep satisfies CFL condition
    pub fn check_cfl_stability(&self, dt: f64, max_sound_speed: f64) -> bool {
        let max_dt = self.max_stable_dt(max_sound_speed);
        dt <= max_dt
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> &FdtdMetrics {
        &self.metrics
    }

    /// Merge metrics from another solver instance
    pub fn merge_metrics(&mut self, other_metrics: &FdtdMetrics) {
        self.metrics.merge(other_metrics);
    }

    /// Extract recorded sensor data as Array2<f64>
    /// Returns None if no sensors are configured or no data has been recorded
    pub fn extract_recorded_sensor_data(&self) -> Option<ndarray::Array2<f64>> {
        self.sensor_recorder.extract_pressure_data()
    }

    pub fn run_orchestrated(&mut self, steps: usize) -> KwaversResult<Option<ndarray::Array2<f64>>> {
        // Record initial state t=0 to match k-Wave's convention (returning Nt+1 points)
        if self.time_step_index == 0 {
            self.sensor_recorder.record_step(&self.fields.p)?;
        }
        for _ in 0..steps {
            self.step_forward()?;
        }
        Ok(self.sensor_recorder.extract_pressure_data())
    }

    pub fn add_source_arc(&mut self, source: Arc<dyn Source>) -> KwaversResult<()> {
        let mask = source.create_mask(&self.grid);

        // Determine injection mode once and cache it
        let mode = Self::determine_injection_mode(&mask, &self.grid);

        self.dynamic_sources.push((source, mask));
        self.source_injection_modes.push(mode);
        Ok(())
    }

    /// Determine injection mode based on source mask spatial distribution
    ///
    /// # Mathematical Specification
    /// - **Boundary Plane Source**: Mask is non-zero only on a single grid plane
    ///   (x=0, x=Nx-1, y=0, y=Ny-1, z=0, or z=Nz-1)
    ///   → Use Dirichlet enforcement: p(boundary) = amplitude(t)
    /// - **Interior Source**: Mask is non-zero in interior or distributed volume
    ///   → Use additive injection: p += (mask / ||mask||) * amplitude(t)
    ///   where ||mask|| is the L1 norm to preserve energy scaling
    fn determine_injection_mode(mask: &Array3<f64>, _grid: &Grid) -> SourceInjectionMode {
        let shape = mask.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        // Count non-zero mask elements
        let mut mask_sum = 0.0;
        let mut nonzero_count = 0;

        // Check if mask is concentrated on a single boundary plane
        let mut is_boundary_plane = false;

        // X boundaries (planes at x=0 or x=nx-1)
        let x0_count = mask
            .slice(s![0, .., ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let xn_count = mask
            .slice(s![nx - 1, .., ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Y boundaries (planes at y=0 or y=ny-1)
        let y0_count = mask
            .slice(s![.., 0, ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let yn_count = mask
            .slice(s![.., ny - 1, ..])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Z boundaries (planes at z=0 or z=nz-1)
        let z0_count = mask
            .slice(s![.., .., 0])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();
        let zn_count = mask
            .slice(s![.., .., nz - 1])
            .iter()
            .filter(|&&v| v > 0.0)
            .count();

        // Compute total mask statistics
        for &val in mask.iter() {
            if val > 0.0 {
                nonzero_count += 1;
                mask_sum += val;
            }
        }

        // If all non-zero elements are on a single boundary plane, use Boundary mode
        if nonzero_count > 0
            && (x0_count == nonzero_count
                || xn_count == nonzero_count
                || y0_count == nonzero_count
                || yn_count == nonzero_count
                || z0_count == nonzero_count
                || zn_count == nonzero_count)
        {
            is_boundary_plane = true;
        }

        if is_boundary_plane {
            SourceInjectionMode::Boundary
        } else {
            // Additive mode: normalize by mask L1 norm to preserve energy
            let scale = if mask_sum > 0.0 { 1.0 / mask_sum } else { 1.0 };
            SourceInjectionMode::Additive { scale }
        }
    }
}

impl std::fmt::Debug for FdtdSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FdtdSolver")
            .field("config", &self.config)
            .field("grid", &self.grid)
            .field("central_operator", &self.central_operator)
            .field("staggered_operator", &self.staggered_operator)
            .field("metrics", &self.metrics)
            .field("cpml_boundary", &self.cpml_boundary)
            .field("spatial_order", &self.spatial_order)
            .field(
                "gpu_accelerator",
                &self.gpu_accelerator.as_ref().map(|_| "GpuAccelerator"),
            )
            .field("source_handler", &self.source_handler)
            // dynamic_sources contains Arc<dyn Source> which might not impl Debug properly if not supertrait
            // If it worked with derive, it should work here.
            // But to be safe and minimalistic, I'll print count.
            .field("dynamic_sources_count", &self.dynamic_sources.len())
            .field("sensor_recorder", &self.sensor_recorder)
            .field("time_step_index", &self.time_step_index)
            .field("fields", &self.fields)
            .field("materials", &self.materials)
            .finish()
    }
}

impl crate::solver::interface::Solver for FdtdSolver {
    fn name(&self) -> &str {
        "FDTD"
    }

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        // Core initialization happens in new(), this could be used for re-init
        Ok(())
    }

    fn add_source(&mut self, source: Box<dyn crate::domain::source::Source>) -> KwaversResult<()> {
        self.add_source_arc(Arc::from(source))
    }

    fn add_sensor(&mut self, _sensor: &crate::domain::sensor::GridSensorSet) -> KwaversResult<()> {
        // Map GridSensorSet to SensorRecorder logic
        // self.sensor_recorder.add_sensor(sensor);
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
            computation_time: std::time::Duration::default(), // Metrics need to track this
            memory_usage: 0,                                  // Estimator needed
            max_pressure,
            max_velocity,
        }
    }

    fn supports_feature(&self, feature: crate::solver::interface::SolverFeature) -> bool {
        matches!(
            feature,
            crate::solver::interface::SolverFeature::MultiThreaded
        ) || (matches!(
            feature,
            crate::solver::interface::SolverFeature::GpuAcceleration
        ) && self.gpu_accelerator.is_some())
    }

    fn enable_feature(
        &mut self,
        feature: crate::solver::interface::SolverFeature,
        enable: bool,
    ) -> KwaversResult<()> {
        match feature {
            crate::solver::interface::SolverFeature::GpuAcceleration => {
                if enable && self.gpu_accelerator.is_none() {
                    return Err(KwaversError::Config(
                        crate::core::error::ConfigError::InvalidValue {
                            parameter: "enable_gpu_acceleration".to_string(),
                            value: "true".to_string(),
                            constraint: "GPU accelerator must be configured".to_string(),
                        },
                    ));
                }
                self.config.enable_gpu_acceleration = enable;
                Ok(())
            }
            _ => Ok(()), // Ignore unsupported for now or error
        }
    }
}

pub trait FdtdGpuAccelerator: Send + Sync + std::fmt::Debug {
    fn propagate_acoustic_wave(
        &self,
        pressure: &Array3<f64>,
        velocity_x: &Array3<f64>,
        velocity_y: &Array3<f64>,
        velocity_z: &Array3<f64>,
        density: &Array3<f64>,
        sound_speed: &Array3<f64>,
        dt: f64,
        dx: f64,
        dy: f64,
        dz: f64,
    ) -> KwaversResult<Array3<f64>>;
}
