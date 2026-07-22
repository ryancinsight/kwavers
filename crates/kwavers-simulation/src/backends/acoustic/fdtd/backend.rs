//! `FdtdBackend` struct definition and construction.

use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::mechanics::acoustic_wave::AcousticSpatialOrder;
use kwavers_solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};
use kwavers_source::GridSource;
use leto::Array3;

/// FDTD solver backend adapter.
///
/// Wraps the low-level `FdtdSolver` to provide the `AcousticSolverBackend`
/// interface required by simulation orchestration.
///
/// ## CFL Time Step
///
/// `dt = 0.5 * dx_min / (c_max * √3)`  (conservative, below 3D stability limit 1/√3)
#[derive(Debug)]
pub struct FdtdBackend {
    /// Underlying FDTD solver.
    pub(super) solver: FdtdSolver,
    pub(super) pressure: Array3<f64>,
    pub(super) ux: Array3<f64>,
    pub(super) uy: Array3<f64>,
    pub(super) uz: Array3<f64>,
    /// Current simulation time (s).
    pub(super) current_time: f64,
    /// Cached grid dimensions (nx, ny, nz).
    pub(super) grid_dims: (usize, usize, usize),
}

impl FdtdBackend {
    /// Create a new FDTD backend.
    ///
    /// Configures the solver with CFL-stable time step, PML absorbing boundaries,
    /// and the specified spatial accuracy order.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        spatial_order: AcousticSpatialOrder,
    ) -> KwaversResult<Self> {
        let c_max = Self::estimate_max_sound_speed(medium, grid)?;
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let dt = Self::compute_stable_timestep(dx_min, c_max);

        let spatial_order_value = match spatial_order {
            AcousticSpatialOrder::Second => 2,
            AcousticSpatialOrder::Fourth => 4,
            AcousticSpatialOrder::Sixth => 6,
        };

        let config = FdtdConfig {
            spatial_order: spatial_order_value,
            staggered_grid: true,
            cfl_factor: 0.5,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            enable_nonlinear: false,
            kspace_correction: KSpaceCorrectionMode::None,
            nt: 1,
            dt,
            sensor_mask: None,
            geometry: Default::default(),
        };

        let source = GridSource::new_empty();
        let solver = FdtdSolver::new(config, grid, medium, source)?;
        let shape = (grid.nx, grid.ny, grid.nz);

        let mut backend = Self {
            solver,
            pressure: Array3::zeros(shape),
            ux: Array3::zeros(shape),
            uy: Array3::zeros(shape),
            uz: Array3::zeros(shape),
            current_time: 0.0,
            grid_dims: (grid.nx, grid.ny, grid.nz),
        };
        backend.sync_shadow_fields();
        Ok(backend)
    }

    /// Compute a CFL-stable FDTD time step.
    ///
    /// `dt = CFL_safety * dx / (c_max * √3)` with `CFL_safety = 0.5`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(crate) fn compute_stable_timestep(dx: f64, c_max: f64) -> f64 {
        const CFL_SAFETY_FACTOR: f64 = 0.5;
        const SQRT_3: f64 = 1.732050807568877;
        CFL_SAFETY_FACTOR * dx / (c_max * SQRT_3)
    }

    /// Estimate the maximum sound speed by sparse grid sampling.
    ///
    /// Samples an 8×8×8 sparse grid for efficiency. Over-estimating c_max
    /// is conservative: it produces a smaller (more stable) dt.
    /// # Errors
    /// - Returns `KwaversError::InvalidInput` if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(crate) fn estimate_max_sound_speed(medium: &dyn Medium, grid: &Grid) -> KwaversResult<f64> {
        const SAMPLE_POINTS: usize = 8;
        let mut c_max = 0.0;

        let di = (grid.nx / SAMPLE_POINTS).max(1);
        let dj = (grid.ny / SAMPLE_POINTS).max(1);
        let dk = (grid.nz / SAMPLE_POINTS).max(1);

        for k in (0..grid.nz).step_by(dk.max(1)) {
            for j in (0..grid.ny).step_by(dj.max(1)) {
                for i in (0..grid.nx).step_by(di.max(1)) {
                    let (x, y, z) = grid.indices_to_coordinates(i, j, k);
                    let c = kwavers_medium::sound_speed_at(medium, x, y, z, grid);
                    if c > c_max {
                        c_max = c;
                    }
                }
            }
        }

        if c_max <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Maximum sound speed must be positive".into(),
            ));
        }

        Ok(c_max)
    }

    pub(super) fn sync_shadow_fields(&mut self) {
        for (dst, src) in self.pressure.iter_mut().zip(self.solver.fields.p.iter()) {
            *dst = *src;
        }
        for (dst, src) in self.ux.iter_mut().zip(self.solver.fields.ux.iter()) {
            *dst = *src;
        }
        for (dst, src) in self.uy.iter_mut().zip(self.solver.fields.uy.iter()) {
            *dst = *src;
        }
        for (dst, src) in self.uz.iter_mut().zip(self.solver.fields.uz.iter()) {
            *dst = *src;
        }
    }
}