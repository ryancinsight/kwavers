//! `FdtdBackend` struct definition and construction.

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;
use crate::domain::medium::Medium;
use crate::domain::source::GridSource;
use crate::physics::acoustics::mechanics::acoustic_wave::SpatialOrder;
use crate::solver::forward::fdtd::{FdtdConfig, FdtdSolver, KSpaceCorrectionMode};

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
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        grid: &Grid,
        medium: &dyn Medium,
        spatial_order: SpatialOrder,
    ) -> KwaversResult<Self> {
        let c_max = Self::estimate_max_sound_speed(medium, grid)?;
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let dt = Self::compute_stable_timestep(dx_min, c_max);

        let spatial_order_value = match spatial_order {
            SpatialOrder::Second => 2,
            SpatialOrder::Fourth => 4,
            SpatialOrder::Sixth => 6,
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

        Ok(Self {
            solver,
            current_time: 0.0,
            grid_dims: (grid.nx, grid.ny, grid.nz),
        })
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
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
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
                    let c = crate::domain::medium::sound_speed_at(medium, x, y, z, grid);
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
}
