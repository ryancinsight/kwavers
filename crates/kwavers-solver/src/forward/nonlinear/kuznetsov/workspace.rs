//! Workspace management for Kuznetsov solver
//!
//! This module provides pre-allocated workspace arrays to eliminate
//! heap allocations in the main simulation loop.

use super::spectral::KuznetsovSpectralOperator;
use kwavers_core::error::KwaversResult;
use kwavers_domain::grid::Grid;
use crate::workspace::ScratchArena;
use ndarray::Array3;

/// Comprehensive workspace for Kuznetsov equation solver
///
/// Pre-allocates all temporary arrays needed for computation to avoid
/// allocations in the hot loop.
///
/// ## Memory layout
///
/// | Group | Arrays | Per-array |
/// |-------|--------|-----------|
/// | History (prev₁/₂/₃) | 3 | N × f64 |
/// | Term buffers (nl, diff, lap) | 3 | N × f64 |
/// | Gradient (∇p) | 3 | N × f64 |
/// | RK4 stages (k1–k4, temp) | 5 | N × f64 |
/// | Medium property cache (hetero) | 4 | N × f64 |
/// | **Total** | **18** | **N × f64** |
///
/// The medium property cache (`cache_density`, `cache_sound_speed`,
/// `cache_nonlinearity`, `cache_source`) is filled serially at the start
/// of `compute_rhs` for heterogeneous media, then the RHS combination is
/// computed in parallel without accessing the trait object.
#[derive(Debug)]
pub struct KuznetsovWorkspace {
    /// Spectral operator for FFT-based derivatives
    pub spectral_op: KuznetsovSpectralOperator,

    /// Pressure field at previous time steps (for finite differences)
    pub pressure_prev: Array3<f64>,
    pub pressure_prev2: Array3<f64>,
    pub pressure_prev3: Array3<f64>,

    /// Workspace for nonlinear term computation
    pub nonlinear_term: Array3<f64>,

    /// Workspace for diffusive term computation
    pub diffusive_term: Array3<f64>,

    /// Workspace for Laplacian computation
    pub laplacian: Array3<f64>,

    /// Workspace for gradient computation
    pub grad_x: Array3<f64>,
    pub grad_y: Array3<f64>,
    pub grad_z: Array3<f64>,

    /// RK4 intermediate stages
    pub k1: Array3<f64>,
    pub k2: Array3<f64>,
    pub k3: Array3<f64>,
    pub k4: Array3<f64>,

    /// Temporary field for RK4 stages
    pub temp_field: Array3<f64>,

    /// Medium property cache for heterogeneous RHS parallelisation.
    ///
    /// Filled serially once per `compute_rhs` call (trait objects are not
    /// `Sync`); thereafter the mathematical combination uses these arrays
    /// through `Zip::indexed().par_for_each()` without touching the trait
    /// objects — achieving full Rayon parallelism over the grid volume.
    pub cache_density: Array3<f64>,
    pub cache_sound_speed: Array3<f64>,
    pub cache_nonlinearity: Array3<f64>,
    pub cache_source: Array3<f64>,
}

impl KuznetsovWorkspace {
    /// Create a new workspace for the given grid
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        let shape = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            spectral_op: KuznetsovSpectralOperator::new(grid),

            // Time history buffers
            pressure_prev: Array3::zeros(shape),
            pressure_prev2: Array3::zeros(shape),
            pressure_prev3: Array3::zeros(shape),

            // Term computation buffers
            nonlinear_term: Array3::zeros(shape),
            diffusive_term: Array3::zeros(shape),
            laplacian: Array3::zeros(shape),

            // Gradient buffers
            grad_x: Array3::zeros(shape),
            grad_y: Array3::zeros(shape),
            grad_z: Array3::zeros(shape),

            // RK4 buffers
            k1: Array3::zeros(shape),
            k2: Array3::zeros(shape),
            k3: Array3::zeros(shape),
            k4: Array3::zeros(shape),
            temp_field: Array3::zeros(shape),

            // Medium property cache (heterogeneous path)
            cache_density: Array3::zeros(shape),
            cache_sound_speed: Array3::zeros(shape),
            cache_nonlinearity: Array3::zeros(shape),
            cache_source: Array3::zeros(shape),
        })
    }

    /// Update time history buffers
    pub fn update_time_history(&mut self, current_pressure: &Array3<f64>) {
        // Shift history: prev3 <- prev2 <- prev <- current
        self.pressure_prev3.assign(&self.pressure_prev2);
        self.pressure_prev2.assign(&self.pressure_prev);
        self.pressure_prev.assign(current_pressure);
    }

    /// Zero all 18 scratch buffers without reallocating.
    ///
    /// The `KuznetsovSpectralOperator` is not a scratch buffer — it holds grid-derived
    /// wavenumber constants and is excluded from this reset.
    /// The four medium-property cache arrays (`cache_*`) are included because
    /// stale cache values from a previous heterogeneous-medium step must not
    /// leak into a subsequent homogeneous step.
    pub fn clear(&mut self) {
        self.pressure_prev.fill(0.0);
        self.pressure_prev2.fill(0.0);
        self.pressure_prev3.fill(0.0);
        self.nonlinear_term.fill(0.0);
        self.diffusive_term.fill(0.0);
        self.laplacian.fill(0.0);
        self.grad_x.fill(0.0);
        self.grad_y.fill(0.0);
        self.grad_z.fill(0.0);
        self.k1.fill(0.0);
        self.k2.fill(0.0);
        self.k3.fill(0.0);
        self.k4.fill(0.0);
        self.temp_field.fill(0.0);
        self.cache_density.fill(0.0);
        self.cache_sound_speed.fill(0.0);
        self.cache_nonlinearity.fill(0.0);
        self.cache_source.fill(0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kwavers_domain::grid::Grid;
    use crate::workspace::ScratchArena;

    /// Workspace must allocate exactly 18 Array3<f64> buffers.
    ///
    /// Breakdown:
    ///   pressure_prev×3, nonlinear_term/diffusive_term/laplacian×3,
    ///   grad_x/y/z×3, k1–k4 + temp_field×5, cache_density/c0/nl/src×4
    ///   = 18 total.
    ///
    /// Analytical: 18 × 8×8×8 × 8 = 18 × 512 × 8 = 73 728 bytes.
    #[test]
    fn scratch_arena_memory_bytes_is_18_per_voxel() {
        let grid = Grid::new(8, 8, 8, 1e-4, 1e-4, 1e-4).unwrap();
        let ws = KuznetsovWorkspace::new(&grid).unwrap();
        let n = 8 * 8 * 8;
        let expected = 18 * n * std::mem::size_of::<f64>();
        assert_eq!(
            ws.memory_bytes(),
            expected,
            "KuznetsovWorkspace footprint must equal 18 × N × sizeof(f64)"
        );
    }

    #[test]
    fn scratch_arena_clear_zeros_all_buffers() {
        let grid = Grid::new(4, 4, 4, 1e-4, 1e-4, 1e-4).unwrap();
        let mut ws = KuznetsovWorkspace::new(&grid).unwrap();

        // Dirty every scratch buffer with a distinct non-zero sentinel.
        ws.pressure_prev.fill(1.0);
        ws.pressure_prev2.fill(2.0);
        ws.pressure_prev3.fill(3.0);
        ws.nonlinear_term.fill(4.0);
        ws.diffusive_term.fill(5.0);
        ws.laplacian.fill(6.0);
        ws.grad_x.fill(7.0);
        ws.grad_y.fill(8.0);
        ws.grad_z.fill(9.0);
        ws.k1.fill(10.0);
        ws.k2.fill(11.0);
        ws.k3.fill(12.0);
        ws.k4.fill(13.0);
        ws.temp_field.fill(14.0);

        ws.clear();

        // Every element of every scratch buffer must be exactly 0.0 after clear().
        for (arr, label) in [
            (ws.pressure_prev.view(), "pressure_prev"),
            (ws.pressure_prev2.view(), "pressure_prev2"),
            (ws.pressure_prev3.view(), "pressure_prev3"),
            (ws.nonlinear_term.view(), "nonlinear_term"),
            (ws.diffusive_term.view(), "diffusive_term"),
            (ws.laplacian.view(), "laplacian"),
            (ws.grad_x.view(), "grad_x"),
            (ws.grad_y.view(), "grad_y"),
            (ws.grad_z.view(), "grad_z"),
            (ws.k1.view(), "k1"),
            (ws.k2.view(), "k2"),
            (ws.k3.view(), "k3"),
            (ws.k4.view(), "k4"),
            (ws.temp_field.view(), "temp_field"),
            (ws.cache_density.view(), "cache_density"),
            (ws.cache_sound_speed.view(), "cache_sound_speed"),
            (ws.cache_nonlinearity.view(), "cache_nonlinearity"),
            (ws.cache_source.view(), "cache_source"),
        ] {
            assert!(
                arr.iter().all(|&v| v == 0.0),
                "{label} not zeroed after clear()"
            );
        }
    }

    #[test]
    fn scratch_arena_memory_bytes_stable_after_clear() {
        let grid = Grid::new(6, 6, 6, 1e-4, 1e-4, 1e-4).unwrap();
        let mut ws = KuznetsovWorkspace::new(&grid).unwrap();
        let bytes_before = ws.memory_bytes();
        ws.pressure_prev.fill(42.0);
        ws.clear();
        assert_eq!(
            ws.memory_bytes(),
            bytes_before,
            "memory_bytes() must be stable across clear()"
        );
    }
}

impl ScratchArena for KuznetsovWorkspace {
    /// Returns the byte footprint of the 18 pre-allocated `Array3<f64>` scratch
    /// buffers.  The `KuznetsovSpectralOperator` (wavenumber tables) is a grid constant,
    /// not scratch storage, and is excluded.
    ///
    /// Footprint = 18 × N × 8  where N = nx × ny × nz.
    ///
    /// The four additional buffers (vs. the original 14) are the medium
    /// property caches that enable full Rayon parallelism in the heterogeneous
    /// RHS path without requiring the `Medium` and `Source` traits to be `Sync`.
    fn memory_bytes(&self) -> usize {
        // pressure_prev, pressure_prev2, pressure_prev3          (3)
        // nonlinear_term, diffusive_term, laplacian              (3)
        // grad_x, grad_y, grad_z                                 (3)
        // k1, k2, k3, k4, temp_field                            (5)
        // cache_density, cache_sound_speed, cache_nonlinearity,
        //   cache_source                                          (4)
        //                                         total = 18 Array3<f64>
        18 * self.pressure_prev.len() * std::mem::size_of::<f64>()
    }

    fn clear(&mut self) {
        self.pressure_prev.fill(0.0);
        self.pressure_prev2.fill(0.0);
        self.pressure_prev3.fill(0.0);
        self.nonlinear_term.fill(0.0);
        self.diffusive_term.fill(0.0);
        self.laplacian.fill(0.0);
        self.grad_x.fill(0.0);
        self.grad_y.fill(0.0);
        self.grad_z.fill(0.0);
        self.k1.fill(0.0);
        self.k2.fill(0.0);
        self.k3.fill(0.0);
        self.k4.fill(0.0);
        self.temp_field.fill(0.0);
        self.cache_density.fill(0.0);
        self.cache_sound_speed.fill(0.0);
        self.cache_nonlinearity.fill(0.0);
        self.cache_source.fill(0.0);
    }
}
