//! Workspace management for Kuznetsov solver
//!
//! This module provides pre-allocated workspace arrays to eliminate
//! heap allocations in the main simulation loop.

use super::spectral::SpectralOperator;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::solver::workspace::ScratchArena;
use ndarray::Array3;

/// Comprehensive workspace for Kuznetsov equation solver
///
/// Pre-allocates all temporary arrays needed for computation to avoid
/// allocations in the hot loop.
#[derive(Debug)]
pub struct KuznetsovWorkspace {
    /// Spectral operator for FFT-based derivatives
    pub spectral_op: SpectralOperator,

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
}

impl KuznetsovWorkspace {
    /// Create a new workspace for the given grid
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        let shape = (grid.nx, grid.ny, grid.nz);

        Ok(Self {
            spectral_op: SpectralOperator::new(grid),

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
        })
    }

    /// Update time history buffers
    pub fn update_time_history(&mut self, current_pressure: &Array3<f64>) {
        // Shift history: prev3 <- prev2 <- prev <- current
        self.pressure_prev3.assign(&self.pressure_prev2);
        self.pressure_prev2.assign(&self.pressure_prev);
        self.pressure_prev.assign(current_pressure);
    }

    /// Zero all 14 scratch buffers without reallocating.
    ///
    /// The `SpectralOperator` is not a scratch buffer — it holds grid-derived
    /// wavenumber constants and is excluded from this reset.
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::solver::workspace::ScratchArena;

    #[test]
    fn scratch_arena_memory_bytes_is_14n_f64() {
        let grid = Grid::new(8, 8, 8, 1e-4, 1e-4, 1e-4).unwrap();
        let ws = KuznetsovWorkspace::new(&grid).unwrap();
        let n = 8 * 8 * 8;
        let expected = 14 * n * std::mem::size_of::<f64>();
        assert_eq!(
            ws.memory_bytes(),
            expected,
            "KuznetsovWorkspace footprint must equal 14 × N × sizeof(f64)"
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
        assert!(
            ws.pressure_prev.iter().all(|&v| v == 0.0),
            "pressure_prev not zeroed"
        );
        assert!(
            ws.pressure_prev2.iter().all(|&v| v == 0.0),
            "pressure_prev2 not zeroed"
        );
        assert!(
            ws.pressure_prev3.iter().all(|&v| v == 0.0),
            "pressure_prev3 not zeroed"
        );
        assert!(
            ws.nonlinear_term.iter().all(|&v| v == 0.0),
            "nonlinear_term not zeroed"
        );
        assert!(
            ws.diffusive_term.iter().all(|&v| v == 0.0),
            "diffusive_term not zeroed"
        );
        assert!(
            ws.laplacian.iter().all(|&v| v == 0.0),
            "laplacian not zeroed"
        );
        assert!(ws.grad_x.iter().all(|&v| v == 0.0), "grad_x not zeroed");
        assert!(ws.grad_y.iter().all(|&v| v == 0.0), "grad_y not zeroed");
        assert!(ws.grad_z.iter().all(|&v| v == 0.0), "grad_z not zeroed");
        assert!(ws.k1.iter().all(|&v| v == 0.0), "k1 not zeroed");
        assert!(ws.k2.iter().all(|&v| v == 0.0), "k2 not zeroed");
        assert!(ws.k3.iter().all(|&v| v == 0.0), "k3 not zeroed");
        assert!(ws.k4.iter().all(|&v| v == 0.0), "k4 not zeroed");
        assert!(
            ws.temp_field.iter().all(|&v| v == 0.0),
            "temp_field not zeroed"
        );
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
    /// Returns the byte footprint of the 14 pre-allocated `Array3<f64>` scratch
    /// buffers.  The `SpectralOperator` (wavenumber tables) is a grid constant,
    /// not scratch storage, and is excluded.
    ///
    /// Footprint = 14 × N × 8  where N = nx × ny × nz.
    fn memory_bytes(&self) -> usize {
        // pressure_prev, pressure_prev2, pressure_prev3 (3)
        // nonlinear_term, diffusive_term, laplacian     (3)
        // grad_x, grad_y, grad_z                       (3)
        // k1, k2, k3, k4, temp_field                   (5)
        //                              total = 14 Array3<f64>
        14 * self.pressure_prev.len() * std::mem::size_of::<f64>()
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
    }
}
