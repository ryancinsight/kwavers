//! `Grid` pyclass — computational domain wrapper.

use kwavers_grid::Grid as KwaversGrid;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Computational grid for acoustic simulation.
///
/// Mathematical Specification:
/// - Domain: Ω = [0, Lx] × [0, Ly] × [0, Lz]
/// - Grid points: (xi, yj, zk) where i ∈ [0, Nx), j ∈ [0, Ny), k ∈ [0, Nz)
/// - Spacing: xi = i·dx, yj = j·dy, zk = k·dz
/// - Total points: N = Nx × Ny × Nz
///
/// Equivalent to k-Wave's kWaveGrid:
/// ```python
/// grid = Grid(nx=128, ny=128, nz=128, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
/// ```
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Grid {
    /// Internal kwavers grid
    pub(crate) inner: KwaversGrid,
}

#[pymethods]
impl Grid {
    /// Create a new 3D grid.
    ///
    /// Parameters
    /// ----------
    /// nx : int
    ///     Number of grid points in x-direction
    /// ny : int
    ///     Number of grid points in y-direction
    /// nz : int
    ///     Number of grid points in z-direction
    /// dx : float
    ///     Grid spacing in x-direction [m]
    /// dy : float
    ///     Grid spacing in y-direction [m]
    /// dz : float
    ///     Grid spacing in z-direction [m]
    ///
    /// Returns
    /// -------
    /// Grid
    ///     Computational grid
    ///
    /// Examples
    /// --------
    /// >>> grid = Grid(128, 128, 128, 0.1e-3, 0.1e-3, 0.1e-3)
    /// >>> print(grid.total_points())
    /// 2097152
    #[new]
    #[pyo3(signature = (nx, ny, nz, dx, dy, dz))]
    pub fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> PyResult<Self> {
        let inner = KwaversGrid::new(nx, ny, nz, dx, dy, dz)
            .map_err(|e| PyRuntimeError::new_err(format!("Grid creation failed: {:?}", e)))?;
        Ok(Grid { inner })
    }

    /// Number of grid points in x-direction.
    #[getter]
    fn nx(&self) -> usize {
        self.inner.nx
    }

    /// Number of grid points in y-direction.
    #[getter]
    fn ny(&self) -> usize {
        self.inner.ny
    }

    /// Number of grid points in z-direction.
    #[getter]
    fn nz(&self) -> usize {
        self.inner.nz
    }

    /// Grid spacing in x-direction [m].
    #[getter]
    fn dx(&self) -> f64 {
        self.inner.dx
    }

    /// Grid spacing in y-direction [m].
    #[getter]
    fn dy(&self) -> f64 {
        self.inner.dy
    }

    /// Grid spacing in z-direction [m].
    #[getter]
    fn dz(&self) -> f64 {
        self.inner.dz
    }

    /// Domain size in x-direction [m].
    #[getter]
    fn lx(&self) -> f64 {
        self.inner.nx as f64 * self.inner.dx
    }

    /// Domain size in y-direction [m].
    #[getter]
    fn ly(&self) -> f64 {
        self.inner.ny as f64 * self.inner.dy
    }

    /// Domain size in z-direction [m].
    #[getter]
    fn lz(&self) -> f64 {
        self.inner.nz as f64 * self.inner.dz
    }

    /// Total number of grid points.
    fn total_points(&self) -> usize {
        self.inner.size()
    }

    /// Grid dimensions as tuple (nx, ny, nz).
    fn dimensions(&self) -> (usize, usize, usize) {
        (self.inner.nx, self.inner.ny, self.inner.nz)
    }

    /// Grid spacing as tuple (dx, dy, dz).
    fn spacing(&self) -> (f64, f64, f64) {
        (self.inner.dx, self.inner.dy, self.inner.dz)
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "Grid(nx={}, ny={}, nz={}, dx={:.2e}, dy={:.2e}, dz={:.2e})",
            self.inner.nx,
            self.inner.ny,
            self.inner.nz,
            self.inner.dx,
            self.inner.dy,
            self.inner.dz
        )
    }

    /// Human-readable string.
    fn __str__(&self) -> String {
        format!(
            "Grid: {}×{}×{} points, {:.2e}×{:.2e}×{:.2e} m spacing, {:.2e}×{:.2e}×{:.2e} m domain",
            self.inner.nx,
            self.inner.ny,
            self.inner.nz,
            self.inner.dx,
            self.inner.dy,
            self.inner.dz,
            self.inner.nx as f64 * self.inner.dx,
            self.inner.ny as f64 * self.inner.dy,
            self.inner.nz as f64 * self.inner.dz
        )
    }
}
