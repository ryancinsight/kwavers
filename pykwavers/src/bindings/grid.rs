use kwavers::domain::grid::Grid as KwaversGrid;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

/// Solver type selection.
///
/// Mathematical Specifications:
/// - FDTD: Finite Difference Time Domain (2nd/4th/6th order spatial accuracy)
/// - PSTD: Pseudospectral Time Domain (spectral spatial accuracy)
/// - Hybrid: Adaptive switching between FDTD and PSTD
///
/// References:
/// - Treeby & Cox (2010) for PSTD implementation
/// - Taflove & Hagness (2005) for FDTD fundamentals
#[pyclass]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolverType {
    /// Finite Difference Time Domain solver
    FDTD,
    /// Pseudospectral Time Domain solver
    PSTD,
    /// Hybrid FDTD/PSTD solver
    Hybrid,
    /// GPU-resident Pseudospectral Time Domain solver (requires `gpu` feature).
    /// Falls back to CPU PSTD if no GPU adapter is available.
    PstdGpu,
}

#[pymethods]
impl SolverType {
    pub(crate) fn __repr__(&self) -> String {
        match self {
            SolverType::FDTD => "SolverType.FDTD".to_string(),
            SolverType::PSTD => "SolverType.PSTD".to_string(),
            SolverType::Hybrid => "SolverType.Hybrid".to_string(),
            SolverType::PstdGpu => "SolverType.PstdGpu".to_string(),
        }
    }

    fn __str__(&self) -> String {
        match self {
            SolverType::FDTD => "FDTD".to_string(),
            SolverType::PSTD => "PSTD".to_string(),
            SolverType::Hybrid => "Hybrid".to_string(),
            SolverType::PstdGpu => "PstdGpu".to_string(),
        }
    }

    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
}

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
#[pyclass]
#[derive(Clone)]
pub struct Grid {
    pub(crate) inner: KwaversGrid,
}

#[pymethods]
impl Grid {
    #[new]
    #[pyo3(signature = (nx, ny, nz, dx, dy, dz))]
    fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> PyResult<Self> {
        let inner = KwaversGrid::new(nx, ny, nz, dx, dy, dz)
            .map_err(|e| PyRuntimeError::new_err(format!("Grid creation failed: {:?}", e)))?;
        Ok(Grid { inner })
    }

    #[getter]
    fn nx(&self) -> usize {
        self.inner.nx
    }

    #[getter]
    fn ny(&self) -> usize {
        self.inner.ny
    }

    #[getter]
    fn nz(&self) -> usize {
        self.inner.nz
    }

    #[getter]
    fn dx(&self) -> f64 {
        self.inner.dx
    }

    #[getter]
    fn dy(&self) -> f64 {
        self.inner.dy
    }

    #[getter]
    fn dz(&self) -> f64 {
        self.inner.dz
    }

    #[getter]
    fn lx(&self) -> f64 {
        self.inner.nx as f64 * self.inner.dx
    }

    #[getter]
    fn ly(&self) -> f64 {
        self.inner.ny as f64 * self.inner.dy
    }

    #[getter]
    fn lz(&self) -> f64 {
        self.inner.nz as f64 * self.inner.dz
    }

    fn total_points(&self) -> usize {
        self.inner.size()
    }

    fn dimensions(&self) -> (usize, usize, usize) {
        (self.inner.nx, self.inner.ny, self.inner.nz)
    }

    fn spacing(&self) -> (f64, f64, f64) {
        (self.inner.dx, self.inner.dy, self.inner.dz)
    }

    pub(crate) fn __repr__(&self) -> String {
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
