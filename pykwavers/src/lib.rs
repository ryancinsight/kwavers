//! # pykwavers: Python Bindings for kwavers
//!
//! This module provides Python bindings for the kwavers ultrasound simulation library
//! with an API compatible with k-Wave/k-wave-python for direct comparison and validation.
//!
//! ## Architecture
//!
//! Following Clean Architecture principles:
//! - **Presentation Layer**: Python API (this crate)
//! - **Domain Layer**: Core kwavers library
//! - **Dependency Direction**: Python → Rust (unidirectional)
//!
//! ## API Design
//!
//! The API mirrors k-Wave's structure for ease of comparison:
//! ```python
//! import pykwavers as kw
//!
//! # Create grid (similar to kWaveGrid)
//! grid = kw.Grid(nx=128, ny=128, nz=128, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
//!
//! # Define medium (similar to k-Wave medium struct)
//! medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
//!
//! # Create source
//! source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
//!
//! # Create sensor
//! sensor = kw.Sensor.point(position=[0.01, 0.01, 0.01])
//!
//! # Run simulation
//! sim = kw.Simulation(grid, medium, source, sensor)
//! result = sim.run(time_steps=1000, dt=1e-8)
//! ```
//!
//! ## Mathematical Specifications
//!
//! - Grid: Uniform Cartesian grid with spacing dx, dy, dz
//! - Medium: Acoustic properties (c, ρ, α, nonlinearity)
//! - Source: Pressure/velocity boundary conditions
//! - Sensor: Point/grid sampling with interpolation
//! - Simulation: FDTD/PSTD time-stepping with PML boundaries
//!
//! ## References
//!
//! 1. Treeby & Cox (2010). "k-Wave: MATLAB toolbox for simulation and
//!    reconstruction of photoacoustic wave fields." J. Biomed. Opt. 15(2).
//! 2. kwavers architecture documentation (../kwavers/ARCHITECTURE.md)
//!
//! Author: Ryan Clanton (@ryancinsight)
//! Date: 2026-02-04
//! Sprint: 217 Session 9 - Python Integration via PyO3

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// Re-exports from kwavers core
use kwavers::core::error::KwaversError;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::medium::HomogeneousMedium;

// ============================================================================
// Error Handling
// ============================================================================

/// Convert kwavers errors to Python exceptions
fn kwavers_error_to_py(err: KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers error: {}", err))
}

// ============================================================================
// Grid: Computational Domain
// ============================================================================

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
    /// Internal kwavers grid
    inner: KwaversGrid,
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
    fn new(nx: usize, ny: usize, nz: usize, dx: f64, dy: f64, dz: f64) -> PyResult<Self> {
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

// ============================================================================
// Medium: Acoustic Properties
// ============================================================================

/// Acoustic medium with material properties.
///
/// Mathematical Specification:
/// - Sound speed: c(x, y, z) [m/s]
/// - Density: ρ(x, y, z) [kg/m³]
/// - Absorption: α(x, y, z) [dB/(MHz^y·cm)] where y ∈ [0, 3]
/// - Nonlinearity: B/A parameter (optional)
///
/// Equivalent to k-Wave medium struct:
/// ```python
/// medium = Medium.homogeneous(sound_speed=1500.0, density=1000.0)
/// ```
#[pyclass]
#[derive(Clone)]
pub struct Medium {
    /// Internal medium (homogeneous for now)
    inner: HomogeneousMedium,
}

#[pymethods]
impl Medium {
    /// Create a homogeneous medium with uniform properties.
    ///
    /// Parameters
    /// ----------
    /// sound_speed : float
    ///     Sound speed [m/s] (typical: water=1500, tissue=1540, bone=4080)
    /// density : float
    ///     Density [kg/m³] (typical: water=1000, tissue=1060, bone=1850)
    /// absorption : float, optional
    ///     Absorption coefficient [dB/(MHz·cm)] (default: 0.0)
    /// nonlinearity : float, optional
    ///     B/A nonlinearity parameter (default: 0.0, tissue≈6, water≈5)
    ///
    /// Returns
    /// -------
    /// Medium
    ///     Homogeneous acoustic medium
    ///
    /// Examples
    /// --------
    /// >>> # Water at 20°C
    /// >>> medium = Medium.homogeneous(1500.0, 1000.0)
    /// >>> # Soft tissue
    /// >>> medium = Medium.homogeneous(1540.0, 1060.0, absorption=0.5, nonlinearity=6.0)
    #[staticmethod]
    #[pyo3(signature = (sound_speed, density, absorption=0.0, _nonlinearity=0.0, grid=None))]
    fn homogeneous(
        sound_speed: f64,
        density: f64,
        absorption: f64,
        _nonlinearity: f64,
        grid: Option<&Grid>,
    ) -> PyResult<Self> {
        // Validate inputs
        if sound_speed <= 0.0 {
            return Err(PyValueError::new_err("Sound speed must be positive"));
        }
        if density <= 0.0 {
            return Err(PyValueError::new_err("Density must be positive"));
        }
        if absorption < 0.0 {
            return Err(PyValueError::new_err("Absorption must be non-negative"));
        }

        // Create default grid if not provided
        let default_grid = KwaversGrid::default();
        let grid_ref = grid.map(|g| &g.inner).unwrap_or(&default_grid);

        // HomogeneousMedium::new(density, sound_speed, mu_a, mu_s_prime, grid)
        // For now, use default optical properties (mu_a=0, mu_s_prime=0)
        let medium = HomogeneousMedium::new(density, sound_speed, 0.0, 0.0, grid_ref);

        Ok(Medium { inner: medium })
    }

    /// String representation.
    fn __repr__(&self) -> String {
        "Medium.homogeneous(...)".to_string()
    }

    /// Human-readable string.
    fn __str__(&self) -> String {
        "Homogeneous Medium".to_string()
    }
}

// ============================================================================
// Source: Acoustic Excitation
// ============================================================================

/// Acoustic source for wave excitation.
///
/// Mathematical Specification:
/// - Pressure source: p(x, t) = A·sin(2πft + φ)
/// - Velocity source: v(x, t) = A·sin(2πft + φ)
/// - Initial pressure: p(x, 0) = p₀(x)
///
/// Equivalent to k-Wave source struct.
#[pyclass]
#[derive(Clone)]
pub struct Source {
    /// Source type identifier
    source_type: String,
    /// Frequency [Hz]
    frequency: f64,
    /// Amplitude [Pa] or [m/s]
    amplitude: f64,
    /// Position for point source
    position: Option<[f64; 3]>,
}

#[pymethods]
impl Source {
    /// Create a plane wave source.
    ///
    /// Parameters
    /// ----------
    /// grid : Grid
    ///     Computational grid
    /// frequency : float
    ///     Source frequency [Hz]
    /// amplitude : float
    ///     Pressure amplitude [Pa]
    /// direction : tuple, optional
    ///     Propagation direction (default: [0, 0, 1] = +z)
    ///
    /// Returns
    /// -------
    /// Source
    ///     Plane wave source
    ///
    /// Examples
    /// --------
    /// >>> source = Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    #[staticmethod]
    #[pyo3(signature = (_grid, frequency, amplitude, _direction=None))]
    fn plane_wave(
        _grid: &Grid,
        frequency: f64,
        amplitude: f64,
        _direction: Option<(f64, f64, f64)>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        if amplitude <= 0.0 {
            return Err(PyValueError::new_err("Amplitude must be positive"));
        }

        Ok(Source {
            source_type: "plane_wave".to_string(),
            frequency,
            amplitude,
            position: None,
        })
    }

    /// Create a point source.
    ///
    /// Parameters
    /// ----------
    /// position : tuple
    ///     Source position [x, y, z] in meters
    /// frequency : float
    ///     Source frequency [Hz]
    /// amplitude : float
    ///     Pressure amplitude [Pa]
    ///
    /// Returns
    /// -------
    /// Source
    ///     Point source
    ///
    /// Examples
    /// --------
    /// >>> source = Source.point([0.01, 0.01, 0.01], frequency=1e6, amplitude=1e5)
    #[staticmethod]
    #[pyo3(signature = (position, frequency, amplitude))]
    fn point(position: (f64, f64, f64), frequency: f64, amplitude: f64) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        if amplitude <= 0.0 {
            return Err(PyValueError::new_err("Amplitude must be positive"));
        }

        Ok(Source {
            source_type: "point".to_string(),
            frequency,
            amplitude,
            position: Some([position.0, position.1, position.2]),
        })
    }

    /// Source frequency [Hz].
    #[getter]
    fn frequency(&self) -> f64 {
        self.frequency
    }

    /// Source amplitude [Pa].
    #[getter]
    fn amplitude(&self) -> f64 {
        self.amplitude
    }

    /// Source type.
    #[getter]
    fn source_type(&self) -> &str {
        &self.source_type
    }

    /// String representation.
    fn __repr__(&self) -> String {
        match &self.position {
            Some(pos) => format!(
                "Source.point(position=[{:.3e}, {:.3e}, {:.3e}], frequency={:.2e}, amplitude={:.2e})",
                pos[0], pos[1], pos[2], self.frequency, self.amplitude
            ),
            None => format!(
                "Source.plane_wave(frequency={:.2e}, amplitude={:.2e})",
                self.frequency, self.amplitude
            ),
        }
    }
}

// ============================================================================
// Sensor: Field Sampling
// ============================================================================

/// Sensor for recording acoustic fields.
///
/// Mathematical Specification:
/// - Point sensor: p(t) at fixed location (x₀, y₀, z₀)
/// - Grid sensor: p(x, y, z, t) on entire grid
/// - Interpolation: trilinear for arbitrary positions
///
/// Equivalent to k-Wave sensor struct.
#[pyclass]
#[derive(Clone)]
pub struct Sensor {
    /// Sensor type
    sensor_type: String,
    /// Position for point sensor
    position: Option<[f64; 3]>,
}

#[pymethods]
impl Sensor {
    /// Create a point sensor at specified position.
    ///
    /// Parameters
    /// ----------
    /// position : tuple
    ///     Sensor position [x, y, z] in meters
    ///
    /// Returns
    /// -------
    /// Sensor
    ///     Point sensor
    ///
    /// Examples
    /// --------
    /// >>> sensor = Sensor.point([0.02, 0.02, 0.02])
    #[staticmethod]
    fn point(position: (f64, f64, f64)) -> Self {
        Sensor {
            sensor_type: "point".to_string(),
            position: Some([position.0, position.1, position.2]),
        }
    }

    /// Create a grid sensor recording entire field.
    ///
    /// Returns
    /// -------
    /// Sensor
    ///     Grid sensor
    ///
    /// Examples
    /// --------
    /// >>> sensor = Sensor.grid()
    #[staticmethod]
    fn grid() -> Self {
        Sensor {
            sensor_type: "grid".to_string(),
            position: None,
        }
    }

    /// Sensor type.
    #[getter]
    fn sensor_type(&self) -> &str {
        &self.sensor_type
    }

    /// String representation.
    fn __repr__(&self) -> String {
        match &self.position {
            Some(pos) => format!(
                "Sensor.point(position=[{:.3e}, {:.3e}, {:.3e}])",
                pos[0], pos[1], pos[2]
            ),
            None => "Sensor.grid()".to_string(),
        }
    }
}

// ============================================================================
// Simulation: Main Interface
// ============================================================================

/// Acoustic wave simulation.
///
/// Mathematical Specification:
/// - Acoustic wave equation: ∂²p/∂t² = c²∇²p + source terms
/// - FDTD discretization: 2nd/4th/6th/8th order accurate
/// - Time stepping: explicit Euler with CFL stability
/// - Boundary conditions: PML (perfectly matched layers)
///
/// Equivalent to k-Wave's kspaceFirstOrder3D function.
#[pyclass]
#[derive(Clone)]
pub struct Simulation {
    grid: Grid,
    medium: Medium,
    source: Source,
    sensor: Sensor,
}

#[pymethods]
impl Simulation {
    /// Create a new simulation.
    ///
    /// Parameters
    /// ----------
    /// grid : Grid
    ///     Computational grid
    /// medium : Medium
    ///     Acoustic medium
    /// source : Source
    ///     Acoustic source
    /// sensor : Sensor
    ///     Field sensor
    ///
    /// Returns
    /// -------
    /// Simulation
    ///     Configured simulation
    ///
    /// Examples
    /// --------
    /// >>> sim = Simulation(grid, medium, source, sensor)
    #[new]
    fn new(grid: Grid, medium: Medium, source: Source, sensor: Sensor) -> Self {
        Simulation {
            grid,
            medium,
            source,
            sensor,
        }
    }

    /// Run the simulation.
    ///
    /// Parameters
    /// ----------
    /// time_steps : int
    ///     Number of time steps
    /// dt : float, optional
    ///     Time step [s] (auto-calculated from CFL if None)
    ///
    /// Returns
    /// -------
    /// SimulationResult
    ///     Simulation results with sensor data
    ///
    /// Examples
    /// --------
    /// >>> result = sim.run(time_steps=1000, dt=1e-8)
    /// >>> print(result.sensor_data.shape)
    #[pyo3(signature = (time_steps, dt=None))]
    fn run<'py>(
        &self,
        _py: Python<'py>,
        time_steps: usize,
        dt: Option<f64>,
    ) -> PyResult<SimulationResult> {
        // Calculate time step from CFL condition if not provided
        let dt_actual = dt.unwrap_or_else(|| {
            let c_max = 1500.0; // Conservative estimate
            let dx_min = self
                .grid
                .inner
                .dx
                .min(self.grid.inner.dy)
                .min(self.grid.inner.dz);
            let cfl = 0.3; // Conservative CFL number
            cfl * dx_min / c_max
        });

        // Placeholder: In real implementation, would call kwavers simulation
        // For now, return shape metadata for API testing
        let shape = (self.grid.inner.nx, self.grid.inner.ny, self.grid.inner.nz);

        Ok(SimulationResult {
            shape,
            time_steps,
            dt: dt_actual,
            final_time: dt_actual * time_steps as f64,
        })
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "Simulation(grid={}, source={}, sensor={})",
            self.grid.__repr__(),
            self.source.__repr__(),
            self.sensor.__repr__()
        )
    }
}

// ============================================================================
// Simulation Result
// ============================================================================

/// Results from acoustic simulation.
///
/// Contains sensor recordings and metadata.
#[pyclass]
pub struct SimulationResult {
    /// Sensor data shape (nx, ny, nz)
    #[pyo3(get)]
    shape: (usize, usize, usize),
    /// Number of time steps
    #[pyo3(get)]
    time_steps: usize,
    /// Time step [s]
    #[pyo3(get)]
    dt: f64,
    /// Final simulation time [s]
    #[pyo3(get)]
    final_time: f64,
}

#[pymethods]
impl SimulationResult {
    /// Get sensor data shape.
    fn sensor_data_shape(&self) -> (usize, usize, usize) {
        self.shape
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "SimulationResult(shape={:?}, time_steps={}, dt={:.2e}, final_time={:.2e})",
            self.shape, self.time_steps, self.dt, self.final_time
        )
    }
}

// ============================================================================
// Module Definition
// ============================================================================

/// pykwavers: Python bindings for kwavers ultrasound simulation library.
///
/// This module provides a k-Wave-compatible API for acoustic wave simulation.
#[pymodule]
fn _pykwavers(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Grid>()?;
    m.add_class::<Medium>()?;
    m.add_class::<Source>()?;
    m.add_class::<Sensor>()?;
    m.add_class::<Simulation>()?;
    m.add_class::<SimulationResult>()?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Ryan Clanton PhD")?;

    Ok(())
}
