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
//! - Simulation: FDTD/PSTD time-stepping with CPML boundaries
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

use kwavers::domain::sensor::recorder::config::RecordingMode;
use kwavers::domain::sensor::recorder::pressure_statistics::SampledStatistics;
use kwavers::domain::sensor::recorder::simple::SensorRecorder;
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

// Re-exports from kwavers core
use kwavers::core::error::{KwaversError, KwaversResult};
use kwavers::domain::boundary::cpml::CPMLConfig;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::medium::core::CoreMedium;
use kwavers::domain::medium::heterogeneous::HeterogeneousMedium;
use kwavers::domain::medium::traits::Medium as MediumTrait;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::signal::Signal;
use kwavers::domain::source::array_2d::{
    ApodizationType as KwaversApodizationType, TransducerArray2D as KwaversTransducerArray2D,
    TransducerArray2DConfig,
};
use kwavers::domain::source::custom::FunctionSource;
use kwavers::domain::source::wavefront::plane_wave::{
    InjectionMode, PlaneWaveConfig, PlaneWaveSource,
};
use kwavers::domain::source::{GridSource, Source as KwaversSource, SourceField};
use kwavers::physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers::solver::forward::fdtd::config::{FdtdConfig, KSpaceCorrectionMode};
use kwavers::solver::forward::fdtd::solver::FdtdSolver;
use kwavers::solver::forward::pstd::config::PSTDConfig;
use kwavers::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers::solver::interface::solver::Solver as SolverTrait;
use ndarray::{Array1, Array3};
use std::sync::Arc;
// ============================================================================
// Utility Function Bindings
// ============================================================================

mod utils_bindings;

// ============================================================================
// Error Handling
// ============================================================================

/// Convert kwavers errors to Python exceptions
fn kwavers_error_to_py(err: KwaversError) -> PyErr {
    PyRuntimeError::new_err(format!("kwavers error: {}", err))
}

/// Convert k-Wave absorption units dB/(MHz^y·cm) to Np/m at the given frequency.
///
/// This follows the standard scalar conversion
/// `alpha_np_m = alpha_db_cm * f_mhz^y * 100 / (20 / ln(10))`.
/// It is used by the GPU PSTD paths, which currently apply absorption using a
/// centre-frequency attenuation model rather than the full spectral Treeby/Cox
/// formulation used by the CPU PSTD solver.
fn alpha_db_cm_to_np_m(alpha_db_cm: f64, frequency_mhz: f64, alpha_power: f64) -> f64 {
    let db_to_np = 20.0 / std::f64::consts::LN_10;
    alpha_db_cm * frequency_mhz.powf(alpha_power) * 100.0 / db_to_np
}

#[cfg(test)]
mod physics_unit_tests {
    use super::alpha_db_cm_to_np_m;

    #[test]
    fn test_alpha_db_cm_to_np_m_matches_scalar_reference_at_1mhz() {
        let alpha_db_cm = 0.75;
        let got = alpha_db_cm_to_np_m(alpha_db_cm, 1.0, 1.5);
        let expected = alpha_db_cm * 100.0 / (20.0 / std::f64::consts::LN_10);
        assert!(
            (got - expected).abs() < 1e-12,
            "conversion mismatch: got {got}, expected {expected}"
        );
    }

    #[test]
    fn test_alpha_db_cm_to_np_m_respects_power_law_frequency_scaling() {
        let alpha_db_cm = 0.5;
        let at_1mhz = alpha_db_cm_to_np_m(alpha_db_cm, 1.0, 1.5);
        let at_2mhz = alpha_db_cm_to_np_m(alpha_db_cm, 2.0, 1.5);
        let expected_ratio = 2.0_f64.powf(1.5);
        let got_ratio = at_2mhz / at_1mhz;
        assert!(
            (got_ratio - expected_ratio).abs() < 1e-12,
            "power-law scaling mismatch: got {got_ratio}, expected {expected_ratio}"
        );
    }
}

// ============================================================================
// Solver Type Enum
// ============================================================================

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
    /// String representation.
    fn __repr__(&self) -> String {
        match self {
            SolverType::FDTD => "SolverType.FDTD".to_string(),
            SolverType::PSTD => "SolverType.PSTD".to_string(),
            SolverType::Hybrid => "SolverType.Hybrid".to_string(),
            SolverType::PstdGpu => "SolverType.PstdGpu".to_string(),
        }
    }

    /// Human-readable string.
    fn __str__(&self) -> String {
        match self {
            SolverType::FDTD => "FDTD".to_string(),
            SolverType::PSTD => "PSTD".to_string(),
            SolverType::Hybrid => "Hybrid".to_string(),
            SolverType::PstdGpu => "PstdGpu".to_string(),
        }
    }

    /// Equality comparison.
    fn __eq__(&self, other: &Self) -> bool {
        self == other
    }
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
/// Supports both homogeneous (uniform) and heterogeneous (spatially varying)
/// acoustic media.
///
/// Equivalent to k-Wave medium struct:
/// ```python
/// # Homogeneous
/// medium = Medium.homogeneous(sound_speed=1500.0, density=1000.0)
///
/// # Heterogeneous
/// c = np.ones((32, 32, 32)) * 1500.0
/// c[16:, :, :] = 2000.0
/// rho = np.ones((32, 32, 32)) * 1000.0
/// medium = Medium(sound_speed=c, density=rho)
/// ```
///
/// Internal enum holding either homogeneous or heterogeneous medium.
#[derive(Clone, Debug)]
enum MediumInner {
    Homogeneous(Box<HomogeneousMedium>),
    Heterogeneous(Box<HeterogeneousMedium>),
}

impl MediumInner {
    /// Get a reference to the inner medium as a trait object.
    fn as_medium(&self) -> &dyn MediumTrait {
        match self {
            MediumInner::Homogeneous(h) => h.as_ref(),
            MediumInner::Heterogeneous(h) => h.as_ref(),
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct Medium {
    /// Internal medium (homogeneous or heterogeneous)
    inner: MediumInner,
}

#[pymethods]
impl Medium {
    /// Create a heterogeneous (spatially varying) medium from 3D arrays.
    ///
    /// Parameters
    /// ----------
    /// sound_speed : ndarray (3D float64)
    ///     Spatially varying sound speed [m/s].  Shape must match the grid.
    /// density : ndarray (3D float64)
    ///     Spatially varying density [kg/m³].  Shape must match the grid.
    /// absorption : ndarray (3D float64), optional
    ///     Spatially varying absorption [dB/(MHz·cm)] (default: zeros).
    /// nonlinearity : ndarray (3D float64), optional
    ///     Spatially varying B/A parameter (default: zeros).
    ///
    /// Returns
    /// -------
    /// Medium
    ///     Heterogeneous acoustic medium
    ///
    /// Examples
    /// --------
    /// >>> c = np.ones((32, 32, 32)) * 1500.0
    /// >>> c[16:, :, :] = 2000.0
    /// >>> rho = np.ones((32, 32, 32)) * 1000.0
    /// >>> medium = Medium(sound_speed=c, density=rho)
    #[new]
    #[pyo3(signature = (sound_speed, density, absorption=None, nonlinearity=None))]
    fn new(
        sound_speed: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        absorption: Option<PyReadonlyArray3<f64>>,
        nonlinearity: Option<PyReadonlyArray3<f64>>,
    ) -> PyResult<Self> {
        let c_arr = sound_speed.as_array().to_owned();
        let rho_arr = density.as_array().to_owned();

        let shape = c_arr.shape().to_vec();
        if shape.len() != 3 {
            return Err(PyValueError::new_err("sound_speed must be a 3D array"));
        }
        if rho_arr.shape() != shape.as_slice() {
            return Err(PyValueError::new_err(
                "density shape must match sound_speed shape",
            ));
        }

        // Validate physical ranges
        if c_arr.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err(
                "All sound_speed values must be positive",
            ));
        }
        if rho_arr.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err("All density values must be positive"));
        }

        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
        // Use acoustic-only constructor: skips 22 non-acoustic zero arrays (~740 MB saved)
        let mut het = HeterogeneousMedium::new_acoustic_only(nx, ny, nz, true);
        het.sound_speed = c_arr;
        het.density = rho_arr;

        if let Some(abs) = absorption {
            let abs_arr = abs.as_array().to_owned();
            if abs_arr.shape() != [nx, ny, nz] {
                return Err(PyValueError::new_err(
                    "absorption shape must match sound_speed shape",
                ));
            }
            het.absorption = abs_arr;
        }

        if let Some(nl) = nonlinearity {
            let nl_arr = nl.as_array().to_owned();
            if nl_arr.shape() != [nx, ny, nz] {
                return Err(PyValueError::new_err(
                    "nonlinearity shape must match sound_speed shape",
                ));
            }
            het.nonlinearity = nl_arr;
        }

        Ok(Medium {
            inner: MediumInner::Heterogeneous(Box::new(het)),
        })
    }

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
    /// alpha_power : float, optional
    ///     Power law exponent for absorption (default: 1.0)
    /// grid : Grid, optional
    ///     Grid for material field pre-computation
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
    #[pyo3(signature = (sound_speed, density, absorption=0.0, nonlinearity=0.0, alpha_power=1.0, grid=None))]
    fn homogeneous(
        sound_speed: f64,
        density: f64,
        absorption: f64,
        nonlinearity: f64,
        alpha_power: f64,
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

        let mut medium = HomogeneousMedium::new(density, sound_speed, 0.0, 0.0, grid_ref);

        // Wire absorption and nonlinearity if provided
        if absorption > 0.0 || nonlinearity > 0.0 {
            medium
                .set_acoustic_properties(absorption, alpha_power, nonlinearity)
                .map_err(|e| {
                    PyValueError::new_err(format!("Invalid acoustic properties: {}", e))
                })?;
        }

        Ok(Medium {
            inner: MediumInner::Homogeneous(Box::new(medium)),
        })
    }

    /// Sound speed [m/s].
    /// For homogeneous media returns the uniform value.
    /// For heterogeneous media returns the maximum sound speed.
    #[getter]
    fn sound_speed(&self) -> f64 {
        self.inner.as_medium().max_sound_speed()
    }

    /// Density [kg/m³].
    /// For homogeneous media returns the uniform value.
    /// For heterogeneous media returns the density at the origin.
    #[getter]
    fn density(&self) -> f64 {
        self.inner.as_medium().density(0, 0, 0)
    }

    /// Whether the medium is homogeneous.
    #[getter]
    fn is_homogeneous(&self) -> bool {
        matches!(self.inner, MediumInner::Homogeneous(_))
    }

    /// String representation.
    fn __repr__(&self) -> String {
        match &self.inner {
            MediumInner::Homogeneous(h) => {
                format!(
                    "Medium.homogeneous(sound_speed={:.1}, density={:.1})",
                    h.max_sound_speed(),
                    h.density(0, 0, 0)
                )
            }
            MediumInner::Heterogeneous(h) => {
                let shape = h.sound_speed.shape();
                format!(
                    "Medium(heterogeneous, shape=({}, {}, {}), c_max={:.1})",
                    shape[0],
                    shape[1],
                    shape[2],
                    h.max_sound_speed()
                )
            }
        }
    }

    /// Human-readable string.
    fn __str__(&self) -> String {
        match &self.inner {
            MediumInner::Homogeneous(_) => "Homogeneous Medium".to_string(),
            MediumInner::Heterogeneous(h) => {
                let shape = h.sound_speed.shape();
                format!(
                    "Heterogeneous Medium ({}x{}x{})",
                    shape[0], shape[1], shape[2]
                )
            }
        }
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
    /// Spatial mask for grid sources
    mask: Option<Array3<f64>>,
    /// Time signal for grid sources (pressure)
    signal: Option<Array1<f64>>,
    /// Source injection mode ("additive", "additive_no_correction", or "dirichlet")
    source_mode: String,
    /// Initial pressure distribution (for p0 / IVP sources)
    initial_pressure: Option<Array3<f64>>,
    /// Velocity signal [3, num_sources, time_steps] for velocity sources
    velocity_signal: Option<ndarray::Array3<f64>>,
    /// Propagation direction for plane wave sources
    direction: Option<(f64, f64, f64)>,
    /// KWaveArray for custom transducer geometry sources
    kwave_array: Option<kwavers::domain::source::kwave_array::KWaveArray>,
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
    #[pyo3(signature = (grid, frequency, amplitude, direction=None))]
    fn plane_wave(
        grid: &Grid,
        frequency: f64,
        amplitude: f64,
        direction: Option<(f64, f64, f64)>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        if amplitude < 0.0 {
            return Err(PyValueError::new_err("Amplitude must be non-negative"));
        }

        // Validate and normalize direction
        let dir = direction.unwrap_or((0.0, 0.0, 1.0));
        let mag = (dir.0 * dir.0 + dir.1 * dir.1 + dir.2 * dir.2).sqrt();
        if mag < 1e-12 {
            return Err(PyValueError::new_err("Direction vector must be non-zero"));
        }
        let norm_dir = (dir.0 / mag, dir.1 / mag, dir.2 / mag);
        let _ = &grid.inner; // retained for wavelength computation in future

        Ok(Source {
            source_type: "plane_wave".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: None,
            velocity_signal: None,
            direction: Some(norm_dir),
            kwave_array: None,
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
        if amplitude < 0.0 {
            return Err(PyValueError::new_err("Amplitude must be non-negative"));
        }

        Ok(Source {
            source_type: "point".to_string(),
            frequency,
            amplitude,
            position: Some([position.0, position.1, position.2]),
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: None,
        })
    }

    /// Create a grid source from a spatial mask and time signal.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray
    ///     3D spatial mask (same shape as grid)
    /// signal : ndarray
    ///     1D time signal [Pa]
    /// frequency : float
    ///     Source frequency [Hz]
    /// Create a grid source from a spatial mask and time signal.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray
    ///     3D spatial mask (same shape as grid)
    /// signal : ndarray
    ///     1D time signal [Pa]
    /// frequency : float
    ///     Source frequency [Hz]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction", or "dirichlet"
    #[staticmethod]
    #[pyo3(signature = (mask, signal, frequency, mode=None))]
    fn from_mask(
        mask: PyReadonlyArray3<f64>,
        signal: PyReadonlyArray1<f64>,
        frequency: f64,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }

        let mask_arr = mask.as_array().to_owned();
        if mask_arr.ndim() != 3 {
            return Err(PyValueError::new_err("Mask must be a 3D array"));
        }

        let signal_arr = signal.as_array().to_owned();
        if signal_arr.ndim() != 1 {
            return Err(PyValueError::new_err("Signal must be a 1D array"));
        }

        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }

        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            Some("additive") | None => "additive".to_string(),
            Some(other) => return Err(PyValueError::new_err(format!(
                "Invalid source mode '{}'. Use 'additive', 'additive_no_correction', or 'dirichlet'",
                other
            ))),
        };

        let amplitude = signal_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));

        Ok(Source {
            source_type: "mask".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: Some(mask_arr),
            signal: Some(signal_arr),
            source_mode,
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: None,
        })
    }

    /// Create an initial pressure (initial value problem) source.
    ///
    /// Equivalent to k-Wave's `source.p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : ndarray
    ///     3D initial pressure distribution [Pa]
    #[staticmethod]
    fn from_initial_pressure(p0: PyReadonlyArray3<f64>) -> PyResult<Self> {
        let p0_arr = p0.as_array().to_owned();
        if p0_arr.iter().all(|&v| v == 0.0) {
            return Err(PyValueError::new_err("Initial pressure is all zeros"));
        }
        let amplitude = p0_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Source {
            source_type: "p0".to_string(),
            frequency: 0.0,
            amplitude,
            position: None,
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: Some(p0_arr),
            velocity_signal: None,
            direction: None,
            kwave_array: None,
        })
    }

    /// Create a velocity source from a spatial mask and directional signals.
    ///
    /// Equivalent to k-Wave's `source.u_mask` / `source.ux` / `source.uy` / `source.uz`.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray (3D float64)
    ///     Spatial mask marking velocity source locations (nonzero = active)
    /// ux : ndarray (1D or 2D), optional
    ///     Velocity signal in x-direction [m/s]
    /// uy : ndarray (1D or 2D), optional
    ///     Velocity signal in y-direction [m/s]
    /// uz : ndarray (1D or 2D), optional
    ///     Velocity signal in z-direction [m/s]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction",
    ///     or "dirichlet"
    #[staticmethod]
    #[pyo3(signature = (mask, ux=None, uy=None, uz=None, mode=None))]
    fn from_velocity_mask(
        mask: PyReadonlyArray3<f64>,
        ux: Option<PyReadonlyArray1<f64>>,
        uy: Option<PyReadonlyArray1<f64>>,
        uz: Option<PyReadonlyArray1<f64>>,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        let mask_arr = mask.as_array().to_owned();
        let num_sources = mask_arr.iter().filter(|v| **v != 0.0).count();
        if num_sources == 0 {
            return Err(PyValueError::new_err(
                "Velocity mask contains no active points",
            ));
        }

        // At least one velocity component must be provided
        if ux.is_none() && uy.is_none() && uz.is_none() {
            return Err(PyValueError::new_err(
                "At least one velocity component (ux, uy, uz) must be provided",
            ));
        }

        // Determine time steps from first available signal
        let nt = if let Some(ref s) = ux {
            s.as_array().len()
        } else if let Some(ref s) = uy {
            s.as_array().len()
        } else if let Some(ref s) = uz {
            s.as_array().len()
        } else {
            unreachable!()
        };

        // Build [3, 1, nt] velocity signal array (broadcast to all sources)
        let mut u_signal = ndarray::Array3::<f64>::zeros((3, 1, nt));
        if let Some(ref sx) = ux {
            let arr = sx.as_array();
            for t in 0..nt {
                u_signal[[0, 0, t]] = arr[t];
            }
        }
        if let Some(ref sy) = uy {
            let arr = sy.as_array();
            if arr.len() != nt {
                return Err(PyValueError::new_err(format!(
                    "uy length {} differs from first signal length {}",
                    arr.len(),
                    nt
                )));
            }
            for t in 0..nt {
                u_signal[[1, 0, t]] = arr[t];
            }
        }
        if let Some(ref sz) = uz {
            let arr = sz.as_array();
            if arr.len() != nt {
                return Err(PyValueError::new_err(format!(
                    "uz length {} differs from first signal length {}",
                    arr.len(),
                    nt
                )));
            }
            for t in 0..nt {
                u_signal[[2, 0, t]] = arr[t];
            }
        }

        let source_mode = match mode {
            Some("dirichlet") => "dirichlet".to_string(),
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            _ => "additive".to_string(),
        };

        let max_amp = u_signal.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));

        Ok(Source {
            source_type: "velocity".to_string(),
            frequency: 0.0,
            amplitude: max_amp,
            position: None,
            mask: Some(mask_arr),
            signal: None,
            source_mode,
            initial_pressure: None,
            velocity_signal: Some(u_signal),
            direction: None,
            kwave_array: None,
        })
    }

    /// Create a velocity source from a 3D mask with per-source-point 2D signal arrays.
    ///
    /// This method supports beamforming delays where each source point gets a
    /// uniquely time-shifted velocity signal, matching k-wave's NotATransducer
    /// focused transducer behavior.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray (3D)
    ///     3D binary mask of source locations (non-zero = source point).
    ///     Source points are iterated in row-major (C-order): x varies slowest.
    /// ux : ndarray (2D), optional
    ///     Velocity signal in x-direction, shape (n_sources, n_timesteps) [m/s]
    /// uy : ndarray (2D), optional
    ///     Velocity signal in y-direction, shape (n_sources, n_timesteps) [m/s]
    /// uz : ndarray (2D), optional
    ///     Velocity signal in z-direction, shape (n_sources, n_timesteps) [m/s]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction",
    ///     or "dirichlet"
    #[staticmethod]
    #[pyo3(signature = (mask, ux=None, uy=None, uz=None, mode=None))]
    fn from_velocity_mask_2d(
        mask: PyReadonlyArray3<f64>,
        ux: Option<PyReadonlyArray2<f64>>,
        uy: Option<PyReadonlyArray2<f64>>,
        uz: Option<PyReadonlyArray2<f64>>,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        let mask_arr = mask.as_array().to_owned();
        let num_sources = mask_arr.iter().filter(|v| **v != 0.0).count();
        if num_sources == 0 {
            return Err(PyValueError::new_err(
                "Velocity mask contains no active points",
            ));
        }

        if ux.is_none() && uy.is_none() && uz.is_none() {
            return Err(PyValueError::new_err(
                "At least one velocity component (ux, uy, uz) must be provided",
            ));
        }

        // Determine (n_sources, nt) from first available 2D signal
        let (n_sig, nt) = if let Some(ref s) = ux {
            let shape = s.as_array().shape().to_vec();
            (shape[0], shape[1])
        } else if let Some(ref s) = uy {
            let shape = s.as_array().shape().to_vec();
            (shape[0], shape[1])
        } else if let Some(ref s) = uz {
            let shape = s.as_array().shape().to_vec();
            (shape[0], shape[1])
        } else {
            unreachable!()
        };

        if n_sig != num_sources {
            return Err(PyValueError::new_err(format!(
                "Signal rows ({}) must match number of active mask points ({})",
                n_sig, num_sources
            )));
        }

        // Build [3, n_sources, nt] velocity signal array
        let mut u_signal = ndarray::Array3::<f64>::zeros((3, num_sources, nt));
        if let Some(ref sx) = ux {
            let arr = sx.as_array();
            for s in 0..num_sources {
                for t in 0..nt {
                    u_signal[[0, s, t]] = arr[[s, t]];
                }
            }
        }
        if let Some(ref sy) = uy {
            let arr = sy.as_array();
            if arr.shape()[0] != num_sources || arr.shape()[1] != nt {
                return Err(PyValueError::new_err(format!(
                    "uy shape {:?} must be ({}, {})",
                    arr.shape(),
                    num_sources,
                    nt
                )));
            }
            for s in 0..num_sources {
                for t in 0..nt {
                    u_signal[[1, s, t]] = arr[[s, t]];
                }
            }
        }
        if let Some(ref sz) = uz {
            let arr = sz.as_array();
            if arr.shape()[0] != num_sources || arr.shape()[1] != nt {
                return Err(PyValueError::new_err(format!(
                    "uz shape {:?} must be ({}, {})",
                    arr.shape(),
                    num_sources,
                    nt
                )));
            }
            for s in 0..num_sources {
                for t in 0..nt {
                    u_signal[[2, s, t]] = arr[[s, t]];
                }
            }
        }

        let source_mode = match mode {
            Some("dirichlet") => "dirichlet".to_string(),
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            _ => "additive".to_string(),
        };

        let max_amp = u_signal.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));

        Ok(Source {
            source_type: "velocity".to_string(),
            frequency: 0.0,
            amplitude: max_amp,
            position: None,
            mask: Some(mask_arr),
            signal: None,
            source_mode,
            initial_pressure: None,
            velocity_signal: Some(u_signal),
            direction: None,
            kwave_array: None,
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

    /// Source injection mode ("additive", "additive_no_correction", or "dirichlet").
    #[getter]
    fn source_mode(&self) -> &str {
        &self.source_mode
    }

    /// Create a source from a KWaveArray with a driving signal.
    ///
    /// The array geometry is rasterized onto the simulation grid at run time.
    ///
    /// Parameters
    /// ----------
    /// array : KWaveArray
    ///     Custom transducer array
    /// signal : ndarray
    ///     1D driving signal [Pa]
    /// frequency : float
    ///     Source frequency [Hz]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction", "dirichlet"
    ///
    /// Examples
    /// --------
    /// >>> arr = KWaveArray()
    /// >>> arr.add_disc_element((0.015, 0.015, 0.0), 0.01)
    /// >>> source = Source.from_kwave_array(arr, signal, frequency=1e6)
    #[staticmethod]
    #[pyo3(signature = (array, signal, frequency, mode=None))]
    fn from_kwave_array(
        array: &KWaveArray,
        signal: PyReadonlyArray1<f64>,
        frequency: f64,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        let signal_arr = signal.as_array().to_owned();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            _ => "additive".to_string(),
        };
        let amplitude = signal_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Source {
            source_type: "kwave_array".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: None,
            signal: Some(signal_arr),
            source_mode,
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: Some(array.inner.clone()),
        })
    }

    /// String representation.
    fn __repr__(&self) -> String {
        match &self.position {
            Some(pos) => format!(
                "Source.point(position=[{:.3e}, {:.3e}, {:.3e}], frequency={:.2e}, amplitude={:.2e})",
                pos[0], pos[1], pos[2], self.frequency, self.amplitude
            ),
            None if self.source_type == "mask" => format!(
                "Source.from_mask(frequency={:.2e}, amplitude={:.2e})",
                self.frequency, self.amplitude
            ),
            None if self.source_type == "kwave_array" => format!(
                "Source.from_kwave_array(frequency={:.2e}, amplitude={:.2e})",
                self.frequency, self.amplitude
            ),
            None => format!(
                "Source.plane_wave(frequency={:.2e}, amplitude={:.2e})",
                self.frequency, self.amplitude
            ),
        }
    }
}

// ============================================================================
// KWaveArray: Custom Transducer Geometry
// ============================================================================

/// Custom transducer array with mixed element geometries.
///
/// Matches k-wave-python's `KWaveArray` API for building arbitrary transducer
/// arrays from arc, disc, rectangular, and bowl elements.
///
/// Examples
/// --------
/// >>> arr = KWaveArray()
/// >>> arr.add_disc_element(position=(0.015, 0.015, 0.0), diameter=0.01)
/// >>> source = Source.from_kwave_array(arr, signal)
#[pyclass]
#[derive(Clone)]
pub struct KWaveArray {
    inner: kwavers::domain::source::kwave_array::KWaveArray,
}

#[pymethods]
impl KWaveArray {
    #[new]
    fn new() -> Self {
        KWaveArray {
            inner: kwavers::domain::source::kwave_array::KWaveArray::new(),
        }
    }

    /// Set the operating frequency [Hz].
    fn set_frequency(&mut self, frequency: f64) {
        self.inner = kwavers::domain::source::kwave_array::KWaveArray::with_params(
            frequency,
            self.inner.frequency(),
        );
    }

    /// Set the sound speed [m/s].
    fn set_sound_speed(&mut self, sound_speed: f64) {
        self.inner = kwavers::domain::source::kwave_array::KWaveArray::with_params(
            self.inner.frequency(),
            sound_speed,
        );
    }

    /// Add a disc-shaped element.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Element center [x, y, z] in meters
    /// diameter : float
    ///     Disc diameter [m]
    fn add_disc_element(&mut self, position: (f64, f64, f64), diameter: f64) {
        self.inner.add_disc_element(position, diameter);
    }

    /// Add an arc-shaped element.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Arc center [x, y, z] in meters
    /// radius : float
    ///     Arc radius [m]
    /// diameter : float
    ///     Arc aperture diameter [m]
    /// start_angle : float, optional
    ///     Start angle in degrees (default: -45)
    /// end_angle : float, optional
    ///     End angle in degrees (default: 45)
    #[pyo3(signature = (position, radius, diameter, start_angle=-45.0, end_angle=45.0))]
    fn add_arc_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameter: f64,
        start_angle: f64,
        end_angle: f64,
    ) {
        self.inner
            .add_arc_element_with_angles(position, radius, diameter, start_angle, end_angle);
    }

    /// Add a rectangular element.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Center position [x, y, z] in meters
    /// dims : tuple[float, float, float]
    ///     Dimensions [width, height, length] in meters
    fn add_rect_element(&mut self, position: (f64, f64, f64), dims: (f64, f64, f64)) {
        self.inner
            .add_rect_element(position, dims.0, dims.1, dims.2);
    }

    /// Add a bowl-shaped element (focused transducer).
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Bowl center position [x, y, z] in meters
    /// radius : float
    ///     Radius of curvature [m]
    /// diameter : float
    ///     Bowl aperture diameter [m]
    fn add_bowl_element(&mut self, position: (f64, f64, f64), radius: f64, diameter: f64) {
        self.inner.add_bowl_element(position, radius, diameter);
    }

    /// Number of elements in the array.
    #[getter]
    fn num_elements(&self) -> usize {
        self.inner.num_elements()
    }

    /// Get element centroid positions as list of (x, y, z) tuples.
    fn get_element_positions(&self) -> Vec<(f64, f64, f64)> {
        self.inner.get_element_positions()
    }

    /// Compute focus delays [s] for each element to focus at a point.
    fn get_focus_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        self.inner.get_focus_delays(focus_point)
    }

    /// Compute time delays [s] for electronic focusing at a point.
    ///
    /// Returns per-element delays such that `τᵢ = (d_max − dᵢ) / c`, where
    /// `dᵢ` is the distance from element `i` to `focus_point` and `d_max = max(dᵢ)`.
    /// The farthest element has delay 0; closer elements are delayed so all wavefronts
    /// arrive at the focus simultaneously.
    ///
    /// Parameters
    /// ----------
    /// focus_point : tuple[float, float, float]
    ///     Focus position (x, y, z) in metres
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Per-element delays in seconds. All values ≥ 0.
    ///
    /// Reference: Selfridge et al. (1980) Appl. Phys. Lett. 37(1):35–36.
    fn get_element_delays(&self, focus_point: (f64, f64, f64)) -> Vec<f64> {
        self.inner.get_element_delays(focus_point)
    }

    /// Compute per-element amplitude weights for array apodization.
    ///
    /// Parameters
    /// ----------
    /// window : str
    ///     Window type: ``"Rectangular"`` (uniform), ``"Hann"``, or ``"Hamming"``
    ///
    /// Returns
    /// -------
    /// list[float]
    ///     Apodization weights in ``[0, 1]``, one per element.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If `window` is not one of the recognised strings.
    ///
    /// Reference: Harris (1978) Proc. IEEE 66(1):51–83.
    fn get_apodization(&self, window: &str) -> PyResult<Vec<f64>> {
        use kwavers::domain::source::kwave_array::ApodizationWindow;
        let w = match window {
            "Rectangular" => ApodizationWindow::Rectangular,
            "Hann" => ApodizationWindow::Hann,
            "Hamming" => ApodizationWindow::Hamming,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown apodization window '{}'. Choose from: Rectangular, Hann, Hamming",
                    other
                )));
            }
        };
        Ok(self.inner.get_apodization(w))
    }

    fn __repr__(&self) -> String {
        format!("KWaveArray(num_elements={})", self.inner.num_elements())
    }
}

// ============================================================================
// TransducerArray2D: 2D Transducer Array Source
// ============================================================================

/// Convert Python apodization string to kwavers type
fn parse_apodization_type(apodization: &str) -> PyResult<KwaversApodizationType> {
    match apodization {
        "Rectangular" => Ok(KwaversApodizationType::Rectangular),
        "Hanning" => Ok(KwaversApodizationType::Hanning),
        "Hamming" => Ok(KwaversApodizationType::Hamming),
        "Blackman" => Ok(KwaversApodizationType::Blackman),
        _ => Err(PyValueError::new_err(
            "Apodization must be one of: Rectangular, Hanning, Hamming, Blackman",
        )),
    }
}

/// Convert kwavers apodization type to Python string
fn apodization_to_string(apodization: &KwaversApodizationType) -> String {
    match apodization {
        KwaversApodizationType::Rectangular => "Rectangular".to_string(),
        KwaversApodizationType::Hanning => "Hanning".to_string(),
        KwaversApodizationType::Hamming => "Hamming".to_string(),
        KwaversApodizationType::Blackman => "Blackman".to_string(),
        KwaversApodizationType::Gaussian { sigma } => format!("Gaussian(sigma={})", sigma),
    }
}

/// 2D transducer array with electronic beam control.
///
/// Mathematical Specification:
/// - Linear array geometry with configurable elements
/// - Electronic steering: time-delay beam steering in azimuthal direction
/// - Electronic focusing: focus at arbitrary depths
/// - Apodization: amplitude weighting (transmit and receive)
///
/// Equivalent to k-Wave's kWaveTransducerSimple and NotATransducer.
///
/// References:
/// - Treeby & Cox (2010) k-Wave toolbox
/// - Szabo (2014) Diagnostic Ultrasound Imaging
#[pyclass]
#[derive(Clone)]
pub struct TransducerArray2D {
    /// Internal kwavers transducer array
    inner: KwaversTransducerArray2D,
    /// Amplitude [Pa] (not in kwavers, added for Python API)
    amplitude: f64,
    /// Input signal (optional, overrides sinusoidal)
    input_signal: Option<Array1<f64>>,
}

#[pymethods]
impl TransducerArray2D {
    /// Create a new 2D transducer array.
    ///
    /// Parameters
    /// ----------
    /// number_elements : int
    ///     Number of elements in the array
    /// element_width : float
    ///     Width of each element [m]
    /// element_length : float
    ///     Length of each element in elevation direction [m]
    /// element_spacing : float
    ///     Spacing between element centers [m]
    /// sound_speed : float
    ///     Speed of sound in medium [m/s]
    /// frequency : float
    ///     Operating frequency [Hz]
    ///
    /// Returns
    /// -------
    /// TransducerArray2D
    ///     Configured transducer array
    ///
    /// Examples
    /// --------
    /// >>> array = TransducerArray2D(
    /// ...     number_elements=32,
    /// ...     element_width=0.3e-3,
    /// ...     element_length=10e-3,
    /// ...     element_spacing=0.5e-3,
    /// ...     sound_speed=1540.0,
    /// ...     frequency=1e6
    /// ... )
    #[new]
    #[pyo3(signature = (number_elements, element_width, element_length, element_spacing, sound_speed, frequency))]
    fn new(
        number_elements: usize,
        element_width: f64,
        element_length: f64,
        element_spacing: f64,
        sound_speed: f64,
        frequency: f64,
    ) -> PyResult<Self> {
        if number_elements == 0 {
            return Err(PyValueError::new_err("Number of elements must be positive"));
        }
        if element_width <= 0.0 {
            return Err(PyValueError::new_err("Element width must be positive"));
        }
        if element_length <= 0.0 {
            return Err(PyValueError::new_err("Element length must be positive"));
        }
        if element_spacing < element_width {
            return Err(PyValueError::new_err(
                "Element spacing must be >= element width",
            ));
        }
        if sound_speed <= 0.0 {
            return Err(PyValueError::new_err("Sound speed must be positive"));
        }
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }

        let config = TransducerArray2DConfig {
            number_elements,
            element_width,
            element_length,
            element_spacing,
            radius: f64::INFINITY,
            center_position: (0.0, 0.0, 0.0),
        };

        let inner = KwaversTransducerArray2D::new(config, sound_speed, frequency).map_err(|e| {
            PyValueError::new_err(format!("Failed to create transducer array: {}", e))
        })?;

        Ok(TransducerArray2D {
            inner,
            amplitude: 1.0,
            input_signal: None,
        })
    }

    /// Set focus distance [m].
    ///
    /// Parameters
    /// ----------
    /// distance : float
    ///     Focus distance from array (INF for no focusing)
    #[pyo3(signature = (distance))]
    fn set_focus_distance(&mut self, distance: f64) -> PyResult<()> {
        if distance <= 0.0 && !distance.is_infinite() {
            return Err(PyValueError::new_err("Focus distance must be positive"));
        }
        self.inner.set_focus_distance(distance);
        Ok(())
    }

    /// Set elevation focus distance [m].
    #[pyo3(signature = (distance))]
    fn set_elevation_focus_distance(&mut self, distance: f64) -> PyResult<()> {
        if distance <= 0.0 && !distance.is_infinite() {
            return Err(PyValueError::new_err(
                "Elevation focus distance must be positive",
            ));
        }
        self.inner.set_elevation_focus_distance(distance);
        Ok(())
    }

    /// Set steering angle [degrees].
    ///
    /// Parameters
    /// ----------
    /// angle : float
    ///     Steering angle in degrees (0 = straight ahead)
    #[pyo3(signature = (angle))]
    fn set_steering_angle(&mut self, angle: f64) {
        self.inner.set_steering_angle(angle);
    }

    /// Set transmit apodization type.
    ///
    /// Parameters
    /// ----------
    /// apodization : str
    ///     One of: "Rectangular", "Hanning", "Hamming", "Blackman"
    #[pyo3(signature = (apodization))]
    fn set_transmit_apodization(&mut self, apodization: &str) -> PyResult<()> {
        let apod_type = parse_apodization_type(apodization)?;
        self.inner.set_transmit_apodization(apod_type);
        Ok(())
    }

    /// Set receive apodization type.
    #[pyo3(signature = (apodization))]
    fn set_receive_apodization(&mut self, apodization: &str) -> PyResult<()> {
        let apod_type = parse_apodization_type(apodization)?;
        self.inner.set_receive_apodization(apod_type);
        Ok(())
    }

    /// Set active element mask.
    ///
    /// Parameters
    /// ----------
    /// mask : list[bool]
    ///     Boolean mask of length number_elements
    #[pyo3(signature = (mask))]
    fn set_active_elements(&mut self, mask: Vec<bool>) -> PyResult<()> {
        if mask.len() != self.inner.number_elements() {
            return Err(PyValueError::new_err(format!(
                "Mask length {} does not match number of elements {}",
                mask.len(),
                self.inner.number_elements()
            )));
        }
        self.inner
            .set_active_elements(&mask)
            .map_err(PyValueError::new_err)?;
        Ok(())
    }

    /// Set center position.
    #[pyo3(signature = (x, y, z))]
    fn set_position(&mut self, x: f64, y: f64, z: f64) {
        self.inner.set_center_position((x, y, z));
    }

    /// Set input signal (overrides sinusoidal).
    #[pyo3(signature = (signal))]
    fn set_input_signal(&mut self, signal: PyReadonlyArray1<f64>) -> PyResult<()> {
        let signal_arr = signal.as_array().to_owned();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        self.input_signal = Some(signal_arr);
        Ok(())
    }

    /// Get number of elements.
    #[getter]
    fn number_elements(&self) -> usize {
        self.inner.number_elements()
    }

    /// Get element spacing.
    #[getter]
    fn element_spacing(&self) -> f64 {
        self.inner.element_spacing()
    }

    /// Get total aperture width.
    #[getter]
    fn aperture_width(&self) -> f64 {
        self.inner.aperture_width()
    }

    /// Element width [m].
    #[getter]
    fn element_width(&self) -> f64 {
        self.inner.element_width()
    }

    /// Element length (elevation) [m].
    #[getter]
    fn element_length(&self) -> f64 {
        self.inner.element_length()
    }

    /// Radius of curvature [m].
    #[getter]
    fn radius(&self) -> f64 {
        self.inner.radius()
    }

    /// Operating frequency [Hz].
    #[getter]
    fn frequency(&self) -> f64 {
        self.inner.frequency()
    }

    /// Focus distance [m].
    #[getter]
    fn focus_distance(&self) -> f64 {
        self.inner.focus_distance()
    }

    /// Steering angle [degrees].
    #[getter]
    fn steering_angle(&self) -> f64 {
        self.inner.steering_angle()
    }

    /// Transmit apodization type.
    #[getter]
    fn transmit_apodization(&self) -> String {
        apodization_to_string(self.inner.transmit_apodization())
    }

    /// Amplitude [Pa].
    #[getter]
    fn amplitude(&self) -> f64 {
        self.amplitude
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "TransducerArray2D(elements={}, width={:.2e}m, focus={:.2e}m, steering={:.1} deg)",
            self.inner.number_elements(),
            self.inner.aperture_width(),
            if self.inner.focus_distance().is_infinite() {
                0.0
            } else {
                self.inner.focus_distance()
            },
            self.inner.steering_angle()
        )
    }
}

// ============================================================================
// Sensor: Field Sampling
// ============================================================================

/// Sensor for recording acoustic fields.
///
/// Mathematical Specification:
/// - Point sensor: p(t) at fixed location (x₀, y₀, z₀)
/// - Mask sensor: p(t) at multiple positions defined by binary mask
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
    /// Binary mask for mask-based sensors
    mask: Option<Array3<bool>>,
    /// k-Wave-style recording mode strings (e.g. ["p", "p_max", "p_rms"])
    record_modes: Vec<String>,
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
            mask: None,
            record_modes: Vec::new(),
        }
    }

    /// Create a mask-based sensor from a binary 3D mask.
    ///
    /// Records pressure at all True positions in the mask.
    /// Equivalent to k-Wave's kSensor(mask).
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray
    ///     3D boolean mask (same shape as grid)
    ///
    /// Returns
    /// -------
    /// Sensor
    ///     Mask-based sensor recording at multiple points
    ///
    /// Examples
    /// --------
    /// >>> mask = np.zeros((32, 32, 32), dtype=bool)
    /// >>> mask[16, 16, 16] = True
    /// >>> sensor = Sensor.from_mask(mask)
    #[staticmethod]
    fn from_mask(mask: PyReadonlyArray3<bool>) -> PyResult<Self> {
        let mask_arr = mask.as_array().to_owned();
        if mask_arr.ndim() != 3 {
            return Err(PyValueError::new_err("Mask must be a 3D array"));
        }
        let num_sensors = mask_arr.iter().filter(|&&v| v).count();
        if num_sensors == 0 {
            return Err(PyValueError::new_err(
                "Sensor mask must have at least one active sensor",
            ));
        }
        Ok(Sensor {
            sensor_type: "mask".to_string(),
            position: None,
            mask: Some(mask_arr),
            record_modes: Vec::new(),
        })
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
            mask: None,
            record_modes: Vec::new(),
        }
    }

    /// Set k-Wave-style recording modes.
    ///
    /// Parameters
    /// ----------
    /// modes : list[str]
    ///     Recording mode strings. Supported: "p", "p_max", "p_min", "p_rms", "p_final", "all"
    ///
    /// Examples
    /// --------
    /// >>> sensor = Sensor.from_mask(mask)
    /// >>> sensor.set_record(["p", "p_max", "p_rms"])
    fn set_record(&mut self, modes: Vec<String>) {
        self.record_modes = modes;
    }

    /// Get current recording modes.
    #[getter]
    fn record(&self) -> Vec<String> {
        self.record_modes.clone()
    }

    /// Number of active sensor points.
    #[getter]
    fn num_sensors(&self) -> usize {
        match &self.mask {
            Some(m) => m.iter().filter(|&&v| v).count(),
            None if self.sensor_type == "point" => 1,
            _ => 0,
        }
    }

    /// Sensor type.
    #[getter]
    fn sensor_type(&self) -> &str {
        &self.sensor_type
    }

    /// String representation.
    fn __repr__(&self) -> String {
        match &self.sensor_type {
            t if t == "point" => {
                let pos = self.position.unwrap_or([0.0, 0.0, 0.0]);
                format!(
                    "Sensor.point(position=[{:.3e}, {:.3e}, {:.3e}])",
                    pos[0], pos[1], pos[2]
                )
            }
            t if t == "mask" => {
                let n = self
                    .mask
                    .as_ref()
                    .map_or(0, |m| m.iter().filter(|&&v| v).count());
                format!("Sensor.from_mask(num_sensors={})", n)
            }
            _ => "Sensor.grid()".to_string(),
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
/// - Boundary conditions: CPML (convolutional perfectly matched layers)
///
/// Equivalent to k-Wave's kspaceFirstOrder3D function.
#[pyclass]
#[derive(Clone)]
pub struct Simulation {
    grid: Grid,
    medium: Medium,
    sources: Vec<Source>,
    transducers: Vec<TransducerArray2D>,
    sensor: Option<Sensor>,
    transducer_sensor: Option<TransducerArray2D>,
    solver_type: SolverType,
    pml_size: Option<usize>,
    pml_size_xyz: Option<(usize, usize, usize)>,
    pml_inside: bool,
    /// Per-dimension PML absorption factor (k-Wave `pml_alpha`): [x, y, z]
    pml_alpha_xyz: Option<(f64, f64, f64)>,
    /// Enable Westervelt nonlinear source term in FDTD solver
    enable_nonlinear: bool,
    /// Medium absorption coefficient [dB/(MHz^y·cm)] — k-Wave convention (0 = lossless)
    alpha_coeff: f64,
    /// Medium absorption power law exponent (k-Wave default: 1.5 for tissue)
    alpha_power: f64,
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
    /// source : Source or list[Source]
    ///     Acoustic source(s)
    /// sensor : Sensor
    ///     Field sensor
    /// solver : SolverType, optional
    ///     Solver type (default: FDTD)
    ///
    /// Returns
    /// -------
    /// Simulation
    ///     Configured simulation
    ///
    /// Examples
    /// --------
    /// >>> sim = Simulation(grid, medium, source, sensor, solver=SolverType.PSTD)
    #[new]
    #[pyo3(signature = (grid, medium, source, sensor, solver=None, pml_size=None))]
    fn new(
        grid: Grid,
        medium: Medium,
        source: &Bound<'_, PyAny>,
        sensor: &Bound<'_, PyAny>,
        solver: Option<SolverType>,
        pml_size: Option<usize>,
    ) -> PyResult<Self> {
        let mut sources = Vec::new();
        let mut transducers = Vec::new();

        if let Ok(src) = source.extract::<Source>() {
            sources.push(src);
        } else if let Ok(trans) = source.extract::<TransducerArray2D>() {
            transducers.push(trans);
        } else if let Ok(list) = source.extract::<Vec<Bound<'_, PyAny>>>() {
            for item in list {
                if let Ok(src) = item.extract::<Source>() {
                    sources.push(src);
                } else if let Ok(trans) = item.extract::<TransducerArray2D>() {
                    transducers.push(trans);
                } else {
                    return Err(PyValueError::new_err(
                        "sources list must contain only Source or TransducerArray2D objects",
                    ));
                }
            }
        } else {
            return Err(PyValueError::new_err(
                "sources must be a Source, TransducerArray2D, or a list of these",
            ));
        }

        if sources.is_empty() && transducers.is_empty() {
            return Err(PyValueError::new_err("At least one source is required"));
        }

        let mut sensor_opt = None;
        let mut transducer_sensor = None;

        if let Ok(s) = sensor.extract::<Sensor>() {
            sensor_opt = Some(s);
        } else if let Ok(ts) = sensor.extract::<TransducerArray2D>() {
            transducer_sensor = Some(ts);
        } else {
            return Err(PyValueError::new_err(
                "sensor must be a Sensor or TransducerArray2D object",
            ));
        }

        Ok(Simulation {
            grid,
            medium,
            sources,
            transducers,
            sensor: sensor_opt,
            transducer_sensor,
            solver_type: solver.unwrap_or(SolverType::FDTD),
            pml_size,
            pml_size_xyz: None,
            pml_inside: true,
            pml_alpha_xyz: None,
            enable_nonlinear: false,
            alpha_coeff: 0.0,
            alpha_power: 1.5,
        })
    }

    /// Set PML (perfectly matched layer) absorbing boundary thickness.
    ///
    /// Parameters
    /// ----------
    /// size : int
    ///     Number of grid points for PML absorbing boundary on each face.
    ///     Typical values: 10-20 for small grids, 20-40 for large grids.
    ///
    /// Examples
    /// --------
    /// >>> sim.set_pml_size(20)
    fn set_pml_size(&mut self, size: usize) {
        self.pml_size = Some(size);
    }

    /// Get the current PML size, or None if using automatic sizing.
    #[getter]
    fn pml_size(&self) -> Option<usize> {
        self.pml_size
    }

    /// Set per-axis PML absorbing boundary thickness for k-Wave parity.
    ///
    /// Parameters
    /// ----------
    /// x : int
    ///     PML thickness in x-direction [grid points]
    /// y : int
    ///     PML thickness in y-direction [grid points]
    /// z : int
    ///     PML thickness in z-direction [grid points]
    ///
    /// Examples
    /// --------
    /// >>> sim.set_pml_size_xyz(20, 10, 10)  # k-Wave default per-axis
    fn set_pml_size_xyz(&mut self, x: usize, y: usize, z: usize) {
        self.pml_size_xyz = Some((x, y, z));
        self.pml_size = Some(x.max(y).max(z));
    }

    /// Set uniform PML absorption factor (equivalent to k-Wave scalar `pml_alpha`, default 2.0).
    ///
    /// Parameters
    /// ----------
    /// alpha : float
    ///     PML absorption coefficient (k-Wave default: 2.0 Np/m)
    ///
    /// Examples
    /// --------
    /// >>> sim.set_pml_alpha(2.0)  # k-Wave default
    fn set_pml_alpha(&mut self, alpha: f64) {
        self.pml_alpha_xyz = Some((alpha, alpha, alpha));
    }

    /// Set per-axis PML absorption factors (equivalent to k-Wave vector `pml_alpha`).
    ///
    /// Parameters
    /// ----------
    /// ax : float
    ///     PML absorption coefficient in x-direction
    /// ay : float
    ///     PML absorption coefficient in y-direction
    /// az : float
    ///     PML absorption coefficient in z-direction
    ///
    /// Examples
    /// --------
    /// >>> sim.set_pml_alpha_xyz(2.0, 1.5, 1.5)  # reduce absorption on y/z faces
    fn set_pml_alpha_xyz(&mut self, ax: f64, ay: f64, az: f64) {
        self.pml_alpha_xyz = Some((ax, ay, az));
    }

    /// Set whether PML is inside the computational domain.
    ///
    /// Parameters
    /// ----------
    /// inside : bool
    ///     If True (default), PML absorbing layers are placed inside the grid,
    ///     reducing the effective domain size. If False, PML layers are placed
    ///     outside the grid, preserving the full domain size (k-Wave default).
    ///
    /// Examples
    /// --------
    /// >>> sim.set_pml_inside(False)  # k-Wave default: PML outside domain
    fn set_pml_inside(&mut self, inside: bool) {
        self.pml_inside = inside;
    }

    /// Get the current PML inside setting.
    #[getter]
    fn pml_inside(&self) -> bool {
        self.pml_inside
    }

    /// Enable or disable the Westervelt nonlinear acoustic source term.
    ///
    /// When enabled, the FDTD solver adds the nonlinear pressure correction
    /// ``(β/ρ₀c₀⁴) ∂²p²/∂t²`` at each time step, enabling harmonic generation
    /// and shock wave simulation.  Currently supported for FDTD solver only.
    ///
    /// Parameters
    /// ----------
    /// enable : bool
    ///     True to enable Westervelt nonlinear term (default: False).
    ///
    /// Examples
    /// --------
    /// >>> sim.set_nonlinear(True)   # enable second-harmonic generation
    fn set_nonlinear(&mut self, enable: bool) {
        self.enable_nonlinear = enable;
    }

    /// Return whether the Westervelt nonlinear term is enabled.
    #[getter]
    fn nonlinear(&self) -> bool {
        self.enable_nonlinear
    }

    /// Set medium absorption coefficient (k-Wave `medium.alpha_coeff`).
    ///
    /// Parameters
    /// ----------
    /// alpha : float
    ///     Absorption coefficient [dB/(MHz^y·cm)].  Set to 0 (default) for lossless.
    ///
    /// Examples
    /// --------
    /// >>> sim.set_alpha_coeff(0.75)   # tissue-like absorption
    fn set_alpha_coeff(&mut self, alpha: f64) {
        self.alpha_coeff = alpha;
    }

    /// Set medium absorption power law exponent (k-Wave `medium.alpha_power`).
    ///
    /// Parameters
    /// ----------
    /// power : float
    ///     Power law exponent (default 1.5 for soft tissue; must not equal 1.0).
    ///
    /// Examples
    /// --------
    /// >>> sim.set_alpha_power(1.5)
    fn set_alpha_power(&mut self, power: f64) {
        self.alpha_power = power;
    }

    /// Get medium absorption coefficient [dB/(MHz^y·cm)].
    #[getter]
    fn alpha_coeff(&self) -> f64 {
        self.alpha_coeff
    }

    /// Get medium absorption power law exponent.
    #[getter]
    fn alpha_power(&self) -> f64 {
        self.alpha_power
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
        py: Python<'py>,
        time_steps: usize,
        dt: Option<f64>,
    ) -> PyResult<SimulationResult> {
        // Calculate time step from CFL condition if not provided
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let cfl = 0.3; // Conservative CFL number for 3D
        let dt_actual = dt.unwrap_or_else(|| cfl * dx_min / (c_max * 3.0_f64.sqrt()));

        let mut grid_source = GridSource::new_empty();
        let mut dynamic_sources: Vec<Box<dyn KwaversSource>> = Vec::new();
        let mut has_mask_source = false;

        for src in &self.sources {
            // Handle KWaveArray source: rasterize geometry onto grid
            if src.source_type == "kwave_array" {
                if has_mask_source {
                    return Err(PyValueError::new_err(
                        "Only one mask/kwave_array source is supported per simulation",
                    ));
                }
                has_mask_source = true;
                let arr = src
                    .kwave_array
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("KWaveArray missing from source"))?;
                let signal = src
                    .signal
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("Source signal missing"))?;
                if signal.len() != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.len(),
                        time_steps
                    )));
                }
                // Rasterize the array geometry onto the simulation grid
                let bool_mask = arr.get_array_binary_mask(&self.grid.inner);
                let float_mask = bool_mask.mapv(|v| if v { 1.0_f64 } else { 0.0_f64 });
                let num_active = float_mask.iter().filter(|&&v| v > 0.0).count();
                if num_active == 0 {
                    return Err(PyValueError::new_err(
                        "KWaveArray mask has no active grid points",
                    ));
                }
                let mut p_signal = ndarray::Array2::<f64>::zeros((1, time_steps));
                for t in 0..time_steps {
                    p_signal[[0, t]] = signal[t];
                }
                let p_mode = match src.source_mode.as_str() {
                    "additive_no_correction" => {
                        kwavers::domain::source::grid_source::SourceMode::AdditiveNoCorrection
                    }
                    "dirichlet" => kwavers::domain::source::grid_source::SourceMode::Dirichlet,
                    _ => kwavers::domain::source::grid_source::SourceMode::Additive,
                };
                grid_source = GridSource {
                    p_mask: Some(float_mask),
                    p_signal: Some(p_signal),
                    p_mode,
                    ..GridSource::new_empty()
                };
                continue;
            }

            if src.source_type == "mask" {
                if has_mask_source {
                    return Err(PyValueError::new_err(
                        "Only one mask source is supported per simulation",
                    ));
                }
                has_mask_source = true;

                let mask = src
                    .mask
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("Source mask missing"))?;
                let signal = src
                    .signal
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("Source signal missing"))?;

                if signal.len() != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.len(),
                        time_steps
                    )));
                }

                let num_sources = mask.iter().filter(|v| **v != 0.0).count();
                if num_sources == 0 {
                    return Err(PyValueError::new_err(
                        "Source mask contains no active points",
                    ));
                }

                // Use a single signal row and let the source handler broadcast if needed
                let mut p_signal = ndarray::Array2::<f64>::zeros((1, time_steps));
                for t in 0..time_steps {
                    p_signal[[0, t]] = signal[t];
                }

                let p_mode = match src.source_mode.as_str() {
                    "additive_no_correction" => {
                        kwavers::domain::source::grid_source::SourceMode::AdditiveNoCorrection
                    }
                    "dirichlet" => kwavers::domain::source::grid_source::SourceMode::Dirichlet,
                    _ => kwavers::domain::source::grid_source::SourceMode::Additive,
                };

                grid_source = GridSource {
                    p_mask: Some(mask.clone()),
                    p_signal: Some(p_signal),
                    p_mode,
                    ..GridSource::new_empty()
                };
                continue;
            }

            // Handle initial pressure (p0) source
            if src.source_type == "p0" {
                if let Some(ref p0) = src.initial_pressure {
                    grid_source.p0 = Some(p0.clone());
                }
                continue;
            }

            // Handle velocity source
            if src.source_type == "velocity" {
                let mask = src
                    .mask
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("Velocity source mask missing"))?;
                let u_sig = src
                    .velocity_signal
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("Velocity signal missing"))?;

                let u_mode = match src.source_mode.as_str() {
                    "additive_no_correction" => {
                        kwavers::domain::source::grid_source::SourceMode::AdditiveNoCorrection
                    }
                    "dirichlet" => kwavers::domain::source::grid_source::SourceMode::Dirichlet,
                    _ => kwavers::domain::source::grid_source::SourceMode::Additive,
                };

                grid_source.u_mask = Some(mask.clone());
                grid_source.u_signal = Some(u_sig.clone());
                grid_source.u_mode = u_mode;
                continue;
            }

            // Create sine wave signal
            let freq = src.frequency;
            let amp = src.amplitude;
            let signal = SineSignal::new(freq, amp);

            // Create source from type
            let function_source: Box<dyn KwaversSource> = if src.source_type == "plane_wave" {
                let dir = src.direction.unwrap_or((0.0, 0.0, 1.0));
                let wavelength = c_max / freq;

                let config = PlaneWaveConfig {
                    direction: dir,
                    wavelength,
                    phase: 0.0,
                    source_type: SourceField::Pressure,
                    injection_mode: InjectionMode::BoundaryOnly,
                };
                Box::new(PlaneWaveSource::new(config, Arc::new(signal)))
            } else {
                // Point source
                let pos_arr = src.position.unwrap_or([0.0, 0.0, 0.0]);
                let px = pos_arr[0];
                let py = pos_arr[1];
                let pz = pos_arr[2];
                let dx = self.grid.inner.dx;
                let dy = self.grid.inner.dy;
                let dz = self.grid.inner.dz;
                Box::new(FunctionSource::new(
                    move |x, y, z, _t| {
                        if (x - px).abs() < dx * 0.5
                            && (y - py).abs() < dy * 0.5
                            && (z - pz).abs() < dz * 0.5
                        {
                            1.0
                        } else {
                            0.0
                        }
                    },
                    Arc::new(signal),
                    SourceField::Pressure,
                ))
            };

            dynamic_sources.push(function_source);
        }

        for trans in &self.transducers {
            let mut inner_trans = trans.inner.clone();
            if let Some(ref sig_arr) = trans.input_signal {
                let sampled_sig = SampledSignal::new(sig_arr.clone(), dt_actual);
                inner_trans.set_signal(Arc::new(sampled_sig));
            }
            dynamic_sources.push(Box::new(inner_trans));
        }

        // Run simulation based on solver type
        // Release GIL during the CPU-intensive simulation to allow other Python threads
        let shape = (self.grid.inner.nx, self.grid.inner.ny, self.grid.inner.nz);
        let grid_clone = self.grid.inner.clone();
        let medium_clone = self.medium.inner.clone();
        let sensor_opt = self.sensor.clone();
        let transducer_sensor_opt = self.transducer_sensor.clone();
        let pml_size = self.pml_size;
        let pml_size_xyz = self.pml_size_xyz;
        let pml_inside = self.pml_inside;
        let pml_alpha_xyz = self.pml_alpha_xyz;
        let solver_type = self.solver_type;
        let enable_nonlinear = self.enable_nonlinear;
        let alpha_coeff = self.alpha_coeff;
        let alpha_power = self.alpha_power;

        // Collect recording modes from sensor (empty vec if no sensor or no modes set)
        let sensor_record_modes: Vec<String> = sensor_opt
            .as_ref()
            .map(|s| s.record_modes.clone())
            .unwrap_or_default();

        let (sensor_data_2d, stats_opt) = py
            .detach(move || match solver_type {
                SolverType::FDTD => Self::run_fdtd_impl(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    grid_source,
                    dynamic_sources,
                    sensor_opt.as_ref(),
                    transducer_sensor_opt.as_ref(),
                    pml_size,
                    pml_size_xyz,
                    pml_inside,
                    pml_alpha_xyz,
                    enable_nonlinear,
                    &sensor_record_modes,
                ),
                SolverType::PSTD | SolverType::Hybrid => Self::run_pstd_impl(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    enable_nonlinear,
                    alpha_coeff,
                    alpha_power,
                    grid_source,
                    dynamic_sources,
                    sensor_opt.as_ref(),
                    transducer_sensor_opt.as_ref(),
                    pml_size,
                    pml_size_xyz,
                    pml_inside,
                    pml_alpha_xyz,
                    &sensor_record_modes,
                ),
                SolverType::PstdGpu => Self::run_gpu_pstd_or_cpu_fallback(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    alpha_coeff,
                    alpha_power,
                    grid_source,
                    dynamic_sources,
                    sensor_opt.as_ref(),
                    transducer_sensor_opt.as_ref(),
                    pml_size,
                    pml_size_xyz,
                    pml_inside,
                    pml_alpha_xyz,
                    enable_nonlinear,
                    &sensor_record_modes,
                ),
            })
            .map_err(kwavers_error_to_py)?;

        // Convert optional stats to numpy arrays
        let p_max = stats_opt
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_max.clone()).into());
        let p_min = stats_opt
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_min.clone()).into());
        let p_rms = stats_opt
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_rms.clone()).into());
        let p_final = stats_opt
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_final.clone()).into());

        // Return 1D array for single sensor, 2D for multi-sensor
        let n_sensors = sensor_data_2d.nrows();
        let time_arr = PyArray1::from_owned_array(
            py,
            Array1::from_iter((0..time_steps).map(|i| i as f64 * dt_actual)),
        )
        .into();

        if n_sensors <= 1 {
            // Single sensor: return 1D for backward compatibility
            let sensor_1d = sensor_data_2d.row(0).to_owned();
            Ok(SimulationResult {
                sensor_data_1d: Some(PyArray1::from_owned_array(py, sensor_1d).into()),
                sensor_data_2d: None,
                time: time_arr,
                shape,
                sensor_data_shape: (1, time_steps),
                time_steps,
                dt: dt_actual,
                final_time: dt_actual * time_steps as f64,
                p_max,
                p_min,
                p_rms,
                p_final,
            })
        } else {
            // Multi-sensor: return 2D array (n_sensors, n_timesteps)
            Ok(SimulationResult {
                sensor_data_1d: None,
                sensor_data_2d: Some(PyArray2::from_owned_array(py, sensor_data_2d).into()),
                time: time_arr,
                shape,
                sensor_data_shape: (n_sensors, time_steps),
                time_steps,
                dt: dt_actual,
                final_time: dt_actual * time_steps as f64,
                p_max,
                p_min,
                p_rms,
                p_final,
            })
        }
    }

    /// String representation.
    fn __repr__(&self) -> String {
        format!(
            "Simulation(grid={}, sources={}, transducers={}, solver={}, sensor={})",
            self.grid.__repr__(),
            self.sources.len(),
            self.transducers.len(),
            self.solver_type.__repr__(),
            if let Some(ref s) = self.sensor {
                s.__repr__()
            } else {
                "Transducer".to_string()
            }
        )
    }
}

// Non-PyO3 implementation block for internal simulation logic
impl Simulation {
    fn create_sensor_mask(
        grid: &KwaversGrid,
        sensor: Option<&Sensor>,
        transducer: Option<&TransducerArray2D>,
    ) -> Array3<bool> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        if let Some(trans) = transducer {
            let mut mask = Array3::<bool>::from_elem((nx, ny, nz), false);
            let width_pts = (trans.inner.element_width() / grid.dx).round() as isize;
            let length_pts = (trans.inner.element_length() / grid.dz).round() as isize;

            for pos in trans.inner.element_positions() {
                // Find grid center index for this element
                let cx = ((pos.0 - grid.origin[0]) / grid.dx).round() as isize;
                let cy = ((pos.1 - grid.origin[1]) / grid.dy).round() as isize;
                let cz = ((pos.2 - grid.origin[2]) / grid.dz).round() as isize;

                let ix_start = cx - (width_pts / 2);
                let ix_end = ix_start + width_pts - 1;
                let iz_start = cz - (length_pts / 2);
                let iz_end = iz_start + length_pts - 1;

                for i in ix_start..=ix_end {
                    for k in iz_start..=iz_end {
                        if i >= 0
                            && i < nx as isize
                            && cy >= 0
                            && cy < ny as isize
                            && k >= 0
                            && k < nz as isize
                        {
                            mask[[i as usize, cy as usize, k as usize]] = true;
                        }
                    }
                }
            }
            return mask;
        }

        let sensor =
            sensor.expect("Simulation must have either a Sensor or TransducerArray2D sensor");

        // Use mask directly if provided
        if let Some(ref mask) = sensor.mask {
            return mask.clone();
        }

        let mut mask = Array3::<bool>::from_elem((nx, ny, nz), false);

        if sensor.sensor_type == "grid" {
            mask.fill(true);
            return mask;
        }

        let pos = sensor.position.unwrap_or([
            (nx as f64 * grid.dx) * 0.5,
            (ny as f64 * grid.dy) * 0.5,
            (nz as f64 * grid.dz) * 0.5,
        ]);

        let ix = (pos[0] / grid.dx).round() as isize;
        let iy = (pos[1] / grid.dy).round() as isize;
        let iz = (pos[2] / grid.dz).round() as isize;

        let ix = ix.clamp(0, (nx - 1) as isize) as usize;
        let iy = iy.clamp(0, (ny - 1) as isize) as usize;
        let iz = iz.clamp(0, (nz - 1) as isize) as usize;

        mask[[ix, iy, iz]] = true;
        mask
    }

    /// Build an ordered list of sensor grid indices for a transducer, element by element.
    ///
    /// Each element's nodes are enumerated in the same X→Z inner order as k-Wave's
    /// transducer sensor mask, so the resulting recording is grouped as:
    ///   [elem0_node0, elem0_node1, …, elem1_node0, …]
    /// enabling direct per-element averaging without reordering.
    fn create_transducer_ordered_indices(
        grid: &KwaversGrid,
        trans: &kwavers::domain::source::array_2d::TransducerArray2D,
    ) -> Vec<(usize, usize, usize)> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let width_pts = (trans.element_width() / grid.dx).round() as isize;
        let length_pts = (trans.element_length() / grid.dz).round() as isize;

        let mut indices = Vec::new();
        for pos in trans.element_positions() {
            let cx = ((pos.0 - grid.origin[0]) / grid.dx).round() as isize;
            let cy = ((pos.1 - grid.origin[1]) / grid.dy).round() as isize;
            let cz = ((pos.2 - grid.origin[2]) / grid.dz).round() as isize;

            let ix_start = cx - (width_pts / 2);
            let iz_start = cz - (length_pts / 2);

            // k-Wave iterates X fastest within each element's footprint
            for ii in 0..width_pts {
                for kk in 0..length_pts {
                    let i = ix_start + ii;
                    let k = iz_start + kk;
                    if i >= 0
                        && i < nx as isize
                        && cy >= 0
                        && cy < ny as isize
                        && k >= 0
                        && k < nz as isize
                    {
                        indices.push((i as usize, cy as usize, k as usize));
                    }
                }
            }
        }
        indices
    }

    /// Convert k-Wave-style record strings to RecordingMode variants.
    fn recording_modes_from_strings(modes: &[String]) -> Vec<RecordingMode> {
        modes
            .iter()
            .filter_map(|s| match s.as_str() {
                "p_max" => Some(RecordingMode::MaxPressure),
                "p_min" => Some(RecordingMode::MinPressure),
                "p_rms" => Some(RecordingMode::RmsPressure),
                "p_final" => Some(RecordingMode::FinalPressure),
                "all" => Some(RecordingMode::AllStatistics),
                _ => None, // "p" (time series) is always recorded
            })
            .collect()
    }

    /// Trim the initial `t=0` recorder column when a backend stores `Nt+1`
    /// samples but the Python-facing API contract is exactly `time_steps`.
    fn trim_initial_recorder_sample(
        recorded_data: ndarray::Array2<f64>,
        time_steps: usize,
    ) -> ndarray::Array2<f64> {
        if recorded_data.ncols() > time_steps {
            recorded_data.slice(ndarray::s![.., 1..]).to_owned()
        } else {
            recorded_data
        }
    }

    /// Run FDTD simulation (internal).
    #[allow(clippy::too_many_arguments)]
    fn run_fdtd_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        _pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        enable_nonlinear: bool,
        record_modes: &[String],
    ) -> KwaversResult<(ndarray::Array2<f64>, Option<SampledStatistics>)> {
        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);

        // For transducer sensors, override the recorder with element-ordered indices
        // so node recordings are grouped [elem0_nodes…, elem1_nodes…] matching k-Wave.
        let transducer_ordered_indices = transducer_sensor
            .map(|trans| Self::create_transducer_ordered_indices(grid, &trans.inner));

        // Create FDTD configuration with sensor mask
        let config = FdtdConfig {
            dt,
            nt: time_steps,
            spatial_order: 4,
            staggered_grid: true,
            cfl_factor: 0.3,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            enable_nonlinear,
            kspace_correction: KSpaceCorrectionMode::None,
            sensor_mask: Some(sensor_mask.clone()),
        };

        // Create solver
        let mut solver = FdtdSolver::new(config, grid, medium.as_medium(), grid_source)?;

        // Build recorder with statistical modes if requested, replacing the default one.
        let modes = Self::recording_modes_from_strings(record_modes);
        if !modes.is_empty() {
            let shape = (grid.nx, grid.ny, grid.nz);
            solver.sensor_recorder =
                SensorRecorder::with_modes(Some(&sensor_mask), shape, time_steps + 1, &modes)?;
        } else if let Some(ordered) = transducer_ordered_indices {
            // Transducer override (no stats)
            solver.sensor_recorder = SensorRecorder::from_ordered_indices(ordered, time_steps + 1)?;
        }

        // Enable CPML boundary if requested
        let min_dim = grid.nx.min(grid.ny).min(grid.nz);
        let max_allowed = (min_dim.saturating_sub(2)) / 2;
        let default_thickness = (min_dim / 6).max(2);
        let thickness = pml_size.unwrap_or(default_thickness).min(max_allowed);

        if thickness > 0 && max_allowed > 0 {
            let mut cpml_config = if let Some((px, py, pz)) = pml_size_xyz {
                CPMLConfig::with_per_dimension_thickness(px, py, pz)
            } else {
                CPMLConfig::with_thickness(thickness)
            };
            if let Some((ax, ay, az)) = pml_alpha_xyz {
                cpml_config = cpml_config.with_alpha_xyz(ax, ay, az);
            }
            let max_c = medium.as_medium().max_sound_speed();
            solver.enable_cpml(cpml_config, dt, max_c)?;
        }

        for source in sources {
            SolverTrait::add_source(&mut solver, source)?;
        }

        // Run simulation - SensorRecorder records pressure at each step
        solver.run_orchestrated(time_steps)?;

        // Extract statistics before consuming recorder
        let stats = solver.sensor_recorder.extract_all_stats();

        // Extract recorded time series: shape (n_sensors, nt+1) due to t=0 initial recording.
        // Slice to (n_sensors, nt) to match k-Wave's convention of exactly Nt output samples.
        let full_data = solver.extract_recorded_sensor_data().ok_or_else(|| {
            kwavers::core::error::KwaversError::Io(std::io::Error::other("No sensor data recorded"))
        })?;
        let recorded_data = Self::trim_initial_recorder_sample(full_data, time_steps);

        Ok((recorded_data, stats))
    }

    /// Run PSTD simulation (internal).
    #[allow(clippy::too_many_arguments)]
    fn run_pstd_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        enable_nonlinear: bool,
        alpha_coeff_db: f64,
        alpha_power: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        record_modes: &[String],
    ) -> KwaversResult<(ndarray::Array2<f64>, Option<SampledStatistics>)> {
        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);

        // For transducer sensors, build element-ordered index list to match k-Wave.
        let transducer_ordered_indices = transducer_sensor
            .map(|trans| Self::create_transducer_ordered_indices(grid, &trans.inner));

        // Create PSTD configuration with sensor mask
        let min_dim = grid.nx.min(grid.ny).min(grid.nz);
        let max_allowed = (min_dim.saturating_sub(2)) / 2;
        let default_thickness = (min_dim / 6).max(2);
        let thickness = pml_size.unwrap_or(default_thickness).min(max_allowed);

        let boundary = if thickness > 0 && max_allowed > 0 {
            let mut cpml_config = if let Some((px, py, pz)) = pml_size_xyz {
                CPMLConfig::with_per_dimension_thickness(px, py, pz)
            } else {
                CPMLConfig::with_thickness(thickness)
            };
            if let Some((ax, ay, az)) = pml_alpha_xyz {
                cpml_config = cpml_config.with_alpha_xyz(ax, ay, az);
            }
            kwavers::solver::forward::pstd::config::BoundaryConfig::CPML(cpml_config)
        } else {
            kwavers::solver::forward::pstd::config::BoundaryConfig::None
        };

        // If caller did not set alpha_coeff explicitly (0.0), fall back to medium's
        // stored absorption coefficient (set via Medium.homogeneous(absorption=...) or
        // Medium.heterogeneous(absorption=...)).  Returns 0.0 for heterogeneous media
        // that don't override alpha_coefficient().
        let effective_alpha_db = if alpha_coeff_db > 0.0 {
            alpha_coeff_db
        } else {
            medium.as_medium().alpha_coefficient(0.0, 0.0, 0.0, grid)
        };

        // Also fall back to medium's alpha_power if caller used default 1.5 and medium has a value.
        let effective_alpha_power = {
            let y_medium = medium.as_medium().alpha_power(0.0, 0.0, 0.0, grid);
            // Use medium's power if it was explicitly set (not the default 1.0 value
            // from HomogeneousMedium constructor).
            if alpha_coeff_db <= 0.0 && y_medium > 0.0 && (y_medium - 1.0).abs() > 1e-12 {
                y_medium
            } else {
                alpha_power
            }
        };

        // Convert absorption from dB/(MHz^y·cm) → Np/(m·(rad/s)^y) using k-Wave db2neper formula.
        // k-Wave Python kwave/utils/conversion.py, db2neper():
        //   alpha = 100.0 * alpha * (1e-6 / (2.0 * pi)) ** y / (20.0 * log10(exp(1)))
        // Note: (1e-6/(2π))^y is the INVERSE of frequency scaling: converts from (rad/s)^y units
        // back to dB/(MHz^y·cm) units for the spectral operator which uses rad/m wavenumbers.
        // For y=1.5: (1e-6/(2π))^1.5 ≈ 6.35e-11, giving alpha_0 ≈ 3.65e-10 Np/((rad/s)^1.5·m).
        // Verification: alpha_0 * (2π*1e6)^1.5 * 0.01m = 0.5 dB/cm ✓ for alpha=0.5, f=1 MHz.
        let absorption_mode =
            if effective_alpha_db > 0.0 && (effective_alpha_power - 1.0).abs() > 1e-12 {
                // 20 * log10(e) = 20 / ln(10) ≈ 8.6859
                let twenty_log10_e = 20.0 / std::f64::consts::LN_10;
                let dbn = 100.0
                    * effective_alpha_db
                    * (1e-6 / (2.0 * std::f64::consts::PI)).powf(effective_alpha_power)
                    / twenty_log10_e;
                AbsorptionMode::PowerLaw {
                    alpha_coeff: dbn,
                    alpha_power: effective_alpha_power,
                }
            } else {
                AbsorptionMode::Lossless
            };

        let config = PSTDConfig {
            dt,
            nt: time_steps,
            sensor_mask: Some(sensor_mask.clone()),
            boundary,
            pml_inside,
            absorption_mode,
            nonlinearity: enable_nonlinear,
            ..Default::default()
        };

        // Create solver
        let mut solver = PSTDSolver::new(config, grid.clone(), medium.as_medium(), grid_source)?;

        // Build recorder with statistical modes if requested, replacing the default one.
        let modes = Self::recording_modes_from_strings(record_modes);
        if !modes.is_empty() {
            let shape = (grid.nx, grid.ny, grid.nz);
            solver.sensor_recorder =
                SensorRecorder::with_modes(Some(&sensor_mask), shape, time_steps + 1, &modes)?;
        } else if let Some(ordered) = transducer_ordered_indices {
            // Transducer override (no stats)
            solver.sensor_recorder = SensorRecorder::from_ordered_indices(ordered, time_steps + 1)?;
        }

        for source in sources {
            SolverTrait::add_source(&mut solver, source)?;
        }

        // Run simulation - SensorRecorder records pressure at each step
        solver.run_orchestrated(time_steps)?;

        // Extract statistics before consuming recorder
        let stats = solver.sensor_recorder.extract_all_stats();

        // Extract recorded time series: solver stores Nt+1 columns when the
        // initial state is recorded, but the Python API returns exactly Nt.
        let full_data = solver.extract_pressure_data().ok_or_else(|| {
            kwavers::core::error::KwaversError::Io(std::io::Error::other("No sensor data recorded"))
        })?;
        let recorded_data = Self::trim_initial_recorder_sample(full_data, time_steps);

        Ok((recorded_data, stats))
    }

    /// Dispatch GPU-resident PSTD if the `gpu` feature is enabled and GPU is
    /// available; otherwise fall back to the CPU PSTD implementation.
    #[allow(clippy::too_many_arguments)]
    fn run_gpu_pstd_or_cpu_fallback(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        alpha_coeff_db: f64,
        alpha_power: f64,
        grid_source: GridSource,
        sources: Vec<Box<dyn KwaversSource>>,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
        enable_nonlinear: bool,
        record_modes: &[String],
    ) -> KwaversResult<(ndarray::Array2<f64>, Option<SampledStatistics>)> {
        #[cfg(feature = "gpu")]
        {
            // Attempt GPU path — fall back to CPU on any error.
            match Self::run_gpu_pstd_impl(
                grid,
                medium,
                time_steps,
                dt,
                alpha_coeff_db,
                alpha_power,
                &grid_source,
                sensor,
                transducer_sensor,
                pml_size,
                pml_size_xyz,
                pml_inside,
                pml_alpha_xyz,
            ) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    eprintln!("[PstdGpu] GPU path failed ({e}), falling back to CPU PSTD");
                }
            }
        }
        // CPU fallback (always compiled)
        Self::run_pstd_impl(
            grid,
            medium,
            time_steps,
            dt,
            enable_nonlinear,
            alpha_coeff_db,
            alpha_power,
            grid_source,
            sources,
            sensor,
            transducer_sensor,
            pml_size,
            pml_size_xyz,
            pml_inside,
            pml_alpha_xyz,
            record_modes,
        )
    }

    /// GPU-resident PSTD implementation (requires `gpu` feature).
    ///
    /// Builds GPU buffers, runs the PSTD time loop entirely on-GPU, and
    /// returns sensor pressure data as `(Array2<f64>, None)`.
    ///
    /// Limitations (current implementation):
    /// - Grid dimensions must be power-of-2 and ≤ 256.
    /// - Absorption (fractional Laplacian) is not yet implemented on GPU.
    /// - Only the static p_mask / p_signal source is supported.
    #[cfg(feature = "gpu")]
    #[allow(clippy::too_many_arguments, unused_variables)]
    fn run_gpu_pstd_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        alpha_coeff_db: f64,
        alpha_power: f64,
        grid_source: &GridSource,
        sensor: Option<&Sensor>,
        transducer_sensor: Option<&TransducerArray2D>,
        pml_size: Option<usize>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        pml_inside: bool,
        pml_alpha_xyz: Option<(f64, f64, f64)>,
    ) -> KwaversResult<(ndarray::Array2<f64>, Option<SampledStatistics>)> {
        use kwavers::domain::boundary::cpml::CPMLProfiles;
        use kwavers::solver::forward::pstd::gpu_pstd::GpuPstdSolver;
        use ndarray::Array2;

        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;
        let total = nx * ny * nz;

        // Power-of-2 and ≤256 check
        if !nx.is_power_of_two() || !ny.is_power_of_two() || !nz.is_power_of_two() {
            return Err(KwaversError::Io(std::io::Error::other(format!(
                "GPU PSTD requires power-of-2 grid dimensions, got {nx}×{ny}×{nz}"
            ))));
        }
        if nx > 256 || ny > 256 || nz > 256 {
            return Err(KwaversError::Io(std::io::Error::other(format!(
                "GPU PSTD supports N≤256 per axis, got {nx}×{ny}×{nz}"
            ))));
        }

        // ── Build CPML profiles ──────────────────────────────────────────────
        let min_dim = nx.min(ny).min(nz);
        let max_allowed = (min_dim.saturating_sub(2)) / 2;
        let default_thickness = (min_dim / 6).max(2);
        let thickness = pml_size.unwrap_or(default_thickness).min(max_allowed);

        let cpml_config = if let Some((px, py, pz)) = pml_size_xyz {
            let mut cfg = CPMLConfig::with_per_dimension_thickness(px, py, pz);
            if let Some((ax, ay, az)) = pml_alpha_xyz {
                cfg = cfg.with_alpha_xyz(ax, ay, az);
            }
            cfg
        } else {
            let mut cfg = CPMLConfig::with_thickness(thickness);
            if let Some((ax, ay, az)) = pml_alpha_xyz {
                cfg = cfg.with_alpha_xyz(ax, ay, az);
            }
            cfg
        };

        let c_ref = medium.as_medium().max_sound_speed();
        let profiles = CPMLProfiles::new(&cpml_config, grid, c_ref, dt)?;

        // ── Broadcast 1D PML arrays to 3D flat (row-major ix*ny*nz + iy*nz + iz) ──
        let mut pml_sgx_3d = vec![1.0f32; total];
        let mut pml_sgy_3d = vec![1.0f32; total];
        let mut pml_sgz_3d = vec![1.0f32; total];
        let mut pml_x_3d = vec![1.0f32; total];
        let mut pml_y_3d = vec![1.0f32; total];
        let mut pml_z_3d = vec![1.0f32; total];

        if pml_inside {
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let flat = ix * ny * nz + iy * nz + iz;
                        pml_sgx_3d[flat] = profiles.pml_x_sgx[ix] as f32;
                        pml_sgy_3d[flat] = profiles.pml_y_sgy[iy] as f32;
                        pml_sgz_3d[flat] = profiles.pml_z_sgz[iz] as f32;
                        pml_x_3d[flat] = profiles.pml_x[ix] as f32;
                        pml_y_3d[flat] = profiles.pml_y[iy] as f32;
                        pml_z_3d[flat] = profiles.pml_z[iz] as f32;
                    }
                }
            }
        }
        // If pml_inside=false, PML is outside the domain — all coefficients stay 1.0
        // (no absorption). TODO: handle exterior PML.

        // ── Medium arrays (f32 flat, row-major ix*ny*nz + iy*nz + iz) ──────────
        let mut c0_flat = vec![c_ref as f32; total];
        let mut rho0_flat = vec![1000.0f32; total];
        for ix in 0..nx {
            for iy in 0..ny {
                for iz in 0..nz {
                    let flat = ix * ny * nz + iy * nz + iz;
                    c0_flat[flat] = medium.as_medium().sound_speed(ix, iy, iz) as f32;
                    rho0_flat[flat] = medium.as_medium().density(ix, iy, iz) as f32;
                }
            }
        }

        // ── Sensor indices ────────────────────────────────────────────────────
        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);
        let mut sensor_indices: Vec<u32> = Vec::new();
        {
            let flat = sensor_mask
                .as_slice()
                .expect("sensor_mask must be C-contiguous");
            for (i, &v) in flat.iter().enumerate() {
                if v {
                    sensor_indices.push(i as u32);
                }
            }
        }

        // ── Source indices and signals ────────────────────────────────────────
        let n_dim_active = [nx > 1, ny > 1, nz > 1]
            .iter()
            .filter(|&&d| d)
            .count()
            .max(1);
        let dx_min = grid.dx.min(grid.dy).min(grid.dz);
        let mass_source_scale = 2.0 * dt / (n_dim_active as f64 * c_ref * dx_min);
        let density_scale = n_dim_active as f64 / 3.0;
        let combined_scale = (mass_source_scale * density_scale) as f32;

        let mut source_indices: Vec<u32> = Vec::new();
        let mut source_signals: Vec<f32> = Vec::new();

        if let (Some(p_mask), Some(p_signal)) = (&grid_source.p_mask, &grid_source.p_signal) {
            // Collect non-zero mask positions
            let mask_flat = p_mask.as_slice().expect("p_mask must be C-contiguous");
            for (i, &v) in mask_flat.iter().enumerate() {
                if v != 0.0 {
                    source_indices.push(i as u32);
                }
            }
            let n_src = source_indices.len();
            let n_sig_rows = p_signal.shape()[0];
            let n_sig_cols = p_signal.shape()[1].min(time_steps);

            // signals layout: [n_src * time_steps] row-major
            source_signals = vec![0.0f32; n_src * time_steps];
            for (src_idx, _) in source_indices.iter().enumerate() {
                let sig_row = if n_sig_rows == 1 {
                    0
                } else {
                    src_idx.min(n_sig_rows - 1)
                };
                for step in 0..n_sig_cols {
                    source_signals[src_idx * time_steps + step] =
                        (p_signal[[sig_row, step]] * combined_scale as f64) as f32;
                }
            }
        }

        // ── Velocity-x source indices and signals ─────────────────────────────
        let mut vel_x_indices: Vec<u32> = Vec::new();
        let mut vel_x_signals: Vec<f32> = Vec::new();

        if let (Some(u_mask), Some(u_signal)) = (&grid_source.u_mask, &grid_source.u_signal) {
            // u_signal shape: [3, num_sources, time_steps]; axis 0 = ux
            let mask_flat = u_mask.as_slice().expect("u_mask must be C-contiguous");
            for (i, &v) in mask_flat.iter().enumerate() {
                if v != 0.0 {
                    vel_x_indices.push(i as u32);
                }
            }
            let n_vel = vel_x_indices.len();
            let n_sig_srcs = u_signal.shape()[1];
            let n_sig_cols = u_signal.shape()[2].min(time_steps);

            // signals layout: [n_vel * time_steps] row-major; read ux component (axis 0)
            vel_x_signals = vec![0.0f32; n_vel * time_steps];
            for src_idx in 0..n_vel {
                let sig_row = src_idx.min(n_sig_srcs.saturating_sub(1));
                for step in 0..n_sig_cols {
                    vel_x_signals[src_idx * time_steps + step] =
                        u_signal[[0, sig_row, step]] as f32;
                }
            }
        }

        // ── Physics flags ─────────────────────────────────────────────────────
        // Check first voxel; works correctly for homogeneous media.
        // Heterogeneous media with spatially-varying properties handled per-voxel below.
        let has_nonlinear = medium.as_medium().nonlinearity(0, 0, 0) > 0.0;
        let has_absorption = alpha_coeff_db > 0.0;

        // ── BonA: B/(2A) per voxel ────────────────────────────────────────────
        let bon_a_flat: Vec<f32> = if has_nonlinear {
            (0..total)
                .map(|flat| {
                    let ix = flat / (ny * nz);
                    let iy = (flat % (ny * nz)) / nz;
                    let iz = flat % nz;
                    // k-Wave stores BonA as B/A; pressure EOS needs B/(2A)
                    (medium.as_medium().nonlinearity(ix, iy, iz) / 2.0) as f32
                })
                .collect()
        } else {
            vec![0.0f32; total]
        };

        // ── Alpha decay: exp(-alpha_Np_m * c0 * dt) per voxel ────────────────
        // Frequency-centred approximation at 1 MHz (centre frequency).
        let alpha_decay_flat: Vec<f32> = if has_absorption {
            let f0_mhz = 1.0_f64; // 1 MHz centre frequency approximation
            (0..total)
                .map(|flat| {
                    let ix = flat / (ny * nz);
                    let iy = (flat % (ny * nz)) / nz;
                    let iz = flat % nz;
                    let alpha_db_cm = medium.as_medium().absorption(ix, iy, iz);
                    let alpha_np_m = alpha_db_cm_to_np_m(alpha_db_cm, f0_mhz, alpha_power);
                    let c0_local = medium.as_medium().sound_speed(ix, iy, iz);
                    ((-alpha_np_m * c0_local * dt).exp()) as f32
                })
                .collect()
        } else {
            vec![1.0f32; total]
        };

        // ── Construct GPU solver (device created internally via with_auto_device) ──
        let mut solver = GpuPstdSolver::with_auto_device(
            grid,
            &c0_flat,
            &rho0_flat,
            dt,
            time_steps,
            c_ref,
            &pml_x_3d,
            &pml_y_3d,
            &pml_z_3d,
            &pml_sgx_3d,
            &pml_sgy_3d,
            &pml_sgz_3d,
            &bon_a_flat,
            &alpha_decay_flat,
            has_nonlinear,
            has_absorption,
        )
        .map_err(|e| KwaversError::Io(std::io::Error::other(e)))?;

        let sensor_data_f32 = solver.run(
            &sensor_indices,
            &source_indices,
            &source_signals,
            &vel_x_indices,
            &vel_x_signals,
        );

        // ── Convert Vec<f32> → Array2<f64> shape (n_sensors, time_steps) ─────
        let n_sensors = sensor_indices.len();
        let mut out = Array2::<f64>::zeros((n_sensors, time_steps));
        for s in 0..n_sensors {
            for t in 0..time_steps {
                out[[s, t]] = sensor_data_f32[s * time_steps + t] as f64;
            }
        }

        Ok((out, None))
    }
}

// ============================================================================
// GPU PSTD Session (persistent solver — reuses compiled pipelines across scan
// lines, eliminating ~500 ms shader-recompilation overhead per scan line)
// ============================================================================

/// Persistent GPU PSTD session for efficient B-mode scan-line loops.
///
/// Creating a new `GpuPstdSolver` per scan line is expensive (~500 ms) because
/// wgpu must compile ~13 WGSL compute pipelines from scratch.  `GpuPstdSession`
/// creates the solver **once** and re-uses compiled pipelines.  Between scan
/// lines you only re-upload the medium arrays via `run_scan_line()`.
///
/// Parameters (constructor)
/// ------------------------
/// grid : Grid
///     Simulation grid.
/// sound_speed : ndarray (nx, ny, nz)
///     Initial sound speed [m/s].
/// density : ndarray (nx, ny, nz)
///     Initial density [kg/m³].
/// absorption : ndarray (nx, ny, nz) or None
///     Absorption coefficient [dB/(MHz^y·cm)].  None → lossless.
/// nonlinearity : ndarray (nx, ny, nz) or None
///     B/A nonlinearity parameter.  None → linear.
/// dt : float
///     Time step [s].
/// time_steps : int
///     Number of time steps per scan line.
/// pml_size_xyz : (int, int, int), optional
///     PML thickness in grid points along each axis.  Default (10, 10, 10).
/// alpha_power : float, optional
///     Power-law exponent for absorption (default 1.5).
///
/// Examples
/// --------
/// >>> import pykwavers as pkw, numpy as np
/// >>> grid = pkw.Grid(64, 64, 64, 1e-4, 1e-4, 1e-4)
/// >>> ss   = np.full((64,64,64), 1540.0)
/// >>> rho  = np.full((64,64,64), 1060.0)
/// >>> session = pkw.GpuPstdSession(grid, ss, rho, dt=1e-8, time_steps=200)
/// >>> mask = np.zeros((64,64,64), dtype=bool); mask[32,32,32] = True
/// >>> ux_sig = np.zeros((1, 200))
/// >>> sd = session.run_scan_line(ss, rho, mask, ux_sig)  # shape (1, 200)
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
#[pyclass(unsendable)]
pub struct GpuPstdSession {
    // GPU solver — Some(_) when the gpu feature is enabled and construction succeeded.
    #[cfg(feature = "gpu")]
    solver: kwavers::solver::forward::pstd::gpu_pstd::GpuPstdSolver,

    // Grid dimensions (needed for flat-index computation)
    nx: usize,
    ny: usize,
    nz: usize,

    // Cached bon_a array (nonlinearity rarely changes between scan lines)
    bon_a_flat: Vec<f32>,
    // Cached alpha decay factor numerator: alpha_Np_m * dt  (c0 from new medium
    // is multiplied in at run time; precomputed to avoid redundant pow() calls)
    alpha_np_dt_flat: Vec<f32>,
    has_absorption: bool,

    // Time loop parameters
    time_steps: usize,

    // Source/sensor indices pre-computed from the initial mask (constant per session)
    // Source rows follow ndarray C-order to match velocity-mask source expansion.
    // Sensor rows follow MATLAB/Fortran order to match k-Wave transducer beamforming.
    sensor_indices: Vec<u32>,
    vel_x_indices: Vec<u32>,
    vel_x_signals: Vec<f32>,
    last_medium_upload_ns: u64,
    last_solver_run_ns: u64,
    last_materialize_ns: u64,
    last_total_ns: u64,

}

impl GpuPstdSession {
    fn rebuild_source_sensor_indices(
        &mut self,
        mask_arr: ndarray::ArrayView3<'_, f64>,
    ) -> PyResult<()> {
        if mask_arr.shape() != [self.nx, self.ny, self.nz] {
            return Err(PyValueError::new_err(format!(
                "mask shape {:?} must match session grid ({}, {}, {})",
                mask_arr.shape(),
                self.nx,
                self.ny,
                self.nz
            )));
        }

        // Source indices follow ndarray C-order to match Source.from_velocity_mask_2d
        // and the kwavers SourceHandler iteration order.
        self.vel_x_indices.clear();
        for ix in 0..self.nx {
            for iy in 0..self.ny {
                for iz in 0..self.nz {
                    if mask_arr[[ix, iy, iz]] != 0.0 {
                        let flat = ix * self.ny * self.nz + iy * self.nz + iz;
                        self.vel_x_indices.push(flat as u32);
                    }
                }
            }
        }

        // Sensor indices follow MATLAB/Fortran order (x-fastest) so the returned
        // pressure matrix is directly compatible with k-Wave transducer beamforming.
        self.sensor_indices.clear();
        for iz in 0..self.nz {
            for iy in 0..self.ny {
                for ix in 0..self.nx {
                    if mask_arr[[ix, iy, iz]] != 0.0 {
                        let flat = ix * self.ny * self.nz + iy * self.nz + iz;
                        self.sensor_indices.push(flat as u32);
                    }
                }
            }
        }

        Ok(())
    }

    fn update_velocity_signal_rows(
        &mut self,
        sig_arr: ndarray::ArrayView2<'_, f64>,
    ) -> PyResult<()> {
        let time_steps = self.time_steps;
        let signal_shape = sig_arr.shape();
        if signal_shape.len() != 2 {
            return Err(PyValueError::new_err(format!(
                "ux_signals must be 2D, got shape {:?}",
                signal_shape
            )));
        }

        let n_vel = self.vel_x_indices.len();
        let n_sig_srcs = signal_shape[0];
        let n_sig_cols = signal_shape[1].min(time_steps);

        if n_vel > 0 && n_sig_srcs == 0 {
            return Err(PyValueError::new_err(
                "ux_signals must contain at least one source row for a non-empty mask",
            ));
        }

        self.vel_x_signals.clear();
        self.vel_x_signals.resize(n_vel * time_steps, 0.0f32);
        for src_idx in 0..n_vel {
            let sig_row = src_idx.min(n_sig_srcs.saturating_sub(1));
            for step in 0..n_sig_cols {
                self.vel_x_signals[src_idx * time_steps + step] = sig_arr[[sig_row, step]] as f32;
            }
        }

        Ok(())
    }
}

#[pymethods]
impl GpuPstdSession {
    /// Create a persistent GPU PSTD session.
    ///
    /// `sound_speed`, `density`, `absorption`, `nonlinearity` are the *initial*
    /// medium arrays (shape nx×ny×nz).  You can provide updated `sound_speed`
    /// and `density` arrays per scan line via `run_scan_line()`.
    #[new]
    #[pyo3(signature = (
        grid, sound_speed, density,
        dt, time_steps,
        absorption=None, nonlinearity=None,
        pml_size_xyz=None, alpha_power=1.5
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        _py: Python<'_>,
        grid: &Grid,
        sound_speed: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        dt: f64,
        time_steps: usize,
        absorption: Option<PyReadonlyArray3<f64>>,
        nonlinearity: Option<PyReadonlyArray3<f64>>,
        pml_size_xyz: Option<(usize, usize, usize)>,
        alpha_power: f64,
    ) -> PyResult<Self> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = (
                _py,
                grid,
                sound_speed,
                density,
                dt,
                time_steps,
                absorption,
                nonlinearity,
                pml_size_xyz,
                alpha_power,
            );
            return Err(PyRuntimeError::new_err(
                "GpuPstdSession requires the 'gpu' feature.  \
                 Rebuild pykwavers with --features gpu.",
            ));
        }

        #[cfg(feature = "gpu")]
        {
            use kwavers::domain::boundary::cpml::CPMLConfig;
            use kwavers::domain::boundary::cpml::CPMLProfiles;
            use kwavers::solver::forward::pstd::gpu_pstd::GpuPstdSolver;

            let kgrid = &grid.inner;
            let nx = kgrid.nx;
            let ny = kgrid.ny;
            let nz = kgrid.nz;
            let total = nx * ny * nz;

            // ── Validate grid is power-of-2 and small enough for GPU ──────────
            if !nx.is_power_of_two() || !ny.is_power_of_two() || !nz.is_power_of_two() {
                return Err(PyValueError::new_err(format!(
                    "GpuPstdSession requires power-of-2 grid; got {}x{}x{}",
                    nx, ny, nz
                )));
            }
            if nx > 256 || ny > 256 || nz > 256 {
                return Err(PyValueError::new_err(format!(
                    "GpuPstdSession: grid axis max 256 pts; got {}x{}x{}",
                    nx, ny, nz
                )));
            }

            // ── Medium arrays ─────────────────────────────────────────────────
            let ss_arr = sound_speed.as_array();
            let rho_arr = density.as_array();

            // C-contiguous ndarray: flat index = ix*ny*nz + iy*nz + iz, matching GPU layout.
            let c0_flat: Vec<f32> = ss_arr.iter().map(|&v| v as f32).collect();
            let rho0_flat: Vec<f32> = rho_arr.iter().map(|&v| v as f32).collect();

            // Detect c_ref as max sound speed for stability
            let c_ref = c0_flat.iter().cloned().fold(0.0f32, f32::max) as f64;

            // ── Nonlinearity (bon_a) ──────────────────────────────────────────
            let bon_a_flat: Vec<f32> = if let Some(ref nl) = nonlinearity {
                let nl_arr = nl.as_array();
                nl_arr.iter().map(|&v| (v / 2.0) as f32).collect() // B/(2A)
            } else {
                vec![0.0f32; total]
            };

            // ── Absorption (alpha_decay) ──────────────────────────────────────
            let has_absorption = absorption.is_some();
            // alpha_np_dt_flat[i] = alpha_Np_m[i] * dt  (c0 multiplied in at run time)
            // The GPU session currently uses a centre-frequency attenuation model
            // at f0 = 1 MHz, matching the phased-array parity examples.
            let alpha_np_dt_flat: Vec<f32> = if let Some(ref ab) = absorption {
                let ab_arr = ab.as_array();
                let f0_mhz = 1.0_f64;
                ab_arr.iter()
                    .map(|&v| (alpha_db_cm_to_np_m(v, f0_mhz, alpha_power) * dt) as f32)
                    .collect()
            } else {
                vec![0.0f32; total]
            };

            // ── Alpha decay from initial c0 ───────────────────────────────────
            let alpha_decay_flat: Vec<f32> = if has_absorption {
                (0..total)
                    .map(|i| (-alpha_np_dt_flat[i] * c0_flat[i]).exp())
                    .collect()
            } else {
                vec![1.0f32; total]
            };

            // ── PML profiles ──────────────────────────────────────────────────
            let (pml_x_sz, pml_y_sz, pml_z_sz) = pml_size_xyz.unwrap_or((10, 10, 10));
            let pml_config = CPMLConfig::with_per_dimension_thickness(pml_x_sz, pml_y_sz, pml_z_sz);
            let profiles = CPMLProfiles::new(&pml_config, kgrid, c_ref, dt)
                .map_err(|e| PyRuntimeError::new_err(format!("PML init failed: {e}")))?;

            let mut pml_x_3d = vec![1.0f32; total];
            let mut pml_y_3d = vec![1.0f32; total];
            let mut pml_z_3d = vec![1.0f32; total];
            let mut pml_sgx_3d = vec![1.0f32; total];
            let mut pml_sgy_3d = vec![1.0f32; total];
            let mut pml_sgz_3d = vec![1.0f32; total];
            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let flat = ix * ny * nz + iy * nz + iz;
                        pml_sgx_3d[flat] = profiles.pml_x_sgx[ix] as f32;
                        pml_sgy_3d[flat] = profiles.pml_y_sgy[iy] as f32;
                        pml_sgz_3d[flat] = profiles.pml_z_sgz[iz] as f32;
                        pml_x_3d[flat] = profiles.pml_x[ix] as f32;
                        pml_y_3d[flat] = profiles.pml_y[iy] as f32;
                        pml_z_3d[flat] = profiles.pml_z[iz] as f32;
                    }
                }
            }

            // ── Construct GPU solver (compiles pipelines once here) ───────────
            let solver = GpuPstdSolver::with_auto_device(
                kgrid,
                &c0_flat,
                &rho0_flat,
                dt,
                time_steps,
                c_ref,
                &pml_x_3d,
                &pml_y_3d,
                &pml_z_3d,
                &pml_sgx_3d,
                &pml_sgy_3d,
                &pml_sgz_3d,
                &bon_a_flat,
                &alpha_decay_flat,
                nonlinearity.is_some(),
                has_absorption,
            )
            .map_err(|e| PyRuntimeError::new_err(format!("GPU solver init failed: {e}")))?;

            Ok(Self {
                solver,
                nx,
                ny,
                nz,
                bon_a_flat,
                alpha_np_dt_flat,
                has_absorption,
                time_steps,
                sensor_indices: Vec::new(),
                vel_x_indices: Vec::new(),
                vel_x_signals: Vec::new(),
                last_medium_upload_ns: 0,
                last_solver_run_ns: 0,
                last_materialize_ns: 0,
                last_total_ns: 0,
            })
        }
    }

    /// Set the source and sensor mask for all scan lines (constant per session).
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray bool (nx, ny, nz)
    ///     True at source+sensor grid positions.
    /// ux_signals : ndarray f64 (n_sources, time_steps)
    ///     Per-source x-velocity signal [m/s].  Pass shape (0, Nt) for no source.
    fn set_source_sensor(
        &mut self,
        _py: Python<'_>,
        mask: PyReadonlyArray3<f64>,
        ux_signals: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let mask_arr = mask.as_array();
        let sig_arr = ux_signals.as_array();
        self.rebuild_source_sensor_indices(mask_arr)?;
        self.update_velocity_signal_rows(sig_arr)
    }

    /// Cache the source/sensor mask when the geometry is invariant across runs.
    ///
    /// This is the preferred setup for phased-array steering loops where only
    /// the source delays change between scan lines.
    fn set_source_sensor_mask(
        &mut self,
        _py: Python<'_>,
        mask: PyReadonlyArray3<f64>,
    ) -> PyResult<()> {
        self.rebuild_source_sensor_indices(mask.as_array())
    }

    /// Update only the x-velocity source signals for a previously cached mask.
    ///
    /// This avoids rebuilding source/sensor indices when the aperture geometry is
    /// fixed and only steering delays or apodization weights change per scan line.
    fn set_velocity_signals(
        &mut self,
        _py: Python<'_>,
        ux_signals: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        self.update_velocity_signal_rows(ux_signals.as_array())
    }

    /// Return the timing profile from the most recent scan-line execution.
    ///
    /// Durations are host-side wall-clock measurements in nanoseconds.
    #[getter]
    fn last_run_profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let profile = PyDict::new(py);
        profile.set_item("medium_upload_ns", self.last_medium_upload_ns)?;
        profile.set_item("solver_run_ns", self.last_solver_run_ns)?;
        profile.set_item("materialize_ns", self.last_materialize_ns)?;
        profile.set_item("total_ns", self.last_total_ns)?;
        profile.set_item("n_sensors", self.sensor_indices.len())?;
        profile.set_item("n_velocity_sources", self.vel_x_indices.len())?;
        Ok(profile)
    }

    /// Run one scan line with updated medium (sound_speed, density).
    ///
    /// Uploads the new medium to the GPU (5 buffer writes, ~ms), then runs the
    /// full time loop.  Returns sensor pressure as an ndarray (n_sensors, Nt).
    ///
    /// Parameters
    /// ----------
    /// sound_speed : ndarray f64 (nx, ny, nz)
    ///     Updated sound speed for this scan line.
    /// density : ndarray f64 (nx, ny, nz)
    ///     Updated density for this scan line.
    ///
    /// Use `run_scan_line_cached()` when the medium is unchanged between scan lines.
    fn run_scan_line<'py>(
        &mut self,
        _py: Python<'py>,
        _sound_speed: PyReadonlyArray3<f64>,
        _density: PyReadonlyArray3<f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        #[cfg(not(feature = "gpu"))]
        {
            return Err(PyRuntimeError::new_err("GPU feature not enabled"));
        }

        #[cfg(feature = "gpu")]
        {
            let total_t0 = std::time::Instant::now();
            let nx = self.nx;
            let ny = self.ny;
            let nz = self.nz;
            let total = nx * ny * nz;

            let ss_arr = _sound_speed.as_array();
            let rho_arr = _density.as_array();

            // Build updated flat medium arrays.
            // numpy arrays passed from Python are C-contiguous; ss_arr.iter() visits
            // elements in flat C-order (ix*ny*nz + iy*nz + iz), matching the GPU layout.
            let c0_flat: Vec<f32> = ss_arr.iter().map(|&v| v as f32).collect();
            let rho0_flat: Vec<f32> = rho_arr.iter().map(|&v| v as f32).collect();

            // Recompute alpha_decay with new c0 (alpha_np_dt_flat already has dt baked in)
            let alpha_decay_flat: Vec<f32> = if self.has_absorption {
                (0..total)
                    .map(|i| (-self.alpha_np_dt_flat[i] * c0_flat[i]).exp())
                    .collect()
            } else {
                vec![1.0f32; total]
            };

            let upload_t0 = std::time::Instant::now();
            // Re-upload medium buffers to GPU
            self.solver
                .update_medium(&c0_flat, &rho0_flat, &self.bon_a_flat, &alpha_decay_flat);
            self.last_medium_upload_ns = upload_t0.elapsed().as_nanos() as u64;

            let result = self.run_scan_line_cached(_py);
            self.last_total_ns = total_t0.elapsed().as_nanos() as u64;
            result
        }
    }

    /// Run one scan line using the currently resident medium buffers.
    ///
    /// This is intended for repeated steering/focusing runs in a fixed medium.
    fn run_scan_line_cached<'py>(&mut self, _py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        #[cfg(not(feature = "gpu"))]
        {
            return Err(PyRuntimeError::new_err("GPU feature not enabled"));
        }

        #[cfg(feature = "gpu")]
        {
            let total_t0 = std::time::Instant::now();
            self.last_medium_upload_ns = 0;
            let time_steps = self.time_steps;

            // No pressure sources (velocity-only B-mode)
            let source_indices: Vec<u32> = Vec::new();
            let source_signals: Vec<f32> = Vec::new();

            // Run time loop
            let solver_t0 = std::time::Instant::now();
            let sensor_data_f32 = self.solver.run(
                &self.sensor_indices,
                &source_indices,
                &source_signals,
                &self.vel_x_indices,
                &self.vel_x_signals,
            );
            self.last_solver_run_ns = solver_t0.elapsed().as_nanos() as u64;

            // Shape → (n_sensors, time_steps) f64 numpy array
            // Use flat iterator for SIMD-vectorisable f32→f64 cast (no index arithmetic).
            let materialize_t0 = std::time::Instant::now();
            let n_sensors = self.sensor_indices.len();
            let out_flat: Vec<f64> = sensor_data_f32.iter().map(|&v| v as f64).collect();
            let out = ndarray::Array2::from_shape_vec((n_sensors, time_steps), out_flat)
                .expect("sensor_data shape mismatch");
            self.last_materialize_ns = materialize_t0.elapsed().as_nanos() as u64;
            if self.last_medium_upload_ns == 0 {
                self.last_total_ns = total_t0.elapsed().as_nanos() as u64;
            }

            Ok(PyArray2::from_owned_array(_py, out))
        }
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
    /// 1D sensor data (single sensor) [Pa]
    sensor_data_1d: Option<Py<PyArray1<f64>>>,
    /// 2D sensor data (multi-sensor, shape: n_sensors x n_timesteps) [Pa]
    sensor_data_2d: Option<Py<PyArray2<f64>>>,
    /// Time vector [s]
    #[pyo3(get)]
    time: Py<PyArray1<f64>>,
    /// Grid shape (nx, ny, nz)
    #[pyo3(get)]
    shape: (usize, usize, usize),
    /// Sensor data shape as `(num_sensors, time_steps)`.
    #[pyo3(get)]
    sensor_data_shape: (usize, usize),
    /// Number of time steps
    #[pyo3(get)]
    time_steps: usize,
    /// Time step [s]
    #[pyo3(get)]
    dt: f64,
    /// Final simulation time [s]
    #[pyo3(get)]
    final_time: f64,
    /// Maximum pressure at each sensor position over all time steps [Pa] (None if not recorded)
    #[pyo3(get)]
    p_max: Option<Py<PyArray1<f64>>>,
    /// Minimum pressure at each sensor position over all time steps [Pa] (None if not recorded)
    #[pyo3(get)]
    p_min: Option<Py<PyArray1<f64>>>,
    /// RMS pressure at each sensor position over all time steps [Pa] (None if not recorded)
    #[pyo3(get)]
    p_rms: Option<Py<PyArray1<f64>>>,
    /// Final pressure at each sensor position [Pa] (None if not recorded)
    #[pyo3(get)]
    p_final: Option<Py<PyArray1<f64>>>,
}

#[pymethods]
impl SimulationResult {
    /// Get the sensor data as a numpy array.
    /// Returns a 1D array for single-sensor simulations, 2D (n_sensors, n_timesteps) for multi-sensor.
    #[getter]
    fn sensor_data<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        if let Some(ref data_2d) = self.sensor_data_2d {
            data_2d.clone_ref(py).into_any()
        } else if let Some(ref data_1d) = self.sensor_data_1d {
            data_1d.clone_ref(py).into_any()
        } else {
            py.None()
        }
    }

    /// Number of sensor points.
    #[getter]
    fn num_sensors(&self) -> usize {
        self.sensor_data_shape.0
    }

    /// String representation.
    fn __repr__(&self) -> String {
        let data_desc = if self.sensor_data_2d.is_some() {
            "multi-sensor 2D"
        } else {
            "single-sensor 1D"
        };
        format!(
            "SimulationResult(data={}, shape={:?}, time_steps={}, dt={:.2e}, final_time={:.2e})",
            data_desc, self.shape, self.time_steps, self.dt, self.final_time
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
fn _pykwavers(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Grid>()?;
    m.add_class::<Medium>()?;
    m.add_class::<Source>()?;
    m.add_class::<KWaveArray>()?;
    m.add_class::<TransducerArray2D>()?;
    m.add_class::<Sensor>()?;
    m.add_class::<Simulation>()?;
    m.add_class::<SimulationResult>()?;
    m.add_class::<SolverType>()?;
    m.add_class::<GpuPstdSession>()?;

    // Phase 22 bindings
    m.add_class::<PyPIDController>()?;
    m.add_class::<PyBubbleField>()?;
    m.add_function(wrap_pyfunction!(resample_to_target_grid, m)?)?;

    // Register utility functions
    utils_bindings::register_utils(m)?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "Ryan Clanton PhD")?;

    Ok(())
}

// ============================================================================
// Signal Implementation
// ============================================================================

/// Simple sine wave signal for testing
#[derive(Clone)]
struct SineSignal {
    frequency: f64,
    amplitude: f64,
}

impl SineSignal {
    fn new(frequency: f64, amplitude: f64) -> Self {
        Self {
            frequency,
            amplitude,
        }
    }
}

impl Signal for SineSignal {
    fn amplitude(&self, t: f64) -> f64 {
        self.amplitude * (2.0 * std::f64::consts::PI * self.frequency * t).sin()
    }

    fn duration(&self) -> Option<f64> {
        None // Continuous signal
    }

    fn frequency(&self, _t: f64) -> f64 {
        self.frequency
    }

    fn phase(&self, t: f64) -> f64 {
        2.0 * std::f64::consts::PI * self.frequency * t
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

impl std::fmt::Debug for SineSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SineSignal")
            .field("frequency", &self.frequency)
            .field("amplitude", &self.amplitude)
            .finish()
    }
}

/// Sampled signal from Python array
#[derive(Clone)]
struct SampledSignal {
    values: Array1<f64>,
    dt: f64,
}

impl SampledSignal {
    fn new(values: Array1<f64>, dt: f64) -> Self {
        Self { values, dt }
    }
}

impl Signal for SampledSignal {
    fn amplitude(&self, t: f64) -> f64 {
        if self.dt <= 0.0 || self.values.is_empty() {
            return 0.0;
        }
        let index = (t / self.dt).round() as isize;
        if index >= 0 && (index as usize) < self.values.len() {
            self.values[index as usize]
        } else {
            0.0
        }
    }

    fn duration(&self) -> Option<f64> {
        Some(self.values.len() as f64 * self.dt)
    }

    fn frequency(&self, _t: f64) -> f64 {
        0.0
    }

    fn phase(&self, _t: f64) -> f64 {
        0.0
    }

    fn clone_box(&self) -> Box<dyn Signal> {
        Box::new(self.clone())
    }
}

impl std::fmt::Debug for SampledSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SampledSignal")
            .field("len", &self.values.len())
            .field("dt", &self.dt)
            .finish()
    }
}

#[cfg(test)]
mod simulation_contract_tests {
    use super::Simulation;
    use ndarray::array;

    #[test]
    fn trim_initial_recorder_sample_discards_t0_column_only_when_present() {
        let nt_plus_one = array![[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]];
        let trimmed = Simulation::trim_initial_recorder_sample(nt_plus_one, 3);
        assert_eq!(trimmed.shape(), &[2, 3]);
        assert_eq!(trimmed[[0, 0]], 1.0);
        assert_eq!(trimmed[[1, 2]], 13.0);

        let exact_nt = array![[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]];
        let untouched = Simulation::trim_initial_recorder_sample(exact_nt.clone(), 3);
        assert_eq!(untouched, exact_nt);
    }
}

// ============================================================================
// Phase 22 Wrappers: PID Controller, Registration, and Bubble Field
// ============================================================================

#[pyclass(name = "PIDController")]
pub struct PyPIDController {
    inner: kwavers::physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDController,
}

#[pymethods]
impl PyPIDController {
    #[new]
    #[pyo3(signature = (kp, ki, kd, setpoint, sample_time=0.001, output_min=0.0, output_max=1.0, integral_limit=100.0))]
    fn new(
        kp: f64,
        ki: f64,
        kd: f64,
        setpoint: f64,
        sample_time: f64,
        output_min: f64,
        output_max: f64,
        integral_limit: f64,
    ) -> Self {
        let gains = kwavers::physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDGains { kp, ki, kd };
        let mut config = kwavers::physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDConfig::default();
        config.gains = gains;
        config.sample_time = sample_time;
        config.output_min = output_min;
        config.output_max = output_max;
        config.integral_limit = integral_limit;
        let mut controller = kwavers::physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDController::new(config);
        controller.set_setpoint(setpoint);
        Self { inner: controller }
    }

    fn update(&mut self, measurement: f64) -> (f64, f64, f64, f64) {
        let out = self.inner.update(measurement);
        (
            out.control_signal,
            out.proportional_term,
            out.integral_term,
            out.derivative_term,
        )
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

#[pyfunction]
fn resample_to_target_grid<'py>(
    py: Python<'py>,
    source_image: PyReadonlyArray3<f64>,
    transform: [f64; 16],
    target_dims: (usize, usize, usize),
) -> Py<PyArray3<f64>> {
    use kwavers::physics::acoustics::imaging::fusion::registration::resample_to_target_grid as kwavers_resample;
    let arr = source_image.as_array().to_owned();
    let resampled = py.detach(|| kwavers_resample(&arr, &transform, target_dims));
    PyArray3::from_owned_array(py, resampled).into()
}

#[pyclass(name = "BubbleField")]
pub struct PyBubbleField {
    inner: kwavers::physics::acoustics::bubble_dynamics::bubble_field::BubbleField,
}

#[pymethods]
impl PyBubbleField {
    #[new]
    fn new(nx: usize, ny: usize, nz: usize) -> Self {
        let params =
            kwavers::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters::default();
        Self {
            inner: kwavers::physics::acoustics::bubble_dynamics::bubble_field::BubbleField::new(
                (nx, ny, nz),
                params,
            ),
        }
    }

    fn add_center_bubble(&mut self) {
        let params =
            kwavers::physics::acoustics::bubble_dynamics::bubble_state::BubbleParameters::default();
        self.inner.add_center_bubble(&params);
    }

    fn num_bubbles(&self) -> usize {
        self.inner.bubbles.len()
    }
}
