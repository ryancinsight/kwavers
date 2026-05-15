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
use kwavers::domain::sensor::recorder::fields::{SensorRecordField, SensorRecordSpec};
use kwavers::domain::sensor::recorder::pressure_statistics::SampledStatistics;
use kwavers::domain::sensor::recorder::simple::SensorRecorder;
use kwavers::domain::sensor::recorder::velocity_statistics::SampledVelocityStats;
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict, PyTuple};

mod pam_bindings;

// Re-exports from kwavers core
use kwavers::core::error::{KwaversError, KwaversResult};
use kwavers::domain::boundary::cpml::CPMLConfig;
use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::medium::core::CoreMedium;
use kwavers::domain::medium::heterogeneous::{HeterogeneousFactory, HeterogeneousMedium};
use kwavers::domain::medium::traits::Medium as MediumTrait;
use kwavers::domain::medium::HomogeneousMedium;
use kwavers::domain::signal::Signal;
use kwavers::domain::source::array_2d::{
    ApodizationType as KwaversApodizationType, TransducerArray2D as KwaversTransducerArray2D,
    TransducerArray2DConfig,
};
use kwavers::domain::source::custom::FunctionSource;
use kwavers::domain::source::grid_source::SourceMode;
use kwavers::domain::source::wavefront::plane_wave::{
    InjectionMode, PlaneWaveConfig, PlaneWaveSource,
};
use kwavers::domain::source::{GridSource, Source as KwaversSource, SourceField};
#[cfg(feature = "gpu")]
use kwavers::physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
use kwavers::physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers::solver::forward::fdtd::config::{FdtdConfig, KSpaceCorrectionMode};
use kwavers::solver::forward::fdtd::solver::FdtdSolver;
use kwavers::solver::forward::pstd::config::{BoundaryConfig, CompatibilityMode, PSTDConfig};
use kwavers::solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers::solver::geometry::Geometry;
use kwavers::solver::interface::solver::Solver as SolverTrait;
use kwavers::solver::inverse::reconstruction::photoacoustic::{
    kspace_line_recon as kwavers_kspace_line_recon, LineReconDataOrder, LineReconInterpolation,
};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
#[cfg(feature = "gpu")]
use std::borrow::Cow;
use std::sync::Arc;
// ============================================================================
// Utility Function Bindings
// ============================================================================

mod fft_bindings;
mod field_surrogate_bindings;
mod ritk_image;
mod seismic_bindings;
mod theranostic_bindings;
mod thermal_bindings;
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
/// The helper remains local for the frequency-domain utility tests; the GPU
/// PSTD path uses the shared kwavers spectral conversion helper.
#[allow(dead_code)]
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
    /// Elastic-wave solver (4th-order FD with velocity-Verlet integration,
    /// PML boundary, supports compressional + shear waves). The Python-level
    /// equivalent of k-Wave's `pstdElastic2D` / `pstdElastic3D`.
    ///
    /// Phase A.2 of ADR 007 — current capability: initial-displacement IVP
    /// only (single component, configurable axis); records the chosen
    /// displacement component at sensor mask positions. Stress / velocity
    /// source masks and multi-component recording land in Phases A.3 and
    /// A.2.5 respectively.
    Elastic,
    /// Pseudospectral elastic solver — drives the canonical PSTD step loop
    /// with the [`pstd::extensions::PstdElasticPlugin`] for full elastic
    /// (μ ≥ 0) propagation. With μ = 0 reduces exactly to baseline acoustic
    /// PSTD per the plugin's acoustic-fluid-limit theorem.
    ///
    /// Currently velocity-source + sensor-mask only; no PML yet (short-
    /// propagation diagnostics + cross-engine parity validation against
    /// KWave.jl's `pstd_elastic_2d`). Adds boundary absorption in a follow-
    /// up step. See `kwavers::solver::forward::pstd::extensions` and the
    /// canonical solver matrix in `solver::forward` module docs.
    ElasticPSTD,
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
            SolverType::Elastic => "SolverType.Elastic".to_string(),
            SolverType::ElasticPSTD => "SolverType.ElasticPSTD".to_string(),
        }
    }

    /// Human-readable string.
    fn __str__(&self) -> String {
        match self {
            SolverType::FDTD => "FDTD".to_string(),
            SolverType::PSTD => "PSTD".to_string(),
            SolverType::Hybrid => "Hybrid".to_string(),
            SolverType::PstdGpu => "PstdGpu".to_string(),
            SolverType::Elastic => "Elastic".to_string(),
            SolverType::ElasticPSTD => "ElasticPSTD".to_string(),
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
    #[pyo3(signature = (sound_speed, density, absorption=None, alpha_power=None, nonlinearity=None))]
    fn new(
        sound_speed: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        absorption: Option<PyReadonlyArray3<f64>>,
        // Power-law exponent y for absorption: α(f) = α₀·(f/f_ref)^y.
        // Pass a scalar float (broadcast to all voxels) or a 3D array
        // matching the shape of `sound_speed`.  Default: 1.0.
        alpha_power: Option<&pyo3::Bound<'_, pyo3::PyAny>>,
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

        if c_arr.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err(
                "All sound_speed values must be positive",
            ));
        }
        if rho_arr.iter().any(|&v| v <= 0.0) {
            return Err(PyValueError::new_err("All density values must be positive"));
        }

        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);
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

        // alpha_power: accept scalar float OR 3D ndarray.
        if let Some(py_ap) = alpha_power {
            use ndarray::Array3 as A3;
            if let Ok(scalar) = py_ap.extract::<f64>() {
                het.alpha_power = A3::from_elem((nx, ny, nz), scalar);
            } else if let Ok(arr) = py_ap.extract::<PyReadonlyArray3<f64>>() {
                let ap_arr = arr.as_array().to_owned();
                if ap_arr.shape() != [nx, ny, nz] {
                    return Err(PyValueError::new_err(
                        "alpha_power shape must match sound_speed shape",
                    ));
                }
                het.alpha_power = ap_arr;
            } else {
                return Err(PyValueError::new_err(
                    "alpha_power must be a float or a 3D ndarray matching sound_speed shape",
                ));
            }
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

    /// Create a homogeneous **elastic** medium parameterised by physical wave
    /// speeds.
    ///
    /// This is the natural pykwavers equivalent of k-Wave's
    /// ``medium.sound_speed_compression`` / ``medium.sound_speed_shear``
    /// inputs to ``pstdElastic2D`` / ``pstdElastic3D``. The Lamé parameters
    /// are derived in closed form from the elastic-wave dispersion relations:
    ///
    /// ::
    ///
    ///     μ = ρ · c_s²                     (shear modulus)
    ///     λ = ρ · (c_p² − 2 · c_s²)         (first Lamé parameter)
    ///
    /// Parameters
    /// ----------
    /// c_compression : float
    ///     Compressional (P-wave) speed [m/s]. Must be positive.
    /// c_shear : float
    ///     Shear (S-wave) speed [m/s]. Must be ≥ 0 and satisfy
    ///     ``2·c_shear² ≤ c_compression²`` (thermodynamic stability,
    ///     equivalent to ``ν ≥ 0``).
    /// density : float
    ///     Mass density [kg/m³]. Must be positive.
    /// grid : Grid, optional
    ///     Computational grid; when omitted a default grid is used (the
    ///     elastic medium itself is uniform, so grid only sizes the cached
    ///     property arrays).
    ///
    /// Returns
    /// -------
    /// Medium
    ///     Homogeneous elastic medium with Lamé parameters set from the
    ///     supplied wave speeds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If any input is non-finite, non-positive (where required), or the
    ///     stability bound ``2·c_s² ≤ c_p²`` is violated.
    ///
    /// Examples
    /// --------
    /// >>> # k-Wave example_ewp_layered_medium upper layer (water)
    /// >>> water = pkw.Medium.elastic(1500.0, 0.0, 1000.0)
    /// >>> # k-Wave example_ewp_layered_medium lower layer (bone-like)
    /// >>> bone  = pkw.Medium.elastic(2000.0, 800.0, 1200.0)
    #[staticmethod]
    #[pyo3(signature = (c_compression, c_shear, density, grid=None))]
    fn elastic(
        c_compression: f64,
        c_shear: f64,
        density: f64,
        grid: Option<&Grid>,
    ) -> PyResult<Self> {
        let default_grid = KwaversGrid::default();
        let grid_ref = grid.map(|g| &g.inner).unwrap_or(&default_grid);

        let medium =
            HomogeneousMedium::elastic_homogeneous(density, c_compression, c_shear, grid_ref)
                .ok_or_else(|| {
                    PyValueError::new_err(
                        "Invalid elastic parameters. Requirements: density > 0, \
                 c_compression > 0, c_shear ≥ 0, 2·c_shear² ≤ c_compression². \
                 (Stability bound: ν ≥ 0; recovers fluid medium when c_shear = 0.)",
                    )
                })?;

        Ok(Medium {
            inner: MediumInner::Homogeneous(Box::new(medium)),
        })
    }

    /// Create a heterogeneous elastic medium from per-voxel wave-speed and density arrays.
    ///
    /// Lamé parameters are computed per voxel:
    ///   μ   = ρ · c_s²
    ///   λ   = ρ · (c_p² − 2·c_s²)
    ///
    /// Parameters
    /// ----------
    /// c_compression : ndarray (3D float64)
    ///     P-wave speed [m/s] at every voxel.
    /// c_shear : ndarray (3D float64)
    ///     S-wave speed [m/s] at every voxel; set to 0 for fluid voxels.
    /// density : ndarray (3D float64)
    ///     Density [kg/m³] at every voxel.
    /// reference_frequency : float, optional
    ///     Reference frequency for absorption [Hz] (default 1 MHz).
    ///
    /// Returns
    /// -------
    /// Medium
    ///     Heterogeneous elastic medium with Lamé parameters set from the wave speeds.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If array shapes mismatch, any density ≤ 0, any c_compression ≤ 0,
    ///     any c_shear < 0, or stability is violated (2·c_s² > c_p² at any voxel).
    #[staticmethod]
    #[pyo3(signature = (c_compression, c_shear, density, reference_frequency=1.0e6))]
    fn elastic_heterogeneous(
        c_compression: PyReadonlyArray3<f64>,
        c_shear: PyReadonlyArray3<f64>,
        density: PyReadonlyArray3<f64>,
        reference_frequency: f64,
    ) -> PyResult<Self> {
        let cp = c_compression.as_array();
        let cs = c_shear.as_array();
        let rho = density.as_array();

        let medium = HeterogeneousFactory::from_elastic_arrays(cp, cs, rho, reference_frequency)
            .map_err(PyValueError::new_err)?;

        Ok(Medium {
            inner: MediumInner::Heterogeneous(Box::new(medium)),
        })
    }

    /// Compressional (P-wave) speed [m/s].
    ///
    /// Computed from the stored Lamé parameters and density via
    /// ``c_p = sqrt((λ + 2μ) / ρ)``. For a fluid medium this collapses to
    /// the acoustic sound speed.
    #[getter]
    fn c_compression(&self) -> f64 {
        let m = self.inner.as_medium();
        // Use the centre voxel; HomogeneousMedium is uniform so any (i,j,k) works.
        // For heterogeneous media this returns the centre value which is a
        // documented limitation of this scalar getter (see also `density`).
        let lambda = match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_lambda_value(),
            MediumInner::Heterogeneous(_) => {
                // Heterogeneous λ access via grid+coords, not exposed here.
                0.0
            }
        };
        let mu = match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_mu_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        };
        let rho = m.density(0, 0, 0);
        if rho > 0.0 {
            ((lambda + 2.0 * mu) / rho).sqrt()
        } else {
            0.0
        }
    }

    /// Shear (S-wave) speed [m/s].
    ///
    /// Computed from the shear modulus and density: ``c_s = sqrt(μ / ρ)``.
    /// Returns 0 for fluid media (μ = 0).
    #[getter]
    fn c_shear(&self) -> f64 {
        let m = self.inner.as_medium();
        let mu = match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_mu_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        };
        let rho = m.density(0, 0, 0);
        if rho > 0.0 {
            (mu / rho).sqrt()
        } else {
            0.0
        }
    }

    /// First Lamé parameter λ (Pa).
    ///
    /// Stored on the homogeneous medium directly. For fluid media this
    /// equals the bulk modulus ``K = ρ · c_p²``.
    #[getter]
    fn lame_lambda(&self) -> f64 {
        match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_lambda_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        }
    }

    /// Second Lamé parameter μ (shear modulus, Pa).
    ///
    /// Zero for fluid media; positive for elastic solids supporting shear waves.
    #[getter]
    fn lame_mu(&self) -> f64 {
        match &self.inner {
            MediumInner::Homogeneous(h) => h.lame_mu_value(),
            MediumInner::Heterogeneous(_) => 0.0,
        }
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
    /// Time signal matrix for grid sources (pressure), shape `[num_sources, time_steps]`
    signal: Option<Array2<f64>>,
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
    /// Per-axis 1-D velocity-signal time series for the elastic
    /// velocity-source path (Phase A.3 of ADR 007). Each entry is `Some`
    /// when the corresponding component is to be driven; `None` otherwise.
    /// The `mask` field above carries the `u_mask` for this source path.
    elastic_ux_signal_1d: Option<ndarray::Array1<f64>>,
    elastic_uy_signal_1d: Option<ndarray::Array1<f64>>,
    elastic_uz_signal_1d: Option<ndarray::Array1<f64>>,
}

fn pressure_signal_to_matrix(signal: &Bound<'_, PyAny>) -> PyResult<Array2<f64>> {
    if let Ok(signal_1d) = signal.extract::<PyReadonlyArray1<f64>>() {
        let signal_arr = signal_1d.as_array();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        return Ok(signal_arr.insert_axis(Axis(0)).to_owned());
    }

    if let Ok(signal_2d) = signal.extract::<PyReadonlyArray2<f64>>() {
        let signal_arr = signal_2d.as_array().to_owned();
        if signal_arr.is_empty() {
            return Err(PyValueError::new_err("Signal must not be empty"));
        }
        return Ok(signal_arr);
    }

    Err(PyValueError::new_err(
        "Signal must be a 1D or 2D ndarray of float64 values",
    ))
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create a grid source from a spatial mask and time signal.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray
    ///     3D spatial mask (same shape as grid)
    /// signal : ndarray
    ///     1D time signal [Pa] or 2D matrix `[num_sources, time_steps]`
    ///     For multi-row pressure sources, rows must follow MATLAB / Fortran-
    ///     order active-point enumeration to match k-wave-python.
    /// frequency : float
    ///     Source frequency [Hz]
    /// Create a grid source from a spatial mask and time signal.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray
    ///     3D spatial mask (same shape as grid)
    /// signal : ndarray
    ///     1D time signal [Pa] or 2D matrix `[num_sources, time_steps]`
    /// frequency : float
    ///     Source frequency [Hz]
    /// mode : str, optional
    ///     Source injection mode: "additive" (default), "additive_no_correction", or "dirichlet"
    #[staticmethod]
    #[pyo3(signature = (mask, signal, frequency, mode=None))]
    fn from_mask(
        mask: PyReadonlyArray3<f64>,
        signal: &Bound<'_, PyAny>,
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

        let signal_arr = pressure_signal_to_matrix(signal)?;

        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            Some("additive") | None => "additive".to_string(),
            Some(other) => return Err(PyValueError::new_err(format!(
                "Invalid source mode '{}'. Use 'additive', 'additive_no_correction', or 'dirichlet'",
                other
            ))),
        };

        let num_sources = mask_arr.iter().filter(|&&v| v != 0.0).count();
        if num_sources == 0 {
            return Err(PyValueError::new_err(
                "Source mask contains no active points",
            ));
        }

        let n_signal_rows = signal_arr.shape()[0];
        if n_signal_rows != 1 && n_signal_rows != num_sources {
            return Err(PyValueError::new_err(format!(
                "Signal rows must be 1 or match active source points: got {}, expected 1 or {}",
                n_signal_rows, num_sources
            )));
        }

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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create an initial pressure (initial value problem) source.
    ///
    /// Equivalent to k-Wave's `source.p0`.
    ///
    /// Parameters
    /// ----------
    /// p0 : ndarray
    ///     2D or 3D initial pressure distribution [Pa]. A 2D field is lifted
    ///     to a single-slice 3D volume with `nz=1` to match the solver layout.
    #[staticmethod]
    fn from_initial_pressure(p0: &Bound<'_, PyAny>) -> PyResult<Self> {
        let p0_arr: Array3<f64> = if let Ok(p0_3d) = p0.extract::<PyReadonlyArray3<f64>>() {
            p0_3d.as_array().to_owned()
        } else if let Ok(p0_2d) = p0.extract::<PyReadonlyArray2<f64>>() {
            p0_2d.as_array().insert_axis(Axis(2)).to_owned()
        } else {
            return Err(PyValueError::new_err(
                "Initial pressure must be a 2D or 3D ndarray of float64 values",
            ));
        };
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create a particle-velocity source mask for the **elastic** solver.
    ///
    /// Equivalent to k-Wave's ``source.u_mask`` / ``source.ux`` /
    /// ``source.uy`` / ``source.uz`` inputs to ``pstdElastic2D`` /
    /// ``pstdElastic3D``. At each time step, the integrator's post-step
    /// velocity field is **assigned** at every grid point inside ``mask``
    /// with the supplied component signal sample for that step (Dirichlet
    /// override semantics — matches k-Wave's default for velocity sources
    /// in pstdElastic).
    ///
    /// Phase A.3 of ADR 007. Signals are 1-D ndarrays (broadcast across
    /// all mask points); per-point signal matrices ship in Phase A.4.
    ///
    /// Parameters
    /// ----------
    /// mask : ndarray (3D bool)
    ///     Boolean grid mask marking source-active points.
    /// ux : ndarray (1D float64), optional
    ///     Time signal for vx at each step. ``None`` disables vx injection.
    /// uy : ndarray (1D float64), optional
    ///     Time signal for vy at each step.
    /// uz : ndarray (1D float64), optional
    ///     Time signal for vz at each step.
    /// mode : {"additive", "dirichlet"}, default "additive"
    ///     Injection mode. ``"additive"`` adds the signal to the
    ///     integrator's post-step velocity (matches MATLAB k-Wave and
    ///     KWave.jl's elastic-solver defaults). ``"dirichlet"``
    ///     overwrites velocity at masked points with the signal sample.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``mask`` has no active points, has wrong dim, all three
    ///     signals are ``None``, or ``mode`` is not one of
    ///     ``"additive"``/``"dirichlet"``.
    ///
    /// Examples
    /// --------
    /// >>> # Additive plane-wave ux source (k-Wave default)
    /// >>> u_mask = np.zeros((Nx, Ny, Nz), dtype=bool)
    /// >>> u_mask[35, :, :] = True
    /// >>> ux_sig = 1e-6 * np.sin(2*np.pi*1e6*np.arange(Nt)*dt)
    /// >>> src = pkw.Source.from_elastic_velocity_source(
    /// ...     u_mask, ux=ux_sig, mode="additive")
    #[staticmethod]
    #[pyo3(signature = (mask, ux=None, uy=None, uz=None, mode=None))]
    fn from_elastic_velocity_source(
        mask: PyReadonlyArray3<bool>,
        ux: Option<PyReadonlyArray1<f64>>,
        uy: Option<PyReadonlyArray1<f64>>,
        uz: Option<PyReadonlyArray1<f64>>,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        // Normalise the mode string up-front so we can fail fast with a
        // helpful error message before touching the signal arrays.
        let normalised_mode = match mode.unwrap_or("additive").to_ascii_lowercase().as_str() {
            "additive" => "additive",
            "dirichlet" => "dirichlet",
            other => {
                return Err(PyValueError::new_err(format!(
                    "mode must be 'additive' or 'dirichlet'; got '{}'",
                    other
                )));
            }
        };
        let mask_arr = mask.as_array();
        if mask_arr.ndim() != 3 {
            return Err(PyValueError::new_err("mask must be a 3D bool ndarray"));
        }
        let n_active = mask_arr.iter().filter(|&&v| v).count();
        if n_active == 0 {
            return Err(PyValueError::new_err(
                "mask must have at least one active point",
            ));
        }
        if ux.is_none() && uy.is_none() && uz.is_none() {
            return Err(PyValueError::new_err(
                "At least one of ux, uy, uz must be provided",
            ));
        }
        let convert = |opt: Option<PyReadonlyArray1<f64>>| -> Option<ndarray::Array1<f64>> {
            opt.map(|sig| sig.as_array().to_owned())
        };
        // Carry the bool mask through `mask: Option<Array3<f64>>` (the
        // existing carrier slot) by converting True/False to 1.0/0.0; the
        // dispatch reads non-zero as active.
        let mask_f64 = mask_arr.mapv(|b| if b { 1.0 } else { 0.0 });
        let amplitude = [&ux, &uy, &uz]
            .iter()
            .filter_map(|sig| {
                sig.as_ref()
                    .map(|s| s.as_array().iter().fold(0.0_f64, |a, &v| a.max(v.abs())))
            })
            .fold(0.0_f64, f64::max);
        Ok(Source {
            source_type: "elastic_velocity_source".to_string(),
            frequency: 0.0,
            amplitude,
            position: None,
            mask: Some(mask_f64),
            signal: None,
            // Carry mode through the existing string-typed source_mode slot;
            // the elastic-routing branch in Simulation::run reads it.
            source_mode: normalised_mode.to_string(),
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: None,
            elastic_ux_signal_1d: convert(ux),
            elastic_uy_signal_1d: convert(uy),
            elastic_uz_signal_1d: convert(uz),
        })
    }

    /// Create an initial-displacement source for the **elastic** solver.
    ///
    /// Sets the initial value of one displacement component (`ux`, `uy`, or
    /// `uz`) on the elastic wavefield while the other two components and
    /// all three velocity components are initialised to zero. The elastic
    /// solver then propagates this initial-value-problem under
    /// `ρ·∂²u/∂t² = (λ+μ)·∇(∇·u) + μ·∇²u`.
    ///
    /// This is the elastic analogue of `Source.from_initial_pressure`. It
    /// is required because the elastic field state vector is
    /// `(ux, uy, uz, vx, vy, vz)` rather than `p`, so a single
    /// initial-pressure scalar is not the natural input.
    ///
    /// Parameters
    /// ----------
    /// field : ndarray
    ///     2D or 3D initial displacement [m] for the chosen axis.
    ///     A 2D field is lifted to a single-slice 3D volume with ``nz=1``.
    /// axis : {"x", "y", "z"}, default "z"
    ///     Which displacement component to initialise. Other components
    ///     start at zero. Must be lower-case "x", "y", or "z".
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``field`` is not a 2-D or 3-D float64 ndarray, is all zeros,
    ///     or ``axis`` is not one of ``"x"``, ``"y"``, ``"z"``.
    ///
    /// Notes
    /// -----
    /// Currently only consumed by ``SolverType.Elastic``. Phase A.2 of
    /// ADR 007.
    #[staticmethod]
    #[pyo3(signature = (field, axis="z"))]
    fn from_initial_displacement(field: &Bound<'_, PyAny>, axis: &str) -> PyResult<Self> {
        let field_arr: Array3<f64> = if let Ok(f3) = field.extract::<PyReadonlyArray3<f64>>() {
            f3.as_array().to_owned()
        } else if let Ok(f2) = field.extract::<PyReadonlyArray2<f64>>() {
            f2.as_array().insert_axis(Axis(2)).to_owned()
        } else {
            return Err(PyValueError::new_err(
                "Initial displacement must be a 2D or 3D ndarray of float64 values",
            ));
        };
        if field_arr.iter().all(|&v| v == 0.0) {
            return Err(PyValueError::new_err("Initial displacement is all zeros"));
        }
        let axis_norm = match axis {
            "x" | "X" => "x",
            "y" | "Y" => "y",
            "z" | "Z" => "z",
            other => {
                return Err(PyValueError::new_err(format!(
                    "axis must be 'x', 'y', or 'z'; got '{}'",
                    other
                )))
            }
        };
        let amplitude = field_arr.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        // source_type encodes both the elastic-IVP role and the axis target.
        // The dispatch path inspects the prefix `elastic_u0_` to route into
        // run_elastic_impl, then reads the suffix to choose the component.
        let source_type = format!("elastic_u0_{}", axis_norm);
        Ok(Source {
            source_type,
            frequency: 0.0,
            amplitude,
            position: None,
            mask: None,
            signal: None,
            source_mode: "additive".to_string(),
            initial_pressure: Some(field_arr),
            velocity_signal: None,
            direction: None,
            kwave_array: None,
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
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
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
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

    /// Initial pressure field, if this source was constructed from `p0`.
    #[getter]
    fn initial_pressure<'py>(&self, py: Python<'py>) -> Option<Py<PyArray3<f64>>> {
        self.initial_pressure
            .as_ref()
            .map(|arr| PyArray3::from_owned_array(py, arr.clone()).into())
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
        let signal_matrix = signal_arr.clone().insert_axis(Axis(0)).to_owned();
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
            signal: Some(signal_matrix),
            source_mode,
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: Some(array.inner.clone()),
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
        })
    }

    /// Create a source from a KWaveArray with per-element driving signals.
    ///
    /// Parameters
    /// ----------
    /// array : KWaveArray
    /// signals : ndarray, shape (n_elements, n_time)
    ///     Per-element driving waveforms. Matches the output shape of
    ///     `kwave.utils.signals.create_cw_signals(t, f, amps, phases)`.
    /// frequency : float
    /// mode : str, optional
    ///
    /// Notes
    /// -----
    /// At run time, pykwavers pre-expands these into a per-active-cell signal
    /// matrix `s_cell[c, t] = Σ_i W_i[c] · s_i[t]` using each element's BLI
    /// weighted mask in MATLAB / Fortran-order active-cell enumeration,
    /// matching k-wave-python's `get_distributed_source_signal`.
    #[staticmethod]
    #[pyo3(signature = (array, signals, frequency, mode=None))]
    fn from_kwave_array_per_element(
        array: &KWaveArray,
        signals: PyReadonlyArray2<f64>,
        frequency: f64,
        mode: Option<&str>,
    ) -> PyResult<Self> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err("Frequency must be positive"));
        }
        let signal_matrix = signals.as_array().to_owned();
        if signal_matrix.is_empty() {
            return Err(PyValueError::new_err("Signals must not be empty"));
        }
        let n_elements = array.inner.num_elements();
        if signal_matrix.shape()[0] != n_elements {
            return Err(PyValueError::new_err(format!(
                "signals has {} rows but array has {} elements",
                signal_matrix.shape()[0],
                n_elements
            )));
        }
        let source_mode = match mode {
            Some("additive_no_correction") => "additive_no_correction".to_string(),
            Some("dirichlet") => "dirichlet".to_string(),
            _ => "additive".to_string(),
        };
        let amplitude = signal_matrix
            .iter()
            .fold(0.0_f64, |acc, v| acc.max(v.abs()));
        Ok(Source {
            source_type: "kwave_array_per_element".to_string(),
            frequency,
            amplitude,
            position: None,
            mask: None,
            signal: Some(signal_matrix),
            source_mode,
            initial_pressure: None,
            velocity_signal: None,
            direction: None,
            kwave_array: Some(array.inner.clone()),
            elastic_ux_signal_1d: None,
            elastic_uy_signal_1d: None,
            elastic_uz_signal_1d: None,
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
/// arrays from arc, disc, rectangular, and bowl elements. For 3-D disc
/// elements, `focus_position` selects the beam-axis normal; `None` keeps the
/// canonical x-y plane.
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
        self.inner.set_frequency(frequency);
    }

    /// Set the sound speed [m/s].
    fn set_sound_speed(&mut self, sound_speed: f64) {
        self.inner.set_sound_speed(sound_speed);
    }

    /// Add a disc-shaped element.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Element center [x, y, z] in meters
    /// diameter : float
    ///     Disc diameter [m]
    /// focus_position : tuple[float, float, float], optional
    ///     Optional point on the beam axis defining the disc normal
    #[pyo3(signature = (position, diameter, focus_position=None))]
    fn add_disc_element(
        &mut self,
        position: (f64, f64, f64),
        diameter: f64,
        focus_position: Option<(f64, f64, f64)>,
    ) -> PyResult<()> {
        if matches!(focus_position, Some(focus) if focus == position) {
            return Err(PyValueError::new_err(
                "focus_position must differ from position for a 3D disc",
            ));
        }
        self.inner
            .add_disc_element(position, diameter, focus_position);
        Ok(())
    }

    /// Generate a binary mask on a computational grid.
    fn get_array_binary_mask<'py>(
        &self,
        py: Python<'py>,
        grid: &Grid,
    ) -> PyResult<Py<PyArray3<bool>>> {
        Ok(PyArray3::from_owned_array(py, self.inner.get_array_binary_mask(&grid.inner)).into())
    }

    /// Generate a weighted mask on a computational grid.
    fn get_array_weighted_mask<'py>(
        &self,
        py: Python<'py>,
        grid: &Grid,
    ) -> PyResult<Py<PyArray3<f64>>> {
        Ok(PyArray3::from_owned_array(py, self.inner.get_array_weighted_mask(&grid.inner)).into())
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

    /// Add a rectangular element rotated about its center by intrinsic X-Y-Z
    /// Euler angles (degrees). Matches the upstream k-wave-python
    /// ``KWaveArray.add_rect_element`` rotation used by the
    /// ``at_linear_array_transducer`` example.
    ///
    /// Parameters
    /// ----------
    /// position : tuple[float, float, float]
    ///     Center position [x, y, z] in meters.
    /// dims : tuple[float, float, float]
    ///     Dimensions [width, height, length] in meters.
    /// euler_xyz_deg : tuple[float, float, float]
    ///     Intrinsic X-Y-Z Euler angles in degrees.
    fn add_rect_rot_element(
        &mut self,
        position: (f64, f64, f64),
        dims: (f64, f64, f64),
        euler_xyz_deg: (f64, f64, f64),
    ) {
        self.inner
            .add_rect_rot_element(position, dims.0, dims.1, dims.2, euler_xyz_deg);
    }

    /// Install a global translation + intrinsic X-Y-Z Euler rotation (degrees)
    /// applied to every element at rasterization time. Mirrors
    /// ``kWaveArray.set_array_position`` in k-wave-python.
    fn set_array_position(&mut self, translation: (f64, f64, f64), euler_xyz_deg: (f64, f64, f64)) {
        self.inner.set_array_position(translation, euler_xyz_deg);
    }

    /// Remove the global array transform if one was previously installed.
    fn clear_array_position(&mut self) {
        self.inner.clear_array_position();
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

    /// Add a single annular (spherical-ring) element.
    ///
    /// Mirrors k-wave-python's `kWaveArray.add_annular_element`: a spherical
    /// cap between `inner_diameter` and `outer_diameter` apertures on a bowl
    /// of curvature `radius`.
    fn add_annular_element(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        inner_diameter: f64,
        outer_diameter: f64,
    ) {
        self.inner
            .add_annular_element(position, radius, inner_diameter, outer_diameter);
    }

    /// Add a concentric annular array — one `ElementShape::Annulus` per
    /// `(inner_diameter, outer_diameter)` pair, all sharing `position` and
    /// `radius`. Mirrors `kWaveArray.add_annular_array`.
    fn add_annular_array(
        &mut self,
        position: (f64, f64, f64),
        radius: f64,
        diameters: Vec<(f64, f64)>,
    ) {
        self.inner.add_annular_array(position, radius, &diameters);
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
///
/// # `record_start_index` convention
///
/// Mirrors k-Wave's `sensor.record_start_index` (1-based).  The default `1`
/// records from the first time step.  Setting it to `N` (1 ≤ N ≤ Nt) starts
/// recording at step N and the output has `Nt - N + 1` columns.
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
    /// k-Wave 1-based start step for recording (default 1 = all steps).
    record_start_index: usize,
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
            record_start_index: 1,
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
            record_start_index: 1,
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
            record_start_index: 1,
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

    /// Set the first time step to record (k-Wave 1-based convention).
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     First time step at which to start recording (≥ 1).
    ///     Equivalent to k-Wave `sensor.record_start_index`.
    ///     Setting this to `N` yields output with `Nt - N + 1` time samples.
    ///
    /// Examples
    /// --------
    /// >>> # Record only the last 300 time steps of a 1000-step simulation
    /// >>> sensor.set_record_start_index(701)
    fn set_record_start_index(&mut self, index: usize) -> PyResult<()> {
        if index < 1 {
            return Err(PyValueError::new_err(
                "record_start_index must be ≥ 1 (k-Wave 1-based convention)",
            ));
        }
        self.record_start_index = index;
        Ok(())
    }

    /// First time step to record (k-Wave 1-based, default 1).
    #[getter]
    fn record_start_index(&self) -> usize {
        self.record_start_index
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
    kspace_correction: KSpaceCorrectionMode,
    compatibility_mode: CompatibilityMode,
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
    /// Axisymmetric (CylindricalAS) geometry: 2-D simulation in the (axial, radial) plane.
    /// Grid convention: nx=Nz_axial, ny=1, nz=Nr_radial. Only valid for PSTD and FDTD solvers.
    axisymmetric: bool,
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
            kspace_correction: KSpaceCorrectionMode::None,
            compatibility_mode: CompatibilityMode::Optimal,
            pml_size,
            pml_size_xyz: None,
            pml_inside: true,
            pml_alpha_xyz: None,
            enable_nonlinear: false,
            alpha_coeff: 0.0,
            alpha_power: 1.5,
            axisymmetric: false,
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

    /// Set the FDTD k-space correction mode.
    ///
    /// Parameters
    /// ----------
    /// mode : str
    ///     Either `"none"` or `"spectral"`.
    ///
    /// Examples
    /// --------
    /// >>> sim.set_kspace_correction("spectral")
    fn set_kspace_correction(&mut self, mode: &str) -> PyResult<()> {
        self.kspace_correction = match mode.to_ascii_lowercase().as_str() {
            "none" => KSpaceCorrectionMode::None,
            "spectral" => KSpaceCorrectionMode::Spectral,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported k-space correction mode: {other}"
                )))
            }
        };
        Ok(())
    }

    /// Get the current FDTD k-space correction mode.
    #[getter]
    fn kspace_correction(&self) -> String {
        match self.kspace_correction {
            KSpaceCorrectionMode::None => "none".to_string(),
            KSpaceCorrectionMode::Spectral => "spectral".to_string(),
        }
    }

    /// Set the PSTD compatibility mode.
    ///
    /// Parameters
    /// ----------
    /// mode : str
    ///     Either `"optimal"` or `"reference"`.
    ///
    /// Examples
    /// --------
    /// >>> sim.set_compatibility_mode("reference")
    fn set_compatibility_mode(&mut self, mode: &str) -> PyResult<()> {
        self.compatibility_mode = match mode.to_ascii_lowercase().as_str() {
            "optimal" => CompatibilityMode::Optimal,
            "reference" => CompatibilityMode::Reference,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported PSTD compatibility mode: {other}"
                )))
            }
        };
        Ok(())
    }

    /// Get the current PSTD compatibility mode.
    #[getter]
    fn compatibility_mode(&self) -> String {
        match self.compatibility_mode {
            CompatibilityMode::Optimal => "optimal".to_string(),
            CompatibilityMode::Reference => "reference".to_string(),
        }
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

    /// Enable axisymmetric (CylindricalAS) geometry for 2-D radial simulations.
    ///
    /// When enabled, the grid must have ny=1, nx=Nz_axial, nz=Nr_radial.
    /// Uses WSWA-FFT radial operators for PSTD, staggered cylindrical divergence for FDTD.
    fn set_axisymmetric(&mut self, enable: bool) {
        self.axisymmetric = enable;
    }

    /// Return whether axisymmetric geometry is enabled.
    #[getter]
    fn axisymmetric(&self) -> bool {
        self.axisymmetric
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
        // Elastic IVP axis tag carried out-of-band from the source-routing
        // layer to the elastic dispatch (Phase A.2.5 of ADR 007: per-axis
        // IVP routing). None for non-elastic runs; "x" / "y" / "z" when an
        // `elastic_u0_*` source has been supplied.
        let mut elastic_ivp_axis: Option<String> = None;
        // Optional elastic velocity-source bundle (Phase A.3 of ADR 007).
        // None unless `Source.from_elastic_velocity_source` was supplied;
        // when present, fed to `run_elastic_impl` and assigned to
        // `ElasticWaveConfig.velocity_source`.
        let mut elastic_velocity_source: Option<(
            ndarray::Array3<bool>,
            Option<ndarray::Array1<f64>>,
            Option<ndarray::Array1<f64>>,
            Option<ndarray::Array1<f64>>,
            String, // mode: "additive" or "dirichlet"
        )> = None;

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
                if signal.shape()[1] != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.shape()[1],
                        time_steps
                    )));
                }
                if signal.shape()[0] != 1 {
                    return Err(PyValueError::new_err(
                        "Source.from_kwave_array expects a single waveform row",
                    ));
                }
                // Rasterize the array geometry onto the simulation grid.
                // Use weighted mask normalised so sum(weights) = m_grid = surface_area/dx²,
                // matching k-wave-python's BLI distributed-source convention.
                let float_mask = arr.get_array_weighted_mask(&self.grid.inner);
                let num_active = float_mask.iter().filter(|&&v| v > 0.0).count();
                if num_active == 0 {
                    return Err(PyValueError::new_err(
                        "KWaveArray mask has no active grid points",
                    ));
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
                    p_signal: Some(signal.clone()),
                    p_mode,
                    ..GridSource::new_empty()
                };
                continue;
            }

            // Per-element-signals kwave_array source
            if src.source_type == "kwave_array_per_element" {
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
                if signal.shape()[1] != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.shape()[1],
                        time_steps
                    )));
                }
                if signal.shape()[0] != arr.num_elements() {
                    return Err(PyValueError::new_err(format!(
                        "Per-element signal rows {} != array elements {}",
                        signal.shape()[0],
                        arr.num_elements()
                    )));
                }
                let (mask, per_cell_signal) = arr
                    .build_per_element_source(&self.grid.inner, signal)
                    .map_err(PyValueError::new_err)?;
                let num_active = mask.iter().filter(|&&v| v != 0.0).count();
                if num_active == 0 {
                    return Err(PyValueError::new_err(
                        "KWaveArray per-element mask has no active grid points",
                    ));
                }
                let p_mode = match src.source_mode.as_str() {
                    "additive_no_correction" => {
                        kwavers::domain::source::grid_source::SourceMode::AdditiveNoCorrection
                    }
                    "dirichlet" => kwavers::domain::source::grid_source::SourceMode::Dirichlet,
                    _ => kwavers::domain::source::grid_source::SourceMode::Additive,
                };
                grid_source = GridSource {
                    p_mask: Some(mask),
                    p_signal: Some(per_cell_signal),
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

                if signal.shape()[1] != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.shape()[1],
                        time_steps
                    )));
                }

                let num_sources = mask.iter().filter(|v| **v != 0.0).count();
                if num_sources == 0 {
                    return Err(PyValueError::new_err(
                        "Source mask contains no active points",
                    ));
                }

                let n_signal_rows = signal.shape()[0];
                if n_signal_rows != 1 && n_signal_rows != num_sources {
                    return Err(PyValueError::new_err(format!(
                        "Signal rows must be 1 or match active source points: got {}, expected 1 or {}",
                        n_signal_rows, num_sources
                    )));
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
                    p_signal: Some(signal.clone()),
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

            // Handle elastic initial-displacement source.
            //
            // The Source.from_initial_displacement(...) static method tags
            // its source_type as "elastic_u0_{x|y|z}". The displacement field
            // is carried in `initial_pressure` (re-using the carrier slot
            // rather than expanding the Source struct surface). Routing into
            // run_elastic_impl is gated on `solver_type == SolverType::Elastic`
            // — see the dispatch match below. Here we just shuttle the field
            // through `grid_source.p0` so the elastic dispatch can read it.
            if src.source_type.starts_with("elastic_u0_") {
                if !matches!(self.solver_type, SolverType::Elastic) {
                    return Err(PyValueError::new_err(format!(
                        "Source.from_initial_displacement(..., axis='{}') requires \
                         SolverType.Elastic; got {:?}. Use Source.from_initial_pressure \
                         for fluid-acoustic IVP.",
                        &src.source_type[11..],
                        self.solver_type
                    )));
                }
                if let Some(ref u0) = src.initial_pressure {
                    grid_source.p0 = Some(u0.clone());
                }
                // Stash the axis suffix so the elastic dispatch can route
                // the IVP to the correct displacement component.
                elastic_ivp_axis = Some(src.source_type[11..].to_string());
                continue;
            }

            // Handle elastic velocity-source mask (Phase A.3 of ADR 007).
            // Tagged source_type = "elastic_velocity_source"; carries the
            // mask in `Source.mask` (as f64 with 0/1 values) and the
            // per-axis 1-D time signals in dedicated fields.
            if src.source_type == "elastic_velocity_source" {
                if !matches!(
                    self.solver_type,
                    SolverType::Elastic | SolverType::ElasticPSTD
                ) {
                    return Err(PyValueError::new_err(format!(
                        "Source.from_elastic_velocity_source requires \
                         SolverType.Elastic or SolverType.ElasticPSTD; got {:?}.",
                        self.solver_type
                    )));
                }
                let mask_f64 = src
                    .mask
                    .as_ref()
                    .ok_or_else(|| PyValueError::new_err("elastic velocity-source mask missing"))?;
                // Convert f64 mask → bool mask for the kwavers config layer.
                let mask_bool = mask_f64.mapv(|v| v != 0.0);
                elastic_velocity_source = Some((
                    mask_bool,
                    src.elastic_ux_signal_1d.clone(),
                    src.elastic_uy_signal_1d.clone(),
                    src.elastic_uz_signal_1d.clone(),
                    src.source_mode.clone(),
                ));
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
        let kspace_correction = self.kspace_correction.clone();
        let compatibility_mode = self.compatibility_mode;
        let enable_nonlinear = self.enable_nonlinear;
        let alpha_coeff = self.alpha_coeff;
        let alpha_power = self.alpha_power;
        let axisymmetric = self.axisymmetric;

        // Collect recording modes from sensor (empty vec if no sensor or no modes set)
        let sensor_record_modes: Vec<String> = sensor_opt
            .as_ref()
            .map(|s| s.record_modes.clone())
            .unwrap_or_default();
        // k-Wave 1-based start index (default 1 = record all steps).
        let sensor_record_start_index: usize = sensor_opt
            .as_ref()
            .map(|s| s.record_start_index)
            .unwrap_or(1);

        let run_result = py
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
                    kspace_correction,
                    enable_nonlinear,
                    axisymmetric,
                    &sensor_record_modes,
                    sensor_record_start_index,
                ),
                SolverType::PSTD | SolverType::Hybrid => Self::run_pstd_impl(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    compatibility_mode,
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
                    axisymmetric,
                    &sensor_record_modes,
                    sensor_record_start_index,
                ),
                SolverType::PstdGpu => Self::run_gpu_pstd_or_cpu_fallback(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    compatibility_mode,
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
                    axisymmetric,
                    &sensor_record_modes,
                    sensor_record_start_index,
                ),
                SolverType::Elastic => Self::run_elastic_impl(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    grid_source,
                    sensor_opt.as_ref(),
                    pml_size,
                    pml_inside,
                    elastic_ivp_axis.as_deref(),
                    elastic_velocity_source,
                ),
                SolverType::ElasticPSTD => Self::run_elastic_pstd_impl(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    sensor_opt.as_ref(),
                    elastic_velocity_source,
                ),
            })
            .map_err(kwavers_error_to_py)?;

        Self::simulation_run_result_to_py(py, run_result, shape, time_steps, dt_actual)
    }

    /// Run PSTD for `checkpoint_steps` steps and save state to `checkpoint_path`.
    ///
    /// Resumes the simulation by calling `run_from_checkpoint` with the same
    /// parameters.  Only supports `SolverType.PSTD`.
    ///
    /// Parameters
    /// ----------
    /// time_steps : int
    ///     Total simulation time steps (used to size the sensor recorder).
    /// checkpoint_steps : int
    ///     Number of steps to run before saving the checkpoint (`≤ time_steps`).
    /// checkpoint_path : str
    ///     File path for the checkpoint.  Must not already exist.
    /// dt : float, optional
    ///     Time step size.  Defaults to CFL-derived value.
    #[pyo3(signature = (time_steps, checkpoint_steps, checkpoint_path, dt=None))]
    fn run_to_checkpoint(
        &self,
        py: Python<'_>,
        time_steps: usize,
        checkpoint_steps: usize,
        checkpoint_path: String,
        dt: Option<f64>,
    ) -> PyResult<()> {
        if !matches!(self.solver_type, SolverType::PSTD) {
            return Err(PyValueError::new_err(
                "run_to_checkpoint only supports SolverType.PSTD",
            ));
        }
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let dt_actual = dt.unwrap_or_else(|| 0.3 * dx_min / (c_max * 3.0_f64.sqrt()));

        let (grid_source, dynamic_sources) = self.build_sources(time_steps, dt_actual, c_max)?;

        let grid_clone = self.grid.inner.clone();
        let medium_clone = self.medium.inner.clone();
        let sensor_opt = self.sensor.clone();
        let transducer_sensor_opt = self.transducer_sensor.clone();
        let pml_size = self.pml_size;
        let pml_size_xyz = self.pml_size_xyz;
        let pml_inside = self.pml_inside;
        let pml_alpha_xyz = self.pml_alpha_xyz;
        let compatibility_mode = self.compatibility_mode;
        let enable_nonlinear = self.enable_nonlinear;
        let alpha_coeff = self.alpha_coeff;
        let alpha_power = self.alpha_power;
        let path = std::path::PathBuf::from(checkpoint_path);

        py.detach(move || {
            Self::run_pstd_to_checkpoint(
                &grid_clone,
                &medium_clone,
                time_steps,
                checkpoint_steps,
                dt_actual,
                compatibility_mode,
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
                &path,
            )
        })
        .map_err(kwavers_error_to_py)
    }

    /// Resume a PSTD simulation from a checkpoint and return sensor data.
    ///
    /// Creates a fresh solver with identical configuration, restores the field
    /// state from `checkpoint_path`, runs the remaining steps to completion, and
    /// returns the full sensor time series.  The checkpoint file is deleted after
    /// a successful restore.
    ///
    /// Parameters
    /// ----------
    /// time_steps : int
    ///     Total simulation time steps (must match the value used in `run_to_checkpoint`).
    /// checkpoint_path : str
    ///     File path written by a prior `run_to_checkpoint` call.
    /// dt : float, optional
    ///     Time step size (must match the value used in `run_to_checkpoint`).
    #[pyo3(signature = (time_steps, checkpoint_path, dt=None))]
    fn run_from_checkpoint(
        &self,
        py: Python<'_>,
        time_steps: usize,
        checkpoint_path: String,
        dt: Option<f64>,
    ) -> PyResult<SimulationResult> {
        if !matches!(self.solver_type, SolverType::PSTD) {
            return Err(PyValueError::new_err(
                "run_from_checkpoint only supports SolverType.PSTD",
            ));
        }
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let dt_actual = dt.unwrap_or_else(|| 0.3 * dx_min / (c_max * 3.0_f64.sqrt()));

        let (grid_source, dynamic_sources) = self.build_sources(time_steps, dt_actual, c_max)?;

        let grid_clone = self.grid.inner.clone();
        let medium_clone = self.medium.inner.clone();
        let sensor_opt = self.sensor.clone();
        let transducer_sensor_opt = self.transducer_sensor.clone();
        let pml_size = self.pml_size;
        let pml_size_xyz = self.pml_size_xyz;
        let pml_inside = self.pml_inside;
        let pml_alpha_xyz = self.pml_alpha_xyz;
        let compatibility_mode = self.compatibility_mode;
        let enable_nonlinear = self.enable_nonlinear;
        let alpha_coeff = self.alpha_coeff;
        let alpha_power = self.alpha_power;
        let path = std::path::PathBuf::from(checkpoint_path);
        // Collect recording modes from sensor — same as the regular PSTD path.
        let sensor_record_modes: Vec<String> = sensor_opt
            .as_ref()
            .map(|s| s.record_modes.clone())
            .unwrap_or_default();
        let checkpoint = kwavers::solver::forward::pstd::checkpoint::PSTDCheckpoint::load(&path)
            .map_err(kwavers_error_to_py)?;
        checkpoint
            .validate_restore_contract(
                self.grid.inner.nx,
                self.grid.inner.ny,
                self.grid.inner.nz,
                time_steps,
                dt_actual,
            )
            .map_err(kwavers_error_to_py)?;
        let remaining_steps = time_steps
            .checked_sub(checkpoint.time_step_index)
            .ok_or_else(|| {
                kwavers_error_to_py(KwaversError::InvalidInput(format!(
                    "checkpoint time_step_index {} exceeds solver total_steps {}",
                    checkpoint.time_step_index, time_steps
                )))
            })?;
        let shape = (self.grid.inner.nx, self.grid.inner.ny, self.grid.inner.nz);

        let run_result = py
            .detach(move || {
                Self::run_pstd_from_checkpoint_loaded(
                    &grid_clone,
                    &medium_clone,
                    time_steps,
                    dt_actual,
                    compatibility_mode,
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
                    checkpoint,
                    remaining_steps,
                    &path,
                    &sensor_record_modes,
                )
            })
            .map_err(kwavers_error_to_py)?;

        Self::simulation_run_result_to_py(py, run_result, shape, time_steps, dt_actual)
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
    /// Build `GridSource` and dynamic source list from `self.sources` and `self.transducers`.
    ///
    /// Extracted from `run` to allow checkpoint methods to share source setup.
    /// `c_max` is the maximum sound speed of the medium (used for plane-wave wavelength).
    fn build_sources(
        &self,
        time_steps: usize,
        dt_actual: f64,
        c_max: f64,
    ) -> PyResult<(GridSource, Vec<Box<dyn KwaversSource>>)> {
        let mut grid_source = GridSource::new_empty();
        let mut dynamic_sources: Vec<Box<dyn KwaversSource>> = Vec::new();
        let mut has_mask_source = false;

        for src in &self.sources {
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
                if signal.shape()[1] != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.shape()[1],
                        time_steps
                    )));
                }
                if signal.shape()[0] != 1 {
                    return Err(PyValueError::new_err(
                        "Source.from_kwave_array expects a single waveform row",
                    ));
                }
                let float_mask = arr.get_array_weighted_mask(&self.grid.inner);
                let num_active = float_mask.iter().filter(|&&v| v > 0.0).count();
                if num_active == 0 {
                    return Err(PyValueError::new_err(
                        "KWaveArray mask has no active grid points",
                    ));
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
                    p_signal: Some(signal.clone()),
                    p_mode,
                    ..GridSource::new_empty()
                };
                continue;
            }

            if src.source_type == "kwave_array_per_element" {
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
                if signal.shape()[1] != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.shape()[1],
                        time_steps
                    )));
                }
                if signal.shape()[0] != arr.num_elements() {
                    return Err(PyValueError::new_err(format!(
                        "Per-element signal rows {} != array elements {}",
                        signal.shape()[0],
                        arr.num_elements()
                    )));
                }
                let (mask, per_cell_signal) = arr
                    .build_per_element_source(&self.grid.inner, signal)
                    .map_err(PyValueError::new_err)?;
                let num_active = mask.iter().filter(|&&v| v != 0.0).count();
                if num_active == 0 {
                    return Err(PyValueError::new_err(
                        "KWaveArray per-element mask has no active grid points",
                    ));
                }
                let p_mode = match src.source_mode.as_str() {
                    "additive_no_correction" => {
                        kwavers::domain::source::grid_source::SourceMode::AdditiveNoCorrection
                    }
                    "dirichlet" => kwavers::domain::source::grid_source::SourceMode::Dirichlet,
                    _ => kwavers::domain::source::grid_source::SourceMode::Additive,
                };
                grid_source = GridSource {
                    p_mask: Some(mask),
                    p_signal: Some(per_cell_signal),
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
                if signal.shape()[1] != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.shape()[1],
                        time_steps
                    )));
                }
                let num_sources = mask.iter().filter(|v| **v != 0.0).count();
                if num_sources == 0 {
                    return Err(PyValueError::new_err(
                        "Source mask contains no active points",
                    ));
                }
                let n_signal_rows = signal.shape()[0];
                if n_signal_rows != 1 && n_signal_rows != num_sources {
                    return Err(PyValueError::new_err(format!(
                        "Signal rows must be 1 or match active source points: got {}, expected 1 or {}",
                        n_signal_rows, num_sources
                    )));
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
                    p_signal: Some(signal.clone()),
                    p_mode,
                    ..GridSource::new_empty()
                };
                continue;
            }

            if src.source_type == "p0" {
                if let Some(ref p0) = src.initial_pressure {
                    grid_source.p0 = Some(p0.clone());
                }
                continue;
            }

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

            let freq = src.frequency;
            let amp = src.amplitude;
            let signal = SineSignal::new(freq, amp);
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
                let pos_arr = src.position.unwrap_or([0.0, 0.0, 0.0]);
                let px = pos_arr[0];
                let py_coord = pos_arr[1];
                let pz = pos_arr[2];
                let dx = self.grid.inner.dx;
                let dy = self.grid.inner.dy;
                let dz = self.grid.inner.dz;
                Box::new(FunctionSource::new(
                    move |x, y, z, _t| {
                        if (x - px).abs() < dx * 0.5
                            && (y - py_coord).abs() < dy * 0.5
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

        Ok((grid_source, dynamic_sources))
    }

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

    /// Map k-Wave-style `sensor.record` strings to a [`SensorRecordSpec`].
    ///
    /// ## Theorem (completeness)
    /// Every string accepted by k-Wave's `sensor.record` cell array is mapped
    /// to the corresponding `SensorRecordField` variant. Unrecognised strings
    /// are silently ignored (k-Wave behaviour). `"p"` is always implicitly
    /// included — it maps to `SensorRecordField::Pressure` which is the default.
    ///
    /// ## Mapping (k-Wave string → SensorRecordField)
    /// | k-Wave string       | Variant                       |
    /// |---------------------|-------------------------------|
    /// | `p`                 | `Pressure`                    |
    /// | `p_max`             | `PressureMax`                 |
    /// | `p_min`             | `PressureMin`                 |
    /// | `p_rms`             | `PressureRms`                 |
    /// | `p_final`           | `PressureFinal`               |
    /// | `all`               | all four pressure stats above  |
    /// | `ux`                | `VelocityX`                   |
    /// | `uy`                | `VelocityY`                   |
    /// | `uz`                | `VelocityZ`                   |
    /// | `ux_max`            | `VelocityMaxX`                |
    /// | `uy_max`            | `VelocityMaxY`                |
    /// | `uz_max`            | `VelocityMaxZ`                |
    /// | `ux_min`            | `VelocityMinX`                |
    /// | `uy_min`            | `VelocityMinY`                |
    /// | `uz_min`            | `VelocityMinZ`                |
    /// | `ux_rms`            | `VelocityRmsX`                |
    /// | `uy_rms`            | `VelocityRmsY`                |
    /// | `uz_rms`            | `VelocityRmsZ`                |
    /// | `ux_non_staggered`  | `VelocityNonStaggeredX`       |
    /// | `uy_non_staggered`  | `VelocityNonStaggeredY`       |
    /// | `uz_non_staggered`  | `VelocityNonStaggeredZ`       |
    /// | `Ix`                | `IntensityX`                 |
    /// | `Iy`                | `IntensityY`                 |
    /// | `Iz`                | `IntensityZ`                 |
    /// | `I_avg_x`           | `IntensityAvgX`              |
    /// | `I_avg_y`           | `IntensityAvgY`              |
    /// | `I_avg_z`           | `IntensityAvgZ`              |
    fn record_modes_to_spec(modes: &[String]) -> SensorRecordSpec {
        let mut fields = vec![SensorRecordField::Pressure];
        for s in modes {
            match s.as_str() {
                "p" => {} // already included above
                "p_max" => fields.push(SensorRecordField::PressureMax),
                "p_min" => fields.push(SensorRecordField::PressureMin),
                "p_rms" => fields.push(SensorRecordField::PressureRms),
                "p_final" => fields.push(SensorRecordField::PressureFinal),
                "all" => {
                    fields.push(SensorRecordField::PressureMax);
                    fields.push(SensorRecordField::PressureMin);
                    fields.push(SensorRecordField::PressureRms);
                    fields.push(SensorRecordField::PressureFinal);
                }
                "ux" => fields.push(SensorRecordField::VelocityX),
                "uy" => fields.push(SensorRecordField::VelocityY),
                "uz" => fields.push(SensorRecordField::VelocityZ),
                "ux_max" => fields.push(SensorRecordField::VelocityMaxX),
                "uy_max" => fields.push(SensorRecordField::VelocityMaxY),
                "uz_max" => fields.push(SensorRecordField::VelocityMaxZ),
                "ux_min" => fields.push(SensorRecordField::VelocityMinX),
                "uy_min" => fields.push(SensorRecordField::VelocityMinY),
                "uz_min" => fields.push(SensorRecordField::VelocityMinZ),
                "ux_rms" => fields.push(SensorRecordField::VelocityRmsX),
                "uy_rms" => fields.push(SensorRecordField::VelocityRmsY),
                "uz_rms" => fields.push(SensorRecordField::VelocityRmsZ),
                "ux_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredX),
                "uy_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredY),
                "uz_non_staggered" => fields.push(SensorRecordField::VelocityNonStaggeredZ),
                "Ix" => fields.push(SensorRecordField::IntensityX),
                "Iy" => fields.push(SensorRecordField::IntensityY),
                "Iz" => fields.push(SensorRecordField::IntensityZ),
                "I_avg_x" => fields.push(SensorRecordField::IntensityAvgX),
                "I_avg_y" => fields.push(SensorRecordField::IntensityAvgY),
                "I_avg_z" => fields.push(SensorRecordField::IntensityAvgZ),
                _ => {} // unrecognised strings silently ignored (k-Wave convention)
            }
        }
        SensorRecordSpec::from_fields(&fields)
    }

    /// Convert k-Wave-style record strings to RecordingMode variants (pressure only).
    ///
    /// Retained for FDTD path compatibility; PSTD uses `record_modes_to_spec`.
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

    /// Trim the recorder buffer to `Nt` columns aligned with k-Wave's
    /// time-axis convention and apply `record_start_index`.
    ///
    /// # Convention
    ///
    /// Two cases depending on source type:
    ///
    /// **IVP (p0) sources** — `run_orchestrated` records `p(0) = p0` once
    /// before the time loop, then `step_forward` records `p(i·dt)` after each
    /// of `Nt` steps, producing `Nt + 1` columns: `[p0, p(dt), …, p(Nt·dt)]`.
    /// k-Wave records `p0` at column 0 (it overrides the first step with p0),
    /// so the correct output is `[p0, p(dt), …, p((Nt−1)·dt)]` — the last
    /// column is dropped.  The buffer has `Nt + 1` columns; the slice
    /// `[skip..Nt]` removes the redundant final sample.
    ///
    /// **Time-varying sources** — `run_orchestrated` does NOT record before
    /// the loop (no IVP), so only the `Nt` post-step recordings fill the
    /// buffer: `[p(dt), p(2dt), …, p(Nt·dt)]`.  k-Wave records the same
    /// range, so `[skip..]` with `Nt` columns already gives correct alignment.
    /// The `Nt + 1` capacity buffer has one trailing zero column that never
    /// gets filled; `ncols() > time_steps` is still true, so the same
    /// `[skip..time_steps]` slice (which stops before that column) is applied.
    ///
    /// Both cases use the same code path: `ncols() > time_steps → [skip..Nt]`.
    ///
    /// `record_start_index` is the 1-based k-Wave start index. A value of `1`
    /// (default) emits all `Nt` time samples; a value of `k` emits the last
    /// `Nt − k + 1` samples starting from `p((k − 1)·dt)`.
    ///
    /// Output shape: `(n_sensors, Nt − record_start_index + 1)`.
    fn trim_initial_recorder_sample(
        recorded_data: ndarray::Array2<f64>,
        time_steps: usize,
        record_start_index: usize,
    ) -> ndarray::Array2<f64> {
        let start = record_start_index.max(1).min(time_steps);
        let skip = start.saturating_sub(1);
        if recorded_data.ncols() > time_steps {
            // Buffer has Nt+1 columns: keep [p(0), …, p((Nt−1)·dt)] then apply
            // record_start_index. Equivalent to slice [skip..time_steps].
            recorded_data
                .slice(ndarray::s![.., skip..time_steps])
                .to_owned()
        } else {
            // Buffer already has Nt columns (e.g. velocity buffers populated
            // only inside step_forward); just apply the start offset.
            recorded_data.slice(ndarray::s![.., skip..]).to_owned()
        }
    }

    /// Borrowed-view variant of [`trim_initial_recorder_sample`].
    ///
    /// Avoids cloning the full `Nt + 1` recorder matrix before applying the
    /// time-axis alignment and `record_start_index` slicing. See
    /// [`trim_initial_recorder_sample`] for the convention.
    fn trim_initial_recorder_view(
        recorded_data: ArrayView2<'_, f64>,
        time_steps: usize,
        record_start_index: usize,
    ) -> ndarray::Array2<f64> {
        let start = record_start_index.max(1).min(time_steps);
        let skip = start.saturating_sub(1);
        if recorded_data.ncols() > time_steps {
            recorded_data
                .slice(ndarray::s![.., skip..time_steps])
                .to_owned()
        } else {
            recorded_data.slice(ndarray::s![.., skip..]).to_owned()
        }
    }

    /// Convert a [`SimulationRunResult`] into a [`SimulationResult`] exposed to Python.
    ///
    /// Velocity arrays and statistics are included only when they were populated
    /// by the run (i.e. when the corresponding record modes were requested).
    fn simulation_run_result_to_py(
        py: Python<'_>,
        result: SimulationRunResult,
        shape: (usize, usize, usize),
        time_steps: usize,
        dt_actual: f64,
    ) -> PyResult<SimulationResult> {
        let SimulationRunResult {
            sensor_data,
            stats,
            ux_data,
            uy_data,
            uz_data,
            ix_data,
            iy_data,
            iz_data,
            i_avg_x,
            i_avg_y,
            i_avg_z,
            velocity_stats,
            full_grid_stats,
        } = result;

        // Full-grid pressure-statistics arrays (cavitation-kernel use).
        let (p_max_3d, p_min_3d, p_rms_3d, p_final_3d) =
            if let Some((mx, mn, rm, fn_)) = full_grid_stats {
                (
                    Some(PyArray3::from_owned_array(py, mx).into()),
                    Some(PyArray3::from_owned_array(py, mn).into()),
                    Some(PyArray3::from_owned_array(py, rm).into()),
                    Some(PyArray3::from_owned_array(py, fn_).into()),
                )
            } else {
                (None, None, None, None)
            };

        // Pressure statistics → numpy arrays.
        let p_max = stats
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_max.clone()).into());
        let p_min = stats
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_min.clone()).into());
        let p_rms = stats
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_rms.clone()).into());
        let p_final = stats
            .as_ref()
            .map(|s| PyArray1::from_owned_array(py, s.p_final.clone()).into());

        // Velocity time series → numpy arrays (None when not recorded).
        let ux = ux_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let uy = uy_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let uz = uz_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let ix = ix_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let iy = iy_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let iz = iz_data.map(|d| PyArray2::from_owned_array(py, d).into());
        let i_avg_x = i_avg_x.map(|d| PyArray1::from_owned_array(py, d).into());
        let i_avg_y = i_avg_y.map(|d| PyArray1::from_owned_array(py, d).into());
        let i_avg_z = i_avg_z.map(|d| PyArray1::from_owned_array(py, d).into());

        // Velocity statistics → numpy arrays (None when not recorded).
        let (ux_max, ux_min, ux_rms, uy_max, uy_min, uy_rms, uz_max, uz_min, uz_rms) =
            if let Some(vs) = velocity_stats {
                (
                    Some(PyArray1::from_owned_array(py, vs.ux_max).into()),
                    Some(PyArray1::from_owned_array(py, vs.ux_min).into()),
                    Some(PyArray1::from_owned_array(py, vs.ux_rms).into()),
                    Some(PyArray1::from_owned_array(py, vs.uy_max).into()),
                    Some(PyArray1::from_owned_array(py, vs.uy_min).into()),
                    Some(PyArray1::from_owned_array(py, vs.uy_rms).into()),
                    Some(PyArray1::from_owned_array(py, vs.uz_max).into()),
                    Some(PyArray1::from_owned_array(py, vs.uz_min).into()),
                    Some(PyArray1::from_owned_array(py, vs.uz_rms).into()),
                )
            } else {
                (None, None, None, None, None, None, None, None, None)
            };

        let time_arr = PyArray1::from_owned_array(
            py,
            Array1::from_iter((0..time_steps).map(|i| i as f64 * dt_actual)),
        )
        .into();

        let n_sensors = sensor_data.nrows();
        if n_sensors <= 1 {
            let sensor_1d = sensor_data.row(0).to_owned();
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
                p_max_field: p_max_3d
                    .as_ref()
                    .map(|p: &Py<PyArray3<f64>>| p.clone_ref(py)),
                p_min_field: p_min_3d
                    .as_ref()
                    .map(|p: &Py<PyArray3<f64>>| p.clone_ref(py)),
                p_rms_field: p_rms_3d
                    .as_ref()
                    .map(|p: &Py<PyArray3<f64>>| p.clone_ref(py)),
                p_final_field: p_final_3d
                    .as_ref()
                    .map(|p: &Py<PyArray3<f64>>| p.clone_ref(py)),
                ux,
                uy,
                uz,
                ix,
                iy,
                iz,
                i_avg_x,
                i_avg_y,
                i_avg_z,
                ux_max,
                ux_min,
                ux_rms,
                uy_max,
                uy_min,
                uy_rms,
                uz_max,
                uz_min,
                uz_rms,
            })
        } else {
            Ok(SimulationResult {
                sensor_data_1d: None,
                sensor_data_2d: Some(PyArray2::from_owned_array(py, sensor_data).into()),
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
                p_max_field: p_max_3d,
                p_min_field: p_min_3d,
                p_rms_field: p_rms_3d,
                p_final_field: p_final_3d,
                ux,
                uy,
                uz,
                ix,
                iy,
                iz,
                i_avg_x,
                i_avg_y,
                i_avg_z,
                ux_max,
                ux_min,
                ux_rms,
                uy_max,
                uy_min,
                uy_rms,
                uz_max,
                uz_min,
                uz_rms,
            })
        }
    }

    /// Return the minimum active axis length and admissible CPML thickness.
    ///
    /// # Theorem (Singleton-axis neutrality)
    /// A dimension with length `1` is a dummy embedding axis for lower-dimensional
    /// examples and must not constrain the absorbing boundary. The admissible
    /// CPML thickness depends only on axes with length `> 1`.
    ///
    /// # Proof sketch
    /// A singleton axis has no physical extent, so damping on that axis would
    /// introduce absorption not present in the reference lower-dimensional
    /// problem. Excluding singleton axes preserves the discrete operator on the
    /// active axes while keeping the same 3-D storage layout.
    fn cpml_thickness_limits(nx: usize, ny: usize, nz: usize) -> (usize, usize) {
        let mut min_dim = usize::MAX;
        for dim in [nx, ny, nz] {
            if dim > 1 {
                min_dim = min_dim.min(dim);
            }
        }
        let min_dim = if min_dim == usize::MAX { 1 } else { min_dim };
        let max_allowed = (min_dim.saturating_sub(2)) / 2;
        // Default matches k-Wave's fixed 20-cell PML (kspaceFirstOrder default).
        // min_dim/6 over-sized for large 1D grids (512/6=85), placing sensors inside the PML.
        let default_thickness = 20_usize.min(max_allowed).max(2);
        (default_thickness, max_allowed)
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
        kspace_correction: KSpaceCorrectionMode,
        enable_nonlinear: bool,
        axisymmetric: bool,
        record_modes: &[String],
        record_start_index: usize,
    ) -> KwaversResult<SimulationRunResult> {
        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);

        // For transducer sensors, override the recorder with element-ordered indices
        // so node recordings are grouped [elem0_nodes…, elem1_nodes…] matching k-Wave.
        let transducer_ordered_indices = transducer_sensor
            .map(|trans| Self::create_transducer_ordered_indices(grid, &trans.inner));

        let geometry = if axisymmetric {
            Geometry::CylindricalAS
        } else {
            Geometry::Cartesian3D
        };

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
            kspace_correction,
            sensor_mask: Some(sensor_mask.clone()),
            geometry,
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
        let (default_thickness, max_allowed) =
            Self::cpml_thickness_limits(grid.nx, grid.ny, grid.nz);
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
        let sensor_data =
            Self::trim_initial_recorder_sample(full_data, time_steps, record_start_index);

        // FDTD stepper does not call record_velocity_step; velocity fields are None.
        let full_grid_stats = extract_full_grid_stats(&solver.sensor_recorder);
        Ok(SimulationRunResult {
            sensor_data,
            stats,
            ux_data: None,
            uy_data: None,
            uz_data: None,
            ix_data: None,
            iy_data: None,
            iz_data: None,
            i_avg_x: None,
            i_avg_y: None,
            i_avg_z: None,
            velocity_stats: None,
            full_grid_stats,
        })
    }

    /// Build and configure a PSTD solver without running it.
    ///
    /// Handles grid padding for `pml_inside=false`, CPML configuration,
    /// absorption mode selection, sensor recorder setup, and dynamic source
    /// registration.  Returns the ready-to-run solver together with the
    /// (possibly padded) simulation grid and effective sensor mask so callers
    /// can vary only the execution strategy.
    #[allow(clippy::too_many_arguments)]
    fn prepare_pstd_solver(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
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
        axisymmetric: bool,
        record_modes: &[String],
    ) -> KwaversResult<(PSTDSolver, KwaversGrid, ndarray::Array3<bool>)> {
        let sensor_mask = Self::create_sensor_mask(grid, sensor, transducer_sensor);
        let transducer_ordered_indices = transducer_sensor
            .map(|trans| Self::create_transducer_ordered_indices(grid, &trans.inner));

        let (default_thickness, max_allowed) =
            Self::cpml_thickness_limits(grid.nx, grid.ny, grid.nz);
        let thickness = pml_size.unwrap_or(default_thickness).min(max_allowed);

        // pml_inside=false: extend the computational grid by `thickness` cells on each
        // side so the CPML absorbing boundary occupies cells outside the physical domain.
        // Source and sensor masks are embedded at [P..NX+P, P..NY+P, P..NZ+P] in the
        // padded grid; the CPML then applies to the outer P cells of the padded domain.
        // This matches k-Wave semantics for pml_inside=False.
        // Transducer sensors are not supported with pml_inside=false (indices are not remapped).
        let (sim_grid, grid_source, sensor_mask, effective_pml_inside) =
            if !pml_inside && thickness > 0 {
                if transducer_sensor.is_some() {
                    return Err(KwaversError::Validation(
                        kwavers::core::error::ValidationError::FieldValidation {
                            field: "pml_inside".to_string(),
                            value: "false".to_string(),
                            constraint: "pml_inside=false is not supported with transducer sensors"
                                .to_string(),
                        },
                    ));
                }
                let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
                let p = thickness;
                // Degenerate axes (extent 1) carry no wave propagation along that
                // direction, so a PML there serves no physical purpose and would
                // inflate field arrays by 1 + 2·p in a dimension that should
                // remain trivial. For example a quasi-1D problem (ny = nz = 1)
                // with p = 80 would otherwise allocate fields shaped
                // (nx + 160, 161, 161) — a ~360 MB scalar buffer for a 1-D grid
                // — which routinely OOM'd `examples/na_modelling_nonlinearity_compare.py`.
                let pad_x = nx > 1;
                let pad_y = !axisymmetric && ny > 1; // CylindricalAS already protects ny=1
                let pad_z_two_sided = !axisymmetric && nz > 1;
                let pad_z_one_sided = axisymmetric && nz > 1;

                let pnx = if pad_x { nx + 2 * p } else { nx };
                let pny = if pad_y { ny + 2 * p } else { ny };
                // CylindricalAS uses one-sided outer radial PML only (k-Wave pml.py semantics):
                //   pnz = nz + p  (physical axis at k=0, PML at k=nz..nz+p-1)
                // This places the WS-expansion axis at the left boundary of k=0, matching
                // k-Wave's axisymmetric convention. Two-sided would push the axis to k=p,
                // corrupting the whole-sample symmetric expansion's axis boundary condition.
                let pnz = if pad_z_two_sided {
                    nz + 2 * p
                } else if pad_z_one_sided {
                    nz + p
                } else {
                    nz
                };
                let padded_grid = KwaversGrid::new(pnx, pny, pnz, grid.dx, grid.dy, grid.dz)?;

                let px_embed = if pad_x { p } else { 0 };
                let py = if pad_y { p } else { 0 };
                // Physical cells embed at z=0 for AS (axis at k=0); z=p for Cartesian
                // when nz>1; z=0 for nz=1 (no padding).
                let pz_embed = if pad_z_two_sided { p } else { 0 };
                let p = px_embed; // alias retained for the embed slices below

                let embed = |arr: ndarray::Array3<f64>| -> ndarray::Array3<f64> {
                    let mut out = ndarray::Array3::<f64>::zeros((pnx, pny, pnz));
                    out.slice_mut(ndarray::s![p..nx + p, py..ny + py, pz_embed..nz + pz_embed])
                        .assign(&arr);
                    out
                };

                let mut padded_mask = ndarray::Array3::<bool>::from_elem((pnx, pny, pnz), false);
                padded_mask
                    .slice_mut(ndarray::s![p..nx + p, py..ny + py, pz_embed..nz + pz_embed])
                    .assign(&sensor_mask);

                let padded_source = GridSource {
                    p0: grid_source.p0.map(&embed),
                    u0: grid_source
                        .u0
                        .map(|(ux, uy, uz)| (embed(ux), embed(uy), embed(uz))),
                    p_mask: grid_source.p_mask.map(&embed),
                    p_signal: grid_source.p_signal,
                    p_mode: grid_source.p_mode,
                    u_mask: grid_source.u_mask.map(embed),
                    u_signal: grid_source.u_signal,
                    u_mode: grid_source.u_mode,
                };

                // For CylindricalAS with axial+radial PML but no y-padding, pml_inside=true
                // is set so the CPML covers the padded axial/radial boundaries as intended.
                (padded_grid, padded_source, padded_mask, true)
            } else {
                (grid.clone(), grid_source, sensor_mask, pml_inside)
            };

        let alpha_is_zero = pml_alpha_xyz
            .map(|(ax, ay, az)| ax == 0.0 && ay == 0.0 && az == 0.0)
            .unwrap_or(false);
        let boundary = if thickness > 0 && max_allowed > 0 && !alpha_is_zero {
            let mut cpml_config = if let Some((px, py, pz)) = pml_size_xyz {
                CPMLConfig::with_per_dimension_thickness(px, py, pz)
            } else {
                CPMLConfig::with_thickness(thickness)
            };
            if let Some((ax, ay, az)) = pml_alpha_xyz {
                cpml_config = cpml_config.with_alpha_xyz(ax, ay, az);
            }
            // For one-sided axisymmetric radial PML, the CPML left-z profile would absorb
            // physical axis cells k=0..p-1. Suppress inner z-sigma to keep the axis transparent.
            if axisymmetric && !pml_inside {
                cpml_config = cpml_config.with_radial_inner_z_transparent();
            }
            kwavers::solver::forward::pstd::config::BoundaryConfig::CPML(cpml_config)
        } else {
            kwavers::solver::forward::pstd::config::BoundaryConfig::None
        };

        let effective_alpha_db = if alpha_coeff_db > 0.0 {
            alpha_coeff_db
        } else {
            medium.as_medium().alpha_coefficient(0.0, 0.0, 0.0, grid)
        };

        let effective_alpha_power = {
            let y_medium = medium.as_medium().alpha_power(0.0, 0.0, 0.0, grid);
            if alpha_coeff_db <= 0.0 && y_medium > 0.0 && (y_medium - 1.0).abs() > 1e-12 {
                y_medium
            } else {
                alpha_power
            }
        };

        let absorption_mode = if effective_alpha_db > 0.0 {
            AbsorptionMode::PowerLaw {
                alpha_coeff: effective_alpha_db,
                alpha_power: effective_alpha_power,
            }
        } else {
            AbsorptionMode::Lossless
        };

        let geometry = if axisymmetric {
            Geometry::CylindricalAS
        } else {
            Geometry::Cartesian3D
        };

        let config = PSTDConfig {
            dt,
            nt: time_steps,
            compatibility_mode,
            sensor_mask: Some(sensor_mask.clone()),
            boundary,
            pml_inside: effective_pml_inside,
            absorption_mode,
            nonlinearity: enable_nonlinear,
            geometry,
            ..Default::default()
        };

        let mut solver =
            PSTDSolver::new(config, sim_grid.clone(), medium.as_medium(), grid_source)?;

        // Always use with_spec so velocity recording is available when requested.
        // with_spec allocates pressure stats and velocity buffers according to
        // what the spec requests; an empty record_modes list produces a
        // pressure-only spec (default, equivalent to the former with_modes path).
        let spec = Self::record_modes_to_spec(record_modes);
        let shape = (sim_grid.nx, sim_grid.ny, sim_grid.nz);
        if let Some(ordered) = transducer_ordered_indices {
            // Transducer override: element-ordered indices, no velocity stats.
            solver.sensor_recorder = SensorRecorder::from_ordered_indices(ordered, time_steps + 1)?;
        } else {
            solver.sensor_recorder =
                SensorRecorder::with_spec(Some(&sensor_mask), shape, time_steps + 1, spec)?;
        }

        for source in sources {
            SolverTrait::add_source(&mut solver, source)?;
        }

        Ok((solver, sim_grid, sensor_mask))
    }

    /// Run PSTD simulation (internal).
    ///
    /// Returns a [`SimulationRunResult`] containing pressure time series,
    /// pressure statistics, and — when velocity modes were requested — staggered
    /// velocity time series and per-component velocity statistics.
    #[allow(clippy::too_many_arguments)]
    fn run_pstd_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
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
        axisymmetric: bool,
        record_modes: &[String],
        record_start_index: usize,
    ) -> KwaversResult<SimulationRunResult> {
        let (mut solver, _sim_grid, _sensor_mask) = Self::prepare_pstd_solver(
            grid,
            medium,
            time_steps,
            dt,
            compatibility_mode,
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
            axisymmetric,
            record_modes,
        )?;

        solver.run_orchestrated(time_steps)?;

        // Extract pressure outputs.
        let stats = solver.sensor_recorder.extract_all_stats();
        let full_data = solver
            .sensor_recorder
            .recorded_pressure_view()
            .ok_or_else(|| {
                kwavers::core::error::KwaversError::Io(std::io::Error::other(
                    "No sensor data recorded",
                ))
            })?;
        let sensor_data =
            Self::trim_initial_recorder_view(full_data, time_steps, record_start_index);

        // Extract velocity time series (None when no velocity mode was requested).
        let ux_data = solver
            .sensor_recorder
            .recorded_ux_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let uy_data = solver
            .sensor_recorder
            .recorded_uy_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let uz_data = solver
            .sensor_recorder
            .recorded_uz_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let ix_data = solver
            .sensor_recorder
            .recorded_ix_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let iy_data = solver
            .sensor_recorder
            .recorded_iy_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let iz_data = solver
            .sensor_recorder
            .recorded_iz_view()
            .map(|d| Self::trim_initial_recorder_view(d, time_steps, record_start_index));
        let i_avg_x = solver.sensor_recorder.extract_i_avg_x();
        let i_avg_y = solver.sensor_recorder.extract_i_avg_y();
        let i_avg_z = solver.sensor_recorder.extract_i_avg_z();

        // Velocity statistics sampled at sensor positions.
        let velocity_stats = solver.sensor_recorder.extract_sampled_velocity_stats();
        let full_grid_stats = extract_full_grid_stats(&solver.sensor_recorder);

        Ok(SimulationRunResult {
            sensor_data,
            stats,
            ux_data,
            uy_data,
            uz_data,
            ix_data,
            iy_data,
            iz_data,
            i_avg_x,
            i_avg_y,
            i_avg_z,
            velocity_stats,
            full_grid_stats,
        })
    }

    /// Run PSTD for `checkpoint_steps` steps and save state to `checkpoint_path`.
    ///
    /// The checkpoint file can be resumed with [`run_from_checkpoint`].
    /// `total_steps` is the full simulation length and must match the value used
    /// on the resume call so the sensor recorder is allocated to the correct size.
    #[allow(clippy::too_many_arguments)]
    fn run_pstd_to_checkpoint(
        grid: &KwaversGrid,
        medium: &MediumInner,
        total_steps: usize,
        checkpoint_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
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
        checkpoint_path: &std::path::Path,
    ) -> KwaversResult<()> {
        let (mut solver, _sim_grid, _sensor_mask) = Self::prepare_pstd_solver(
            grid,
            medium,
            total_steps,
            dt,
            compatibility_mode,
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
            false, // axisymmetric: checkpoint functions default to Cartesian3D
            &[],
        )?;
        solver.run_to_checkpoint(checkpoint_steps, checkpoint_path)
    }

    /// Resume a checkpointed PSTD simulation from a preloaded checkpoint and return sensor data.
    ///
    /// `record_modes` mirrors the sensor's recording spec: if it includes velocity
    /// modes (`"ux"`, `"uy"`, `"uz"`) the recorder allocates the corresponding
    /// buffers and `run_from_checkpoint_loaded` populates them step-by-step.
    #[allow(clippy::too_many_arguments)]
    fn run_pstd_from_checkpoint_loaded(
        grid: &KwaversGrid,
        medium: &MediumInner,
        total_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
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
        checkpoint: kwavers::solver::forward::pstd::checkpoint::PSTDCheckpoint,
        remaining_steps: usize,
        checkpoint_path: &std::path::Path,
        record_modes: &[String],
    ) -> KwaversResult<SimulationRunResult> {
        let (mut solver, _sim_grid, _sensor_mask) = Self::prepare_pstd_solver(
            grid,
            medium,
            total_steps,
            dt,
            compatibility_mode,
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
            false, // axisymmetric: checkpoint functions default to Cartesian3D
            record_modes,
        )?;

        checkpoint.validate_restore_contract(grid.nx, grid.ny, grid.nz, total_steps, dt)?;

        let full_data = solver
            .run_from_checkpoint_loaded(checkpoint, checkpoint_path, remaining_steps)?
            .ok_or_else(|| {
                kwavers::core::error::KwaversError::Io(std::io::Error::other(
                    "No sensor data recorded",
                ))
            })?;

        let stats = solver.sensor_recorder.extract_all_stats();
        // Checkpoint path does not propagate record_start_index — always strips only t=0.
        let sensor_data = Self::trim_initial_recorder_sample(full_data, total_steps, 1);

        // Extract velocity time series using the same pattern as run_pstd_impl.
        let ux_data = solver
            .sensor_recorder
            .recorded_ux_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let uy_data = solver
            .sensor_recorder
            .recorded_uy_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let uz_data = solver
            .sensor_recorder
            .recorded_uz_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let ix_data = solver
            .sensor_recorder
            .recorded_ix_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let iy_data = solver
            .sensor_recorder
            .recorded_iy_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let iz_data = solver
            .sensor_recorder
            .recorded_iz_view()
            .map(|d| Self::trim_initial_recorder_view(d, total_steps, 1));
        let i_avg_x = solver.sensor_recorder.extract_i_avg_x();
        let i_avg_y = solver.sensor_recorder.extract_i_avg_y();
        let i_avg_z = solver.sensor_recorder.extract_i_avg_z();
        let velocity_stats = solver.sensor_recorder.extract_sampled_velocity_stats();
        let full_grid_stats = extract_full_grid_stats(&solver.sensor_recorder);

        Ok(SimulationRunResult {
            sensor_data,
            stats,
            ux_data,
            uy_data,
            uz_data,
            ix_data,
            iy_data,
            iz_data,
            i_avg_x,
            i_avg_y,
            i_avg_z,
            velocity_stats,
            full_grid_stats,
        })
    }

    /// Run the elastic-wave solver (Phase A.2 of ADR 007).
    ///
    /// Drives `kwavers::solver::forward::elastic::swe::ElasticWaveSolver`:
    ///   - reads the initial-displacement field from `grid_source.p0`
    ///     (carrier slot used by `Source.from_initial_displacement`); the axis
    ///     is decoded from the leading source's `source_type` suffix.
    ///   - assigns the field to the chosen displacement component on a fresh
    ///     `ElasticWaveField` (other components zero-initialised).
    ///   - records the same component to `sensor_data` at sensor-mask points.
    ///
    /// **Phase A.2 / A.2.5 / A.3 status** (see ADR 007):
    ///   - **A.2** ✅ initial-displacement IVP routed per-axis to the
    ///     chosen `ElasticWaveField` component.
    ///   - **A.2.5** ✅ ux / uy / uz traces all populated separately via
    ///     the SensorRecorder's per-component buffers.
    ///   - **A.3** ✅ optional velocity-source mask: post-step Dirichlet
    ///     override on vx / vy / vz at `mask` points using the supplied
    ///     1-D signals.
    ///   - **Remaining**: stress-tensor sources (A.3.5),
    ///     heterogeneous elastic media (A.4), PML alpha / anisotropic
    ///     thickness.
    #[allow(clippy::too_many_arguments)]
    fn run_elastic_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        grid_source: GridSource,
        sensor: Option<&Sensor>,
        pml_size: Option<usize>,
        _pml_inside: bool,
        elastic_ivp_axis: Option<&str>,
        elastic_velocity_source: Option<(
            ndarray::Array3<bool>,
            Option<ndarray::Array1<f64>>,
            Option<ndarray::Array1<f64>>,
            Option<ndarray::Array1<f64>>,
            String,
        )>,
    ) -> KwaversResult<SimulationRunResult> {
        // Local alias to keep the existing axis-decode block readable.
        let grid_source_axis_suffix: Option<String> = elastic_ivp_axis.map(|s| s.to_string());
        use kwavers::solver::forward::elastic::swe::{
            ElasticWaveConfig, ElasticWaveField, ElasticWaveSolver,
        };

        let (nx, ny, nz) = grid.dimensions();

        // The elastic solver requires at least one source: either an IVP
        // (initial-displacement field via grid_source.p0) or a velocity-
        // source mask (supplied via elastic_velocity_source). A zero-input
        // simulation is meaningless; surface a clear error.
        let has_ivp = grid_source.p0.is_some();
        let has_vel_source = elastic_velocity_source.is_some();
        if !has_ivp && !has_vel_source {
            return Err(kwavers::core::error::KwaversError::InvalidInput(
                "SolverType.Elastic requires either Source.from_initial_displacement(...) \
                 (initial-value problem) or Source.from_elastic_velocity_source(...) \
                 (driven velocity source). No source was supplied."
                    .to_string(),
            ));
        }

        // If an IVP was supplied, validate its shape.
        let u0_opt = match grid_source.p0 {
            Some(u0) => {
                if u0.dim() != (nx, ny, nz) {
                    return Err(kwavers::core::error::KwaversError::InvalidInput(format!(
                        "Elastic initial displacement shape {:?} must equal grid ({}, {}, {})",
                        u0.dim(),
                        nx,
                        ny,
                        nz
                    )));
                }
                Some(u0)
            }
            None => None,
        };

        // Sensor mask (optional) — records the chosen displacement component.
        let sensor_mask: Option<Array3<bool>> = sensor.and_then(|s| s.mask.clone());

        // Validate and convert the velocity-source bundle into the
        // kwavers `ElasticVelocitySource` type.
        let elastic_vsrc_kw: Option<kwavers::solver::forward::elastic::swe::ElasticVelocitySource> =
            if let Some((mask, ux_sig, uy_sig, uz_sig, mode_str)) = elastic_velocity_source {
                if mask.dim() != (nx, ny, nz) {
                    return Err(kwavers::core::error::KwaversError::InvalidInput(format!(
                        "Elastic velocity-source mask shape {:?} must equal grid ({}, {}, {})",
                        mask.dim(),
                        nx,
                        ny,
                        nz
                    )));
                }
                let validate_signal = |sig: &Option<ndarray::Array1<f64>>,
                                       name: &str|
                 -> KwaversResult<()> {
                    if let Some(s) = sig {
                        if s.len() != time_steps {
                            return Err(kwavers::core::error::KwaversError::InvalidInput(format!(
                            "Elastic velocity-source {} signal length {} must equal time_steps {}",
                            name,
                            s.len(),
                            time_steps,
                        )));
                        }
                    }
                    Ok(())
                };
                validate_signal(&ux_sig, "ux")?;
                validate_signal(&uy_sig, "uy")?;
                validate_signal(&uz_sig, "uz")?;
                let kw_mode = match mode_str.as_str() {
                    "dirichlet" => {
                        kwavers::solver::forward::elastic::swe::ElasticVelocitySourceMode::Dirichlet
                    }
                    _ => {
                        kwavers::solver::forward::elastic::swe::ElasticVelocitySourceMode::Additive
                    }
                };
                Some(
                    kwavers::solver::forward::elastic::swe::ElasticVelocitySource {
                        mask,
                        ux_signal: ux_sig,
                        uy_signal: uy_sig,
                        uz_signal: uz_sig,
                        mode: kw_mode,
                    },
                )
            } else {
                None
            };

        // Build the elastic config — single uniform PML thickness (Phase A.2).
        let pml_thickness = pml_size.unwrap_or(10);
        let mut config = ElasticWaveConfig::default();
        config.time_step = dt; // honor the dt computed by Simulation::run
        config.simulation_time = dt * (time_steps as f64);
        config.pml_thickness = pml_thickness;
        config.save_every = 1; // record every step to match k-Wave behavior
        config.sensor_mask = sensor_mask;
        config.velocity_source = elastic_vsrc_kw;

        // Construct the solver; the Medium trait object covers ElasticProperties.
        let medium_ref: &dyn kwavers::domain::medium::traits::Medium = medium.as_medium();
        let mut solver = ElasticWaveSolver::new(grid, medium_ref, config)?;

        // Initialise the field on the requested axis component (Phase A.2.5
        // of ADR 007: per-axis IVP routing). All other components and all
        // velocities start at zero. When no IVP is supplied (driven-velocity
        // source case), the field starts identically zero.
        let mut initial_field = ElasticWaveField::new(nx, ny, nz);
        if let Some(u0) = u0_opt {
            let axis_suffix = grid_source_axis_suffix.as_deref().unwrap_or("z");
            match axis_suffix {
                "x" => initial_field.ux.assign(&u0),
                "y" => initial_field.uy.assign(&u0),
                "z" | _ => initial_field.uz.assign(&u0),
            }
        }

        let duration = dt * (time_steps as f64);
        let _final_field = solver.propagate(&initial_field, duration, None)?;

        // Extract pressure-buffer back-compat trace (uz, named "sensor_data"
        // for parity with the acoustic API).
        let recorded_p = solver.extract_recorded_data();
        let sensor_data = recorded_p.unwrap_or_else(|| ndarray::Array2::zeros((1, 0)));

        // Extract per-component **particle-velocity** traces (Phase A.2.5)
        // via the public accessor — `sensor_recorder` itself is pub(crate).
        // Despite the legacy method name, the recorder is fed
        // `field.{vx, vy, vz}`. See the theorem block on
        // `extract_recorded_velocity_components` in
        // kwavers/src/solver/forward/elastic/swe/core/solver/propagation.rs.
        let (ux_data, uy_data, uz_data) = solver.extract_recorded_velocity_components();

        Ok(SimulationRunResult {
            sensor_data,
            stats: None,
            ux_data,
            uy_data,
            uz_data,
            ix_data: None,
            iy_data: None,
            iz_data: None,
            i_avg_x: None,
            i_avg_y: None,
            i_avg_z: None,
            velocity_stats: None,
            full_grid_stats: None, // elastic-wave path doesn't use pressure stats
        })
    }

    /// Dispatch the pseudospectral elastic path
    /// (`SolverType::ElasticPSTD`) — drives the canonical PSTD kernel via
    /// `pstd::extensions::ElasticPstdOrchestrator`. Currently velocity-source
    /// + sensor-mask only; no PML yet (boundary absorption is the next
    /// extension on the elastic-as-PSTD-plugin roadmap; see canonical solver
    /// matrix in `kwavers::solver::forward` module docs and the `[arch]`
    /// ElasticPSTD entry in `backlog.md`).
    #[allow(clippy::too_many_arguments)]
    fn run_elastic_pstd_impl(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        sensor: Option<&Sensor>,
        elastic_velocity_source: Option<(
            ndarray::Array3<bool>,
            Option<ndarray::Array1<f64>>,
            Option<ndarray::Array1<f64>>,
            Option<ndarray::Array1<f64>>,
            String,
        )>,
    ) -> KwaversResult<SimulationRunResult> {
        use kwavers::solver::forward::pstd::extensions::{
            ElasticPstdMedium, ElasticPstdOrchestrator, ElasticPstdSourceMode,
            ElasticPstdVelocitySource,
        };

        let medium_ref: &dyn kwavers::domain::medium::traits::Medium = medium.as_medium();
        let lame_lambda = medium_ref.lame_lambda_array();
        let lame_mu = medium_ref.lame_mu_array();
        let density = medium_ref.density_array().to_owned();

        let pstd_medium = ElasticPstdMedium {
            lame_lambda,
            lame_mu,
            density,
        };
        let mut orch = ElasticPstdOrchestrator::new(grid, pstd_medium, dt)?;

        let source = elastic_velocity_source.map(|(mask, ux, uy, uz, mode_str)| {
            let mode = match mode_str.as_str() {
                "dirichlet" => ElasticPstdSourceMode::Dirichlet,
                _ => ElasticPstdSourceMode::Additive,
            };
            ElasticPstdVelocitySource {
                mask,
                ux,
                uy,
                uz,
                mode,
            }
        });

        let sensor_mask: Option<ndarray::Array3<bool>> = sensor.and_then(|s| s.mask.clone());
        let recorded = orch.propagate(time_steps, source.as_ref(), sensor_mask.as_ref())?;

        // The orchestrator records velocity components (vx, vy, vz). To match
        // the SimulationRunResult contract used by SolverType::Elastic, the
        // legacy "sensor_data" pressure buffer carries vz (preserves the
        // back-compat ordering enforced for ElasticWaveSolver).
        let sensor_data = recorded
            .vz
            .clone()
            .unwrap_or_else(|| ndarray::Array2::zeros((1, 0)));

        Ok(SimulationRunResult {
            sensor_data,
            stats: None,
            ux_data: recorded.vx,
            uy_data: recorded.vy,
            uz_data: recorded.vz,
            ix_data: None,
            iy_data: None,
            iz_data: None,
            i_avg_x: None,
            i_avg_y: None,
            i_avg_z: None,
            velocity_stats: None,
            full_grid_stats: None,
        })
    }

    /// Dispatch GPU-resident PSTD if the `gpu` feature is enabled and GPU is
    /// available; otherwise fall back to the CPU PSTD implementation.
    #[allow(clippy::too_many_arguments)]
    fn run_gpu_pstd_or_cpu_fallback(
        grid: &KwaversGrid,
        medium: &MediumInner,
        time_steps: usize,
        dt: f64,
        compatibility_mode: CompatibilityMode,
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
        axisymmetric: bool,
        record_modes: &[String],
        record_start_index: usize,
    ) -> KwaversResult<SimulationRunResult> {
        #[cfg(feature = "gpu")]
        {
            // Attempt GPU path — fall back to CPU on any error.
            // The GPU path returns `(Array2<f64>, Option<SampledStatistics>)`; wrap into
            // `SimulationRunResult`. GPU path does not support velocity recording.
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
                Ok((sensor_data, stats)) => {
                    // GPU returns exactly `time_steps` columns with no t=0 column.
                    // Apply record_start_index via the same trim helper (else branch).
                    let sensor_data = Self::trim_initial_recorder_sample(
                        sensor_data,
                        time_steps,
                        record_start_index,
                    );
                    return Ok(SimulationRunResult {
                        sensor_data,
                        stats,
                        ux_data: None,
                        uy_data: None,
                        uz_data: None,
                        ix_data: None,
                        iy_data: None,
                        iz_data: None,
                        i_avg_x: None,
                        i_avg_y: None,
                        i_avg_z: None,
                        velocity_stats: None,
                    });
                }
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
            compatibility_mode,
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
            axisymmetric,
            record_modes,
            record_start_index,
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
        let (default_thickness, max_allowed) = Self::cpml_thickness_limits(nx, ny, nz);
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
            // Convert sigma (s⁻¹) → exp(-sigma·dt/2) ∈ (0,1].
            // The WGSL velocity/density updates expect: pml = exp(-σ·dt/2).
            // Using raw sigma causes f32 overflow (σ_max ≈ 3×10⁷) → NaN after ~4 steps.
            let pml_sgx_1d: Vec<f32> = profiles
                .sigma_x_sgx
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_sgy_1d: Vec<f32> = profiles
                .sigma_y_sgy
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_sgz_1d: Vec<f32> = profiles
                .sigma_z_sgz
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_x_1d: Vec<f32> = profiles
                .sigma_x
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_y_1d: Vec<f32> = profiles
                .sigma_y
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_z_1d: Vec<f32> = profiles
                .sigma_z
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();

            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let flat = ix * ny * nz + iy * nz + iz;
                        pml_sgx_3d[flat] = pml_sgx_1d[ix];
                        pml_sgy_3d[flat] = pml_sgy_1d[iy];
                        pml_sgz_3d[flat] = pml_sgz_1d[iz];
                        pml_x_3d[flat] = pml_x_1d[ix];
                        pml_y_3d[flat] = pml_y_1d[iy];
                        pml_z_3d[flat] = pml_z_1d[iz];
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

        // ── Effective absorption (fall back to medium's stored coefficient) ────
        let effective_alpha_db = if alpha_coeff_db > 0.0 {
            alpha_coeff_db
        } else {
            medium.as_medium().alpha_coefficient(0.0, 0.0, 0.0, grid)
        };
        let alpha_power = {
            let y_medium = medium.as_medium().alpha_power(0.0, 0.0, 0.0, grid);
            if alpha_coeff_db <= 0.0 && y_medium > 0.0 && (y_medium - 1.0).abs() > 1e-12 {
                y_medium
            } else {
                alpha_power
            }
        };

        // ── Physics flags ─────────────────────────────────────────────────────
        // Check first voxel; works correctly for homogeneous media.
        // Heterogeneous media with spatially-varying properties handled per-voxel below.
        let has_nonlinear = medium.as_medium().nonlinearity(0, 0, 0) > 0.0;
        let has_absorption = effective_alpha_db > 0.0;

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

        // ── Fractional-Laplacian absorption operators (Treeby & Cox 2010 Eqs. 9-10) ──
        // nabla1 = |k|^(y-2), nabla2 = |k|^(y-1) in FFT order.
        // tau = -2*alpha0*c0^(y-1), eta = 2*alpha0*c0^y*tan(pi*y/2) per voxel.
        let (absorb_nabla1_flat, absorb_nabla2_flat, absorb_tau_flat, absorb_eta_flat) =
            if has_absorption {
                use std::f64::consts::PI;
                let dk_x = 2.0 * PI / (nx as f64 * grid.dx);
                let dk_y = 2.0 * PI / (ny as f64 * grid.dy);
                let dk_z = 2.0 * PI / (nz as f64 * grid.dz);
                let singularity_thresh: f64 = 1e-8;
                let y = alpha_power;

                let mut n1 = vec![0.0f32; total];
                let mut n2 = vec![0.0f32; total];
                let mut tau_v = vec![0.0f32; total];
                let mut eta_v = vec![0.0f32; total];

                for flat in 0..total {
                    let ix = flat / (ny * nz);
                    let iy = (flat % (ny * nz)) / nz;
                    let iz = flat % nz;

                    // k-magnitude in FFT order
                    let kix = if ix <= nx / 2 {
                        ix as f64
                    } else {
                        (nx - ix) as f64
                    } * dk_x;
                    let kiy = if iy <= ny / 2 {
                        iy as f64
                    } else {
                        (ny - iy) as f64
                    } * dk_y;
                    let kiz = if iz <= nz / 2 {
                        iz as f64
                    } else {
                        (nz - iz) as f64
                    } * dk_z;
                    let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();

                    if k_mag > singularity_thresh {
                        n1[flat] = k_mag.powf(y - 2.0) as f32;
                        n2[flat] = k_mag.powf(y - 1.0) as f32;
                    }

                    let alpha_db_cm = medium.as_medium().absorption(ix, iy, iz);
                    let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
                    let c0_local = medium.as_medium().sound_speed(ix, iy, iz);
                    tau_v[flat] = (-2.0 * alpha_0_si * c0_local.powf(y - 1.0)) as f32;
                    eta_v[flat] =
                        (2.0 * alpha_0_si * c0_local.powf(y) * (PI * y / 2.0).tan()) as f32;
                }
                (n1, n2, tau_v, eta_v)
            } else {
                (
                    vec![0.0f32; total],
                    vec![0.0f32; total],
                    vec![0.0f32; total],
                    vec![0.0f32; total],
                )
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
            &absorb_nabla1_flat,
            &absorb_nabla2_flat,
            &absorb_tau_flat,
            &absorb_eta_flat,
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
    // Fractional-Laplacian absorption coefficients (Treeby & Cox 2010 Eqs. 9-10).
    // tau = -2*alpha_0*c0^(y-1) and eta = 2*alpha_0*c0^y*tan(pi*y/2) per voxel.
    // These are precomputed at session creation and re-uploaded on each scan line.
    absorb_tau_flat: Vec<f32>,
    absorb_eta_flat: Vec<f32>,
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
    last_medium_variable_upload_ns: u64,
    last_medium_static_upload_ns: u64,
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
            Err(PyRuntimeError::new_err(
                "GpuPstdSession requires the 'gpu' feature.  \
                 Rebuild pykwavers with --features gpu.",
            ))
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

            // ── Fractional-Laplacian absorption operators (Treeby & Cox 2010 Eqs. 9-10) ──
            let has_absorption = absorption.is_some();
            // nabla1 = |k|^(y-2), nabla2 = |k|^(y-1) in FFT order (k-space operators).
            // tau = -2*alpha0*c0^(y-1), eta = 2*alpha0*c0^y*tan(pi*y/2) per voxel.
            let (absorb_nabla1_flat, absorb_nabla2_flat, absorb_tau_flat, absorb_eta_flat) =
                if has_absorption {
                    use std::f64::consts::PI;
                    let dk_x = 2.0 * PI / (nx as f64 * kgrid.dx);
                    let dk_y = 2.0 * PI / (ny as f64 * kgrid.dy);
                    let dk_z = 2.0 * PI / (nz as f64 * kgrid.dz);
                    let singularity_thresh: f64 = 1e-8;
                    let y = alpha_power;

                    let mut n1 = vec![0.0f32; total];
                    let mut n2 = vec![0.0f32; total];
                    let mut tau_v = vec![0.0f32; total];
                    let mut eta_v = vec![0.0f32; total];

                    let ab_arr = absorption.as_ref().unwrap().as_array();

                    if (y - 1.0).abs() < 1e-12 && ab_arr.iter().any(|&v| v > 0.0) {
                        return Err(PyValueError::new_err(
                            "alpha_power must not be 1.0 for fractional Laplacian absorption",
                        ));
                    }

                    for flat in 0..total {
                        let ix = flat / (ny * nz);
                        let iy = (flat % (ny * nz)) / nz;
                        let iz = flat % nz;

                        // k-magnitude in FFT order
                        let kix = if ix <= nx / 2 {
                            ix as f64
                        } else {
                            (nx - ix) as f64
                        } * dk_x;
                        let kiy = if iy <= ny / 2 {
                            iy as f64
                        } else {
                            (ny - iy) as f64
                        } * dk_y;
                        let kiz = if iz <= nz / 2 {
                            iz as f64
                        } else {
                            (nz - iz) as f64
                        } * dk_z;
                        let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();

                        if k_mag > singularity_thresh {
                            n1[flat] = k_mag.powf(y - 2.0) as f32;
                            n2[flat] = k_mag.powf(y - 1.0) as f32;
                        }

                        // Spatially-varying absorption coefficients.
                        let alpha_db_cm = ab_arr[[ix, iy, iz]];
                        let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
                        let c0_local = c0_flat[flat] as f64;
                        tau_v[flat] = (-2.0 * alpha_0_si * c0_local.powf(y - 1.0)) as f32;
                        eta_v[flat] =
                            (2.0 * alpha_0_si * c0_local.powf(y) * (PI * y / 2.0).tan()) as f32;
                    }
                    (n1, n2, tau_v, eta_v)
                } else {
                    (
                        vec![0.0f32; total],
                        vec![0.0f32; total],
                        vec![0.0f32; total],
                        vec![0.0f32; total],
                    )
                };

            // ── Absorption diagnostic ─────────────────────────────────────────
            if has_absorption {
                let tau_max = absorb_tau_flat
                    .iter()
                    .cloned()
                    .fold(0.0f32, |a, b| a.abs().max(b.abs()));
                let eta_max = absorb_eta_flat
                    .iter()
                    .cloned()
                    .fold(0.0f32, |a, b| a.abs().max(b.abs()));
                let nabla2_max = absorb_nabla2_flat
                    .iter()
                    .cloned()
                    .fold(0.0f32, |a, b| a.max(b));
                eprintln!("[pykwavers-diag] GpuPstdSession absorbing=true: tau_max={tau_max:.3e}, eta_max={eta_max:.3e}, nabla2_max={nabla2_max:.3e}");
            } else {
                eprintln!("[pykwavers-diag] GpuPstdSession absorbing=false (lossless)");
            }

            // ── PML profiles ──────────────────────────────────────────────────
            let (pml_x_sz, pml_y_sz, pml_z_sz) = pml_size_xyz.unwrap_or((10, 10, 10));
            let pml_config = CPMLConfig::with_per_dimension_thickness(pml_x_sz, pml_y_sz, pml_z_sz);
            let profiles = CPMLProfiles::new(&pml_config, kgrid, c_ref, dt)
                .map_err(|e| PyRuntimeError::new_err(format!("PML init failed: {e}")))?;

            // Convert sigma (s⁻¹) → exp(-sigma·dt/2) ∈ (0,1].
            // The WGSL velocity update is: ux = pml² * ux - dt/ρ₀ * pml * ∇p
            // which matches k-Wave's split-field PML with pml = exp(-σ·dt/2).
            // Passing raw sigma (up to ~3×10⁷) instead causes f32 overflow → NaN.
            let pml_sgx_1d: Vec<f32> = profiles
                .sigma_x_sgx
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_sgy_1d: Vec<f32> = profiles
                .sigma_y_sgy
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_sgz_1d: Vec<f32> = profiles
                .sigma_z_sgz
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_x_1d: Vec<f32> = profiles
                .sigma_x
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_y_1d: Vec<f32> = profiles
                .sigma_y
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();
            let pml_z_1d: Vec<f32> = profiles
                .sigma_z
                .iter()
                .map(|&s| (-s * dt * 0.5).exp() as f32)
                .collect();

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
                        pml_sgx_3d[flat] = pml_sgx_1d[ix];
                        pml_sgy_3d[flat] = pml_sgy_1d[iy];
                        pml_sgz_3d[flat] = pml_sgz_1d[iz];
                        pml_x_3d[flat] = pml_x_1d[ix];
                        pml_y_3d[flat] = pml_y_1d[iy];
                        pml_z_3d[flat] = pml_z_1d[iz];
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
                &absorb_nabla1_flat,
                &absorb_nabla2_flat,
                &absorb_tau_flat,
                &absorb_eta_flat,
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
                absorb_tau_flat,
                absorb_eta_flat,
                has_absorption,
                time_steps,
                sensor_indices: Vec::new(),
                vel_x_indices: Vec::new(),
                vel_x_signals: Vec::new(),
                last_medium_upload_ns: 0,
                last_medium_variable_upload_ns: 0,
                last_medium_static_upload_ns: 0,
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

    /// Disable the k-space source correction (sets source_kappa = 1 everywhere).
    ///
    /// Matches k-wave-python's `u_mode = "additive-no-correction"`, which is the
    /// default for NotATransducer (see kwave/ksource.py:186, ktransducer.py:244).
    /// Without this call, GpuPstdSession applies `sinc(c·dt·|k|/2)` correction
    /// on injected velocity sources (the "additive" mode). Call once after
    /// session construction; effect persists across subsequent `run_scan_line`.
    fn disable_source_correction(&self, _py: Python<'_>) -> PyResult<()> {
        #[cfg(feature = "gpu")]
        {
            self.solver.disable_source_correction();
            Ok(())
        }
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err(
                "GpuPstdSession requires the 'gpu' feature.",
            ))
        }
    }

    /// Return the timing profile from the most recent scan-line execution.
    ///
    /// Durations are host-side wall-clock measurements in nanoseconds.
    /// `medium_variable_upload_ns` and `medium_static_upload_ns` separate the
    /// varying scan-line medium refresh from the resident static buffers.
    #[getter]
    fn last_run_profile<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let profile = PyDict::new(py);
        profile.set_item("medium_upload_ns", self.last_medium_upload_ns)?;
        profile.set_item(
            "medium_variable_upload_ns",
            self.last_medium_variable_upload_ns,
        )?;
        profile.set_item("medium_static_upload_ns", self.last_medium_static_upload_ns)?;
        profile.set_item("solver_run_ns", self.last_solver_run_ns)?;
        profile.set_item("materialize_ns", self.last_materialize_ns)?;
        profile.set_item("total_ns", self.last_total_ns)?;
        profile.set_item("n_sensors", self.sensor_indices.len())?;
        profile.set_item("n_velocity_sources", self.vel_x_indices.len())?;
        Ok(profile)
    }

    /// Return the most recent scan-line timing profile as a compact tuple.
    ///
    /// Field order:
    /// `medium_upload_ns`, `medium_variable_upload_ns`, `medium_static_upload_ns`,
    /// `solver_run_ns`, `materialize_ns`, `total_ns`.
    #[getter]
    fn last_run_profile_ns<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                self.last_medium_upload_ns,
                self.last_medium_variable_upload_ns,
                self.last_medium_static_upload_ns,
                self.last_solver_run_ns,
                self.last_materialize_ns,
                self.last_total_ns,
            ],
        )
    }

    /// Run one scan line with updated medium (sound_speed, density).
    ///
    /// Uploads the varying medium to the GPU, then runs the full time loop.
    /// Returns sensor pressure as an ndarray (n_sensors, Nt).
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
            Err(PyRuntimeError::new_err("GPU feature not enabled"))
        }

        #[cfg(feature = "gpu")]
        {
            let total_t0 = std::time::Instant::now();

            let ss_arr = _sound_speed.as_array();
            let rho_arr = _density.as_array();

            // Borrow contiguous NumPy memory directly when possible; fall back to
            // a one-time host copy only for non-contiguous inputs.
            let c0_flat: Cow<'_, [f64]> = match ss_arr.as_slice() {
                Some(slice) => Cow::Borrowed(slice),
                None => Cow::Owned(ss_arr.iter().copied().collect()),
            };
            let rho0_flat: Cow<'_, [f64]> = match rho_arr.as_slice() {
                Some(slice) => Cow::Borrowed(slice),
                None => Cow::Owned(rho_arr.iter().copied().collect()),
            };

            let upload_t0 = std::time::Instant::now();
            // Only the varying medium tensors are refreshed per scan line.
            // bon_a / tau / eta are resident from session construction and stay
            // unchanged for the linear-transducer example and similar fixed-media scans.
            self.solver
                .update_medium_variable(c0_flat.as_ref(), rho0_flat.as_ref());
            let medium_upload_ns = upload_t0.elapsed().as_nanos() as u64;

            let result = self.run_scan_line_cached(_py);
            self.last_medium_variable_upload_ns = medium_upload_ns;
            self.last_medium_static_upload_ns = 0;
            self.last_medium_upload_ns = medium_upload_ns;
            self.last_total_ns = total_t0.elapsed().as_nanos() as u64;
            result
        }
    }

    /// Run one scan line using the currently resident medium buffers.
    ///
    /// This is intended for repeated steering/focusing runs in a fixed medium.
    fn run_scan_line_cached<'py>(
        &mut self,
        _py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        #[cfg(not(feature = "gpu"))]
        {
            Err(PyRuntimeError::new_err("GPU feature not enabled"))
        }

        #[cfg(feature = "gpu")]
        {
            let total_t0 = std::time::Instant::now();
            self.last_medium_upload_ns = 0;
            self.last_medium_variable_upload_ns = 0;
            self.last_medium_static_upload_ns = 0;
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
// Internal run result bundle
// ============================================================================

/// Bundle returned by every `run_*_impl` function.
///
/// Velocity fields are `None` unless the caller supplied a `record_modes` list
/// that includes at least one velocity component (e.g. `"ux"`, `"ux_max"`).
/// The FDTD path never populates velocity fields; the PSTD path does.
/// Extract full-grid `(p_max, p_min, p_rms, p_final)` from a recorder if
/// any pressure-statistics mode was requested. Returns `None` otherwise.
fn extract_full_grid_stats(
    recorder: &SensorRecorder,
) -> Option<(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)> {
    let stats = recorder.full_pressure_statistics()?;
    Some((
        stats.get_p_max().clone(),
        stats.get_p_min().clone(),
        stats.p_rms(),
        stats.get_p_final().clone(),
    ))
}

struct SimulationRunResult {
    /// Pressure time series at sensor positions: `(n_sensors, time_steps)`.
    sensor_data: ndarray::Array2<f64>,
    /// Pressure spatial statistics (p_max/min/rms/final sampled at sensors).
    stats: Option<SampledStatistics>,
    /// Staggered ux time series at sensor positions: `(n_sensors, time_steps)`.
    ux_data: Option<ndarray::Array2<f64>>,
    /// Staggered uy time series at sensor positions.
    uy_data: Option<ndarray::Array2<f64>>,
    /// Staggered uz time series at sensor positions.
    uz_data: Option<ndarray::Array2<f64>>,
    /// Acoustic x-intensity time series at sensor positions.
    ix_data: Option<ndarray::Array2<f64>>,
    /// Acoustic y-intensity time series at sensor positions.
    iy_data: Option<ndarray::Array2<f64>>,
    /// Acoustic z-intensity time series at sensor positions.
    iz_data: Option<ndarray::Array2<f64>>,
    /// Time-averaged x-intensity at sensor positions.
    i_avg_x: Option<ndarray::Array1<f64>>,
    /// Time-averaged y-intensity at sensor positions.
    i_avg_y: Option<ndarray::Array1<f64>>,
    /// Time-averaged z-intensity at sensor positions.
    i_avg_z: Option<ndarray::Array1<f64>>,
    /// Per-component velocity statistics sampled at sensor positions.
    velocity_stats: Option<SampledVelocityStats>,
    /// Full-grid pressure-statistics field (kernel-generation use).
    /// `(p_max, p_min, p_rms, p_final)` — each `Array3<f64>` shape
    /// `(nx, ny, nz)`. `None` when no `p_*` mode was requested.
    full_grid_stats: Option<(Array3<f64>, Array3<f64>, Array3<f64>, Array3<f64>)>,
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

    /// Full-grid peak compressional pressure [Pa] over all time steps —
    /// shape `(nx, ny, nz)`. None unless a `p_*` recording mode was set.
    #[pyo3(get)]
    p_max_field: Option<Py<PyArray3<f64>>>,
    /// Full-grid peak rarefactional pressure [Pa] (most-negative
    /// pressure per voxel) over all time steps. Shape `(nx, ny, nz)`.
    /// This is the canonical cavitation-kernel field: feed it through
    /// the Maxwell-2013 erf-CDF to obtain per-voxel intrinsic-threshold
    /// cavitation probability.
    #[pyo3(get)]
    p_min_field: Option<Py<PyArray3<f64>>>,
    /// Full-grid RMS pressure [Pa]. Shape `(nx, ny, nz)`.
    #[pyo3(get)]
    p_rms_field: Option<Py<PyArray3<f64>>>,
    /// Full-grid final-time pressure snapshot [Pa]. Shape `(nx, ny, nz)`.
    #[pyo3(get)]
    p_final_field: Option<Py<PyArray3<f64>>>,

    // ── Particle velocity time series ────────────────────────────────────────
    /// Staggered ux time series: `(n_sensors, time_steps)` [m/s] (None if not requested)
    #[pyo3(get)]
    ux: Option<Py<PyArray2<f64>>>,
    /// Staggered uy time series: `(n_sensors, time_steps)` [m/s] (None if not requested)
    #[pyo3(get)]
    uy: Option<Py<PyArray2<f64>>>,
    /// Staggered uz time series: `(n_sensors, time_steps)` [m/s] (None if not requested)
    #[pyo3(get)]
    uz: Option<Py<PyArray2<f64>>>,
    /// Acoustic x-intensity time series: `p * ux` [W/m^2] (None if not requested)
    #[pyo3(get)]
    ix: Option<Py<PyArray2<f64>>>,
    /// Acoustic y-intensity time series: `p * uy` [W/m^2] (None if not requested)
    #[pyo3(get)]
    iy: Option<Py<PyArray2<f64>>>,
    /// Acoustic z-intensity time series: `p * uz` [W/m^2] (None if not requested)
    #[pyo3(get)]
    iz: Option<Py<PyArray2<f64>>>,
    /// Time-averaged x-intensity at each sensor [W/m^2] (None if not requested)
    #[pyo3(get)]
    i_avg_x: Option<Py<PyArray1<f64>>>,
    /// Time-averaged y-intensity at each sensor [W/m^2] (None if not requested)
    #[pyo3(get)]
    i_avg_y: Option<Py<PyArray1<f64>>>,
    /// Time-averaged z-intensity at each sensor [W/m^2] (None if not requested)
    #[pyo3(get)]
    i_avg_z: Option<Py<PyArray1<f64>>>,

    // ── Velocity statistics ──────────────────────────────────────────────────
    /// Maximum ux at each sensor position over all time steps [m/s] (None if not requested)
    #[pyo3(get)]
    ux_max: Option<Py<PyArray1<f64>>>,
    /// Minimum ux at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    ux_min: Option<Py<PyArray1<f64>>>,
    /// RMS ux at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    ux_rms: Option<Py<PyArray1<f64>>>,
    /// Maximum uy at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    uy_max: Option<Py<PyArray1<f64>>>,
    /// Minimum uy at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    uy_min: Option<Py<PyArray1<f64>>>,
    /// RMS uy at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    uy_rms: Option<Py<PyArray1<f64>>>,
    /// Maximum uz at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    uz_max: Option<Py<PyArray1<f64>>>,
    /// Minimum uz at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    uz_min: Option<Py<PyArray1<f64>>>,
    /// RMS uz at each sensor position [m/s] (None if not requested)
    #[pyo3(get)]
    uz_rms: Option<Py<PyArray1<f64>>>,
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
    m.add_function(wrap_pyfunction!(kspace_line_recon, m)?)?;
    m.add_function(wrap_pyfunction!(time_reversal_reconstruction, m)?)?;

    // Register utility functions
    pam_bindings::register_pam(m)?;
    utils_bindings::register_utils(m)?;
    thermal_bindings::register_thermal(m)?;
    field_surrogate_bindings::register(m)?;
    seismic_bindings::register(m)?;
    theranostic_bindings::register(m)?;
    fft_bindings::register(m)?;

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
    use super::{time_reversal_reconstruction_impl, Grid, SensorRecordField, Simulation};
    use ndarray::{array, Array2};

    #[test]
    fn trim_initial_recorder_sample_aligns_with_kwave_time_axis() {
        // Default record_start_index=1: keeps cols [0..Nt) → drops the LAST
        // column (post-final-step value), aligning with k-Wave's Nt-sample
        // window indexed by physical times [0, dt, …, (Nt−1)·dt].
        let nt_plus_one = array![[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]];
        let trimmed = Simulation::trim_initial_recorder_sample(nt_plus_one, 3, 1);
        assert_eq!(trimmed.shape(), &[2, 3]);
        assert_eq!(trimmed[[0, 0]], 0.0); // p(0) preserved
        assert_eq!(trimmed[[0, 1]], 1.0);
        assert_eq!(trimmed[[0, 2]], 2.0); // p(2dt); p(3dt) dropped
        assert_eq!(trimmed[[1, 0]], 10.0);
        assert_eq!(trimmed[[1, 2]], 12.0);

        // Buffer already Nt cols (e.g. velocity buffers populated only inside
        // step_forward): pass through unchanged at start=1.
        let exact_nt = array![[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]];
        let untouched = Simulation::trim_initial_recorder_sample(exact_nt.clone(), 3, 1);
        assert_eq!(untouched, exact_nt);

        // record_start_index=2: skip col 0, keep cols [1..Nt) → Nt-1 cols.
        let nt_plus_one2 = array![[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]];
        let trimmed2 = Simulation::trim_initial_recorder_sample(nt_plus_one2, 3, 2);
        assert_eq!(trimmed2.shape(), &[2, 2]);
        assert_eq!(trimmed2[[0, 0]], 1.0);
        assert_eq!(trimmed2[[0, 1]], 2.0);
        assert_eq!(trimmed2[[1, 1]], 12.0);
    }

    #[test]
    fn trim_initial_recorder_view_aligns_with_kwave_time_axis() {
        let nt_plus_one = array![[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0]];
        let trimmed = Simulation::trim_initial_recorder_view(nt_plus_one.view(), 3, 1);
        assert_eq!(trimmed.shape(), &[2, 3]);
        assert_eq!(trimmed[[0, 0]], 0.0);
        assert_eq!(trimmed[[0, 2]], 2.0);
        assert_eq!(trimmed[[1, 2]], 12.0);

        let exact_nt = array![[1.0, 2.0, 3.0], [11.0, 12.0, 13.0]];
        let untouched = Simulation::trim_initial_recorder_view(exact_nt.view(), 3, 1);
        assert_eq!(untouched, exact_nt);
    }

    #[test]
    fn record_modes_to_spec_maps_acoustic_intensity_fields() {
        let modes = vec!["Ix".to_string(), "I_avg_x".to_string()];
        let spec = Simulation::record_modes_to_spec(&modes);

        assert!(spec.contains(SensorRecordField::IntensityX));
        assert!(spec.contains(SensorRecordField::IntensityAvgX));
        assert!(spec.records_pressure());
        assert!(!spec.records_ux());
        assert!(spec.needs_any_velocity());
        assert!(spec.records_intensity_x());
        assert!(!spec.records_intensity_y());
    }

    #[test]
    fn time_reversal_reconstruction_impl_preserves_zero_field_with_pml_crop() {
        let grid = Grid::new(6, 6, 1, 0.1e-3, 0.1e-3, 0.1e-3).unwrap();
        let sensor_data = Array2::zeros((3, 8));
        let sensor_positions = array![[0.0, 0.0, 0.0], [0.0, 0.1e-3, 0.0], [0.0, 0.2e-3, 0.0],];

        let reconstruction = time_reversal_reconstruction_impl(
            sensor_data,
            sensor_positions,
            &grid.inner,
            1500.0,
            1.0e8,
            Some(2),
        )
        .unwrap();

        assert_eq!(reconstruction.dim(), (6, 6, 1));
        assert!(reconstruction.iter().all(|&value| value == 0.0));
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
    #[allow(clippy::too_many_arguments)]
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
        let config = kwavers::physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDConfig {
            gains,
            sample_time,
            output_min,
            output_max,
            integral_limit,
            ..kwavers::physics::acoustics::bubble_dynamics::cavitation_control::pid_controller::PIDConfig::default()
        };
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

#[pyfunction]
#[pyo3(signature = (sensor_data, dy, dt, c, *, data_order = "ty", interp = "linear", pos_cond = false))]
#[allow(clippy::too_many_arguments)]
fn kspace_line_recon<'py>(
    py: Python<'py>,
    sensor_data: PyReadonlyArray2<f64>,
    dy: f64,
    dt: f64,
    c: f64,
    data_order: &str,
    interp: &str,
    pos_cond: bool,
) -> PyResult<Py<PyArray2<f64>>> {
    let data_order = match data_order.to_ascii_lowercase().as_str() {
        "ty" => LineReconDataOrder::Ty,
        "yt" => LineReconDataOrder::Yt,
        other => {
            return Err(PyValueError::new_err(format!(
                "data_order must be 'ty' or 'yt', got {other}"
            )))
        }
    };
    let interp = match interp.to_ascii_lowercase().as_str() {
        "linear" => LineReconInterpolation::Linear,
        "nearest" => LineReconInterpolation::Nearest,
        other => {
            return Err(PyValueError::new_err(format!(
                "interp must be 'linear' or 'nearest', got {other}"
            )))
        }
    };

    let input = sensor_data.as_array().to_owned();
    let recon = py
        .detach(|| kwavers_kspace_line_recon(input.view(), dy, dt, c, data_order, interp, pos_cond))
        .map_err(|err| PyRuntimeError::new_err(format!("kwavers error: {}", err)))?;

    Ok(PyArray2::from_owned_array(py, recon).into())
}

#[pyfunction]
#[pyo3(signature = (sensor_data, sensor_positions, grid, sound_speed, sampling_frequency, pml_size=None))]
fn time_reversal_reconstruction<'py>(
    py: Python<'py>,
    sensor_data: PyReadonlyArray2<f64>,
    sensor_positions: PyReadonlyArray2<f64>,
    grid: &Grid,
    sound_speed: f64,
    sampling_frequency: f64,
    pml_size: Option<usize>,
) -> PyResult<Py<PyArray3<f64>>> {
    let sensor_data = sensor_data.as_array().to_owned();
    let sensor_positions = sensor_positions.as_array().to_owned();
    let grid_inner = grid.inner.clone();
    let reconstruction = py
        .detach(move || {
            time_reversal_reconstruction_impl(
                sensor_data,
                sensor_positions,
                &grid_inner,
                sound_speed,
                sampling_frequency,
                pml_size,
            )
        })
        .map_err(|err| PyRuntimeError::new_err(format!("kwavers error: {}", err)))?;

    Ok(PyArray3::from_owned_array(py, reconstruction).into())
}

/// Reconstruct an initial pressure field by replaying time-reversed boundary data.
///
/// # Theorem
/// Let `u_n` denote the discrete pressure field after `n` forward FDTD steps,
/// and let `g[t, s]` be the recorded pressure at sensor `s` and time index `t`.
/// If the same grid, medium, timestep, boundary treatment, and source mask are
/// used for replay, then a Dirichlet source driven by `g` reversed in time
/// produces the same discrete time-reversal experiment as the vendored
/// k-Wave `TimeReversal` example. The returned field is the solver's final
/// pressure state cropped back to the physical domain when outer PML layers
/// are requested.
///
/// # Proof sketch
/// The helper constructs the same sensor mask, reverses the same discrete
/// trace matrix, applies the same Dirichlet boundary condition, and advances
/// the same FDTD update operator. Outer PML is represented by embedding the
/// physical domain in a larger computational grid and cropping the final field
/// back to the interior. This preserves the interior discrete operator on the
/// physical domain while matching the source replay used by k-Wave.
fn time_reversal_reconstruction_impl(
    sensor_data: Array2<f64>,
    sensor_positions: Array2<f64>,
    grid: &KwaversGrid,
    sound_speed: f64,
    sampling_frequency: f64,
    pml_size: Option<usize>,
) -> KwaversResult<Array3<f64>> {
    if sound_speed <= 0.0 || !sound_speed.is_finite() {
        return Err(KwaversError::Validation(
            kwavers::core::error::ValidationError::FieldValidation {
                field: "sound_speed".to_string(),
                value: sound_speed.to_string(),
                constraint: "must be a positive finite scalar".to_string(),
            },
        ));
    }
    if sampling_frequency <= 0.0 || !sampling_frequency.is_finite() {
        return Err(KwaversError::Validation(
            kwavers::core::error::ValidationError::FieldValidation {
                field: "sampling_frequency".to_string(),
                value: sampling_frequency.to_string(),
                constraint: "must be a positive finite scalar".to_string(),
            },
        ));
    }
    if sensor_positions.ncols() != 3 || sensor_positions.nrows() == 0 {
        return Err(KwaversError::Validation(
            kwavers::core::error::ValidationError::FieldValidation {
                field: "sensor_positions".to_string(),
                value: format!("{:?}", sensor_positions.dim()),
                constraint: "must have shape (n_sensors, 3) and contain at least one sensor"
                    .to_string(),
            },
        ));
    }

    let n_sensors = sensor_positions.nrows();
    let sensor_data = match sensor_data.dim() {
        (rows, _cols) if rows == n_sensors => sensor_data,
        (_rows, cols) if cols == n_sensors => sensor_data.reversed_axes().to_owned(),
        (rows, cols) => {
            return Err(KwaversError::Validation(
                kwavers::core::error::ValidationError::FieldValidation {
                    field: "sensor_data".to_string(),
                    value: format!("shape=({rows}, {cols})"),
                    constraint: format!(
                        "must align with sensor_positions rows {} along one axis",
                        n_sensors
                    ),
                },
            ))
        }
    };

    if sensor_data.ncols() == 0 {
        return Err(KwaversError::Validation(
            kwavers::core::error::ValidationError::FieldValidation {
                field: "sensor_data".to_string(),
                value: "0 time samples".to_string(),
                constraint: "must contain at least one time sample".to_string(),
            },
        ));
    }
    let nt = sensor_data.ncols();

    let (default_thickness, max_allowed) =
        Simulation::cpml_thickness_limits(grid.nx, grid.ny, grid.nz);
    let pml = pml_size.unwrap_or(default_thickness).min(max_allowed);

    // Expand the grid by `pml` cells on each active side so the TR sensor falls
    // on the first non-PML cell (sigma = 0) of the expanded domain.
    //
    // Architecture note: pykwavers uses split-field PML (exp(−σΔt/2) per half-step),
    // which is direction-agnostic — it absorbs both inward and outward waves.  Placing
    // the Dirichlet TR source inside the PML (sensor at cell 0 of the original domain)
    // causes outward TR waves to be attenuated by cells 1..pml-1 as they propagate toward
    // the interior, collapsing the reconstruction amplitude by >200×.
    //
    // KWave.jl uses CPML (Convolutional PML), which is direction-selective: the
    // convolution memory (ψ) tracks outgoing waves and is transparent to inward waves.
    // When `time_reversal_boundary_data` is active, KWave.jl additionally bypasses CPML
    // at source cells, allowing the forced pressure to drive the interior without split-
    // field interference.  Replicating this with split-field PML would require either:
    //   (a) full CPML implementation (direction-selective absorption), or
    //   (b) bypassing ALL PML cells 0..pml-1 (which removes the left absorbing boundary).
    //
    // The expansion approach is the correct workaround: the sensor at cell `pml` of the
    // expanded domain is at sigma = 0 (first non-PML cell), so no attenuation applies.
    // The residual recon-vs-recon Pearson gap (~0.70 vs KWave.jl's 1.0) reflects the
    // pml×dx spatial offset between the source positions; lifting it to ≥0.99 requires
    // CPML implementation (tracked in project memory as a multi-session effort).
    let expand_x = if grid.nx > 1 { pml } else { 0 };
    let expand_y = if grid.ny > 1 { pml } else { 0 };
    let expand_z = if grid.nz > 1 { pml } else { 0 };
    let expanded_grid = KwaversGrid::new(
        grid.nx + 2 * expand_x,
        grid.ny + 2 * expand_y,
        grid.nz + 2 * expand_z,
        grid.dx,
        grid.dy,
        grid.dz,
    )?;

    let mut p_mask = Array3::<f64>::zeros((expanded_grid.nx, expanded_grid.ny, expanded_grid.nz));
    for row in sensor_positions.outer_iter() {
        let x = row[0] + expand_x as f64 * grid.dx;
        let y = row[1] + expand_y as f64 * grid.dy;
        let z = row[2] + expand_z as f64 * grid.dz;
        let i = (x / grid.dx).round() as isize;
        let j = (y / grid.dy).round() as isize;
        let k = (z / grid.dz).round() as isize;
        if i < 0
            || j < 0
            || k < 0
            || i >= expanded_grid.nx as isize
            || j >= expanded_grid.ny as isize
            || k >= expanded_grid.nz as isize
        {
            return Err(KwaversError::Validation(
                kwavers::core::error::ValidationError::FieldValidation {
                    field: "sensor_positions".to_string(),
                    value: format!("[{x}, {y}, {z}]"),
                    constraint: "must map to a grid node inside the expanded domain".to_string(),
                },
            ));
        }
        let (i, j, k) = (i as usize, j as usize, k as usize);
        if p_mask[[i, j, k]] != 0.0 {
            return Err(KwaversError::Validation(
                kwavers::core::error::ValidationError::FieldValidation {
                    field: "sensor_positions".to_string(),
                    value: format!("duplicate grid node ({i}, {j}, {k})"),
                    constraint: "sensor positions must map to unique grid nodes".to_string(),
                },
            ));
        }
        p_mask[[i, j, k]] = 1.0;
    }

    let mut reversed_signal = sensor_data;
    reversed_signal.invert_axis(Axis(1));

    let grid_source = GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(reversed_signal),
        p_mode: SourceMode::Dirichlet,
        ..GridSource::new_empty()
    };

    let medium = HomogeneousMedium::from_minimal(1000.0, sound_speed, &expanded_grid);
    let dt = 1.0 / sampling_frequency;
    let boundary = if pml > 0 {
        BoundaryConfig::CPML(CPMLConfig::with_thickness(pml))
    } else {
        BoundaryConfig::None
    };

    let config = PSTDConfig {
        nt,
        dt,
        compatibility_mode: CompatibilityMode::Reference,
        boundary,
        sensor_mask: None,
        pml_inside: true,
        ..PSTDConfig::default()
    };

    let mut solver = PSTDSolver::new(config, expanded_grid, &medium, grid_source)?;

    SolverTrait::run(&mut solver, nt)?;
    let pressure = SolverTrait::pressure_field(&solver);
    let mut cropped = Array3::from_shape_fn((grid.nx, grid.ny, grid.nz), |(i, j, k)| {
        pressure[[i + expand_x, j + expand_y, k + expand_z]]
    });
    // Apply the standard Dirichlet half-amplitude compensation (k-Wave
    // convention): the reverse-time enforced-pressure source radiates into
    // both half-spaces; only the inward-traveling wave focuses, so the recon
    // recovers half the original initial pressure and is scaled by 2 here.
    // Signed values are preserved — the non-negativity prior p₀ ≥ 0 is a
    // photoacoustic post-processing choice and is left to the caller because
    // clipping breaks signed-pattern parity against k-Wave's `p_final`.
    cropped.mapv_inplace(|value| 2.0 * value);

    Ok(cropped)
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
