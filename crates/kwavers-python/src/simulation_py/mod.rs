mod checkpoint;
mod config;
mod configuration;
pub(crate) mod gpu;
pub(crate) mod helpers;
mod physics;
mod pml;
mod run;
mod solvers;
mod tests;
pub use gpu::GpuPstdSession;

/// Elastic velocity source bundle: (mask, ux_signal, uy_signal, uz_signal, mode).
pub(crate) type ElasticVelocitySource = Option<(
    leto::Array3<bool>,
    Option<leto::Array1<f64>>,
    Option<leto::Array1<f64>>,
    Option<leto::Array1<f64>>,
    String,
)>;

// ══ Default thermal properties (soft tissue, ICRU Report 44) ══════════════════
const DEFAULT_K: f64 = 0.5;
const DEFAULT_RHO: f64 = 1000.0;
const DEFAULT_CP: f64 = 3600.0;
const DEFAULT_WB: f64 = 5e-3;
const DEFAULT_RHO_B: f64 = 1050.0;
const DEFAULT_CPB: f64 = 3840.0;
const DEFAULT_TA_C: f64 = 37.0; // BODY_TEMPERATURE_C

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use kwavers_simulation::{
    HelmholtzConfig as KwaversHelmholtzConfig, NonlinearConfig as KwaversNonlinearConfig,
    PmlConfig as KwaversPmlConfig, PoroelasticConfig as KwaversPoroelasticConfig,
    SimulationRunRequest, SimulationRunner, ThermalConfig as KwaversThermalConfig,
};
use kwavers_solver::forward::fdtd::config::KSpaceCorrectionMode;
use kwavers_solver::forward::pstd::config::CompatibilityMode;
use kwavers_source::GridSource;

use crate::config_builders::{
    HelmholtzConfig as PyHelmholtzConfig, NonlinearConfig as PyNonlinearConfig,
    PmlConfig as PyPmlConfig, PoroelasticConfig as PyPoroelasticConfig,
    ThermalConfig as PyThermalConfig,
};
use crate::grid_py::Grid;
use crate::medium_py::Medium;
use crate::sensor_py::Sensor;
use crate::solver_type_bindings::{FftBackend, SolverType};
use crate::source_py::Source;
use crate::transducer_array_py::TransducerArray2D;

use self::run::kwavers_error_to_py;

/// Acoustic wave simulation.
///
/// Mathematical Specification:
/// - Acoustic wave equation: ∂²p/∂t² = c²∇²p + source terms
/// - FDTD discretization: 2nd/4th/6th/8th order accurate
/// - Time stepping: explicit Euler with CFL stability
/// - Boundary conditions: CPML (convolutional perfectly matched layers)
///
/// Equivalent to k-Wave's kspaceFirstOrder3D function.
#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct Simulation {
    pub(crate) grid: Grid,
    pub(crate) medium: Medium,
    pub(crate) sources: Vec<Source>,
    pub(crate) transducers: Vec<TransducerArray2D>,
    pub(crate) sensor: Option<Sensor>,
    pub(crate) transducer_sensor: Option<TransducerArray2D>,
    pub(crate) solver_type: SolverType,
    pub(crate) fft_backend: FftBackend,
    pub(crate) kspace_correction: KSpaceCorrectionMode,
    pub(crate) compatibility_mode: CompatibilityMode,
    pub(crate) pml_size: Option<usize>,
    pub(crate) pml_size_xyz: Option<(usize, usize, usize)>,
    pub(crate) pml_inside: bool,
    /// Per-dimension PML absorption factor (k-Wave `pml_alpha`): [x, y, z]
    pub(crate) pml_alpha_xyz: Option<(f64, f64, f64)>,
    /// Enable Westervelt nonlinear source term in FDTD solver
    pub(crate) enable_nonlinear: bool,
    /// Medium absorption coefficient [dB/(MHz^y·cm)] — k-Wave convention (0 = lossless)
    pub(crate) alpha_coeff: f64,
    /// Medium absorption power law exponent (k-Wave default: 1.5 for tissue)
    pub(crate) alpha_power: f64,
    /// Axisymmetric (CylindricalAS) geometry: 2-D simulation in the (axial, radial) plane.
    /// Grid convention: nx=Nz_axial, ny=1, nz=Nr_radial. Only valid for PSTD and FDTD solvers.
    pub(crate) axisymmetric: bool,
    /// Optional acoustic→thermal coupling configuration.
    /// When set, PSTD `run()` drives the coupled thermal loop.
    pub(crate) thermal: Option<KwaversThermalConfig>,
    /// Helmholtz solver frequency override `Hz`.
    /// When set, the wavenumber `k = 2π·f / cₘₐₓ`; when `None` (default),
    /// the wavenumber is derived from `dt` via `k = 2π / (cₘₐₓ · dt)`.
    /// Only used when `solver_type == SolverType::Helmholtz`.
    pub(crate) helmholtz_frequency: Option<f64>,

    /// Poroelastic material configuration.
    /// When set, routes material properties through the Biot solver config.
    /// When `None`, the dispatch derives defaults from the Medium trait.
    pub(crate) poroelastic: Option<KwaversPoroelasticConfig>,

    // ── Config builder objects (replace scattered field-setters) ──────────
    /// PML configuration object.
    pub(crate) pml_config: Option<KwaversPmlConfig>,
    /// Helmholtz frequency-domain config.
    pub(crate) helmholtz_config: Option<KwaversHelmholtzConfig>,
    /// Nonlinear acoustics config.
    pub(crate) nonlinear_config: Option<KwaversNonlinearConfig>,
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
    #[pyo3(signature = (grid, medium, source, sensor, solver=None, fft_backend=None, pml_size=None))]
    fn new(
        grid: Grid,
        medium: Medium,
        source: &Bound<'_, PyAny>,
        sensor: &Bound<'_, PyAny>,
        solver: Option<SolverType>,
        fft_backend: Option<FftBackend>,
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

        // Sync the constructor's `pml_size` into `pml_config` so the run path —
        // which reads `self.pml_config`, not `self.pml_size` — actually honours it.
        // In particular `pml_size = Some(0)` yields `size = Some(0)`, which the
        // dispatch maps to a zero-thickness boundary (`BoundaryConfig::None`),
        // i.e. a transparent/periodic boundary. Leaving it `None` (the previous
        // behaviour) silently fell back to the default ~20-cell absorbing PML
        // regardless of the requested `pml_size`.
        let pml_config = KwaversPmlConfig {
            size: pml_size,
            ..KwaversPmlConfig::default()
        };

        Ok(Simulation {
            grid,
            medium,
            sources,
            transducers,
            sensor: sensor_opt,
            transducer_sensor,
            solver_type: solver.unwrap_or(SolverType::FDTD),
            fft_backend: fft_backend.unwrap_or_default(),
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
            thermal: None,
            helmholtz_frequency: None,
            pml_config: Some(pml_config),
            helmholtz_config: None,
            nonlinear_config: Some(KwaversNonlinearConfig::default()),
            poroelastic: None,
        })
    }

    // ── Config builder setters ────────────────────────────────────────────
}
