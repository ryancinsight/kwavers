pub(crate) mod helpers;
mod config;
mod run;
mod solvers;
pub(crate) mod gpu;
mod tests;

pub use gpu::GpuPstdSession;

/// Elastic velocity source bundle: (mask, ux_signal, uy_signal, uz_signal, mode).
pub(crate) type ElasticVelocitySource = Option<(
    ndarray::Array3<bool>,
    Option<ndarray::Array1<f64>>,
    Option<ndarray::Array1<f64>>,
    Option<ndarray::Array1<f64>>,
    String,
)>;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use kwavers::solver::forward::fdtd::config::KSpaceCorrectionMode;
use kwavers::solver::forward::pstd::config::CompatibilityMode;

use crate::grid_py::Grid;
use crate::medium_py::Medium;
use crate::sensor_py::Sensor;
use crate::solver_type_bindings::SolverType;
use crate::source_py::Source;
use crate::transducer_array_py::TransducerArray2D;

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
    pub(crate) grid: Grid,
    pub(crate) medium: Medium,
    pub(crate) sources: Vec<Source>,
    pub(crate) transducers: Vec<TransducerArray2D>,
    pub(crate) sensor: Option<Sensor>,
    pub(crate) transducer_sensor: Option<TransducerArray2D>,
    pub(crate) solver_type: SolverType,
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
    fn set_pml_size_xyz(&mut self, x: usize, y: usize, z: usize) {
        self.pml_size_xyz = Some((x, y, z));
        self.pml_size = Some(x.max(y).max(z));
    }

    /// Set uniform PML absorption factor (equivalent to k-Wave scalar `pml_alpha`, default 2.0).
    fn set_pml_alpha(&mut self, alpha: f64) {
        self.pml_alpha_xyz = Some((alpha, alpha, alpha));
    }

    /// Set per-axis PML absorption factors (equivalent to k-Wave vector `pml_alpha`).
    fn set_pml_alpha_xyz(&mut self, ax: f64, ay: f64, az: f64) {
        self.pml_alpha_xyz = Some((ax, ay, az));
    }

    /// Set whether PML is inside the computational domain.
    fn set_pml_inside(&mut self, inside: bool) {
        self.pml_inside = inside;
    }

    /// Get the current PML inside setting.
    #[getter]
    fn pml_inside(&self) -> bool {
        self.pml_inside
    }

    /// Enable or disable the Westervelt nonlinear acoustic source term.
    fn set_nonlinear(&mut self, enable: bool) {
        self.enable_nonlinear = enable;
    }

    /// Return whether the Westervelt nonlinear term is enabled.
    #[getter]
    fn nonlinear(&self) -> bool {
        self.enable_nonlinear
    }

    /// Enable axisymmetric (CylindricalAS) geometry for 2-D radial simulations.
    fn set_axisymmetric(&mut self, enable: bool) {
        self.axisymmetric = enable;
    }

    /// Return whether axisymmetric geometry is enabled.
    #[getter]
    fn axisymmetric(&self) -> bool {
        self.axisymmetric
    }

    /// Set medium absorption coefficient (k-Wave `medium.alpha_coeff`).
    fn set_alpha_coeff(&mut self, alpha: f64) {
        self.alpha_coeff = alpha;
    }

    /// Set medium absorption power law exponent (k-Wave `medium.alpha_power`).
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
}
