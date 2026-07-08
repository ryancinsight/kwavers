mod checkpoint;
mod config;
pub(crate) mod gpu;
pub(crate) mod helpers;
mod run;
mod solvers;
mod tests;
use crate::breast_fwi_bindings::complex_compat::{nd_to_leto1, nd_to_leto3};
pub use gpu::GpuPstdSession;

/// Elastic velocity source bundle: (mask, ux_signal, uy_signal, uz_signal, mode).
pub(crate) type ElasticVelocitySource = Option<(
    ndarray::Array3<bool>,
    Option<ndarray::Array1<f64>>,
    Option<ndarray::Array1<f64>>,
    Option<ndarray::Array1<f64>>,
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
use kwavers_solver::forward::pstd::extensions::{ElasticPstdSourceMode, ElasticPstdVelocitySource};
use kwavers_source::GridSource;

use crate::config_builders::{
    HelmholtzConfig as PyHelmholtzConfig, NonlinearConfig as PyNonlinearConfig,
    PmlConfig as PyPmlConfig, PoroelasticConfig as PyPoroelasticConfig,
    ThermalConfig as PyThermalConfig,
};
use crate::grid_py::Grid;
use crate::medium_py::Medium;
use crate::sensor_py::Sensor;
use crate::solver_type_bindings::SolverType;
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
    /// Optional acoustic→thermal coupling configuration.
    /// When set, PSTD `run()` drives the coupled thermal loop.
    pub(crate) thermal: Option<KwaversThermalConfig>,
    /// Helmholtz solver frequency override [Hz].
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

    /// Attach a pre-built PmlConfig object.
    ///
    /// Replaces any PML settings previously set via individual setters.
    ///
    /// Parameters
    /// ----------
    /// config : PmlConfig
    ///     PML configuration built with ``PmlConfig().with_size(20).with_alpha(2.0)``.
    ///
    /// Examples
    /// --------
    /// >>> pml = PmlConfig().with_size(20).with_alpha(2.0)
    /// >>> sim.set_pml_config(pml)
    fn set_pml_config(&mut self, config: PyPmlConfig) {
        self.pml_size = config.inner.size;
        self.pml_size_xyz = config.inner.size_xyz;
        self.pml_inside = config.inner.inside;
        self.pml_alpha_xyz = config.inner.alpha_xyz;
        self.pml_config = Some(config.inner);
    }

    /// Attach a pre-built HelmholtzConfig object.
    ///
    /// Parameters
    /// ----------
    /// config : HelmholtzConfig
    ///     Helmholtz configuration built with
    ///     ``HelmholtzConfig().with_frequency(1e6)``.
    fn set_helmholtz_config(&mut self, config: PyHelmholtzConfig) {
        self.helmholtz_frequency = config.inner.frequency;
        self.helmholtz_config = Some(config.inner);
    }

    /// Attach a pre-built NonlinearConfig object.
    ///
    /// Parameters
    /// ----------
    /// config : NonlinearConfig
    ///     Nonlinear configuration built with
    ///     ``NonlinearConfig().with_enabled().with_alpha_coeff(0.75)``.
    fn set_nonlinear_config(&mut self, config: PyNonlinearConfig) {
        self.enable_nonlinear = config.inner.enabled;
        self.alpha_coeff = config.inner.alpha_coeff;
        self.alpha_power = config.inner.alpha_power;
        self.nonlinear_config = Some(config.inner);
    }

    /// Attach a pre-built ThermalConfig object.
    ///
    /// When set, ``Simulation.run()`` drives the coupled acoustic-thermal
    /// loop via the PSTD solver. The result's ``thermal_temperature`` (°C)
    /// and ``thermal_dose`` (CEM43 min) fields are populated.
    ///
    /// Parameters
    /// ----------
    /// config : ThermalConfig
    ///     Thermal coupling configuration built with
    ///     ``ThermalConfig(center_frequency=1e6).with_bioheat()``.
    fn set_thermal_config(&mut self, config: PyThermalConfig) {
        self.thermal = Some(config.inner);
    }

    /// Attach a pre-built PoroelasticConfig object.
    ///
    /// When set, routes Biot poroelastic material properties (porosity,
    /// permeability, tortuosity, fluid density/bulk-modulus/viscosity)
    /// through the solver config instead of falling back to SSOT defaults
    /// derived from the ``Medium`` trait.
    ///
    /// Parameters
    /// ----------
    /// config : PoroelasticConfig
    ///     Poroelastic material configuration built with
    ///     ``PoroelasticConfig().with_porosity(0.3).with_permeability(1e-9)``.
    fn set_poroelastic_config(&mut self, config: PyPoroelasticConfig) {
        self.poroelastic = Some(config.inner);
    }

    /// Remove poroelastic material configuration, reverting to SSOT defaults.
    pub fn clear_poroelastic(&mut self) {
        self.poroelastic = None;
    }

    /// True if a poroelastic material configuration is attached.
    #[getter]
    pub fn has_poroelastic(&self) -> bool {
        self.poroelastic.is_some()
    }

    // ── Legacy thermal setter (backward-compatible) ───────────────────────

    /// Attach acoustic→thermal coupling to this simulation (legacy API).
    ///
    /// When set, ``Simulation.run()`` with a PSTD solver drives the coupled
    /// time loop: acoustic heat deposition Q = 2α·c·e [W/m³] feeds the Pennes
    /// bioheat / thermal diffusion solver every ``n_acoustic_per_thermal`` steps.
    ///
    /// Prefer ``set_thermal_config(ThermalConfig(...).with_bioheat())`` for
    /// new code — it directly constructs a ``ThermalConfig`` config object.
    #[pyo3(signature = (
        center_frequency,
        n_acoustic_per_thermal = 1,
        thermal_conductivity = DEFAULT_K,
        density = DEFAULT_RHO,
        specific_heat = DEFAULT_CP,
        enable_bioheat = false,
        perfusion_rate = DEFAULT_WB,
        blood_density = DEFAULT_RHO_B,
        blood_specific_heat = DEFAULT_CPB,
        arterial_temperature = DEFAULT_TA_C,
        metabolic_heat = 0.0,
        initial_temperature = DEFAULT_TA_C,
        track_thermal_dose = true,
        dt_thermal = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn set_thermal(
        &mut self,
        center_frequency: f64,
        n_acoustic_per_thermal: usize,
        thermal_conductivity: f64,
        density: f64,
        specific_heat: f64,
        enable_bioheat: bool,
        perfusion_rate: f64,
        blood_density: f64,
        blood_specific_heat: f64,
        arterial_temperature: f64,
        metabolic_heat: f64,
        initial_temperature: f64,
        track_thermal_dose: bool,
        dt_thermal: Option<f64>,
    ) -> PyResult<()> {
        if center_frequency <= 0.0 {
            return Err(PyValueError::new_err("center_frequency must be > 0"));
        }
        if n_acoustic_per_thermal == 0 {
            return Err(PyValueError::new_err("n_acoustic_per_thermal must be >= 1"));
        }
        if thermal_conductivity <= 0.0 || density <= 0.0 || specific_heat <= 0.0 {
            return Err(PyValueError::new_err(
                "thermal_conductivity, density, specific_heat must be > 0",
            ));
        }
        self.thermal = Some(KwaversThermalConfig {
            thermal_conductivity,
            density,
            specific_heat,
            enable_bioheat,
            perfusion_rate,
            blood_density,
            blood_specific_heat,
            arterial_temperature_c: arterial_temperature,
            metabolic_heat,
            initial_temperature_c: initial_temperature,
            track_thermal_dose,
            center_frequency_hz: center_frequency,
            n_acoustic_per_thermal,
            dt_thermal,
        });
        Ok(())
    }

    /// Remove thermal coupling, reverting to acoustic-only simulation.
    pub fn clear_thermal(&mut self) {
        self.thermal = None;
    }

    /// True if thermal coupling is configured.
    #[getter]
    pub fn has_thermal(&self) -> bool {
        self.thermal.is_some()
    }

    // ── Legacy setters (backward-compatible, sync to config builders) ─────

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
        if let Some(ref mut cfg) = self.pml_config {
            cfg.size = Some(size);
        }
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
        if let Some(ref mut cfg) = self.pml_config {
            cfg.size_xyz = Some((x, y, z));
            cfg.size = Some(x.max(y).max(z));
        }
    }

    /// Set uniform PML absorption factor (equivalent to k-Wave scalar `pml_alpha`, default 2.0).
    fn set_pml_alpha(&mut self, alpha: f64) {
        self.pml_alpha_xyz = Some((alpha, alpha, alpha));
        if let Some(ref mut cfg) = self.pml_config {
            cfg.alpha_xyz = Some((alpha, alpha, alpha));
        }
    }

    /// Set per-axis PML absorption factors (equivalent to k-Wave vector `pml_alpha`).
    fn set_pml_alpha_xyz(&mut self, ax: f64, ay: f64, az: f64) {
        self.pml_alpha_xyz = Some((ax, ay, az));
        if let Some(ref mut cfg) = self.pml_config {
            cfg.alpha_xyz = Some((ax, ay, az));
        }
    }

    /// Set whether PML is inside the computational domain.
    fn set_pml_inside(&mut self, inside: bool) {
        self.pml_inside = inside;
        if let Some(ref mut cfg) = self.pml_config {
            cfg.inside = inside;
        }
    }

    /// Get the current PML inside setting.
    #[getter]
    fn pml_inside(&self) -> bool {
        self.pml_inside
    }

    /// Enable or disable the Westervelt nonlinear acoustic source term.
    fn set_nonlinear(&mut self, enable: bool) {
        self.enable_nonlinear = enable;
        if let Some(ref mut cfg) = self.nonlinear_config {
            cfg.enabled = enable;
        }
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
        if let Some(ref mut cfg) = self.nonlinear_config {
            cfg.alpha_coeff = alpha;
        }
    }

    /// Set medium absorption power law exponent (k-Wave `medium.alpha_power`).
    fn set_alpha_power(&mut self, power: f64) {
        self.alpha_power = power;
        if let Some(ref mut cfg) = self.nonlinear_config {
            cfg.alpha_power = power;
        }
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

    /// Set the Helmholtz solver frequency for wavenumber control.
    ///
    /// When a frequency is set, the Helmholtz solver derives the wavenumber
    /// independently from the time step `dt`:
    ///
    /// ```text
    /// k = 2π · frequency / cₘₐₓ
    /// ```
    ///
    /// When no frequency is set (the default), the wavenumber is derived from
    /// `dt` as `k = 2π / (cₘₐₓ · dt)`, which is convenient for quick
    /// prototyping but couples the frequency-domain solve to the time step.
    ///
    /// Parameters
    /// ----------
    /// frequency : float
    ///     Source frequency in Hz (e.g., `1e6` for 1 MHz).
    ///
    /// Examples
    /// --------
    /// >>> sim.set_helmholtz_wavenumber(1e6)  # 1 MHz Helmholtz solve
    fn set_helmholtz_wavenumber(&mut self, frequency: f64) -> PyResult<()> {
        if frequency <= 0.0 {
            return Err(PyValueError::new_err(
                "Helmholtz frequency must be positive (Hz)",
            ));
        }
        self.helmholtz_frequency = Some(frequency);
        if let Some(ref mut cfg) = self.helmholtz_config {
            cfg.frequency = Some(frequency);
        }
        Ok(())
    }

    /// Get the currently configured Helmholtz frequency, if any.
    #[getter]
    fn helmholtz_frequency(&self) -> Option<f64> {
        self.helmholtz_frequency
    }

    // ── Run (delegated to kwavers SimulationRunner) ─────────────────────────

    /// Run the simulation.
    ///
    /// Parameters
    /// ----------
    /// time_steps : int
    ///     Number of time steps to simulate.
    /// dt : float, optional
    ///     Time step size [s]. When ``None``, auto-calculated from the CFL
    ///     condition: ``dt = 0.3 * min(dx,dy,dz) / (c_max * sqrt(3))``.
    /// record_start_index : int, default 1
    ///     Start index for recording (k-Wave convention).
    /// record_modes : list[str] or None
    ///     Recording modes: ``["p_max", "p_min", "p_rms", "p_final", "all",
    ///     "ux", "uy", "uz", "ux_non_staggered", ...]``.
    ///
    /// Returns
    /// -------
    /// SimulationResult
    #[pyo3(signature = (time_steps, dt=None, record_start_index=1, record_modes=None))]
    fn run(
        &mut self,
        time_steps: usize,
        dt: Option<f64>,
        record_start_index: usize,
        record_modes: Option<Vec<String>>,
    ) -> PyResult<crate::simulation_result_py::SimulationResult> {
        use crate::simulation_result_py::build_simulation_result;

        // ── Guard: time_steps must be at least 1 ──────────────────────────
        if time_steps == 0 {
            return Err(PyValueError::new_err("time_steps must be at least 1"));
        }

        // ── Fall back to sensor record modes when not explicitly passed ────
        let record_modes = record_modes
            .or_else(|| self.sensor.as_ref().map(|s| s.record_modes.clone()))
            .unwrap_or_default();

        // ── Compute dt from CFL condition when not provided ───────────────
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let cfl = 0.3;
        let dt = dt.unwrap_or_else(|| cfl * dx_min / (c_max * 3.0_f64.sqrt()));

        // ── Resolve solver_type to kwavers SolverType ───────────────────────
        let solver_type = match self.solver_type {
            SolverType::FDTD => kwavers_solver::config::SolverType::FDTD,
            SolverType::PSTD => kwavers_solver::config::SolverType::PSTD,
            SolverType::Hybrid => kwavers_solver::config::SolverType::Hybrid,
            SolverType::Elastic => kwavers_solver::config::SolverType::Elastic,
            SolverType::ElasticPSTD => kwavers_solver::config::SolverType::ElasticPSTD,
            SolverType::Helmholtz => kwavers_solver::config::SolverType::Helmholtz,
            SolverType::BEM => kwavers_solver::config::SolverType::BEM,
            SolverType::DG => kwavers_solver::config::SolverType::DG,
            SolverType::RayleighSommerfeld => {
                kwavers_solver::config::SolverType::RayleighSommerfeld
            }
            SolverType::Poroelastic => kwavers_solver::config::SolverType::Poroelastic,
            SolverType::PstdGpu => kwavers_solver::config::SolverType::PstdGpu,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unsupported solver type: {:?}",
                    other
                )))
            }
        };

        // ── Process sources ─────────────────────────────────────────────────
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let mut grid_source = GridSource::new_empty();
        let mut dynamic_sources: Vec<Box<dyn kwavers_source::Source>> = Vec::new();
        let mut has_mask_source = false;
        let mut elastic_ivp_axis: Option<String> = None;
        let mut elastic_velocity_source = None;

        for src in &self.sources {
            crate::simulation_py::run::sources::process_source_for_run(
                src,
                &self.grid,
                time_steps,
                c_max,
                &mut grid_source,
                &mut dynamic_sources,
                &mut has_mask_source,
                &mut elastic_ivp_axis,
                &mut elastic_velocity_source,
            )?;
        }

        // ── Sensor mask / transducer indices ────────────────────────────────
        let sensor_mask = Simulation::create_sensor_mask(
            &self.grid.inner,
            self.sensor.as_ref(),
            self.transducer_sensor.as_ref(),
        );

        let transducer_ordered_indices = self
            .transducer_sensor
            .as_ref()
            .map(|ts| Simulation::create_transducer_ordered_indices(&self.grid.inner, &ts.inner));

        // ── Config references (moved from config builders, no clone) ────────
        // ── Extract transducer refs for RS solver ───────────────────────────
        let kwavers_transducers: Vec<kwavers_transducer::array_2d::TransducerArray2D> =
            self.transducers.iter().map(|t| t.inner.clone()).collect();

        // ── Convert elastic velocity source ─────────────────────────────────
        let kwavers_elastic_vsrc = elastic_velocity_source.map(|(mask, ux, uy, uz, mode_str)| {
            let mode = match mode_str.as_str() {
                "dirichlet" => ElasticPstdSourceMode::Dirichlet,
                _ => ElasticPstdSourceMode::Additive,
            };
            ElasticPstdVelocitySource {
                mask: nd_to_leto3(mask),
                ux: ux.map(nd_to_leto1),
                uy: uy.map(nd_to_leto1),
                uz: uz.map(nd_to_leto1),
                mode,
            }
        });

        // ── Build request ───────────────────────────────────────────────────
        let req = SimulationRunRequest {
            grid: &self.grid.inner,
            medium: self.medium.inner.as_medium(),
            time_steps,
            dt,
            solver_type,
            pml: self.pml_config.as_ref(),
            helmholtz: self
                .helmholtz_config
                .as_ref()
                .filter(|cfg| cfg.frequency.is_some()),
            nonlinear: self
                .nonlinear_config
                .as_ref()
                .filter(|cfg| cfg.enabled || cfg.alpha_coeff > 0.0),
            thermal: self.thermal.as_ref(),
            poroelastic: self.poroelastic.as_ref(),
            compatibility_mode: self.compatibility_mode,
            kspace_correction: self.kspace_correction.clone(),
            axisymmetric: self.axisymmetric,
            grid_source,
            sensor_mask: Some(sensor_mask),
            transducer_ordered_indices,
            record_modes,
            record_start_index,
            transducers_for_rs: &kwavers_transducers,
            elastic_velocity_source: kwavers_elastic_vsrc,
            elastic_ivp_axis: elastic_ivp_axis.as_deref().map(|s| match s {
                "x" => 0usize,
                "y" => 1usize,
                _ => 2usize,
            }),
        };

        // ── Run ─────────────────────────────────────────────────────────────
        let result = SimulationRunner::run(&req, dynamic_sources).map_err(kwavers_error_to_py)?;

        // ── Build Python result ─────────────────────────────────────────────
        Python::attach(|py| build_simulation_result(py, &result, &self.grid.inner, time_steps, dt))
    }
}
