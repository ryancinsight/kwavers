//! Thermal diffusion / Pennes bioheat Python bindings.
//!
//! Wraps `kwavers_solver::forward::thermal_diffusion::ThermalDiffusionSolver`
//! and `kwavers_medium::HomogeneousMedium`.  All physics (Laplacian,
//! bioheat perfusion, CEM43 dose) are delegated to the kwavers core; this
//! module only handles PyO3 marshalling, unit conversion (°C ↔ K), and sensor
//! extraction (Python-specific concern with no kwavers equivalent).
//!
//! # Temperature convention
//!
//! The Python API mirrors k-Wave MATLAB/Python (°C throughout). The kwavers
//! `ThermalDiffusionSolver` stores temperatures in Kelvin (the
//! `ThermalDoseCalculator` converts via `T_C = T_K − 273.15` at line 51 of
//! `physics/thermal/diffusion/dose.rs`). Conversions occur at the binding
//! boundary: input °C → K before calling the solver; output K → °C before
//! returning to Python.
//!
//! # External heat source convention
//!
//! Python accepts `heat_source` in W/m³ (same as k-Wave `source.Q`). The
//! binding divides by `ρ·cp` to produce K/s, which is what
//! `ThermalDiffusionSolver::update` expects for its `external_source` argument.

use crate::breast_fwi_bindings::complex_compat::{leto2_to_nd2, leto3_to_nd3, nd_to_leto3};
use leto::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArray3, ToPyArray};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use kwavers_core::constants::fundamental::{DENSITY_TISSUE, SOUND_SPEED_TISSUE};
use kwavers_core::constants::medical::THERMAL_DOSE_REFERENCE_TEMP_C;
use kwavers_core::constants::thermodynamic::{BODY_TEMPERATURE_C, KELVIN_OFFSET_C};
use kwavers_grid::Grid as KwaversGrid;
use kwavers_medium::HomogeneousMedium;
use kwavers_physics::thermal::diffusion::ThermalDiffusionConfig;
use kwavers_solver::forward::thermal_diffusion::ThermalDiffusionSolver;

// ── Defaults (soft tissue, ICRU Report 44) ──────────────────────────────────
const DEFAULT_K: f64 = 0.5; // thermal conductivity [W/(m·K)]
const DEFAULT_RHO: f64 = DENSITY_TISSUE; // density [kg/m³]
const DEFAULT_CP: f64 = 3600.0; // specific heat [J/(kg·K)]
const DEFAULT_WB: f64 = 5e-3; // blood perfusion rate [1/s]
const DEFAULT_RHO_B: f64 = 1050.0; // blood density [kg/m³] — ICRU-44 bioheat value (distinct from DENSITY_BLOOD=1060)
const DEFAULT_CPB: f64 = 3840.0; // blood specific heat [J/(kg·K)]
const DEFAULT_TA_C: f64 = BODY_TEMPERATURE_C; // arterial temperature [°C]

// ── Result ───────────────────────────────────────────────────────────────────

/// Result of a `ThermalSimulation::run()` call.
#[pyclass]
pub struct ThermalResult {
    /// Final temperature field (nx, ny, nz) [°C]
    #[pyo3(get)]
    pub temperature: Py<PyArray3<f64>>,
    /// Temperature time series at sensor positions (n_sensors, time_steps) [°C].
    /// `None` if no sensor mask was provided.
    #[pyo3(get)]
    pub temperature_at_sensors: Option<Py<PyArray2<f64>>>,
    /// CEM43 thermal dose field (nx, ny, nz) [min]. `None` if dose tracking disabled.
    #[pyo3(get)]
    pub thermal_dose: Option<Py<PyArray3<f64>>>,
    /// Time vector (time_steps,) [s]
    #[pyo3(get)]
    pub time: Py<PyArray1<f64>>,
    /// Number of time steps
    #[pyo3(get)]
    pub time_steps: usize,
    /// Time step [s]
    #[pyo3(get)]
    pub dt: f64,
}

#[pymethods]
impl ThermalResult {
    fn __repr__(&self) -> String {
        format!(
            "ThermalResult(time_steps={}, dt={:.3e})",
            self.time_steps, self.dt
        )
    }
}

// ── Simulation ───────────────────────────────────────────────────────────────

/// Standalone thermal diffusion / Pennes bioheat simulation.
///
/// Mirrors `KWaveDiffusion` from k-Wave MATLAB.
///
/// ## Example (Python)
/// ```python
/// sim = ThermalSimulation(
///     nx=64, ny=64, nz=1,
///     dx=5e-4, dy=5e-4, dz=5e-4,
///     thermal_conductivity=0.5,
///     density=1000.0, specific_heat=3600.0,
///     initial_temperature=37.0,
/// )
/// result = sim.run(time_steps=200, dt=0.1)
/// ```
#[pyclass]
pub struct ThermalSimulation {
    nx: usize,
    ny: usize,
    nz: usize,
    dx: f64,
    dy: f64,
    dz: f64,
    // material
    thermal_conductivity: f64,
    density: f64,
    specific_heat: f64,
    // bioheat
    enable_bioheat: bool,
    perfusion_rate: f64,
    blood_density: f64,
    blood_specific_heat: f64,
    arterial_temperature_c: f64, // stored as °C; converted to K when building config
    metabolic_heat: f64,
    // IC (°C)
    initial_temperature_c: f64,
    // options
    track_thermal_dose: bool,
    spatial_order: u8,
}

#[pymethods]
impl ThermalSimulation {
    /// Create a new `ThermalSimulation`.
    ///
    /// Parameters
    /// ----------
    /// nx, ny, nz : grid point counts.
    /// dx, dy, dz : grid spacings [m].
    /// thermal_conductivity : k [W/(m·K)]. Default 0.5 (soft tissue).
    /// density : ρ [kg/m³]. Default 1000.
    /// specific_heat : c_p [J/(kg·K)]. Default 3600.
    /// enable_bioheat : enable Pennes perfusion + metabolic terms. Default False.
    /// perfusion_rate : w_b [1/s]. Default 5e-3.
    /// blood_density : ρ_b [kg/m³]. Default 1050.
    /// blood_specific_heat : c_b [J/(kg·K)]. Default 3840.
    /// arterial_temperature : T_a [°C]. Default 37.
    /// metabolic_heat : Q_m [W/m³]. Default 0.
    /// initial_temperature : T_0 [°C]. Default 37.
    /// track_thermal_dose : compute CEM43 field. Default True.
    /// spatial_order : Laplacian finite-difference order (2 or 4). Default 2.
    #[new]
    #[pyo3(signature = (
        nx, ny, nz, dx, dy, dz,
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
        spatial_order = 2,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
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
        spatial_order: u8,
    ) -> PyResult<Self> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(PyValueError::new_err("Grid dimensions must be > 0"));
        }
        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(PyValueError::new_err("Grid spacings must be > 0"));
        }
        if thermal_conductivity <= 0.0 || density <= 0.0 || specific_heat <= 0.0 {
            return Err(PyValueError::new_err(
                "thermal_conductivity, density, specific_heat must be > 0",
            ));
        }
        Ok(Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
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
            spatial_order,
        })
    }

    /// Run the thermal simulation.
    ///
    /// Parameters
    /// ----------
    /// time_steps : number of time steps.
    /// dt : time step [s].
    /// heat_source : optional ndarray (nx, ny, nz), Q [W/m³], spatially varying
    ///     constant heat source (e.g. acoustic absorption heating).
    /// sensor_mask : optional boolean ndarray (nx, ny, nz); records temperature
    ///     time series [°C] at `True` positions.
    ///
    /// Returns
    /// -------
    /// ThermalResult
    #[pyo3(signature = (time_steps, dt, heat_source = None, sensor_mask = None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        time_steps: usize,
        dt: f64,
        heat_source: Option<PyReadonlyArray3<'py, f64>>,
        sensor_mask: Option<PyReadonlyArray3<'py, bool>>,
    ) -> PyResult<ThermalResult> {
        if time_steps == 0 {
            return Err(PyValueError::new_err("time_steps must be > 0"));
        }
        if dt <= 0.0 {
            return Err(PyValueError::new_err("dt must be > 0"));
        }

        let (nx, ny, nz) = (self.nx, self.ny, self.nz);

        // ── Build kwavers Grid ────────────────────────────────────────────────
        let kgrid = KwaversGrid::new(nx, ny, nz, self.dx, self.dy, self.dz)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // ── Build HomogeneousMedium with correct thermal properties ───────────
        // sound_speed is irrelevant for the thermal solve; use tissue default.
        let mut medium = HomogeneousMedium::new(self.density, SOUND_SPEED_TISSUE, 0.0, 0.0, &kgrid);
        medium
            .set_thermal_properties(self.thermal_conductivity, self.specific_heat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // ── Unit conversion: °C → K ───────────────────────────────────────────
        let initial_temp_k = self.initial_temperature_c + KELVIN_OFFSET_C;
        let arterial_temp_k = self.arterial_temperature_c + KELVIN_OFFSET_C;

        // ── Build ThermalDiffusionConfig ──────────────────────────────────────
        let config = ThermalDiffusionConfig {
            enable_bioheat: self.enable_bioheat,
            perfusion_rate: self.perfusion_rate,
            blood_density: self.blood_density,
            blood_specific_heat: self.blood_specific_heat,
            arterial_temperature: arterial_temp_k,
            enable_hyperbolic: false,
            relaxation_time: 20.0,
            track_thermal_dose: self.track_thermal_dose,
            dose_reference_temperature: THERMAL_DOSE_REFERENCE_TEMP_C,
            spatial_order: self.spatial_order as usize,
        };

        // ── Construct solver and set initial temperature ──────────────────────
        let mut solver = ThermalDiffusionSolver::new(config, &kgrid);
        // Override the default arterial-temperature IC with the user's initial_temperature.
        solver.set_temperature(Array3::from_elem((nx, ny, nz), initial_temp_k));

        // ── Prepare external source in K/s = (Q_acou + Q_m) / (ρ·cp) ─────────
        // ThermalDiffusionSolver::update expects external_source in K/s.
        let rho_cp = self.density * self.specific_heat;
        let q_ks: Option<Array3<f64>> = match heat_source {
            Some(qs) => {
                let arr = qs.as_array();
                if arr.shape() != [nx, ny, nz] {
                    return Err(PyValueError::new_err(format!(
                        "heat_source shape {:?} != ({nx},{ny},{nz})",
                        arr.shape()
                    )));
                }
                let uniform_m = self.metabolic_heat / rho_cp;
                Some(nd_to_leto3(arr.mapv(|q| q / rho_cp + uniform_m)))
            }
            None => {
                if self.metabolic_heat != 0.0 {
                    let val = self.metabolic_heat / rho_cp;
                    Some(Array3::from_elem((nx, ny, nz), val))
                } else {
                    None
                }
            }
        };

        // ── Extract sensor positions (Fortran-order: x-fastest) ──────────────
        let sensor_positions: Vec<(usize, usize, usize)> = match sensor_mask {
            Some(mask) => {
                let m = mask.as_array();
                if m.shape() != [nx, ny, nz] {
                    return Err(PyValueError::new_err(format!(
                        "sensor_mask shape {:?} != ({nx},{ny},{nz})",
                        m.shape()
                    )));
                }
                let mut positions = Vec::new();
                for k in 0..nz {
                    for j in 0..ny {
                        for i in 0..nx {
                            if m[[i, j, k]] {
                                positions.push((i, j, k));
                            }
                        }
                    }
                }
                positions
            }
            None => Vec::new(),
        };
        let n_sensors = sensor_positions.len();
        let mut sensor_data: Array2<f64> = Array2::zeros((n_sensors.max(1), time_steps));

        // ── Time loop ─────────────────────────────────────────────────────────
        for step in 0..time_steps {
            // Record sensor temperatures before this step (k-Wave convention).
            // Convert K → °C at the boundary.
            if n_sensors > 0 {
                let temp = solver.temperature();
                for (s, &(si, sj, sk)) in sensor_positions.iter().enumerate() {
                    sensor_data[[s, step]] = temp[[si, sj, sk]] - KELVIN_OFFSET_C;
                }
            }

            solver
                .update(&medium, &kgrid, dt, q_ks.as_ref().map(|a| a.view()))
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        }

        // ── Build outputs ─────────────────────────────────────────────────────
        // Convert final temperature field K → °C.
        let temp_celsius: Array3<f64> = solver.temperature().mapv(|t| t - KELVIN_OFFSET_C);

        let time_vec = numpy::ndarray::Array1::linspace(dt, dt * time_steps as f64, time_steps);

        let temperature_at_sensors: Option<Py<PyArray2<f64>>> = if n_sensors > 0 {
            let selected = sensor_data
                .slice(&[(0, n_sensors, 1), (0, time_steps, 1)])
                .expect("sensor slice bounds")
                .to_contiguous();
            Some(leto2_to_nd2(selected).to_pyarray(py).into())
        } else {
            None
        };

        let thermal_dose_out: Option<Py<PyArray3<f64>>> = solver
            .thermal_dose()
            .map(|d| leto3_to_nd3(d.to_owned()).to_pyarray(py).into());

        Ok(ThermalResult {
            temperature: leto3_to_nd3(temp_celsius).to_pyarray(py).into(),
            temperature_at_sensors,
            thermal_dose: thermal_dose_out,
            time: time_vec.to_pyarray(py).into(),
            time_steps,
            dt,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ThermalSimulation({}×{}×{}, dx={:.3e}, k={:.3}, rho={:.0}, cp={:.0}, bioheat={})",
            self.nx,
            self.ny,
            self.nz,
            self.dx,
            self.thermal_conductivity,
            self.density,
            self.specific_heat,
            self.enable_bioheat,
        )
    }
}

// ── Module registration ──────────────────────────────────────────────────────
pub fn register_thermal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ThermalSimulation>()?;
    m.add_class::<ThermalResult>()?;
    Ok(())
}
