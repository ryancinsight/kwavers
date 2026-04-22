use std::sync::Arc;

use ndarray::Array1;
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAny;

use crate::bindings::array::TransducerArray2D;
use crate::bindings::{kwavers_error_to_py, Grid, Medium, Sensor, SolverType, Source};
use crate::{SampledSignal, SineSignal};
use kwavers::domain::source::custom::FunctionSource;
use kwavers::domain::source::wavefront::plane_wave::{
    InjectionMode, PlaneWaveConfig, PlaneWaveSource,
};
use kwavers::domain::source::{GridSource, Source as KwaversSource, SourceField};

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
    pub(crate) pml_size: Option<usize>,
    pub(crate) pml_size_xyz: Option<(usize, usize, usize)>,
    pub(crate) pml_inside: bool,
    pub(crate) pml_alpha_xyz: Option<(f64, f64, f64)>,
    pub(crate) enable_nonlinear: bool,
    pub(crate) alpha_coeff: f64,
    pub(crate) alpha_power: f64,
}

#[pymethods]
impl Simulation {
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

        Ok(Self {
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

    fn set_pml_size(&mut self, size: usize) {
        self.pml_size = Some(size);
    }

    #[getter]
    fn pml_size(&self) -> Option<usize> {
        self.pml_size
    }

    fn set_pml_size_xyz(&mut self, x: usize, y: usize, z: usize) {
        self.pml_size_xyz = Some((x, y, z));
        self.pml_size = Some(x.max(y).max(z));
    }

    fn set_pml_alpha(&mut self, alpha: f64) {
        self.pml_alpha_xyz = Some((alpha, alpha, alpha));
    }

    fn set_pml_alpha_xyz(&mut self, ax: f64, ay: f64, az: f64) {
        self.pml_alpha_xyz = Some((ax, ay, az));
    }

    fn set_pml_inside(&mut self, inside: bool) {
        self.pml_inside = inside;
    }

    #[getter]
    fn pml_inside(&self) -> bool {
        self.pml_inside
    }

    fn set_nonlinear(&mut self, enable: bool) {
        self.enable_nonlinear = enable;
    }

    #[getter]
    fn nonlinear(&self) -> bool {
        self.enable_nonlinear
    }

    fn set_alpha_coeff(&mut self, alpha: f64) {
        self.alpha_coeff = alpha;
    }

    fn set_alpha_power(&mut self, power: f64) {
        self.alpha_power = power;
    }

    #[getter]
    fn alpha_coeff(&self) -> f64 {
        self.alpha_coeff
    }

    #[getter]
    fn alpha_power(&self) -> f64 {
        self.alpha_power
    }

    #[pyo3(signature = (time_steps, dt=None))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        time_steps: usize,
        dt: Option<f64>,
    ) -> PyResult<SimulationResult> {
        let c_max = self.medium.inner.as_medium().max_sound_speed();
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let cfl = 0.3;
        let dt_actual = dt.unwrap_or_else(|| cfl * dx_min / (c_max * 3.0_f64.sqrt()));

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
                if signal.len() != time_steps {
                    return Err(PyValueError::new_err(format!(
                        "Signal length {} does not match time_steps {}",
                        signal.len(),
                        time_steps
                    )));
                }
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

        let n_sensors = sensor_data_2d.nrows();
        let time_arr = PyArray1::from_owned_array(
            py,
            Array1::from_iter((0..time_steps).map(|i| i as f64 * dt_actual)),
        )
        .into();

        if n_sensors <= 1 {
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

#[pyclass]
pub struct SimulationResult {
    pub(crate) sensor_data_1d: Option<Py<PyArray1<f64>>>,
    pub(crate) sensor_data_2d: Option<Py<PyArray2<f64>>>,
    #[pyo3(get)]
    pub(crate) time: Py<PyArray1<f64>>,
    #[pyo3(get)]
    pub(crate) shape: (usize, usize, usize),
    #[pyo3(get)]
    pub(crate) sensor_data_shape: (usize, usize),
    #[pyo3(get)]
    pub(crate) time_steps: usize,
    #[pyo3(get)]
    pub(crate) dt: f64,
    #[pyo3(get)]
    pub(crate) final_time: f64,
    #[pyo3(get)]
    pub(crate) p_max: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub(crate) p_min: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub(crate) p_rms: Option<Py<PyArray1<f64>>>,
    #[pyo3(get)]
    pub(crate) p_final: Option<Py<PyArray1<f64>>>,
}

#[pymethods]
impl SimulationResult {
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

    #[getter]
    fn num_sensors(&self) -> usize {
        self.sensor_data_shape.0
    }

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
