use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use kwavers::domain::grid::Grid as KwaversGrid;
use kwavers::domain::sensor::recorder::config::RecordingMode;
use kwavers::domain::sensor::recorder::fields::{SensorRecordField, SensorRecordSpec};
use kwavers::domain::source::{GridSource, Source as KwaversSource, SourceField};
use kwavers::domain::source::custom::FunctionSource;

use kwavers::domain::source::wavefront::plane_wave::{
    InjectionMode, PlaneWaveConfig, PlaneWaveSource,
};

use std::sync::Arc;

use crate::sensor_py::Sensor;
use crate::transducer_array_py::TransducerArray2D;

use super::super::Simulation;
use super::super::helpers::{SampledSignal, SineSignal};
use super::sources::parse_source_mode;

impl Simulation {
    /// Build `GridSource` and dynamic source list from `self.sources` and `self.transducers`.
    ///
    /// Extracted from `run` to allow checkpoint methods to share source setup.
    /// `c_max` is the maximum sound speed of the medium (used for plane-wave wavelength).
    pub(crate) fn build_sources(
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
                let p_mode = parse_source_mode(&src.source_mode);
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
                let p_mode = parse_source_mode(&src.source_mode);
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
                let p_mode = parse_source_mode(&src.source_mode);
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
                let u_mode = parse_source_mode(&src.source_mode);
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

    pub(crate) fn create_sensor_mask(
        grid: &KwaversGrid,
        sensor: Option<&Sensor>,
        transducer: Option<&TransducerArray2D>,
    ) -> ndarray::Array3<bool> {
        let nx = grid.nx;
        let ny = grid.ny;
        let nz = grid.nz;

        if let Some(trans) = transducer {
            let mut mask = ndarray::Array3::<bool>::from_elem((nx, ny, nz), false);
            let width_pts = (trans.inner.element_width() / grid.dx).round() as isize;
            let length_pts = (trans.inner.element_length() / grid.dz).round() as isize;

            for pos in trans.inner.element_positions() {
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

        if let Some(ref mask) = sensor.mask {
            return mask.clone();
        }

        let mut mask = ndarray::Array3::<bool>::from_elem((nx, ny, nz), false);

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
    pub(crate) fn create_transducer_ordered_indices(
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
    pub(crate) fn record_modes_to_spec(modes: &[String]) -> SensorRecordSpec {
        let mut fields = vec![SensorRecordField::Pressure];
        for s in modes {
            match s.as_str() {
                "p" => {}
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
                _ => {}
            }
        }
        SensorRecordSpec::from_fields(&fields)
    }

    /// Convert k-Wave-style record strings to RecordingMode variants (pressure only).
    pub(crate) fn recording_modes_from_strings(modes: &[String]) -> Vec<RecordingMode> {
        modes
            .iter()
            .filter_map(|s| match s.as_str() {
                "p_max" => Some(RecordingMode::MaxPressure),
                "p_min" => Some(RecordingMode::MinPressure),
                "p_rms" => Some(RecordingMode::RmsPressure),
                "p_final" => Some(RecordingMode::FinalPressure),
                "all" => Some(RecordingMode::AllStatistics),
                _ => None,
            })
            .collect()
    }

    /// Trim the recorder buffer to `Nt` columns aligned with k-Wave's time-axis convention.
    pub(crate) fn trim_initial_recorder_sample(
        recorded_data: ndarray::Array2<f64>,
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

    /// Borrowed-view variant of [`trim_initial_recorder_sample`].
    pub(crate) fn trim_initial_recorder_view(
        recorded_data: ndarray::ArrayView2<'_, f64>,
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

    /// Return the minimum active axis length and admissible CPML thickness.
    pub(crate) fn cpml_thickness_limits(nx: usize, ny: usize, nz: usize) -> (usize, usize) {
        let mut min_dim = usize::MAX;
        for dim in [nx, ny, nz] {
            if dim > 1 {
                min_dim = min_dim.min(dim);
            }
        }
        let min_dim = if min_dim == usize::MAX { 1 } else { min_dim };
        let max_allowed = (min_dim.saturating_sub(2)) / 2;
        let default_thickness = 20_usize.min(max_allowed).max(2);
        (default_thickness, max_allowed)
    }
}
