use std::sync::Arc;

use kwavers::domain::source::custom::FunctionSource;
use kwavers::domain::source::wavefront::plane_wave::{
    InjectionMode, PlaneWaveConfig, PlaneWaveSource,
};
use kwavers::domain::source::{GridSource, Source as KwaversSource, SourceField};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::super::super::helpers::{SampledSignal, SineSignal};
use super::super::super::Simulation;
use super::super::sources::parse_source_mode;

impl Simulation {
    /// Build `GridSource` and dynamic source list from `self.sources` and `self.transducers`.
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
}
