// recorder/complex/trait_impl.rs - RecorderTrait implementation

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_field::indices::{LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX};
use kwavers_grid::Grid;
use leto::Array4;
use log::info;

use super::super::config::RecorderChannel;
use super::super::traits::RecorderTrait;
use super::recorder::Recorder;
use crate::{GridSensorSet, SensorTraces};

impl RecorderTrait for Recorder {
    fn initialize(&mut self, grid: &Grid) -> KwaversResult<()> {
        info!(
            "Initializing recorder for grid {}x{}x{}",
            grid.nx, grid.ny, grid.nz
        );

        for p in self.sensor.points() {
            if p.i >= grid.nx || p.j >= grid.ny || p.k >= grid.nz {
                return Err(KwaversError::Validation(
                    ValidationError::ConstraintViolation {
                        message: format!(
                            "Recorder sensor point out of bounds: ({}, {}, {}) for grid ({}, {}, {})",
                            p.i, p.j, p.k, grid.nx, grid.ny, grid.nz
                        ),
                    },
                ));
            }
        }

        let expected_steps = (self.time.duration() / self.time.dt) as usize;
        let expected_snapshots = expected_steps / self.snapshot_interval;

        self.fields_snapshots.reserve(expected_snapshots);
        self.recorded_steps.reserve(expected_steps);

        if self.channels.contains(RecorderChannel::Pressure) {
            self.pressure_sensor_data.reserve_steps(expected_steps);
        }
        if self.channels.contains(RecorderChannel::Light) {
            self.light_sensor_data.reserve_steps(expected_steps);
        }
        if self.channels.contains(RecorderChannel::Temperature) {
            self.temperature_sensor_data.reserve_steps(expected_steps);
        }

        Ok(())
    }

    fn record(&mut self, fields: &Array4<f64>, step: usize) -> KwaversResult<()> {
        let time = step as f64 * self.time.dt;
        self.recorded_steps.push(time);

        for (channel, field_idx, name, traces) in [
            (
                RecorderChannel::Pressure,
                PRESSURE_IDX,
                "pressure",
                &mut self.pressure_sensor_data,
            ),
            (
                RecorderChannel::Light,
                LIGHT_IDX,
                "light",
                &mut self.light_sensor_data,
            ),
            (
                RecorderChannel::Temperature,
                TEMPERATURE_IDX,
                "temperature",
                &mut self.temperature_sensor_data,
            ),
        ] {
            if self.channels.contains(channel) {
                record_channel(&self.sensor, fields, field_idx, name, traces)?;
            }
        }

        self.record_fields(fields, step, time)?;

        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        info!("Finalizing recording");
        self.statistics.print_summary();
        self.save_data()?;
        Ok(())
    }
}

/// Sample one field channel at every sensor and append the row to `traces`.
fn record_channel(
    sensor: &GridSensorSet,
    fields: &Array4<f64>,
    field_idx: usize,
    name: &str,
    traces: &mut SensorTraces,
) -> KwaversResult<()> {
    let field = fields
        .index_axis::<3>(0, field_idx)
        .map_err(|e| KwaversError::InternalError(format!("{name} axis slice failed: {e}")))?
        .to_contiguous();
    traces.try_push_step(
        sensor
            .sample(&field)
            .into_iter()
            .enumerate()
            .map(|(idx, v)| {
                v.ok_or_else(|| {
                    KwaversError::Validation(ValidationError::ConstraintViolation {
                        message: format!("Recorder sampled None for {name} at sensor index {idx}"),
                    })
                })
            }),
    )
}
