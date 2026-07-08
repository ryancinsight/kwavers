// recorder/complex/trait_impl.rs - RecorderTrait implementation

use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_field::indices::{LIGHT_IDX, PRESSURE_IDX, TEMPERATURE_IDX};
use kwavers_grid::Grid;
use log::info;
use leto::Array4;

use super::super::traits::RecorderTrait;
use super::recorder::Recorder;

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

        if self.record_pressure {
            self.pressure_sensor_data.reserve(expected_steps);
        }
        if self.record_light {
            self.light_sensor_data.reserve(expected_steps);
        }
        if self.record_temperature {
            self.temperature_sensor_data.reserve(expected_steps);
        }

        Ok(())
    }

    fn record(&mut self, fields: &Array4<f64>, step: usize) -> KwaversResult<()> {
        let time = step as f64 * self.time.dt;
        self.recorded_steps.push(time);

        if self.record_pressure {
            let pressure_field = fields
                .index_axis::<3>(0, PRESSURE_IDX)
                .map_err(|e| KwaversError::InternalError(format!("pressure axis slice failed: {e}")))?
                .to_contiguous();
            let sensor_data = self.sensor.sample(&pressure_field);
            let sensor_data: Vec<f64> = sensor_data
                .into_iter()
                .enumerate()
                .map(|(idx, v)| {
                    v.ok_or_else(|| {
                        KwaversError::Validation(ValidationError::ConstraintViolation {
                            message: format!(
                                "Recorder sampled None for pressure at sensor index {idx}"
                            ),
                        })
                    })
                })
                .collect::<KwaversResult<Vec<f64>>>()?;
            self.pressure_sensor_data.push(sensor_data);
        }

        if self.record_light {
            let light_field = fields
                .index_axis::<3>(0, LIGHT_IDX)
                .map_err(|e| KwaversError::InternalError(format!("light axis slice failed: {e}")))?
                .to_contiguous();
            let sensor_data = self.sensor.sample(&light_field);
            let sensor_data: Vec<f64> = sensor_data
                .into_iter()
                .enumerate()
                .map(|(idx, v)| {
                    v.ok_or_else(|| {
                        KwaversError::Validation(ValidationError::ConstraintViolation {
                            message: format!(
                                "Recorder sampled None for light at sensor index {idx}"
                            ),
                        })
                    })
                })
                .collect::<KwaversResult<Vec<f64>>>()?;
            self.light_sensor_data.push(sensor_data);
        }

        if self.record_temperature {
            let temp_field = fields
                .index_axis::<3>(0, TEMPERATURE_IDX)
                .map_err(|e| KwaversError::InternalError(format!("temperature axis slice failed: {e}")))?
                .to_contiguous();
            let sensor_data = self.sensor.sample(&temp_field);
            let sensor_data: Vec<f64> = sensor_data
                .into_iter()
                .enumerate()
                .map(|(idx, v)| {
                    v.ok_or_else(|| {
                        KwaversError::Validation(ValidationError::ConstraintViolation {
                            message: format!(
                                "Recorder sampled None for temperature at sensor index {idx}"
                            ),
                        })
                    })
                })
                .collect::<KwaversResult<Vec<f64>>>()?;
            self.temperature_sensor_data.push(sensor_data);
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
