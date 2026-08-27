//! GPU PSTD simulation adapter — batch-mode `Solver` trait wrapper.
//!
//! # Architecture
//!
//! `GpuPstdSimulationAdapter` exposes the GPU-resident PSTD acoustic solver
//! through the simulation `Solver` trait.  The GPU solver is architecturally
//! batch-only: all `nt` time steps are encoded into GPU command buffers and
//! submitted in a single blocking call; single-step access is not supported.
//!
//! ## Trait compliance
//!
//! | Method | Behaviour |
//! |---|---|
//! | `run(nt)` | Executes the provider-owned prepared-medium runner; stores sensor traces |
//! | `step_forward()` | Returns `Err(FeatureNotAvailable)` — batch-only arch |
//! | `pressure_field()` | Returns the final host-read pressure field after `run()` |
//! | `run_peak_pressure(nt)` | Downloads the provider-computed `max_t |p|` field only |
//! | `peak_pressure_field()` | Returns the latest explicit peak-pressure readback |
//! | `velocity_fields()` | Returns final host-read staggered velocity fields after `run()` |
//! | `recorded_sensor_pressure()` | Returns sensor traces after `run()` completes |
//! | `add_source(Box<dyn Source>)` | Returns an explicit waveform-contract error; high-level dispatch samples pressure sources before construction |
//! | `add_sensor(&GridSensorSet)` | Converts `GridPoint` list to boolean sensor mask |
//!
//! ## Source signal
//!
//! `Box<dyn Source>` does not encode the source-major sampled waveform that a
//! GPU PSTD batch requires, so [`Solver::add_source`] rejects it rather than
//! broadcasting or discarding an unknown source signal. Callers with a drive
//! waveform configure [`GpuPstdSimulationAdapter::set_grid_source`] before
//! running the batch. The factory still constructs an empty source for
//! source-free initial-condition propagation.
//!
//! `SimulationSolverFactory` constructs this adapter for `SolverType::PSTD`
//! with `FftBackend::Hephaestus` when the `gpu` feature is enabled. Provider
//! acquisition or execution failures are surfaced and never select the Leto
//! path implicitly.

use kwavers_boundary::cpml::CPMLConfig;
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_gpu::pstd_gpu::{
    run_gpu_pstd_with_snapshot_outputs, GpuPstdRunConfig, PstdFinalFields, PstdMediumSnapshot,
    PstdOutputRequest, PstdRunResult,
};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_receiver::GridSensorSet;
use kwavers_solver::config::SolverConfiguration;
use kwavers_solver::feature::SolverFeature;
use kwavers_solver::interface::{Solver, SolverStatistics};
use kwavers_source::{GridSource, Source};
use leto::{Array2, Array3};
use std::time::{Duration, Instant};

/// GPU-resident PSTD adapter implementing the simulation `Solver` trait.
///
/// See module-level documentation for the full architecture and limitation
/// table.
#[derive(Debug)]
pub struct GpuPstdSimulationAdapter {
    pub(self) grid: Grid,
    pub(self) medium: PstdMediumSnapshot,
    pub(self) dt: f64,
    pub(self) cpml_config: CPMLConfig,
    pub(self) pml_inside: bool,
    pub(self) source: GridSource,
    pub(self) sensor_mask: Array3<bool>,
    /// Sensor traces recorded by the most-recent `run()` call.
    pub(self) recorded: Option<Array2<f64>>,
    pub(self) pressure: Array3<f64>,
    pub(self) velocity_x: Array3<f64>,
    pub(self) velocity_y: Array3<f64>,
    pub(self) velocity_z: Array3<f64>,
    /// Provider-computed `max_t |p|` from the most-recent peak-field batch.
    pub(self) peak_pressure: Option<Array3<f64>>,
    pub(self) current_step: usize,
    pub(self) computation_time: Duration,
}

impl GpuPstdSimulationAdapter {
    /// Construct a GPU PSTD adapter.
    ///
    /// Validates scalar solver parameters and extracts medium data. The GPU
    /// device is **not** acquired here; it is acquired on the first `run()` call.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` when the time step is invalid.
    pub fn new<M: Medium>(
        config: &SolverConfiguration,
        grid: &Grid,
        medium: &M,
    ) -> KwaversResult<Self> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        if !config.dt.is_finite() || config.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD requires finite positive dt; got {}",
                config.dt
            )));
        }

        let cpml_config = config
            .absorbing_boundary
            .as_ref()
            .map(|abc| abc.cpml.clone())
            .unwrap_or_else(|| CPMLConfig::with_thickness(20));
        let pml_inside = true; // canonical default; matches run_gpu_pstd

        let medium_snap = PstdMediumSnapshot::from_medium(grid, medium, 0.0, 1.5)?;
        let shape = (nx, ny, nz);

        Ok(Self {
            grid: grid.clone(),
            medium: medium_snap,
            dt: config.dt,
            cpml_config,
            pml_inside,
            source: GridSource::new_empty(),
            sensor_mask: Array3::from_elem(shape, false),
            recorded: None,
            pressure: Array3::zeros(shape),
            velocity_x: Array3::zeros(shape),
            velocity_y: Array3::zeros(shape),
            velocity_z: Array3::zeros(shape),
            peak_pressure: None,
            current_step: 0,
            computation_time: Duration::ZERO,
        })
    }

    /// Set a complete `GridSource` (mask + signal).
    ///
    /// Bypasses the `Solver::add_source` waveform limitation. Call before
    /// `run()`.
    pub fn set_grid_source(&mut self, source: GridSource) {
        self.source = source;
    }

    /// Run one GPU PSTD batch and retain the provider-computed temporal
    /// pressure envelope `max_t |p|`.
    ///
    /// This requests only sensor traces and the peak-pressure field. It does
    /// not download final pressure or velocity frames, because those frames are
    /// not a valid substitute for a transient treatment envelope.
    ///
    /// # Errors
    ///
    /// Returns an error if the provider cannot acquire a compatible
    /// Hephaestus-owned device, the input source is invalid, or the requested
    /// peak readback violates the adapter's grid/value contract.
    pub fn run_peak_pressure(&mut self, num_steps: usize) -> KwaversResult<()> {
        self.peak_pressure = None;
        let result = self.run_gpu_impl(num_steps, PstdOutputRequest::with_peak_pressure())?;
        let peak_pressure = result.peak_pressure.ok_or_else(|| {
            KwaversError::InternalError(
                "GPU PSTD peak-pressure request completed without an envelope readback".to_owned(),
            )
        })?;
        self.store_peak_pressure(peak_pressure)
    }

    /// Return the explicit `max_t |p|` result from the most recent
    /// [`Self::run_peak_pressure`] call.
    #[must_use]
    pub fn peak_pressure_field(&self) -> Option<&Array3<f64>> {
        self.peak_pressure.as_ref()
    }

    fn host_field(
        shape: (usize, usize, usize),
        values: Vec<f32>,
        field_name: &str,
    ) -> KwaversResult<Array3<f64>> {
        if let Some((index, value)) = values
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD {field_name} readback has non-finite value {value} at flat index {index}"
            )));
        }
        Array3::from_shape_vec(shape, values.into_iter().map(f64::from).collect()).map_err(
            |error| {
                KwaversError::InvalidInput(format!(
                    "GPU PSTD {field_name} readback does not match the simulation grid: {error}"
                ))
            },
        )
    }

    fn store_final_fields(&mut self, fields: PstdFinalFields) -> KwaversResult<()> {
        let PstdFinalFields {
            pressure,
            velocity_x,
            velocity_y,
            velocity_z,
        } = fields;
        let shape = (self.grid.nx, self.grid.ny, self.grid.nz);
        let pressure = Self::host_field(shape, pressure, "pressure")?;
        let velocity_x = Self::host_field(shape, velocity_x, "x-velocity")?;
        let velocity_y = Self::host_field(shape, velocity_y, "y-velocity")?;
        let velocity_z = Self::host_field(shape, velocity_z, "z-velocity")?;
        self.pressure = pressure;
        self.velocity_x = velocity_x;
        self.velocity_y = velocity_y;
        self.velocity_z = velocity_z;
        Ok(())
    }

    fn store_peak_pressure(&mut self, peak_pressure: Vec<f32>) -> KwaversResult<()> {
        if let Some((index, value)) = peak_pressure
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value < 0.0)
        {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD peak-pressure readback has negative value {value} at flat index {index}"
            )));
        }
        let shape = (self.grid.nx, self.grid.ny, self.grid.nz);
        self.peak_pressure = Some(Self::host_field(shape, peak_pressure, "peak pressure")?);
        Ok(())
    }

    fn store_sensor_data(
        &mut self,
        sensor_data: &[f32],
        sensor_count: usize,
        time_steps: usize,
    ) -> KwaversResult<()> {
        let expected = sensor_count.checked_mul(time_steps).ok_or_else(|| {
            KwaversError::InvalidInput(format!(
                "GPU PSTD sensor result shape overflows usize: {sensor_count} sensors × {time_steps} steps"
            ))
        })?;
        if sensor_data.len() != expected {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD sensor readback has {} values; expected {expected} for {sensor_count} sensors × {time_steps} steps",
                sensor_data.len()
            )));
        }
        if let Some((index, value)) = sensor_data
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD sensor readback has non-finite value {value} at flat index {index}"
            )));
        }
        self.recorded = Some(
            Array2::from_shape_vec(
                (sensor_count, time_steps),
                sensor_data.iter().copied().map(f64::from).collect(),
            )
            .map_err(|error| {
                KwaversError::InvalidInput(format!(
                    "GPU PSTD sensor readback does not match the simulation grid: {error}"
                ))
            })?,
        );
        Ok(())
    }

    /// Execute one batch and return only the outputs explicitly requested from
    /// the Hephaestus-owned GPU provider.
    fn run_gpu_impl(
        &mut self,
        nt: usize,
        output_request: PstdOutputRequest,
    ) -> KwaversResult<PstdRunResult> {
        if self.source.u_mask.is_some() || self.source.u_signal.is_some() {
            return Err(KwaversError::FeatureNotAvailable(
                "GpuPstdSimulationAdapter does not expose velocity-source assembly; use run_gpu_pstd_with_outputs for velocity sources"
                    .to_owned(),
            ));
        }
        let sensor_count = self.sensor_mask.iter().filter(|&&enabled| enabled).count();
        let started = Instant::now();
        let result = run_gpu_pstd_with_snapshot_outputs(
            &self.grid,
            &self.medium,
            &self.source,
            &self.sensor_mask,
            GpuPstdRunConfig {
                time_steps: nt,
                dt: self.dt,
                nonlinear: true,
                alpha_coeff_db: 0.0,
                alpha_power: 1.5,
                cpml: Some(self.cpml_config.clone()),
                pml_inside: self.pml_inside,
            },
            output_request,
        )?;
        self.store_sensor_data(&result.sensor_data, sensor_count, nt)?;
        self.computation_time += started.elapsed();
        self.current_step += nt;
        Ok(result)
    }
}

impl Solver for GpuPstdSimulationAdapter {
    fn name(&self) -> &'static str {
        "GpuPstd"
    }

    fn initialize(
        &mut self,
        grid: &Grid,
        _medium: &dyn kwavers_medium::Medium,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        if (nx, ny, nz) != (self.grid.nx, self.grid.ny, self.grid.nz) {
            return Err(KwaversError::DimensionMismatch(format!(
                "GpuPstd adapter initialized for ({}×{}×{}), got ({}×{}×{})",
                self.grid.nx, self.grid.ny, self.grid.nz, nx, ny, nz
            )));
        }
        Ok(())
    }

    /// Reject an unsampled source instead of silently dropping its waveform.
    fn add_source(&mut self, _source: Box<dyn Source>) -> KwaversResult<()> {
        Err(KwaversError::FeatureNotAvailable(
            "GpuPstdSimulationAdapter requires a source-major sampled waveform; use set_grid_source"
                .to_owned(),
        ))
    }

    /// Convert `GridSensorSet` points into a boolean volume sensor mask.
    fn add_sensor(&mut self, sensor: &GridSensorSet) -> KwaversResult<()> {
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        for point in sensor.points() {
            if point.i >= nx || point.j >= ny || point.k >= nz {
                return Err(KwaversError::InvalidInput(format!(
                    "GpuPstd sensor point ({}, {}, {}) is outside grid ({}×{}×{})",
                    point.i, point.j, point.k, nx, ny, nz
                )));
            }
            self.sensor_mask[[point.i, point.j, point.k]] = true;
        }
        Ok(())
    }

    /// Execute the full batch of `num_steps` GPU time steps.
    ///
    /// Acquires the wgpu device, allocates GPU buffers, and submits the
    /// encoded command buffer.  Results are available via
    /// [`Self::recorded_sensor_pressure`] immediately after return.
    fn run(&mut self, num_steps: usize) -> KwaversResult<()> {
        self.peak_pressure = None;
        let result = self.run_gpu_impl(num_steps, PstdOutputRequest::with_final_fields())?;
        let final_fields = result.final_fields.ok_or_else(|| {
            KwaversError::InternalError(
                "GPU PSTD final-field request completed without field readback".to_owned(),
            )
        })?;
        self.store_final_fields(final_fields)
    }

    /// GPU PSTD is batch-only; single-step access is not supported.
    fn step_forward(&mut self) -> KwaversResult<()> {
        Err(KwaversError::FeatureNotAvailable(
            "GpuPstdSimulationAdapter is batch-only: call run(nt) instead of step_forward()"
                .to_owned(),
        ))
    }

    /// Returns the final GPU-read pressure field from the most recent batch.
    fn pressure_field(&self) -> &Array3<f64> {
        &self.pressure
    }

    fn velocity_fields(&self) -> (&Array3<f64>, &Array3<f64>, &Array3<f64>) {
        (&self.velocity_x, &self.velocity_y, &self.velocity_z)
    }

    fn recorded_sensor_pressure(&self) -> Option<Array2<f64>> {
        self.recorded.clone()
    }

    fn statistics(&self) -> SolverStatistics {
        let reported_pressure = self.peak_pressure.as_ref().unwrap_or(&self.pressure);
        SolverStatistics {
            total_steps: self.current_step,
            current_step: self.current_step,
            computation_time: self.computation_time,
            memory_usage: self.medium.resident_bytes(),
            max_pressure: reported_pressure
                .iter()
                .fold(0.0_f64, |max_pressure, &pressure| {
                    max_pressure.max(pressure.abs())
                }),
            max_velocity: self
                .velocity_x
                .iter()
                .chain(self.velocity_y.iter())
                .chain(self.velocity_z.iter())
                .fold(0.0_f64, |max_velocity, &velocity| {
                    max_velocity.max(velocity.abs())
                }),
        }
    }

    fn supports_feature(&self, feature: SolverFeature) -> bool {
        matches!(feature, SolverFeature::GpuAcceleration)
    }

    fn enable_feature(&mut self, feature: SolverFeature, enable: bool) -> KwaversResult<()> {
        if enable && !self.supports_feature(feature) {
            return Err(KwaversError::FeatureNotAvailable(format!(
                "GpuPstd adapter does not support {feature:?}"
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
