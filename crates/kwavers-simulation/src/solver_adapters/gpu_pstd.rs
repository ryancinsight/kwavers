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
//! | `run(nt)` | Builds GPU arrays, calls `GpuPstdSolver::with_auto_device` + `run()`; stores sensor traces |
//! | `step_forward()` | Returns `Err(FeatureNotAvailable)` — batch-only arch |
//! | `pressure_field()` | Returns the final host-read pressure field after `run()` |
//! | `run_peak_pressure(nt)` | Downloads the provider-computed `max_t |p|` field only |
//! | `peak_pressure_field()` | Returns the latest explicit peak-pressure readback |
//! | `velocity_fields()` | Returns final host-read staggered velocity fields after `run()` |
//! | `recorded_sensor_pressure()` | Returns sensor traces after `run()` completes |
//! | `add_source(Box<dyn Source>)` | Returns an explicit waveform-contract error |
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
//! ## Grid constraints
//!
//! GPU PSTD requires power-of-two dimensions with each axis ≤ 1,024.
//! Construction fails with `KwaversError::InvalidInput` if these are
//! violated.  `SimulationSolverFactory::create_solver(SolverType::PstdGpu,
//! ...)` propagates that error to the caller.

use kwavers_core::constants::numerical::TWO_PI;
mod medium;

use medium::GpuMediumSnapshot;

use kwavers_boundary::cpml::{CPMLConfig, CPMLProfiles};
use kwavers_core::error::{KwaversError, KwaversResult};
use kwavers_gpu::pstd_gpu::{
    prepare_pstd_pressure_source, validate_gpu_pstd_dimensions, AbsorptionArrays, GpuPstdSolver,
    MediumArrays, PmlArrays, PstdFinalFields, PstdOutputRequest, PstdRunInputs, PstdRunResult,
    SolverParams, WgpuPstdStateProvider,
};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::mechanics::absorption::power_law_db_cm_to_np_omega_m;
use kwavers_receiver::GridSensorSet;
use kwavers_solver::config::SolverConfiguration;
use kwavers_solver::feature::SolverFeature;
use kwavers_solver::interface::{Solver, SolverStatistics};
use kwavers_source::{GridSource, Source};
use leto::Array1 as LetoArray1;
use leto::{Array2, Array3};
use std::f64::consts::PI;
use std::time::{Duration, Instant};

/// GPU-resident PSTD adapter implementing the simulation `Solver` trait.
///
/// See module-level documentation for the full architecture and limitation
/// table.
#[derive(Debug)]
pub struct GpuPstdSimulationAdapter {
    pub(self) grid: Grid,
    pub(self) medium: GpuMediumSnapshot,
    pub(self) dt: f64,
    /// Absorption power-law exponent `y` for the fractional-Laplacian model.
    /// Defaults to `1.0` (standard tissue attenuation).
    pub(self) alpha_power: f64,
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
    /// Validates GPU PSTD grid constraints and extracts medium data.  The GPU
    /// device is **not** acquired here; it is acquired on the first `run()` call.
    ///
    /// # Errors
    ///
    /// Returns `KwaversError::InvalidInput` when grid dimensions are not
    /// power-of-two or exceed 1,024 per axis.
    pub fn new<M: Medium>(
        config: &SolverConfiguration,
        grid: &Grid,
        medium: &M,
    ) -> KwaversResult<Self> {
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);

        validate_gpu_pstd_dimensions(nx, ny, nz)?;
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

        let medium_snap = GpuMediumSnapshot::from_medium(grid, medium);
        let shape = (nx, ny, nz);

        Ok(Self {
            grid: grid.clone(),
            medium: medium_snap,
            dt: config.dt,
            alpha_power: 1.0,
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
        if nt == 0 || !self.dt.is_finite() || self.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(format!(
                "GPU PSTD requires time_steps > 0 and finite positive dt; got steps={nt} dt={}",
                self.dt
            )));
        }
        if self.source.u_mask.is_some() || self.source.u_signal.is_some() {
            return Err(KwaversError::FeatureNotAvailable(
                "GpuPstdSimulationAdapter does not expose velocity-source assembly; use run_gpu_pstd_with_outputs for velocity sources"
                    .to_owned(),
            ));
        }

        let nx = self.grid.nx;
        let ny = self.grid.ny;
        let nz = self.grid.nz;
        let total = nx * ny * nz;
        let dt = self.dt;
        let alpha_power = self.alpha_power;

        // ── PML profiles ──────────────────────────────────────────────────────
        let profiles = CPMLProfiles::new(&self.cpml_config, &self.grid, self.medium.c_ref, dt)?;

        let mut pml_sgx = vec![1.0f32; total];
        let mut pml_sgy = vec![1.0f32; total];
        let mut pml_sgz = vec![1.0f32; total];
        let mut pml_x = vec![1.0f32; total];
        let mut pml_y = vec![1.0f32; total];
        let mut pml_z = vec![1.0f32; total];

        if self.pml_inside {
            let exp_half = |sigma: &LetoArray1<f64>| -> Vec<f32> {
                sigma
                    .iter()
                    .map(|&s| (-s * dt * 0.5).exp() as f32)
                    .collect()
            };
            let sgx = exp_half(&profiles.sigma_x_sgx);
            let sgy = exp_half(&profiles.sigma_y_sgy);
            let sgz = exp_half(&profiles.sigma_z_sgz);
            let px = exp_half(&profiles.sigma_x);
            let py = exp_half(&profiles.sigma_y);
            let pz = exp_half(&profiles.sigma_z);

            for ix in 0..nx {
                for iy in 0..ny {
                    for iz in 0..nz {
                        let flat = ix * ny * nz + iy * nz + iz;
                        pml_sgx[flat] = sgx[ix];
                        pml_sgy[flat] = sgy[iy];
                        pml_sgz[flat] = sgz[iz];
                        pml_x[flat] = px[ix];
                        pml_y[flat] = py[iy];
                        pml_z[flat] = pz[iz];
                    }
                }
            }
        }

        // ── Absorption operator arrays ────────────────────────────────────────
        let (nabla1, nabla2, tau_v, eta_v) = if self.medium.has_absorption {
            let dk_x = TWO_PI / (nx as f64 * self.grid.dx);
            let dk_y = TWO_PI / (ny as f64 * self.grid.dy);
            let dk_z = TWO_PI / (nz as f64 * self.grid.dz);
            const SINGULARITY: f64 = 1e-8;
            let y = alpha_power;

            let mut n1 = vec![0.0f32; total];
            let mut n2 = vec![0.0f32; total];
            let mut tau = vec![0.0f32; total];
            let mut eta = vec![0.0f32; total];

            for flat in 0..total {
                let ix = flat / (ny * nz);
                let iy = (flat % (ny * nz)) / nz;
                let iz = flat % nz;

                let kix = if ix <= nx / 2 {
                    ix as f64
                } else {
                    (nx - ix) as f64
                } * dk_x;
                let kiy = if iy <= ny / 2 {
                    iy as f64
                } else {
                    (ny - iy) as f64
                } * dk_y;
                let kiz = if iz <= nz / 2 {
                    iz as f64
                } else {
                    (nz - iz) as f64
                } * dk_z;
                let k_mag = (kix * kix + kiy * kiy + kiz * kiz).sqrt();

                if k_mag > SINGULARITY {
                    n1[flat] = k_mag.powf(y - 2.0) as f32;
                    n2[flat] = k_mag.powf(y - 1.0) as f32;
                }

                let alpha_db_cm = self.medium.absorption_flat[flat] as f64;
                let alpha_0_si = power_law_db_cm_to_np_omega_m(alpha_db_cm, alpha_power);
                let c0 = self.medium.c0_flat[flat] as f64;
                tau[flat] = (-2.0 * alpha_0_si * c0.powf(y - 1.0)) as f32;
                eta[flat] = (2.0 * alpha_0_si * c0.powf(y) * (PI * y / 2.0).tan()) as f32;
            }
            (n1, n2, tau, eta)
        } else {
            (
                vec![0.0f32; total],
                vec![0.0f32; total],
                vec![0.0f32; total],
                vec![0.0f32; total],
            )
        };

        // ── Construct GPU solver ──────────────────────────────────────────────
        let mut gpu_solver = GpuPstdSolver::<WgpuPstdStateProvider>::with_auto_device(
            &self.grid,
            MediumArrays {
                c0_flat: &self.medium.c0_flat,
                rho0_flat: &self.medium.rho0_flat,
            },
            SolverParams {
                dt,
                nt,
                c_ref: self.medium.c_ref,
                nonlinear: self.medium.has_nonlinear,
                absorbing: self.medium.has_absorption,
            },
            PmlArrays {
                x: &pml_x,
                y: &pml_y,
                z: &pml_z,
                sgx: &pml_sgx,
                sgy: &pml_sgy,
                sgz: &pml_sgz,
            },
            AbsorptionArrays {
                bon_a_flat: &self.medium.bon_a_flat,
                nabla1: &nabla1,
                nabla2: &nabla2,
                tau: &tau_v,
                eta: &eta_v,
            },
        )
        .map_err(|e| KwaversError::InvalidInput(format!("GPU device init failed: {e}")))?;

        // ── Sensor indices ────────────────────────────────────────────────────
        let sensor_flat = self.sensor_mask.as_slice().ok_or_else(|| {
            KwaversError::InvalidInput("sensor_mask must be C-contiguous".to_owned())
        })?;
        let sensor_indices: Vec<u32> = sensor_flat
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v { Some(i as u32) } else { None })
            .collect();

        // ── Source indices and signals ────────────────────────────────────────
        let pressure_source =
            prepare_pstd_pressure_source(&self.grid, &self.source, &self.medium.c0_flat, dt, nt)?;

        // ── Run GPU time loop ─────────────────────────────────────────────────
        let t0 = Instant::now();
        let result = gpu_solver.run(PstdRunInputs {
            sensor_indices: &sensor_indices,
            source_indices: &pressure_source.indices,
            source_signals: &pressure_source.signals,
            pressure_source_correction: pressure_source.uses_kspace_correction,
            vel_x_indices: &[],
            vel_x_signals: &[],
            velocity_source_correction: false,
            output_request,
        });
        self.store_sensor_data(&result.sensor_data, sensor_indices.len(), nt)?;
        self.computation_time += t0.elapsed();
        self.current_step += nt;
        Ok(result)
    }
}

impl Solver for GpuPstdSimulationAdapter {
    fn name(&self) -> &str {
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
            memory_usage: self.medium.c0_flat.len() * std::mem::size_of::<f32>() * 6,
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
