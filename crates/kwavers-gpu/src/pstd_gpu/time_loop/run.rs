//! Top-level `run` entry point: cache management, field zeroing, batch loop, sensor download.
//!
//! SRP: changes when the batch strategy, TDR throttle, or sensor I/O format changes.

use super::super::{
    state::{
        PstdFinalFields, PstdOutputRequest, PstdRunInputs, PstdRunResult, PstdRunScalars,
        PstdRunState, PstdStateProvider, WgpuPstdState,
    },
    GpuPstdSolver, PstdParams,
};
use super::commands::{PstdCommandProvider, WgpuPstdCommandProvider};
use super::encode::StepCtx;
use super::passes::{PstdPassProvider, WgpuPstdPassProvider};

impl PstdRunScalars {
    #[inline]
    fn total_points(self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn step_context(self, n_sensors: usize, n_src: usize, n_vel_x: usize) -> StepCtx {
        StepCtx {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            dt: self.dt as f32,
            n_sensors: n_sensors as u32,
            nt: self.nt as u32,
            nonlinear: u32::from(self.nonlinear),
            absorbing: u32::from(self.absorbing),
            n_src,
            n_vel_x,
            elem_wg: StepCtx::ceil_div(self.total_points(), 256),
        }
    }

    fn zero_params(self) -> PstdParams {
        PstdParams {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            axis: 0,
            n_fft: 0,
            n_batches: 0,
            log2n: 0,
            inverse: 0,
            step: 0,
            dt: self.dt as f32,
            n_sensors: 0,
            nt: self.nt as u32,
            nonlinear: 0,
            absorbing: 0,
        }
    }
}

impl PstdRunState for WgpuPstdState {
    fn run_pstd(&mut self, scalars: PstdRunScalars, inputs: PstdRunInputs<'_>) -> PstdRunResult {
        let n_sensors = inputs.sensor_indices.len();
        let n_src = inputs.source_indices.len();
        let n_vel_x = inputs.vel_x_indices.len();
        let n_src_safe = n_src.max(1);
        let n_vel_safe = n_vel_x.max(1);

        // Sensor indices/staging buffers are invariant across B-mode scan lines.
        // Source/vel buffers are reused; cache hits refresh only the signal tail.
        let cache_valid = self.run_cache.n_sensors == n_sensors
            && self.run_cache.n_src == n_src
            && self.run_cache.n_vel_x == n_vel_x
            && self.run_cache.sensor_indices_buf.is_some();

        if !cache_valid {
            self.build_run_cache(
                scalars.nt,
                inputs.sensor_indices,
                inputs.source_indices,
                inputs.source_signals,
                inputs.vel_x_indices,
                inputs.vel_x_signals,
            );
        } else {
            self.refresh_signal_tails(
                inputs.source_signals,
                inputs.vel_x_signals,
                n_src_safe,
                n_vel_safe,
            );
        }

        if inputs.output_request.includes_final_fields() {
            self.ensure_field_staging_buffer(scalars.total_points());
        }

        let buf_sensor_data = self
            .run_cache
            .sensor_data_buf
            .as_ref()
            .expect("cache populated above");
        let bg_sensor = self
            .run_cache
            .bg_sensor
            .as_ref()
            .expect("cache populated above");
        let bg_sensor_vel = self
            .run_cache
            .bg_sensor_vel
            .as_ref()
            .expect("cache populated above");
        let commands = WgpuPstdCommandProvider::new(self.device(), self.queue());
        let passes = WgpuPstdPassProvider::new(self);

        let sensor_count = n_sensors.max(1);
        let sensor_bytes = (sensor_count * scalars.nt * std::mem::size_of::<f32>()) as u64;
        commands.clear_buffer(buf_sensor_data, sensor_bytes, "clear_sensor_data");

        let elem_wg = StepCtx::ceil_div(scalars.total_points(), 256);
        let zero_params = scalars.zero_params();
        commands.submit_compute_pass("zero_fields", "zero_fields", |cpass| {
            passes.encode_zero_fields(cpass, &zero_params, bg_sensor, elem_wg);
        });

        let ctx = scalars.step_context(n_sensors, n_src, n_vel_x);

        // Batching reduces wgpu API overhead from O(nt) submits to O(nt/STEP_BATCH).
        // Kept at 32 to avoid Windows TDR on long runs.
        const STEP_BATCH: usize = 32;
        let mut cpu_encode_ns: u64 = 0;
        let mut batch_start = 0usize;
        while batch_start < scalars.nt {
            let batch_end = (batch_start + STEP_BATCH).min(scalars.nt);
            let cpu_t0 = std::time::Instant::now();
            commands.submit_compute_passes(
                "pstd_batch",
                "pstd_step",
                batch_start..batch_end,
                |step, cpass| {
                    passes.encode_time_step(cpass, &ctx, bg_sensor, bg_sensor_vel, step as u32);
                },
            );
            cpu_encode_ns += cpu_t0.elapsed().as_nanos() as u64;
            batch_start = batch_end;

            // Bound queued GPU work so the D3D12 driver does not collapse long
            // simulations into one TDR-sized workload.
            if (batch_start / STEP_BATCH).is_multiple_of(16) {
                commands.poll_wait();
            }
        }

        let gpu_t0 = std::time::Instant::now();
        let sensor_data = if n_sensors == 0 {
            Vec::new()
        } else {
            let sensor_bytes = (n_sensors * scalars.nt * std::mem::size_of::<f32>()) as u64;
            let staging = self
                .run_cache
                .staging_buf
                .as_ref()
                .expect("invariant: sensor cache allocates a staging buffer");
            commands.copy_buffer(buf_sensor_data, staging, sensor_bytes, "sensor_copy");
            commands.read_mapped(staging, sensor_bytes)
        };

        let final_fields = if inputs.output_request.includes_final_fields() {
            let field_bytes = (scalars.total_points() * std::mem::size_of::<f32>()) as u64;
            let staging = self
                .run_cache
                .field_staging_buf
                .as_ref()
                .expect("invariant: full-field request allocates a staging buffer");
            let read_field = |field: &wgpu::Buffer, label| {
                commands.copy_buffer(field, staging, field_bytes, label);
                commands.read_mapped(staging, field_bytes)
            };
            Some(PstdFinalFields {
                pressure: read_field(&self.field_buffers.p, "final_pressure_copy"),
                velocity_x: read_field(&self.field_buffers.ux, "final_velocity_x_copy"),
                velocity_y: read_field(&self.field_buffers.uy, "final_velocity_y_copy"),
                velocity_z: read_field(&self.field_buffers.uz, "final_velocity_z_copy"),
            })
        } else {
            if n_sensors == 0 {
                commands.poll_wait();
            }
            None
        };

        let gpu_wait_ns = gpu_t0.elapsed().as_nanos() as u64;
        let _ = (cpu_encode_ns, gpu_wait_ns);
        #[cfg(test)]
        eprintln!(
            "  CPU encode: {:.1}ms total ({:.2}ms/step), GPU poll-wait: {:.1}ms",
            cpu_encode_ns as f64 / 1e6,
            cpu_encode_ns as f64 / 1e6 / scalars.nt as f64,
            gpu_wait_ns as f64 / 1e6,
        );

        PstdRunResult {
            sensor_data,
            final_fields,
        }
    }
}

impl<P> GpuPstdSolver<P>
where
    P: PstdStateProvider,
    P::State: PstdRunState,
{
    /// Run the GPU PSTD time loop.
    ///
    /// Returns the requested host outputs.
    ///
    /// # Arguments
    /// * `sensor_indices` - flat grid indices of sensor points.
    /// * `source_indices` - flat grid indices of pressure source injection points.
    /// * `source_signals` - source pressure amplitude per `(source_pt, step)`.
    /// * `vel_x_indices` - flat grid indices of `ux` velocity source points.
    /// * `vel_x_signals` - `ux` velocity amplitude per `(source_pt, step)`.
    /// # Panics
    /// Panics if the provider run cache is not populated after cache rebuild.
    pub fn run(
        &mut self,
        sensor_indices: &[u32],
        source_indices: &[u32],
        source_signals: &[f32],
        vel_x_indices: &[u32],
        vel_x_signals: &[f32],
        output_request: PstdOutputRequest,
    ) -> PstdRunResult {
        let scalars = PstdRunScalars {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            nt: self.nt,
            dt: self.dt,
            nonlinear: self.nonlinear,
            absorbing: self.absorbing,
        };
        let inputs = PstdRunInputs {
            sensor_indices,
            source_indices,
            source_signals,
            vel_x_indices,
            vel_x_signals,
            output_request,
        };

        self.state.run_pstd(scalars, inputs)
    }
}
