//! Top-level `run` entry point: cache management, field zeroing, batch loop, sensor download.
//!
//! SRP: changes when the batch strategy, TDR throttle, or sensor I/O format changes.

use super::super::{GpuPstdSolver, PstdParams};
use super::encode::StepCtx;

impl GpuPstdSolver {
    /// Run the GPU PSTD time loop.
    ///
    /// Returns `sensor_data` as a flat `Vec<f32>` in row-major order
    /// `[sensor_id * nt + step]`.
    ///
    /// # Arguments
    /// * `sensor_indices` — flat grid indices of sensor points (u32)
    /// * `source_indices` — flat grid indices of pressure source injection points (u32)
    /// * `source_signals` — source pressure amplitude per (source_pt, step),
    ///   flat `[n_src * nt]` row-major
    /// * `vel_x_indices` — flat grid indices of ux velocity source points (u32); pass `&[]` for none
    /// * `vel_x_signals` — ux velocity amplitude per (source_pt, step),
    ///   flat `[n_vel_x * nt]` row-major; pass `&[]` for none
    /// # Panics
    /// - Panics if `cache populated above`.
    /// - Panics if `staging buffer allocated in cache miss path`.
    ///
    pub fn run(
        &mut self,
        sensor_indices: &[u32],
        source_indices: &[u32],
        source_signals: &[f32],
        vel_x_indices: &[u32],
        vel_x_signals: &[f32],
    ) -> Vec<f32> {
        let nx = self.nx;
        let ny = self.ny;
        let nz = self.nz;
        let nt = self.nt;
        let total = nx * ny * nz;
        let n_sensors = sensor_indices.len();
        let n_src = source_indices.len();
        let n_vel_x = vel_x_indices.len();
        let n_src_safe = n_src.max(1);
        let n_vel_safe = n_vel_x.max(1);

        // ── Cache-key check: reallocate buffers only when layout changes ───────
        // Sensor indices/staging buffers are invariant across B-mode scan lines.
        // Source/vel buffers are reused; on cache hits only the signal tail is
        // refreshed via queue.write_buffer() because the packed index prefix is stable.
        let cache_valid = self.cache_n_sensors == n_sensors
            && self.cache_n_src == n_src
            && self.cache_n_vel_x == n_vel_x
            && self.cache_sensor_indices_buf.is_some();

        if !cache_valid {
            self.build_run_cache(
                sensor_indices,
                source_indices,
                source_signals,
                vel_x_indices,
                vel_x_signals,
            );
        } else {
            self.refresh_signal_tails(source_signals, vel_x_signals, n_src_safe, n_vel_safe);
        }

        // Borrow cached buffers and bind groups
        let buf_sensor_data = self
            .cache_sensor_data_buf
            .as_ref()
            .expect("cache populated above");
        let bg_sensor = self
            .cache_bg_sensor
            .as_ref()
            .expect("cache populated above");
        let bg_sensor_vel = self
            .cache_bg_sensor_vel
            .as_ref()
            .expect("cache populated above");

        // Zero-clear sensor_data output buffer (GPU-side clear, no CPU upload)
        {
            let sensor_count = n_sensors.max(1);
            let sensor_bytes = (sensor_count * nt * std::mem::size_of::<f32>()) as u64;
            let mut clear_enc =
                self.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("clear_sensor_data"),
                    });
            clear_enc.clear_buffer(buf_sensor_data, 0, Some(sensor_bytes));
            self.queue.submit(std::iter::once(clear_enc.finish()));
        }

        let elem_wg = StepCtx::ceil_div(total, 256);

        // ── Reset acoustic field buffers to zero (GPU-side) ──────────────────
        // A single GPU compute dispatch zeroes all field buffers in parallel,
        // replacing 7 × write_buffer() PCIe uploads (~112 MB) with a sub-ms
        // GPU memory fill at ~500 GB/s VRAM bandwidth.
        {
            let zero_params = PstdParams {
                nx: nx as u32,
                ny: ny as u32,
                nz: nz as u32,
                axis: 0,
                n_fft: 0,
                n_batches: 0,
                log2n: 0,
                inverse: 0,
                step: 0,
                dt: self.dt as f32,
                n_sensors: 0,
                nt: nt as u32,
                nonlinear: 0,
                absorbing: 0,
            };
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("zero_fields"),
                });
            {
                let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("zero_fields"),
                    timestamp_writes: None,
                });
                self.dispatch(
                    &mut cpass,
                    &zero_params,
                    &self.pipeline_zero_fields,
                    bg_sensor,
                    elem_wg,
                    "zero_fields",
                );
            }
            self.queue.submit(std::iter::once(enc.finish()));
        }

        let ctx = StepCtx {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            dt: self.dt as f32,
            n_sensors: n_sensors as u32,
            nt: nt as u32,
            nonlinear: if self.nonlinear { 1 } else { 0 },
            absorbing: if self.absorbing { 1 } else { 0 },
            n_src,
            n_vel_x,
            elem_wg,
        };

        // ── Encode time steps: batch STEP_BATCH steps per command buffer ─────
        // Batching reduces wgpu API overhead from O(nt) submits to O(nt/STEP_BATCH).
        // Push constants embed params inline — no write_buffer() overhead per step.
        // Kept at 32 to avoid Windows TDR (2s default); 256 * 40ms = 10s/batch
        // caused GPU reset hangs on long runs (e.g. nt≥1700 phased-array).
        const STEP_BATCH: usize = 32;
        let mut cpu_encode_ns: u64 = 0;
        let mut batch_start = 0usize;
        while batch_start < nt {
            let batch_end = (batch_start + STEP_BATCH).min(nt);
            let cpu_t0 = std::time::Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("pstd_batch"),
                });
            for step in batch_start..batch_end {
                let step_u32 = step as u32;

                // ONE compute pass for the entire time step — keeping all dispatches
                // inside a single ComputePass eliminates the ~250 µs D3D12 UAV barrier
                // inserted at every begin/end_compute_pass boundary.
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("pstd_step"),
                    timestamp_writes: None,
                });

                self.encode_velocity_update(&mut cpass, &ctx, bg_sensor, step_u32);
                self.encode_source_injection(&mut cpass, &ctx, bg_sensor, bg_sensor_vel, step_u32);
                self.encode_nonlinear_snapshot(&mut cpass, &ctx, bg_sensor, step_u32);
                self.encode_density_update(&mut cpass, &ctx, bg_sensor, step_u32);
                self.encode_pressure_record(&mut cpass, &ctx, bg_sensor, step_u32);

                drop(cpass);
            }
            self.queue.submit(std::iter::once(encoder.finish()));
            cpu_encode_ns += cpu_t0.elapsed().as_nanos() as u64;
            batch_start = batch_end;
            // Throttle CPU so submissions don't outrun GPU. Without this, a long
            // simulation (>~60s GPU wall time) queues dozens of command buffers
            // that the D3D12 driver treats as one continuous workload and kills
            // via TDR. Polling every ~16 batches (~20s GPU work at 40ms/step ×
            // STEP_BATCH=32) keeps the in-flight queue bounded without adding
            // per-batch sync overhead on short runs.
            if (batch_start / STEP_BATCH).is_multiple_of(16) {
                let _ = self.device.poll(wgpu::PollType::Wait);
            }
        }

        // ── Download sensor data ──────────────────────────────────────────────
        if n_sensors == 0 {
            return Vec::new();
        }

        let sensor_bytes = (n_sensors * nt * std::mem::size_of::<f32>()) as u64;

        let staging = self
            .cache_staging_buf
            .as_ref()
            .expect("staging buffer allocated in cache miss path");

        let mut copy_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sensor_copy"),
            });
        copy_enc.copy_buffer_to_buffer(buf_sensor_data, 0, staging, 0, sensor_bytes);
        self.queue.submit(std::iter::once(copy_enc.finish()));

        let slice = staging.slice(..sensor_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let gpu_t0 = std::time::Instant::now();
        let _ = self.device.poll(wgpu::PollType::Wait);
        let _ = rx.recv();
        let gpu_wait_ns = gpu_t0.elapsed().as_nanos() as u64;
        let _ = (cpu_encode_ns, gpu_wait_ns);
        #[cfg(test)]
        eprintln!(
            "  CPU encode: {:.1}ms total ({:.2}ms/step), GPU poll-wait: {:.1}ms",
            cpu_encode_ns as f64 / 1e6,
            cpu_encode_ns as f64 / 1e6 / nt as f64,
            gpu_wait_ns as f64 / 1e6,
        );

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();

        result
    }
}
