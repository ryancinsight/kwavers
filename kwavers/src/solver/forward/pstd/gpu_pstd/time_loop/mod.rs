//! GPU PSTD time-marching loop and internal command-encoding helpers.
//!
//! **SRP boundary**: this module changes when the physics time-stepping
//! algorithm changes (operator ordering, batching strategy, etc.).  It does
//! not change when the bind-group layout, buffer allocation, or medium
//! properties change.

use super::{GpuPstdSolver, PstdParams};
use wgpu::util::DeviceExt;

fn packed_signal_len(n_points: usize, signal_len: usize) -> usize {
    n_points.max(1) + signal_len.max(1)
}

fn rewrite_packed_source_buffer(buffer: &mut Vec<f32>, indices: &[u32], signals: &[f32]) {
    let packed_len = packed_signal_len(indices.len(), signals.len());
    buffer.clear();
    buffer.reserve(packed_len.saturating_sub(buffer.capacity()));
    for &idx in indices {
        buffer.push(f32::from_bits(idx));
    }
    buffer.resize(indices.len().max(1), 0.0);
    if signals.is_empty() {
        buffer.push(0.0);
    } else {
        buffer.extend_from_slice(signals);
    }
}

fn overwrite_packed_signal_tail(buffer: &mut Vec<f32>, prefix_len: usize, signals: &[f32]) {
    let required_len = prefix_len + signals.len().max(1);
    if buffer.len() != required_len {
        buffer.resize(required_len, 0.0);
    }
    if signals.is_empty() {
        buffer[prefix_len] = 0.0;
    } else {
        buffer[prefix_len..prefix_len + signals.len()].copy_from_slice(signals);
    }
}

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
    ///    flat `[n_src * nt]` row-major
    /// * `vel_x_indices` — flat grid indices of ux velocity source points (u32); pass `&[]` for none
    /// * `vel_x_signals` — ux velocity amplitude per (source_pt, step),
    ///    flat `[n_vel_x * nt]` row-major; pass `&[]` for none
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
        // Source/vel buffers are reused and, on cache hits, only the signal tail is
        // refreshed via queue.write_buffer() because the packed index prefix is stable.
        let cache_valid = self.cache_n_sensors == n_sensors
            && self.cache_n_src == n_src
            && self.cache_n_vel_x == n_vel_x
            && self.cache_sensor_indices_buf.is_some();

        if !cache_valid {
            rewrite_packed_source_buffer(&mut self.scratch_source_data, source_indices, source_signals);
            rewrite_packed_source_buffer(&mut self.scratch_vel_x_data, vel_x_indices, vel_x_signals);

            // ── Allocate sensor_indices buffer (stable; only changes when n_sensors changes)
            let placeholder_u32 = [0u32];
            let si_data: &[u32] = if sensor_indices.is_empty() { &placeholder_u32 } else { sensor_indices };
            let sensor_count = n_sensors.max(1);
            self.cache_sensor_indices_buf = Some(
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("sensor_indices"),
                    contents: bytemuck::cast_slice(si_data),
                    usage: wgpu::BufferUsages::STORAGE,
                }),
            );

            // ── Allocate sensor_data output buffer (COPY_SRC|COPY_DST for clear+download)
            self.cache_sensor_data_buf = Some(
                self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("sensor_data"),
                    size: (sensor_count * nt * std::mem::size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );

            // ── Allocate source_data buffer (STORAGE|COPY_DST for write_buffer updates)
            self.cache_source_data_buf = Some(
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("source_data"),
                    contents: bytemuck::cast_slice(&self.scratch_source_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
            );

            // ── Allocate vel_x_data buffer (STORAGE|COPY_DST for write_buffer updates)
            self.cache_vel_x_data_buf = Some(
                self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("vel_x_data"),
                    contents: bytemuck::cast_slice(&self.scratch_vel_x_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                }),
            );

            // ── Allocate download staging buffer
            self.cache_staging_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("sensor_staging"),
                size: (sensor_count * nt * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));

            // ── Build bind groups from newly allocated buffers
            let buf_si = self.cache_sensor_indices_buf.as_ref().expect("just allocated above");
            let buf_sd = self.cache_sensor_data_buf.as_ref().expect("just allocated above");
            let buf_src = self.cache_source_data_buf.as_ref().expect("just allocated above");
            let buf_vel = self.cache_vel_x_data_buf.as_ref().expect("just allocated above");

            self.cache_bg_sensor = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_sensor_run"),
                layout: &self.bgl_sensor,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.buf_pml_sgx.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.buf_pml_sgy.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.buf_pml_sgz.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.buf_pml_xyz.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.buf_shifts_all.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_si.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_sd.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_src.as_entire_binding() },
                ],
            }));

            self.cache_bg_sensor_vel = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_sensor_vel_run"),
                layout: &self.bgl_sensor,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: self.buf_pml_sgx.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.buf_pml_sgy.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.buf_pml_sgz.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.buf_pml_xyz.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.buf_shifts_all.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: buf_si.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: buf_sd.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: buf_vel.as_entire_binding() },
                ],
            }));

            // Update cache size keys
            self.cache_n_sensors = n_sensors;
            self.cache_n_src = n_src;
            self.cache_n_vel_x = n_vel_x;
        } else {
            // ── Cache hit: rewrite only the signal tail (no re-allocation, no bind-group rebuild)
            // write_buffer enqueues a PCIe upload; GPU will see updated data before first dispatch.
            overwrite_packed_signal_tail(&mut self.scratch_source_data, n_src_safe, source_signals);
            overwrite_packed_signal_tail(&mut self.scratch_vel_x_data, n_vel_safe, vel_x_signals);
            self.queue.write_buffer(
                self.cache_source_data_buf.as_ref().expect("cache hit implies buffer allocated"),
                (n_src_safe * std::mem::size_of::<f32>()) as u64,
                bytemuck::cast_slice(&self.scratch_source_data[n_src_safe..]),
            );
            self.queue.write_buffer(
                self.cache_vel_x_data_buf.as_ref().expect("cache hit implies buffer allocated"),
                (n_vel_safe * std::mem::size_of::<f32>()) as u64,
                bytemuck::cast_slice(&self.scratch_vel_x_data[n_vel_safe..]),
            );
        }

        // Borrow cached buffers and bind groups
        let buf_sensor_data = self.cache_sensor_data_buf.as_ref().expect("cache populated above");
        let bg_sensor = self.cache_bg_sensor.as_ref().expect("cache populated above");
        let bg_sensor_vel = self.cache_bg_sensor_vel.as_ref().expect("cache populated above");

        // Zero-clear sensor_data output buffer (GPU-side clear, no CPU upload)
        {
            let sensor_count = n_sensors.max(1);
            let sensor_bytes = (sensor_count * nt * std::mem::size_of::<f32>()) as u64;
            let mut clear_enc = self.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("clear_sensor_data") });
            clear_enc.clear_buffer(buf_sensor_data, 0, Some(sensor_bytes));
            self.queue.submit(std::iter::once(clear_enc.finish()));
        }

        // ── Dispatch helpers ──────────────────────────────────────────────────
        let ceil_div = |n: usize, d: usize| -> u32 { ((n + d - 1) / d) as u32 };
        let elem_wg = ceil_div(total, 256);

        // ── Reset acoustic field buffers to zero (GPU-side) ──────────────────
        // A single GPU compute dispatch zeroes all 7 field buffers in parallel,
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
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("zero_fields") });
            self.dispatch(&mut enc, &zero_params, &self.pipeline_zero_fields, &bg_sensor, elem_wg, "zero_fields");
            self.queue.submit(std::iter::once(enc.finish()));
            // No wait needed — GPU zero runs before first step's dispatch.
        }

        let nxu = nx as u32;
        let nyu = ny as u32;
        let nzu = nz as u32;
        let dtu = self.dt as f32;
        let ntu = nt as u32;
        let ns_u = n_sensors as u32;

        let nl_u = if self.nonlinear { 1u32 } else { 0u32 };
        let abs_u = if self.absorbing { 1u32 } else { 0u32 };

        // Helper macro: build a generic PstdParams with only `axis` varying
        macro_rules! p {
            ($step:expr, $axis:expr) => {
                PstdParams {
                    nx: nxu, ny: nyu, nz: nzu,
                    axis: $axis,
                    n_fft: 0, n_batches: 0, log2n: 0, inverse: 0,
                    step: $step, dt: dtu,
                    n_sensors: ns_u, nt: ntu,
                    nonlinear: nl_u, absorbing: abs_u,
                }
            };
        }

        // ── Encode time steps: batch STEP_BATCH steps per command buffer ─────
        // Batching reduces wgpu API overhead from O(nt) submits to O(nt/STEP_BATCH).
        // Push constants embed params inline — no write_buffer() overhead per step.
        const STEP_BATCH: usize = 32;
        let mut cpu_encode_ns: u64 = 0;
        let mut batch_start = 0usize;
        while batch_start < nt {
            let batch_end = (batch_start + STEP_BATCH).min(nt);
            let cpu_t0 = std::time::Instant::now();
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pstd_batch") });
            for step in batch_start..batch_end {
                let step_u32 = step as u32;

                // ─── VELOCITY UPDATE ──────────────────────────────────────────────
                for ax in 0u32..3u32 {
                    // Each staggered derivative must start from the current pressure
                    // field, not the prior axis's already shifted/IFFT-transformed state.
                    self.dispatch(&mut encoder, &p!(step_u32, 0), &self.pipeline_copy_field_to_k, &bg_sensor, elem_wg, "cp_p");
                    self.fft_3d(&mut encoder, &bg_sensor, step_u32, ns_u);
                    // kspace_shift writes in-place to kspace_re/im — no copy needed
                    self.dispatch(&mut encoder, &p!(step_u32, ax), &self.pipeline_kspace_shift, &bg_sensor, elem_wg, "kshift_v");
                    // IFFT writes to kspace_re — vel_update reads kspace_re directly
                    self.ifft_3d(&mut encoder, &bg_sensor, step_u32, ns_u);
                    self.dispatch(&mut encoder, &p!(step_u32, ax), &self.pipeline_vel_update, &bg_sensor, elem_wg, "vel_upd");
                }

                // ─── SOURCE INJECTION ─────────────────────────────────────────────
                if n_src > 0 {
                    self.dispatch(
                        &mut encoder,
                        &PstdParams {
                            nx: nxu, ny: nyu, nz: nzu,
                            axis: n_src as u32,
                            n_fft: 0, n_batches: 0, log2n: 0, inverse: 0,
                            step: step_u32, dt: dtu,
                            n_sensors: ns_u, nt: ntu,
                            nonlinear: nl_u, absorbing: abs_u,
                        },
                        &self.pipeline_inject_src,
                        &bg_sensor,
                        ceil_div(n_src, 256),
                        "inject",
                    );
                }
                if n_vel_x > 0 {
                    encoder.clear_buffer(&self.buf_kspace_re, 0, None);
                    encoder.clear_buffer(&self.buf_kspace_im, 0, None);
                    self.dispatch(
                        &mut encoder,
                        &PstdParams {
                            nx: nxu, ny: nyu, nz: nzu,
                            axis: n_vel_x as u32,
                            n_fft: 0, n_batches: 0, log2n: 0, inverse: 0,
                            step: step_u32, dt: dtu,
                            n_sensors: ns_u, nt: ntu,
                            nonlinear: nl_u, absorbing: abs_u,
                        },
                        &self.pipeline_inject_vel_x,
                        &bg_sensor_vel,
                        ceil_div(n_vel_x, 256),
                        "inject_vx",
                    );
                    self.fft_3d(&mut encoder, &bg_sensor, step_u32, ns_u);
                    self.dispatch(
                        &mut encoder,
                        &p!(step_u32, 0),
                        &self.pipeline_apply_source_kappa,
                        &bg_sensor,
                        elem_wg,
                        "src_kappa",
                    );
                    self.ifft_3d(&mut encoder, &bg_sensor, step_u32, ns_u);
                    self.dispatch(
                        &mut encoder,
                        &p!(step_u32, 0),
                        &self.pipeline_add_kspace_to_field_ux,
                        &bg_sensor,
                        elem_wg,
                        "add_ux",
                    );
                }

                // ─── DENSITY UPDATE ───────────────────────────────────────────────
                for ax in 0u32..3u32 {
                    let field_sel = ax + 1;
                    self.dispatch(&mut encoder, &p!(step_u32, field_sel), &self.pipeline_copy_field_to_k, &bg_sensor, elem_wg, "cp_u");
                    self.fft_3d(&mut encoder, &bg_sensor, step_u32, ns_u);
                    // kspace_shift writes in-place to kspace_re/im — no copy needed
                    self.dispatch(&mut encoder, &p!(step_u32, ax + 3), &self.pipeline_kspace_shift, &bg_sensor, elem_wg, "kshift_d");
                    // IFFT writes to kspace_re — dens_update reads kspace_re directly
                    self.ifft_3d(&mut encoder, &bg_sensor, step_u32, ns_u);
                    self.dispatch(&mut encoder, &p!(step_u32, ax), &self.pipeline_dens_update, &bg_sensor, elem_wg, "dens_upd");
                }

                // ─── ABSORPTION DECAY (optional) ─────────────────────────────────
                // Multiply all density components by exp(-alpha*c0*dt) per voxel.
                // Applied after density update, before pressure EOS.
                if self.absorbing {
                    self.dispatch(&mut encoder, &p!(step_u32, 0), &self.pipeline_absorption, &bg_sensor, elem_wg, "absorb");
                }

                // ─── PRESSURE FROM DENSITY ────────────────────────────────────────
                self.dispatch(&mut encoder, &p!(step_u32, 0), &self.pipeline_pres_density, &bg_sensor, elem_wg, "pres");

                // ─── RECORD SENSORS ───────────────────────────────────────────────
                if n_sensors > 0 {
                    self.dispatch(&mut encoder, &p!(step_u32, 0), &self.pipeline_record, &bg_sensor, ceil_div(n_sensors, 256), "rec");
                }
            } // end per-step encoding (inner loop)
            // Submit batch — GPU executes batch N while CPU encodes batch N+1
            self.queue.submit(std::iter::once(encoder.finish()));
            cpu_encode_ns += cpu_t0.elapsed().as_nanos() as u64;
            batch_start = batch_end;
        } // end batch loop

        // Wait for all GPU work to complete before reading sensor data
        let gpu_t0 = std::time::Instant::now();
        let _ = self.device.poll(wgpu::PollType::Wait);
        let gpu_wait_ns = gpu_t0.elapsed().as_nanos() as u64;
        let _ = (cpu_encode_ns, gpu_wait_ns); // suppress unused warning in non-test builds
        #[cfg(test)]
        eprintln!(
            "  CPU encode: {:.1}ms total ({:.2}ms/step), GPU poll-wait: {:.1}ms",
            cpu_encode_ns as f64 / 1e6,
            cpu_encode_ns as f64 / 1e6 / nt as f64,
            gpu_wait_ns as f64 / 1e6,
        );

        // ── Download sensor data ──────────────────────────────────────────────
        if n_sensors == 0 {
            return Vec::new();
        }

        let sensor_count = n_sensors; // n_sensors > 0 guaranteed by guard above
        let sensor_bytes = (sensor_count * nt * std::mem::size_of::<f32>()) as u64;

        // Reuse cached staging buffer — allocated with sensor_count.max(1) * nt * 4 bytes
        let staging = self.cache_staging_buf.as_ref().expect("staging buffer allocated in cache miss path");

        let mut copy_enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sensor_copy") });
        copy_enc.copy_buffer_to_buffer(buf_sensor_data, 0, staging, 0, sensor_bytes);
        self.queue.submit(std::iter::once(copy_enc.finish()));

        let _ = self.device.poll(wgpu::PollType::Wait);

        let slice = staging.slice(..sensor_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::Wait);
        let _ = rx.recv();

        let mapped = slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&mapped).to_vec();
        drop(mapped);
        staging.unmap();

        result
    }

    // ─── Internal command-encoding helpers ───────────────────────────────────

    /// Encode one compute dispatch using push constants for per-dispatch params.
    ///
    /// Push constants are embedded directly in the command buffer — no write_buffer()
    /// overhead and no coalescing issues. Bind groups: fields(0), kspace(1), sensor(2).
    #[inline]
    pub(super) fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &PstdParams,
        pipeline: &wgpu::ComputePipeline,
        bg_sensor: &wgpu::BindGroup,
        workgroups: u32,
        label: &str,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_push_constants(0, bytemuck::bytes_of(params));
        cpass.set_bind_group(0, &self.bg_fields, &[]);
        cpass.set_bind_group(1, &self.bg_kspace, &[]);
        cpass.set_bind_group(2, bg_sensor, &[]);
        cpass.dispatch_workgroups(workgroups, 1, 1);
    }

    pub(super) fn dispatch_2d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        params: &PstdParams,
        pipeline: &wgpu::ComputePipeline,
        bg_sensor: &wgpu::BindGroup,
        workgroups_x: u32,
        workgroups_y: u32,
        label: &str,
    ) {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        cpass.set_pipeline(pipeline);
        cpass.set_push_constants(0, bytemuck::bytes_of(params));
        cpass.set_bind_group(0, &self.bg_fields, &[]);
        cpass.set_bind_group(1, &self.bg_kspace, &[]);
        cpass.set_bind_group(2, bg_sensor, &[]);
        cpass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
    }

    /// Encode a forward 3D FFT: Z-axis → Y-axis → X-axis.
    pub(super) fn fft_3d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg_sensor: &wgpu::BindGroup,
        step_u32: u32,
        n_sensors: u32,
    ) {
        let nx = self.nx as u32;
        let ny = self.ny as u32;
        let nz = self.nz as u32;
        let dt = self.dt as f32;
        let nt = self.nt as u32;
        let nl = if self.nonlinear { 1u32 } else { 0u32 };
        let abs = if self.absorbing { 1u32 } else { 0u32 };
        let fft_dispatch = |encoder: &mut wgpu::CommandEncoder,
                            params: &PstdParams,
                            batches: u32,
                            label: &str| {
            let workgroups_x = batches.min(65535);
            let workgroups_y = batches.div_ceil(65535);
            self.dispatch_2d(encoder, params, &self.pipeline_fft, bg_sensor, workgroups_x, workgroups_y, label);
        };
        let p = |axis: u32, n_fft: u32, n_batches: u32, log2n: u32| PstdParams {
            nx, ny, nz, axis, n_fft, n_batches, log2n, inverse: 0,
            step: step_u32, dt, n_sensors, nt, nonlinear: nl, absorbing: abs,
        };
        fft_dispatch(encoder, &p(2, nz, nx * ny, nz.trailing_zeros()), nx * ny, "fft_z");
        fft_dispatch(encoder, &p(1, ny, nx * nz, ny.trailing_zeros()), nx * nz, "fft_y");
        fft_dispatch(encoder, &p(0, nx, ny * nz, nx.trailing_zeros()), ny * nz, "fft_x");
    }

    /// Encode an inverse 3D FFT: X-axis → Y-axis → Z-axis.
    pub(super) fn ifft_3d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bg_sensor: &wgpu::BindGroup,
        step_u32: u32,
        n_sensors: u32,
    ) {
        let nx = self.nx as u32;
        let ny = self.ny as u32;
        let nz = self.nz as u32;
        let dt = self.dt as f32;
        let nt = self.nt as u32;
        let nl = if self.nonlinear { 1u32 } else { 0u32 };
        let abs = if self.absorbing { 1u32 } else { 0u32 };
        let fft_dispatch = |encoder: &mut wgpu::CommandEncoder,
                            params: &PstdParams,
                            batches: u32,
                            label: &str| {
            let workgroups_x = batches.min(65535);
            let workgroups_y = batches.div_ceil(65535);
            self.dispatch_2d(encoder, params, &self.pipeline_fft, bg_sensor, workgroups_x, workgroups_y, label);
        };
        let p = |axis: u32, n_fft: u32, n_batches: u32, log2n: u32| PstdParams {
            nx, ny, nz, axis, n_fft, n_batches, log2n, inverse: 1,
            step: step_u32, dt, n_sensors, nt, nonlinear: nl, absorbing: abs,
        };
        fft_dispatch(encoder, &p(0, nx, ny * nz, nx.trailing_zeros()), ny * nz, "ifft_x");
        fft_dispatch(encoder, &p(1, ny, nx * nz, ny.trailing_zeros()), nx * nz, "ifft_y");
        fft_dispatch(encoder, &p(2, nz, nx * ny, nz.trailing_zeros()), nx * ny, "ifft_z");
    }
}
