//! Packed source/sensor buffer helpers and run-cache management.
//!
//! SRP: changes when the GPU buffer packing format or cache invalidation policy changes.

use super::super::GpuPstdSolver;
use wgpu::util::DeviceExt;

const EMPTY_STORAGE_BUFFER_U32: [u32; 1] = [0];

// ─── Packed buffer layout ────────────────────────────────────────────────────
//
// `source_data` GPU buffer: `[bitcast<f32>(indices[n_src]) | signals[n_src * nt]]`.
// Indices occupy a stable prefix; signals occupy the tail and are overwritten each
// scan line (cache hit) without reallocating the buffer or rebuilding bind groups.
// Empty source/sensor sets still bind a one-element zero sentinel because WebGPU
// storage buffers must have non-zero size; dispatch parameters carry the real
// count, so shaders do not consume the sentinel when the count is zero.

pub(super) fn packed_signal_len(n_points: usize, signal_len: usize) -> usize {
    n_points.max(1) + signal_len.max(1)
}

/// Rebuild the packed buffer from scratch (cache miss).
pub(super) fn rewrite_packed_source_buffer(
    buffer: &mut Vec<f32>,
    indices: &[u32],
    signals: &[f32],
) {
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

/// Overwrite only the signal tail (cache hit); no reallocation.
pub(super) fn overwrite_packed_signal_tail(
    buffer: &mut Vec<f32>,
    prefix_len: usize,
    signals: &[f32],
) {
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
    /// Allocate all run-scoped GPU buffers and rebuild bind groups (cache miss).
    ///
    /// Called when `n_sensors`, `n_src`, or `n_vel_x` changes between `run` calls.
    /// Updates `self.cache_*` fields and sets the cache-key counters.
    pub(super) fn build_run_cache(
        &mut self,
        sensor_indices: &[u32],
        source_indices: &[u32],
        source_signals: &[f32],
        vel_x_indices: &[u32],
        vel_x_signals: &[f32],
    ) {
        let n_sensors = sensor_indices.len();
        let n_src = source_indices.len();
        let n_vel_x = vel_x_indices.len();
        let sensor_count = n_sensors.max(1);
        let nt = self.nt;

        rewrite_packed_source_buffer(
            &mut self.scratch_source_data,
            source_indices,
            source_signals,
        );
        rewrite_packed_source_buffer(&mut self.scratch_vel_x_data, vel_x_indices, vel_x_signals);

        let si_data: &[u32] = if sensor_indices.is_empty() {
            &EMPTY_STORAGE_BUFFER_U32
        } else {
            sensor_indices
        };

        self.cache_sensor_indices_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("sensor_indices"),
                contents: bytemuck::cast_slice(si_data),
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));

        self.cache_sensor_data_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sensor_data"),
            size: (sensor_count * nt * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.cache_source_data_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("source_data"),
                contents: bytemuck::cast_slice(&self.scratch_source_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));

        self.cache_vel_x_data_buf = Some(self.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("vel_x_data"),
                contents: bytemuck::cast_slice(&self.scratch_vel_x_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            },
        ));

        self.cache_staging_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sensor_staging"),
            size: (sensor_count * nt * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        let buf_si = self.cache_sensor_indices_buf.as_ref().unwrap();
        let buf_sd = self.cache_sensor_data_buf.as_ref().unwrap();
        let buf_src = self.cache_source_data_buf.as_ref().unwrap();
        let buf_vel = self.cache_vel_x_data_buf.as_ref().unwrap();

        self.cache_bg_sensor = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_sensor_run"),
            layout: &self.bgl_sensor,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buf_pml_sgx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_pml_sgy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_pml_sgz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_pml_xyz.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.buf_shifts_all.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_si.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_sd.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_src.as_entire_binding(),
                },
            ],
        }));

        self.cache_bg_sensor_vel =
            Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bg_sensor_vel_run"),
                layout: &self.bgl_sensor,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buf_pml_sgx.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.buf_pml_sgy.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.buf_pml_sgz.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.buf_pml_xyz.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.buf_shifts_all.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: buf_si.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: buf_sd.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: buf_vel.as_entire_binding(),
                    },
                ],
            }));

        self.cache_n_sensors = n_sensors;
        self.cache_n_src = n_src;
        self.cache_n_vel_x = n_vel_x;
    }

    /// Overwrite signal tails in cached GPU buffers (cache hit).
    ///
    /// `write_buffer` enqueues a PCIe upload; the GPU sees updated data before
    /// the first dispatch of the new run.  The stable index prefix is untouched.
    pub(super) fn refresh_signal_tails(
        &mut self,
        source_signals: &[f32],
        vel_x_signals: &[f32],
        n_src_safe: usize,
        n_vel_safe: usize,
    ) {
        overwrite_packed_signal_tail(&mut self.scratch_source_data, n_src_safe, source_signals);
        overwrite_packed_signal_tail(&mut self.scratch_vel_x_data, n_vel_safe, vel_x_signals);
        self.queue.write_buffer(
            self.cache_source_data_buf.as_ref().expect("cache hit"),
            (n_src_safe * std::mem::size_of::<f32>()) as u64,
            bytemuck::cast_slice(&self.scratch_source_data[n_src_safe..]),
        );
        self.queue.write_buffer(
            self.cache_vel_x_data_buf.as_ref().expect("cache hit"),
            (n_vel_safe * std::mem::size_of::<f32>()) as u64,
            bytemuck::cast_slice(&self.scratch_vel_x_data[n_vel_safe..]),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_signal_len_keeps_storage_buffers_non_empty() {
        assert_eq!(packed_signal_len(0, 0), 2);
        assert_eq!(packed_signal_len(3, 0), 4);
        assert_eq!(packed_signal_len(0, 5), 6);
        assert_eq!(EMPTY_STORAGE_BUFFER_U32, [0]);
    }

    #[test]
    fn rewrite_packed_source_buffer_preserves_indices_and_signal_tail() {
        let mut buffer = Vec::new();
        rewrite_packed_source_buffer(&mut buffer, &[3, 7], &[0.25, -0.5, 1.0]);

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer[0].to_bits(), 3);
        assert_eq!(buffer[1].to_bits(), 7);
        assert_eq!(&buffer[2..], &[0.25, -0.5, 1.0]);
    }

    #[test]
    fn rewrite_packed_source_buffer_uses_zero_signal_sentinel_for_empty_tail() {
        let mut buffer = vec![99.0];
        rewrite_packed_source_buffer(&mut buffer, &[], &[]);

        assert_eq!(buffer, vec![0.0, 0.0]);
    }

    #[test]
    fn overwrite_packed_signal_tail_keeps_index_prefix_stable() {
        let mut buffer = Vec::new();
        rewrite_packed_source_buffer(&mut buffer, &[11, 13], &[1.0, 2.0]);

        overwrite_packed_signal_tail(&mut buffer, 2, &[3.0, 5.0, 8.0]);

        assert_eq!(buffer[0].to_bits(), 11);
        assert_eq!(buffer[1].to_bits(), 13);
        assert_eq!(&buffer[2..], &[3.0, 5.0, 8.0]);
    }
}
