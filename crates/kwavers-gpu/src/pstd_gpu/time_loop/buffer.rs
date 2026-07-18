//! Packed source/sensor buffer helpers and run-cache management.
//!
//! SRP: changes when GPU buffer packing or cache invalidation policy changes.

use super::super::pipeline::{
    PstdBindGroupProvider, PstdBufferProvider, WgpuPstdBindGroupFactory, WgpuPstdBufferFactory,
};
use super::super::state::WgpuPstdState;
use super::commands::{PstdCommandProvider, WgpuPstdCommandProvider};

const EMPTY_STORAGE_BUFFER_U32: [u32; 1] = [0];

pub(super) fn packed_signal_len(n_points: usize, signal_len: usize) -> usize {
    n_points.max(1) + signal_len.max(1)
}

/// Return `(sensor_trace_len, total_output_len)` for one run-cache layout.
pub(super) fn output_storage_lengths(
    n_sensors: usize,
    nt: usize,
    total_points: usize,
    records_peak_pressure: bool,
) -> (usize, usize) {
    let sensor_trace_len = n_sensors.max(1) * nt;
    let total_output_len = sensor_trace_len
        + if records_peak_pressure {
            total_points
        } else {
            0
        };
    (sensor_trace_len, total_output_len)
}

/// Rebuild the packed buffer from scratch for a cache miss.
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

/// Overwrite only the signal tail for a cache hit.
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

impl WgpuPstdState {
    pub(super) fn ensure_field_staging_buffer(&mut self, total_points: usize) {
        if self.run_cache.field_staging_buf.is_none() {
            let buffers = WgpuPstdBufferFactory::new(self.context.device());
            self.run_cache.field_staging_buf =
                Some(buffers.map_read_buffer::<f32>(total_points, "final_field_staging"));
        }
    }

    pub(super) fn build_run_cache(
        &mut self,
        nt: usize,
        total_points: usize,
        records_peak_pressure: bool,
        sensor_indices: &[u32],
        source_indices: &[u32],
        source_signals: &[f32],
        vel_x_indices: &[u32],
        vel_x_signals: &[f32],
    ) {
        let n_sensors = sensor_indices.len();
        let n_src = source_indices.len();
        let n_vel_x = vel_x_indices.len();
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
        let buffers = WgpuPstdBufferFactory::new(self.context.device());
        let (sensor_len, output_storage_len) =
            output_storage_lengths(n_sensors, nt, total_points, records_peak_pressure);

        self.run_cache.sensor_indices_buf = Some(buffers.static_storage(si_data, "sensor_indices"));
        self.run_cache.sensor_data_buf =
            Some(buffers.read_write_storage::<f32>(output_storage_len, "sensor_data"));
        self.run_cache.source_data_buf =
            Some(buffers.upload_storage(&self.scratch_source_data, "source_data"));
        self.run_cache.vel_x_data_buf =
            Some(buffers.upload_storage(&self.scratch_vel_x_data, "vel_x_data"));
        self.run_cache.staging_buf =
            Some(buffers.map_read_buffer::<f32>(sensor_len, "sensor_staging"));

        let buf_si = self.run_cache.sensor_indices_buf.as_ref().unwrap();
        let buf_sd = self.run_cache.sensor_data_buf.as_ref().unwrap();
        let buf_src = self.run_cache.source_data_buf.as_ref().unwrap();
        let buf_vel = self.run_cache.vel_x_data_buf.as_ref().unwrap();

        let bind_groups = WgpuPstdBindGroupFactory::new(self.context.device());
        self.run_cache.bg_sensor = Some(bind_groups.bind_group(
            "bg_sensor_run",
            &self.layouts.sensor,
            [
                &self.pml_shift_buffers.pml_sgx,
                &self.pml_shift_buffers.pml_sgy,
                &self.pml_shift_buffers.pml_sgz,
                &self.pml_shift_buffers.pml_xyz,
                &self.pml_shift_buffers.shifts_all,
                buf_si,
                buf_sd,
                buf_src,
            ],
        ));
        self.run_cache.bg_sensor_vel = Some(bind_groups.bind_group(
            "bg_sensor_vel_run",
            &self.layouts.sensor,
            [
                &self.pml_shift_buffers.pml_sgx,
                &self.pml_shift_buffers.pml_sgy,
                &self.pml_shift_buffers.pml_sgz,
                &self.pml_shift_buffers.pml_xyz,
                &self.pml_shift_buffers.shifts_all,
                buf_si,
                buf_sd,
                buf_vel,
            ],
        ));

        self.run_cache.n_sensors = n_sensors;
        self.run_cache.n_src = n_src;
        self.run_cache.n_vel_x = n_vel_x;
        self.run_cache.output_storage_len = output_storage_len;
        self.run_cache.peak_offset = sensor_len;
        self.run_cache.records_peak_pressure = records_peak_pressure;
    }

    pub(super) fn refresh_signal_tails(
        &mut self,
        source_signals: &[f32],
        vel_x_signals: &[f32],
        n_src_safe: usize,
        n_vel_safe: usize,
    ) {
        overwrite_packed_signal_tail(&mut self.scratch_source_data, n_src_safe, source_signals);
        overwrite_packed_signal_tail(&mut self.scratch_vel_x_data, n_vel_safe, vel_x_signals);
        let commands = WgpuPstdCommandProvider::new(self.device(), self.queue());
        commands.write_buffer(
            self.run_cache.source_data_buf.as_ref().expect("cache hit"),
            (n_src_safe * std::mem::size_of::<f32>()) as u64,
            &self.scratch_source_data[n_src_safe..],
        );
        commands.write_buffer(
            self.run_cache.vel_x_data_buf.as_ref().expect("cache hit"),
            (n_vel_safe * std::mem::size_of::<f32>()) as u64,
            &self.scratch_vel_x_data[n_vel_safe..],
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
    fn peak_output_appends_exactly_one_pressure_volume() {
        assert_eq!(output_storage_lengths(3, 11, 512, false), (33, 33));
        assert_eq!(output_storage_lengths(3, 11, 512, true), (33, 545));
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
