//! Packed source/sensor buffer helpers and run-cache management.
//!
//! SRP: changes when GPU buffer packing or cache invalidation policy changes.

use super::super::pipeline::{
    capacity::{check_byte_counts, BufferNeed, StorageBufferLimits, ELEMENT_BYTES},
    PstdBindGroupProvider, PstdBufferProvider, WgpuPstdBindGroupFactory, WgpuPstdBufferFactory,
};
use super::super::state::{PstdRunInputs, PstdRunScalars, WgpuPstdState};
use super::commands::{PstdCommandProvider, WgpuPstdCommandProvider};
use kwavers_core::error::KwaversResult;

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

    pub(super) fn build_run_cache(&mut self, scalars: PstdRunScalars, inputs: &PstdRunInputs<'_>) {
        let n_sensors = inputs.sensor_indices.len();
        let records_peak_pressure = inputs.output_request.includes_peak_pressure();
        rewrite_packed_source_buffer(
            &mut self.scratch_source_data,
            inputs.source_indices,
            inputs.source_signals,
        );
        rewrite_packed_source_buffer(
            &mut self.scratch_vel_x_data,
            inputs.vel_x_indices,
            inputs.vel_x_signals,
        );

        let si_data: &[u32] = if inputs.sensor_indices.is_empty() {
            &EMPTY_STORAGE_BUFFER_U32
        } else {
            inputs.sensor_indices
        };
        let buffers = WgpuPstdBufferFactory::new(self.context.device());
        let (sensor_len, output_storage_len) = output_storage_lengths(
            n_sensors,
            scalars.nt,
            scalars.total_points(),
            records_peak_pressure,
        );

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

        self.run_cache.sensor_indices.clear();
        self.run_cache
            .sensor_indices
            .extend_from_slice(inputs.sensor_indices);
        self.run_cache.source_indices.clear();
        self.run_cache
            .source_indices
            .extend_from_slice(inputs.source_indices);
        self.run_cache.vel_x_indices.clear();
        self.run_cache
            .vel_x_indices
            .extend_from_slice(inputs.vel_x_indices);
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

/// Inventory every buffer [`build_run_cache`] and
/// [`ensure_field_staging_buffer`] create, with the same lengths, and the
/// subject naming the run for an overflowing count.
///
/// Lengths mirror `output_storage_lengths` and `packed_signal_len` above;
/// all counts are computed in `u64` with checked arithmetic so a
/// degenerate run length is refused instead of panicking on host-side
/// `usize` overflow. Keep this inventory next to the cache builder: a
/// buffer added there belongs here too.
pub(in crate::pstd_gpu) fn run_buffer_needs(
    scalars: PstdRunScalars,
    inputs: &PstdRunInputs<'_>,
) -> (Vec<BufferNeed>, String) {
    let as_u64 = |value: usize| u64::try_from(value).ok();
    let element_bytes =
        |elements: Option<u64>| elements.and_then(|elements| elements.checked_mul(ELEMENT_BYTES));

    let n_sensors = inputs.sensor_indices.len();
    let sensors_or_one = as_u64(n_sensors.max(1));
    let steps = as_u64(scalars.nt);
    let sensor_traces =
        sensors_or_one.and_then(|sensors| steps.and_then(|nt| sensors.checked_mul(nt)));
    let total_points = as_u64(scalars.nx)
        .and_then(|nx| as_u64(scalars.ny).and_then(|ny| nx.checked_mul(ny)))
        .and_then(|plane| as_u64(scalars.nz).and_then(|nz| plane.checked_mul(nz)));
    let records_peak_pressure = inputs.output_request.includes_peak_pressure();
    let output_storage = sensor_traces.and_then(|traces| {
        if records_peak_pressure {
            total_points.and_then(|points| traces.checked_add(points))
        } else {
            Some(traces)
        }
    });
    let packed = |indices: usize, signals: usize| {
        as_u64(indices.max(1))
            .and_then(|prefix| as_u64(signals.max(1)).and_then(|tail| prefix.checked_add(tail)))
    };

    let mut needs = vec![
        BufferNeed {
            name: "sensor_indices",
            bytes: element_bytes(sensors_or_one),
            storage_bound: true,
        },
        BufferNeed {
            name: "sensor_data",
            bytes: element_bytes(output_storage),
            storage_bound: true,
        },
        BufferNeed {
            name: "source_data",
            bytes: element_bytes(packed(
                inputs.source_indices.len(),
                inputs.source_signals.len(),
            )),
            storage_bound: true,
        },
        BufferNeed {
            name: "vel_x_data",
            bytes: element_bytes(packed(
                inputs.vel_x_indices.len(),
                inputs.vel_x_signals.len(),
            )),
            storage_bound: true,
        },
        BufferNeed {
            name: "sensor_staging",
            bytes: element_bytes(sensor_traces),
            storage_bound: false,
        },
    ];
    if inputs.output_request.includes_final_fields() || records_peak_pressure {
        needs.push(BufferNeed {
            name: "final_field_staging",
            bytes: element_bytes(total_points),
            storage_bound: false,
        });
    }
    let subject = format!("of a run with {n_sensors} sensors and {} steps", scalars.nt);
    (needs, subject)
}

/// Check the buffers a run sizes before the provider creates any of them.
///
/// # Errors
///
/// [`KwaversError::ResourceLimitExceeded`] naming the buffer, the bytes it
/// needs, and the limit it exceeds; the same variant when an element or
/// byte count of a buffer overflows.
pub(in crate::pstd_gpu) fn check_run_buffer_capacity(
    scalars: PstdRunScalars,
    inputs: &PstdRunInputs<'_>,
    limits: &StorageBufferLimits,
) -> KwaversResult<()> {
    let (needs, subject) = run_buffer_needs(scalars, inputs);
    check_byte_counts(&needs, limits, &subject)
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

    use super::super::super::state::PstdOutputRequest;
    use kwavers_core::error::KwaversError;

    fn test_scalars(nx: usize, ny: usize, nz: usize, nt: usize) -> PstdRunScalars {
        PstdRunScalars {
            nx,
            ny,
            nz,
            nt,
            dt: 1.0e-7,
            nonlinear: false,
            absorbing: false,
        }
    }

    fn test_inputs<'a>(
        sensor_indices: &'a [u32],
        source_indices: &'a [u32],
        source_signals: &'a [f32],
        vel_x_indices: &'a [u32],
        vel_x_signals: &'a [f32],
        output_request: PstdOutputRequest,
    ) -> PstdRunInputs<'a> {
        PstdRunInputs {
            sensor_indices,
            source_indices,
            source_signals,
            pressure_source_correction: false,
            vel_x_indices,
            vel_x_signals,
            velocity_source_correction: false,
            output_request,
        }
    }

    fn refusal(result: KwaversResult<()>) -> String {
        match result {
            Err(KwaversError::ResourceLimitExceeded { message }) => message,
            other => panic!("expected ResourceLimitExceeded, got {other:?}"),
        }
    }

    fn need_summary(needs: &[BufferNeed]) -> Vec<(&'static str, Option<u64>, bool)> {
        needs
            .iter()
            .map(|need| (need.name, need.bytes, need.storage_bound))
            .collect()
    }

    #[test]
    fn run_buffer_inventory_lists_every_run_sized_buffer() {
        let sensor_indices = [0, 1, 2];
        let source_indices = [0, 1];
        let source_signals = [0.5; 5];
        let inputs = test_inputs(
            &sensor_indices,
            &source_indices,
            &source_signals,
            &[],
            &[],
            PstdOutputRequest::sensor_traces(),
        );

        let (needs, subject) = run_buffer_needs(test_scalars(2, 2, 2, 11), &inputs);

        assert_eq!(
            need_summary(&needs),
            [
                ("sensor_indices", Some(12), true),
                ("sensor_data", Some(132), true),
                ("source_data", Some(28), true),
                ("vel_x_data", Some(8), true),
                ("sensor_staging", Some(132), false),
            ]
        );
        assert_eq!(subject, "of a run with 3 sensors and 11 steps");
    }

    #[test]
    fn run_buffer_inventory_adds_field_staging_for_peak_and_final_fields() {
        let sensor_indices = [0, 1, 2];
        let inputs = test_inputs(
            &sensor_indices,
            &[0, 1],
            &[0.5; 5],
            &[],
            &[],
            PstdOutputRequest::with_final_fields_and_peak_pressure(),
        );

        let (needs, _) = run_buffer_needs(test_scalars(2, 2, 2, 11), &inputs);

        assert_eq!(
            need_summary(&needs),
            [
                ("sensor_indices", Some(12), true),
                ("sensor_data", Some(164), true),
                ("source_data", Some(28), true),
                ("vel_x_data", Some(8), true),
                ("sensor_staging", Some(132), false),
                ("final_field_staging", Some(32), false),
            ]
        );
    }

    /// 4096 sensors x 8193 steps x 4 bytes = 134234112, over the 128 MiB
    /// device buffer limit the provider requests.
    #[test]
    fn run_sensor_data_beyond_device_capacity_names_bytes_and_limit() {
        let sensor_indices = vec![0u32; 4096];
        let inputs = test_inputs(
            &sensor_indices,
            &[],
            &[],
            &[],
            &[],
            PstdOutputRequest::sensor_traces(),
        );
        let limits = StorageBufferLimits {
            max_buffer_size: 134217728,
            max_storage_buffer_binding_size: 134217728,
        };

        assert_eq!(
            refusal(check_run_buffer_capacity(
                test_scalars(16, 16, 16, 8193),
                &inputs,
                &limits
            )),
            "GPU PSTD buffer `sensor_data` needs 134234112 bytes; the device enables max_buffer_size = 134217728 bytes"
        );
    }

    #[test]
    fn run_at_exactly_device_capacity_is_admitted() {
        let sensor_indices = vec![0u32; 4096];
        let inputs = test_inputs(
            &sensor_indices,
            &[],
            &[],
            &[],
            &[],
            PstdOutputRequest::sensor_traces(),
        );
        let limits = StorageBufferLimits {
            max_buffer_size: 134217728,
            max_storage_buffer_binding_size: 134217728,
        };

        let result = check_run_buffer_capacity(test_scalars(16, 16, 16, 8192), &inputs, &limits);
        assert!(matches!(result, Ok(())), "{result:?}");
    }

    #[test]
    fn run_field_staging_skips_the_binding_limit() {
        let sensor_indices = [0];
        let inputs = test_inputs(
            &sensor_indices,
            &[],
            &[],
            &[],
            &[],
            PstdOutputRequest::with_final_fields(),
        );
        let binding_only = StorageBufferLimits {
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: 8,
        };

        let result =
            check_run_buffer_capacity(test_scalars(256, 256, 256, 1), &inputs, &binding_only);
        assert!(matches!(result, Ok(())), "{result:?}");

        let buffer_only = StorageBufferLimits {
            max_buffer_size: 67108863,
            max_storage_buffer_binding_size: u64::MAX,
        };
        assert_eq!(
            refusal(check_run_buffer_capacity(
                test_scalars(256, 256, 256, 1),
                &inputs,
                &buffer_only
            )),
            "GPU PSTD buffer `final_field_staging` needs 67108864 bytes; the device enables max_buffer_size = 67108863 bytes"
        );
    }

    #[test]
    fn run_count_overflow_is_refused_with_the_run_subject() {
        let sensor_indices = [0];
        let inputs = test_inputs(
            &sensor_indices,
            &[],
            &[],
            &[],
            &[],
            PstdOutputRequest::sensor_traces(),
        );
        let limits = StorageBufferLimits {
            max_buffer_size: u64::MAX,
            max_storage_buffer_binding_size: u64::MAX,
        };

        assert_eq!(
            refusal(check_run_buffer_capacity(
                test_scalars(16, 16, 16, usize::MAX),
                &inputs,
                &limits
            )),
            format!(
                "GPU PSTD buffer `sensor_data` of a run with 1 sensors and {} steps has an element or byte count that overflows",
                usize::MAX
            )
        );
    }
}
