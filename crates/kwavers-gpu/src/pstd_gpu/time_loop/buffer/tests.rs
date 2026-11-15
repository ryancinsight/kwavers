//! Unit tests for the run-sized buffer inventory.
//! Moved out of `buffer.rs` so the cache builder stays under the
//! 500-line file budget; no behavior change.

use super::super::super::state::PstdOutputRequest;
use super::*;
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

    let result = check_run_buffer_capacity(test_scalars(256, 256, 256, 1), &inputs, &binding_only);
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
