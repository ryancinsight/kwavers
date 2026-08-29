//! Top-level `run` entry point: cache management, field zeroing, batch loop, sensor download.
//!
//! SRP: changes when the batch strategy, TDR throttle, or sensor I/O format changes.

use super::super::{
    state::{
        PstdFinalFields, PstdRunInputs, PstdRunResult, PstdRunScalars, PstdRunState,
        PstdStateProvider, WgpuPstdRunCache, WgpuPstdState,
    },
    GpuPstdSolver, PstdParams,
};
use super::commands::{PstdCommandProvider, WgpuPstdCommandProvider};
use super::encode::StepCtx;
use super::passes::{PstdPassProvider, SourceActivity, StepBindGroups, WgpuPstdPassProvider};
use hephaestus_core::{GroupedCommandStream, KernelDevice};

impl PstdRunScalars {
    #[inline]
    pub(super) fn total_points(self) -> usize {
        self.nx * self.ny * self.nz
    }

    fn step_context(self, inputs: &PstdRunInputs<'_>, peak_offset: usize) -> StepCtx {
        StepCtx {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            dt: self.dt as f32,
            n_sensors: inputs.sensor_indices.len() as u32,
            nt: self.nt as u32,
            nonlinear: u32::from(self.nonlinear),
            absorbing: u32::from(self.absorbing),
            peak_offset: peak_offset as u32,
            record_peak_pressure: u32::from(inputs.output_request.includes_peak_pressure()),
            n_src: inputs.source_indices.len(),
            n_vel_x: inputs.vel_x_indices.len(),
            pressure_source_correction: inputs.pressure_source_correction,
            velocity_source_correction: inputs.velocity_source_correction,
            elem_wg: StepCtx::ceil_div(self.total_points(), 256),
        }
    }

    fn zero_params(self) -> PstdParams {
        PstdParams {
            nx: self.nx as u32,
            ny: self.ny as u32,
            nz: self.nz as u32,
            axis: 0,
            step: 0,
            dt: self.dt as f32,
            n_sensors: 0,
            nt: self.nt as u32,
            nonlinear: 0,
            absorbing: 0,
            peak_offset: 0,
            record_peak_pressure: 0,
        }
    }
}

/// Mark each time step where any source row has a nonzero amplitude.
///
/// An all-zero source step is an identity operation. Omitting its clear,
/// injection, optional spectral correction, and addition preserves the PSTD
/// state while avoiding unnecessary full-volume transforms for finite bursts.
fn mark_active_source_steps(active: &mut [bool], signals: &[f32], source_count: usize) {
    debug_assert!(!active.is_empty());
    active.fill(false);
    let time_steps = active.len();
    for source_signal in signals.chunks_exact(time_steps).take(source_count) {
        for (is_active, &amplitude) in active.iter_mut().zip(source_signal) {
            *is_active |= amplitude != 0.0;
        }
    }
}

fn validate_activity_time_extent(
    requested_time_steps: usize,
    pressure_time_steps: usize,
    velocity_time_steps: usize,
) -> Result<(), String> {
    if requested_time_steps == pressure_time_steps && requested_time_steps == velocity_time_steps {
        return Ok(());
    }

    Err(format!(
        "GPU PSTD run time-step count {requested_time_steps} does not match retained source-activity extents: pressure {pressure_time_steps}, velocity {velocity_time_steps}"
    ))
}

fn validate_indices(name: &str, indices: &[u32], total_points: usize) -> Result<(), String> {
    u32::try_from(indices.len()).map_err(|_| {
        format!(
            "GPU PSTD {name} count {} exceeds u32 shader addressing",
            indices.len()
        )
    })?;
    if let Some((position, index)) = indices
        .iter()
        .copied()
        .enumerate()
        .find(|(_, index)| *index as usize >= total_points)
    {
        return Err(format!(
            "GPU PSTD {name} index {index} at position {position} exceeds field length {total_points}"
        ));
    }
    Ok(())
}

fn validate_signal_matrix(
    name: &str,
    signals: &[f32],
    source_count: usize,
    time_steps: usize,
) -> Result<(), String> {
    let expected = source_count.checked_mul(time_steps).ok_or_else(|| {
        format!("GPU PSTD {name} shape {source_count} x {time_steps} overflows usize")
    })?;
    if signals.len() != expected {
        return Err(format!(
            "GPU PSTD {name} has {} values; expected {expected} for {source_count} sources and {time_steps} time steps",
            signals.len()
        ));
    }
    if let Some((index, value)) = signals
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(format!(
            "GPU PSTD {name} must be finite at flat index {index}; got {value}"
        ));
    }
    Ok(())
}

fn run_cache_matches(cache: &WgpuPstdRunCache, inputs: &PstdRunInputs<'_>) -> bool {
    cache.sensor_indices == inputs.sensor_indices
        && cache.source_indices == inputs.source_indices
        && cache.vel_x_indices == inputs.vel_x_indices
        && cache.records_peak_pressure == inputs.output_request.includes_peak_pressure()
}

fn validate_run_inputs(
    total_points: usize,
    time_steps: usize,
    inputs: &PstdRunInputs<'_>,
) -> Result<(), String> {
    validate_indices("sensor", inputs.sensor_indices, total_points)?;
    validate_indices("pressure-source", inputs.source_indices, total_points)?;
    validate_indices("x-velocity-source", inputs.vel_x_indices, total_points)?;
    validate_signal_matrix(
        "pressure-source signals",
        inputs.source_signals,
        inputs.source_indices.len(),
        time_steps,
    )?;
    validate_signal_matrix(
        "x-velocity-source signals",
        inputs.vel_x_signals,
        inputs.vel_x_indices.len(),
        time_steps,
    )?;

    let sensor_values = inputs
        .sensor_indices
        .len()
        .max(1)
        .checked_mul(time_steps)
        .ok_or_else(|| "GPU PSTD sensor output length overflows usize".to_owned())?;
    if inputs.output_request.includes_peak_pressure() {
        sensor_values
            .checked_add(total_points)
            .ok_or_else(|| "GPU PSTD combined output length overflows usize".to_owned())?;
    }
    Ok(())
}

impl PstdRunState for WgpuPstdState {
    fn run_pstd(
        &mut self,
        scalars: PstdRunScalars,
        inputs: PstdRunInputs<'_>,
    ) -> Result<PstdRunResult, String> {
        // Retained activity geometry must agree before cache mutation or device work.
        validate_activity_time_extent(
            scalars.nt,
            self.pressure_source_activity.len(),
            self.velocity_source_activity.len(),
        )?;

        let n_sensors = inputs.sensor_indices.len();
        let n_src = inputs.source_indices.len();
        let n_vel_x = inputs.vel_x_indices.len();
        let n_src_safe = n_src.max(1);
        let n_vel_safe = n_vel_x.max(1);
        let records_peak_pressure = inputs.output_request.includes_peak_pressure();

        // Sensor indices/staging buffers are invariant across B-mode scan lines.
        // Source/vel buffers are reused; cache hits refresh only the signal tail.
        let cache_valid = run_cache_matches(&self.run_cache, &inputs)
            && self.run_cache.sensor_indices_buf.is_some();

        if !cache_valid {
            self.build_run_cache(scalars, &inputs);
        } else {
            self.refresh_signal_tails(
                inputs.source_signals,
                inputs.vel_x_signals,
                n_src_safe,
                n_vel_safe,
            );
        }

        if inputs.output_request.includes_final_fields() || records_peak_pressure {
            self.ensure_field_staging_buffer(scalars.total_points());
        }

        mark_active_source_steps(
            &mut self.pressure_source_activity,
            inputs.source_signals,
            n_src,
        );
        mark_active_source_steps(
            &mut self.velocity_source_activity,
            inputs.vel_x_signals,
            n_vel_x,
        );

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

        let output_bytes = (self.run_cache.output_storage_len * std::mem::size_of::<f32>()) as u64;
        commands.clear_buffer(buf_sensor_data, output_bytes, "clear_sensor_data");

        let elem_wg = StepCtx::ceil_div(scalars.total_points(), 256);
        let zero_params = scalars.zero_params();
        let hephaestus_device = self.context.hephaestus_device();
        let mut zero_stream = hephaestus_device
            .stream()
            .map_err(|error| format!("PSTD zero-field stream creation failed: {error}"))?;
        zero_stream
            .encode_grouped_sequence("zero_fields", |sequence| {
                passes.encode_zero_fields(sequence, &zero_params, bg_sensor, elem_wg);
                Ok(())
            })
            .map_err(|error| format!("PSTD zero-field encoding failed: {error}"))?;
        zero_stream
            .submit_grouped()
            .map_err(|error| format!("PSTD zero-field submission failed: {error}"))?;

        let ctx = scalars.step_context(&inputs, self.run_cache.peak_offset);

        // Batching reduces wgpu API overhead from O(nt) submits to O(nt/STEP_BATCH).
        // Kept at 32 to avoid Windows TDR on long runs.
        const STEP_BATCH: usize = 32;
        let mut batch_start = 0usize;
        while batch_start < scalars.nt {
            let batch_end = (batch_start + STEP_BATCH).min(scalars.nt);
            let mut stream = hephaestus_device
                .stream()
                .map_err(|error| format!("PSTD batch stream creation failed: {error}"))?;
            for step in batch_start..batch_end {
                stream
                    .encode_grouped_sequence("pstd_step", |sequence| {
                        passes.encode_time_step(
                            sequence,
                            &ctx,
                            StepBindGroups {
                                sensor: bg_sensor,
                                velocity_sensor: bg_sensor_vel,
                            },
                            step as u32,
                            SourceActivity {
                                pressure: self.pressure_source_activity[step],
                                velocity: self.velocity_source_activity[step],
                            },
                        )
                    })
                    .map_err(|error| format!("PSTD time-step {step} encoding failed: {error}"))?;
            }
            stream
                .submit_grouped()
                .map_err(|error| format!("PSTD batch submission failed: {error}"))?;
            batch_start = batch_end;

            // Bound queued GPU work so the D3D12 driver does not collapse long
            // simulations into one TDR-sized workload.
            if (batch_start / STEP_BATCH).is_multiple_of(16) {
                commands.poll_wait();
            }
        }

        let sensor_data = if n_sensors == 0 {
            Vec::new()
        } else {
            let sensor_trace_bytes = (n_sensors * scalars.nt * std::mem::size_of::<f32>()) as u64;
            let staging = self
                .run_cache
                .staging_buf
                .as_ref()
                .expect("invariant: sensor cache allocates a staging buffer");
            commands.copy_buffer_region(
                buf_sensor_data,
                0,
                staging,
                0,
                sensor_trace_bytes,
                "sensor_copy",
            );
            commands.read_mapped(staging, sensor_trace_bytes)
        };

        let final_fields = if inputs.output_request.includes_final_fields() {
            let field_bytes = (scalars.total_points() * std::mem::size_of::<f32>()) as u64;
            let staging = self
                .run_cache
                .field_staging_buf
                .as_ref()
                .expect("invariant: full-field request allocates a staging buffer");
            let read_field = |field: &wgpu::Buffer, label| {
                commands.copy_buffer_region(field, 0, staging, 0, field_bytes, label);
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

        let peak_pressure = if records_peak_pressure {
            let field_bytes = (scalars.total_points() * std::mem::size_of::<f32>()) as u64;
            let staging = self
                .run_cache
                .field_staging_buf
                .as_ref()
                .expect("invariant: peak request allocates a field staging buffer");
            commands.copy_buffer_region(
                buf_sensor_data,
                (self.run_cache.peak_offset * std::mem::size_of::<f32>()) as u64,
                staging,
                0,
                field_bytes,
                "peak_pressure_copy",
            );
            Some(commands.read_mapped(staging, field_bytes))
        } else {
            None
        };

        Ok(PstdRunResult {
            sensor_data,
            final_fields,
            peak_pressure,
        })
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
    /// Borrowed inputs preserve the source, sensor, correction, and output
    /// request contract as one operation-level value.
    /// # Errors
    ///
    /// Returns an error when run inputs violate their source-major shape or
    /// field-index contract, or when provider command encoding or submission
    /// fails.
    ///
    /// # Panics
    /// Panics if the provider run cache is not populated after cache rebuild.
    pub fn run(&mut self, inputs: PstdRunInputs<'_>) -> Result<PstdRunResult, String> {
        let scalars = PstdRunScalars {
            nx: self.nx,
            ny: self.ny,
            nz: self.nz,
            nt: self.nt,
            dt: self.dt,
            nonlinear: self.nonlinear,
            absorbing: self.absorbing,
        };

        validate_run_inputs(scalars.total_points(), scalars.nt, &inputs)?;

        self.state.run_pstd(scalars, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        mark_active_source_steps, run_cache_matches, validate_activity_time_extent,
        validate_run_inputs,
    };
    use crate::pstd_gpu::{state::WgpuPstdRunCache, PstdOutputRequest, PstdRunInputs};
    use kwavers_alloc_probe::{ThreadScopedAllocator, Window};

    #[global_allocator]
    static GLOBAL: ThreadScopedAllocator = ThreadScopedAllocator;

    fn assert_allocation_free(operation: impl FnOnce()) {
        let window = Window::open();
        operation();
        let change = window.change();
        drop(window);
        assert_eq!(change.allocations, 0);
        assert_eq!(change.reallocations, 0);
    }

    fn cache_inputs<'a>(
        sensor_indices: &'a [u32],
        source_indices: &'a [u32],
        vel_x_indices: &'a [u32],
    ) -> PstdRunInputs<'a> {
        PstdRunInputs {
            sensor_indices,
            source_indices,
            source_signals: &[],
            pressure_source_correction: false,
            vel_x_indices,
            vel_x_signals: &[],
            velocity_source_correction: false,
            output_request: PstdOutputRequest::sensor_traces(),
        }
    }

    #[test]
    fn run_cache_identity_includes_index_values() {
        let mut cache = WgpuPstdRunCache::default();
        cache.sensor_indices.extend_from_slice(&[1, 4]);
        cache.source_indices.extend_from_slice(&[2, 7]);
        cache.vel_x_indices.extend_from_slice(&[3]);

        assert!(run_cache_matches(
            &cache,
            &cache_inputs(&[1, 4], &[2, 7], &[3])
        ));
        assert!(!run_cache_matches(
            &cache,
            &cache_inputs(&[1, 5], &[2, 7], &[3])
        ));
        assert!(!run_cache_matches(
            &cache,
            &cache_inputs(&[1, 4], &[2, 6], &[3])
        ));
        assert!(!run_cache_matches(
            &cache,
            &cache_inputs(&[1, 4], &[2, 7], &[5])
        ));
    }

    #[test]
    fn active_source_steps_preserve_rows_and_clear_stale_flags() {
        let mut active = [true; 3];

        assert_allocation_free(|| {
            mark_active_source_steps(&mut active, &[0.0, 1.0, 0.0, 0.0, 0.0, -2.0], 2);
        });
        assert_eq!(active, [false, true, true]);

        assert_allocation_free(|| {
            mark_active_source_steps(&mut active, &[3.0, 0.0, 0.0], 1);
        });
        assert_eq!(active, [true, false, false]);

        assert_allocation_free(|| mark_active_source_steps(&mut active, &[], 0));
        assert_eq!(active, [false; 3]);
    }

    #[test]
    fn activity_time_extent_rejects_smaller_and_larger_runs() {
        assert_eq!(validate_activity_time_extent(2, 2, 2), Ok(()));

        for requested_time_steps in [1, 3] {
            assert_eq!(
                validate_activity_time_extent(requested_time_steps, 2, 2),
                Err(format!(
                    "GPU PSTD run time-step count {requested_time_steps} does not match retained source-activity extents: pressure 2, velocity 2"
                ))
            );
        }

        assert_eq!(
            validate_activity_time_extent(2, 2, 3),
            Err("GPU PSTD run time-step count 2 does not match retained source-activity extents: pressure 2, velocity 3".to_owned())
        );
    }

    #[test]
    fn run_input_validation_accepts_source_major_signals() {
        let inputs = PstdRunInputs {
            sensor_indices: &[7],
            source_indices: &[0, 7],
            source_signals: &[1.0, 0.0, -0.5, 0.25],
            pressure_source_correction: true,
            vel_x_indices: &[3],
            vel_x_signals: &[0.0, 1.0],
            velocity_source_correction: false,
            output_request: PstdOutputRequest::with_peak_pressure(),
        };

        assert_eq!(validate_run_inputs(8, 2, &inputs), Ok(()));
    }

    #[test]
    fn run_input_validation_rejects_signal_shape_mismatch() {
        let inputs = PstdRunInputs {
            sensor_indices: &[],
            source_indices: &[2, 3],
            source_signals: &[1.0, 2.0, 3.0],
            pressure_source_correction: false,
            vel_x_indices: &[],
            vel_x_signals: &[],
            velocity_source_correction: false,
            output_request: PstdOutputRequest::sensor_traces(),
        };

        assert_eq!(
            validate_run_inputs(8, 2, &inputs),
            Err("GPU PSTD pressure-source signals has 3 values; expected 4 for 2 sources and 2 time steps".to_owned())
        );
    }

    #[test]
    fn run_input_validation_rejects_out_of_range_indices() {
        let inputs = PstdRunInputs {
            sensor_indices: &[8],
            source_indices: &[],
            source_signals: &[],
            pressure_source_correction: false,
            vel_x_indices: &[],
            vel_x_signals: &[],
            velocity_source_correction: false,
            output_request: PstdOutputRequest::sensor_traces(),
        };

        assert_eq!(
            validate_run_inputs(8, 2, &inputs),
            Err("GPU PSTD sensor index 8 at position 0 exceeds field length 8".to_owned())
        );
    }

    #[test]
    fn run_input_validation_rejects_non_finite_amplitudes() {
        let inputs = PstdRunInputs {
            sensor_indices: &[],
            source_indices: &[1],
            source_signals: &[0.0, f32::NAN],
            pressure_source_correction: false,
            vel_x_indices: &[],
            vel_x_signals: &[],
            velocity_source_correction: false,
            output_request: PstdOutputRequest::sensor_traces(),
        };

        let error = validate_run_inputs(8, 2, &inputs)
            .expect_err("non-finite source amplitudes must be rejected");
        assert!(error.starts_with(
            "GPU PSTD pressure-source signals must be finite at flat index 1; got NaN"
        ));
    }
}
