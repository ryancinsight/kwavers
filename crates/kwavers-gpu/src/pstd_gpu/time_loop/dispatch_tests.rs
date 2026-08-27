use super::super::encode::StepCtx;
use crate::pstd_gpu::{
    state::WgpuPstdState, AbsorptionArrays, GpuPstdSolver, MediumArrays, PmlArrays,
    PstdOutputRequest, PstdParams, PstdRunInputs, PstdRunScalars, SolverParams,
    WgpuPstdStateProvider,
};
use hephaestus_core::{GroupedCommandStream, KernelDevice};
use hephaestus_wgpu::WgpuGroupedSequence;
use kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

fn submit_sequence<F>(state: &WgpuPstdState, label: &str, encode: F)
where
    F: for<'sequence> FnOnce(&mut WgpuGroupedSequence<'sequence>) -> hephaestus_core::Result<()>,
{
    let mut stream = state
        .context
        .hephaestus_device()
        .stream()
        .expect("PSTD test command stream");
    stream
        .encode_grouped_sequence(label, encode)
        .expect("PSTD test grouped sequence");
    stream
        .submit_grouped()
        .expect("PSTD test command submission");
}

fn read_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    len: usize,
) -> Vec<T> {
    let byte_size = (len * core::mem::size_of::<T>()) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("pstd_fft_roundtrip_staging"),
        size: byte_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pstd_fft_roundtrip_copy"),
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_size);
    queue.submit(std::iter::once(encoder.finish()));
    let slice = staging.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = sender.send(result);
    });
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    receiver
        .recv()
        .expect("FFT roundtrip readback callback")
        .expect("FFT roundtrip staging map");
    let mapped = slice
        .get_mapped_range()
        .expect("mapped FFT roundtrip staging range");
    let values = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    staging.unmap();
    values
}

#[test]
fn pstd_copy_fft_inverse_roundtrip_preserves_pressure_field() {
    let n = 64usize;
    let total = n * n * n;
    let c0 = SOUND_SPEED_WATER_SIM as f32;
    let rho0 = 1000.0_f32;
    let c0_flat = vec![c0; total];
    let rho0_flat = vec![rho0; total];
    let ones = vec![1.0_f32; total];
    let zeros = vec![0.0_f32; total];
    let grid =
        kwavers_grid::Grid::new(n, n, n, 1e-3, 1e-3, 1e-3).expect("valid FFT-roundtrip grid");
    let solver = GpuPstdSolver::<WgpuPstdStateProvider>::with_auto_device(
        &grid,
        MediumArrays {
            c0_flat: &c0_flat,
            rho0_flat: &rho0_flat,
        },
        SolverParams {
            dt: 0.3e-3 / f64::from(c0),
            nt: 1,
            c_ref: f64::from(c0),
            nonlinear: false,
            absorbing: false,
        },
        PmlArrays {
            x: &ones,
            y: &ones,
            z: &ones,
            sgx: &ones,
            sgy: &ones,
            sgz: &ones,
        },
        AbsorptionArrays {
            bon_a_flat: &zeros,
            nabla1: &zeros,
            nabla2: &zeros,
            tau: &zeros,
            eta: &zeros,
        },
    );
    let Some(mut solver) = solver.ok() else {
        eprintln!("No GPU adapter — skipping PSTD FFT roundtrip test");
        return;
    };

    let mut pressure = vec![0.0_f32; total];
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let flat = ix * n * n + iy * n + iz;
                pressure[flat] = (core::f32::consts::TAU * ix as f32 / n as f32).sin();
            }
        }
    }

    solver.state.queue().write_buffer(
        &solver.state.field_buffers.p,
        0,
        bytemuck::cast_slice(&pressure),
    );
    solver.state.build_run_cache(
        PstdRunScalars {
            nx: n,
            ny: n,
            nz: n,
            nt: 1,
            dt: 0.3e-3 / f64::from(c0),
            nonlinear: false,
            absorbing: false,
        },
        &PstdRunInputs {
            sensor_indices: &[],
            source_indices: &[],
            source_signals: &[],
            pressure_source_correction: false,
            vel_x_indices: &[],
            vel_x_signals: &[],
            velocity_source_correction: false,
            output_request: PstdOutputRequest::sensor_traces(),
        },
    );
    let ctx = StepCtx {
        nx: n as u32,
        ny: n as u32,
        nz: n as u32,
        dt: 0.3e-3 / c0,
        n_sensors: 0,
        nt: 1,
        nonlinear: 0,
        absorbing: 0,
        peak_offset: 0,
        record_peak_pressure: 0,
        n_src: 0,
        n_vel_x: 0,
        pressure_source_correction: false,
        velocity_source_correction: false,
        elem_wg: StepCtx::ceil_div(total, 256),
    };
    let params = PstdParams {
        nx: n as u32,
        ny: n as u32,
        nz: n as u32,
        axis: 0,
        step: 0,
        dt: ctx.dt,
        n_sensors: 0,
        nt: 1,
        nonlinear: 0,
        absorbing: 0,
        peak_offset: 0,
        record_peak_pressure: 0,
    };
    let state = &solver.state;
    let bind_group = state
        .run_cache
        .bg_sensor
        .as_ref()
        .expect("FFT roundtrip run cache creates sensor bind group");
    submit_sequence(state, "pstd_fft_roundtrip", |sequence| {
        state.dispatch(
            sequence,
            &params,
            &state.pipelines.copy_field_to_k,
            bind_group,
            ctx.elem_wg,
            "copy_pressure",
        );
        state.encode_forward_fft(sequence)?;
        state.encode_inverse_fft(sequence)
    });
    let reconstructed = read_buffer::<f32>(
        state.device(),
        state.queue(),
        state.kspace_buffers.re.raw(),
        total,
    );

    let max_error = pressure
        .iter()
        .zip(&reconstructed)
        .map(|(&expected, &actual)| (expected - actual).abs())
        .fold(0.0_f32, f32::max);
    let peak = pressure
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0_f32, f32::max);
    // A three-axis forward/inverse transform traverses 36 radix-2 layers.
    // Budgeting 2,048 f32 operations covers the complex butterflies, the
    // k-space multiplier, and host-oracle evaluation: gamma_2048 < 2.45e-4.
    const TRANSFORM_RELATIVE_ERROR_BOUND: f32 = 3.0e-4;
    assert!(
        max_error < peak * TRANSFORM_RELATIVE_ERROR_BOUND,
        "GPU FFT roundtrip error {max_error:.2e} exceeds derived bound {:.2e}",
        peak * TRANSFORM_RELATIVE_ERROR_BOUND
    );

    submit_sequence(state, "pstd_x_gradient", |sequence| {
        state.dispatch(
            sequence,
            &params,
            &state.pipelines.copy_field_to_k,
            bind_group,
            ctx.elem_wg,
            "copy_pressure",
        );
        state.encode_forward_fft(sequence)?;
        state.dispatch(
            sequence,
            &PstdParams { axis: 0, ..params },
            &state.pipelines.kspace_shift,
            bind_group,
            ctx.elem_wg,
            "x_shift",
        );
        state.encode_inverse_fft(sequence)
    });
    let gradient = read_buffer::<f32>(
        state.device(),
        state.queue(),
        state.kspace_buffers.re.raw(),
        total,
    );
    let wavenumber = core::f32::consts::TAU / (n as f32 * 1e-3);
    let half_step = 0.5 * f64::from(c0) * (0.3e-3 / f64::from(c0)) * f64::from(wavenumber);
    let kappa = (half_step.sin() / half_step) as f32;
    let max_gradient_error = gradient
        .iter()
        .enumerate()
        .map(|(flat, &actual)| {
            let ix = flat / (n * n);
            let expected =
                kappa * wavenumber * (core::f32::consts::TAU * (ix as f32 + 0.5) / n as f32).cos();
            (expected - actual).abs()
        })
        .fold(0.0_f32, f32::max);
    let gradient_peak = kappa * wavenumber;
    assert!(
        max_gradient_error < gradient_peak * TRANSFORM_RELATIVE_ERROR_BOUND,
        "GPU staggered x-gradient error {max_gradient_error:.2e} exceeds derived bound {:.2e}",
        gradient_peak * TRANSFORM_RELATIVE_ERROR_BOUND
    );

    let zero_velocity = vec![0.0_f32; total];
    state.queue().write_buffer(
        &state.field_buffers.ux,
        0,
        bytemuck::cast_slice(&zero_velocity),
    );
    submit_sequence(state, "pstd_x_velocity", |sequence| {
        state.dispatch(
            sequence,
            &PstdParams { axis: 0, ..params },
            &state.pipelines.vel_update,
            bind_group,
            ctx.elem_wg,
            "x_velocity",
        );
        Ok(())
    });
    let velocity = read_buffer::<f32>(
        state.device(),
        state.queue(),
        &state.field_buffers.ux,
        total,
    );
    let dt = 0.3e-3 / c0;
    let velocity_peak = dt * gradient_peak / rho0;
    let max_velocity_error = velocity
        .iter()
        .enumerate()
        .map(|(flat, &actual)| {
            let ix = flat / (n * n);
            let expected = -dt
                * kappa
                * wavenumber
                * (core::f32::consts::TAU * (ix as f32 + 0.5) / n as f32).cos()
                / rho0;
            (expected - actual).abs()
        })
        .fold(0.0_f32, f32::max);
    assert!(
        max_velocity_error < velocity_peak * TRANSFORM_RELATIVE_ERROR_BOUND,
        "GPU velocity update error {max_velocity_error:.2e} exceeds derived bound {:.2e}",
        velocity_peak * TRANSFORM_RELATIVE_ERROR_BOUND
    );

    submit_sequence(state, "pstd_x_divergence", |sequence| {
        state.dispatch(
            sequence,
            &PstdParams { axis: 1, ..params },
            &state.pipelines.copy_field_to_k,
            bind_group,
            ctx.elem_wg,
            "copy_velocity_x",
        );
        state.encode_forward_fft(sequence)?;
        state.dispatch(
            sequence,
            &PstdParams { axis: 3, ..params },
            &state.pipelines.kspace_shift,
            bind_group,
            ctx.elem_wg,
            "x_negative_shift",
        );
        state.encode_inverse_fft(sequence)
    });
    let divergence = read_buffer::<f32>(
        state.device(),
        state.queue(),
        state.kspace_buffers.re.raw(),
        total,
    );
    let divergence_peak = dt * kappa * wavenumber * wavenumber / rho0;
    let max_divergence_error = divergence
        .iter()
        .enumerate()
        .map(|(flat, &actual)| {
            let ix = flat / (n * n);
            let expected = divergence_peak * (core::f32::consts::TAU * ix as f32 / n as f32).sin();
            (expected - actual).abs()
        })
        .fold(0.0_f32, f32::max);
    assert!(
        max_divergence_error < divergence_peak * TRANSFORM_RELATIVE_ERROR_BOUND,
        "GPU staggered x-divergence error {max_divergence_error:.2e} exceeds derived bound {:.2e}",
        divergence_peak * TRANSFORM_RELATIVE_ERROR_BOUND
    );

    state.queue().write_buffer(
        &state.field_buffers.rhox,
        0,
        bytemuck::cast_slice(&zero_velocity),
    );
    submit_sequence(state, "pstd_x_density", |sequence| {
        state.dispatch(
            sequence,
            &PstdParams { axis: 0, ..params },
            &state.pipelines.dens_update,
            bind_group,
            ctx.elem_wg,
            "x_density",
        );
        Ok(())
    });
    let density = read_buffer::<f32>(
        state.device(),
        state.queue(),
        &state.field_buffers.rhox,
        total,
    );
    let density_peak = dt * rho0 * divergence_peak;
    let max_density_error = density
        .iter()
        .enumerate()
        .map(|(flat, &actual)| {
            let ix = flat / (n * n);
            let expected = -density_peak * (core::f32::consts::TAU * ix as f32 / n as f32).sin();
            (expected - actual).abs()
        })
        .fold(0.0_f32, f32::max);
    assert!(
        max_density_error < density_peak * TRANSFORM_RELATIVE_ERROR_BOUND,
        "GPU density update error {max_density_error:.2e} exceeds derived bound {:.2e}",
        density_peak * TRANSFORM_RELATIVE_ERROR_BOUND
    );

    for (axis, velocity_field, density_field) in [
        (1_u32, &state.field_buffers.uy, &state.field_buffers.rhoy),
        (2_u32, &state.field_buffers.uz, &state.field_buffers.rhoz),
    ] {
        let component_velocity = (0..total)
            .map(|flat| {
                let coordinate = if axis == 1 {
                    (flat % (n * n)) / n
                } else {
                    flat % n
                };
                -velocity_peak
                    * (core::f32::consts::TAU * (coordinate as f32 + 0.5) / n as f32).cos()
            })
            .collect::<Vec<_>>();
        state
            .queue()
            .write_buffer(velocity_field, 0, bytemuck::cast_slice(&component_velocity));
        state
            .queue()
            .write_buffer(density_field, 0, bytemuck::cast_slice(&zero_velocity));
        submit_sequence(state, "pstd_component_density", |sequence| {
            state.dispatch(
                sequence,
                &PstdParams {
                    axis: axis + 1,
                    ..params
                },
                &state.pipelines.copy_field_to_k,
                bind_group,
                ctx.elem_wg,
                "copy_component_velocity",
            );
            state.encode_forward_fft(sequence)?;
            state.dispatch(
                sequence,
                &PstdParams {
                    axis: axis + 3,
                    ..params
                },
                &state.pipelines.kspace_shift,
                bind_group,
                ctx.elem_wg,
                "component_negative_shift",
            );
            state.encode_inverse_fft(sequence)?;
            state.dispatch(
                sequence,
                &PstdParams { axis, ..params },
                &state.pipelines.dens_update,
                bind_group,
                ctx.elem_wg,
                "component_density",
            );
            Ok(())
        });
        let component_density =
            read_buffer::<f32>(state.device(), state.queue(), density_field, total);
        let component_density_error = component_density
            .iter()
            .enumerate()
            .map(|(flat, &actual)| {
                let coordinate = if axis == 1 {
                    (flat % (n * n)) / n
                } else {
                    flat % n
                };
                let expected =
                    -density_peak * (core::f32::consts::TAU * coordinate as f32 / n as f32).sin();
                (expected - actual).abs()
            })
            .fold(0.0_f32, f32::max);
        assert!(
                component_density_error < density_peak * TRANSFORM_RELATIVE_ERROR_BOUND,
                "GPU staggered component {axis} density error {component_density_error:.2e} exceeds derived bound {:.2e}",
                density_peak * TRANSFORM_RELATIVE_ERROR_BOUND
            );
    }
}
