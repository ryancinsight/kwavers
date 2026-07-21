//! GPU PSTD versus CPU PSTD differential tests.
//!
//! Each case injects one identical finite pressure burst into both providers
//! and compares the same per-step sensor trace. The CPU path is an independent
//! f64 PSTD execution; the GPU path uses the public provider-owned API.

#![cfg(feature = "gpu")]

use kwavers_gpu::pstd_gpu::{
    run_gpu_pstd, run_gpu_pstd_with_outputs, GpuPstdRunConfig, PstdOutputRequest,
};
use kwavers_grid::Grid;
use kwavers_medium::{heterogeneous::HeterogeneousMedium, HomogeneousMedium, Medium};
use kwavers_solver::forward::pstd::config::{BoundaryConfig, KSpaceMethod, PSTDConfig};
use kwavers_solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_source::{GridSource, SourceMode};
use leto::{Array2, Array3};

/// Construct the shared finite pressure burst used by both PSTD providers.
fn point_pressure_burst(n: usize, nt: usize, source: [usize; 3]) -> GridSource {
    let mut p_mask = Array3::<f64>::zeros((n, n, n));
    p_mask[[source[0], source[1], source[2]]] = 1.0;
    let mut p_signal = Array2::<f64>::zeros((1, nt));
    p_signal[[0, 1]] = 1.0;
    GridSource {
        p_mask: Some(p_mask),
        p_signal: Some(p_signal),
        p_mode: SourceMode::Additive,
        ..GridSource::new_empty()
    }
}

/// CPU-side outputs used as the independent double-precision reference.
struct CpuPstdOutput {
    trace: Vec<f64>,
    pressure: Vec<f64>,
    velocity_x: Vec<f64>,
    velocity_y: Vec<f64>,
    velocity_z: Vec<f64>,
}

/// Run CPU PSTD and retain the sensor trace and final pressure field.
fn run_cpu_pstd<M: Medium>(
    grid: &Grid,
    medium: &M,
    source: &GridSource,
    sensor: [usize; 3],
    nt: usize,
    dt: f64,
) -> CpuPstdOutput {
    let config = PSTDConfig {
        dt,
        nt,
        kspace_method: KSpaceMethod::StandardPSTD,
        boundary: BoundaryConfig::None,
        nonlinearity: medium
            .nonlinearity_array()
            .iter()
            .any(|&coefficient| coefficient != 0.0),
        smooth_sources: false,
        ..Default::default()
    };
    let mut solver =
        PSTDSolver::new(config, grid.clone(), medium, source.clone()).expect("CPU solver");

    let mut trace = Vec::with_capacity(nt);
    for _ in 0..nt {
        solver.step_forward().expect("CPU step");
        trace.push(solver.pressure_field()[[sensor[0], sensor[1], sensor[2]]]);
    }
    let (velocity_x, velocity_y, velocity_z) = solver.velocity_fields();
    CpuPstdOutput {
        trace,
        pressure: solver.pressure_field().iter().copied().collect(),
        velocity_x: velocity_x.iter().copied().collect(),
        velocity_y: velocity_y.iter().copied().collect(),
        velocity_z: velocity_z.iter().copied().collect(),
    }
}

/// Execute independent CPU and Hephaestus/WGPU providers concurrently.
///
/// The two solvers share immutable inputs but no state. Parallel execution
/// preserves the full 128³ × 100-step differential workload while preventing
/// the test wall time from being the serial sum of independent providers.
fn run_cpu_gpu_traces<M: Medium + Sync>(
    grid: &Grid,
    medium: &M,
    source: &GridSource,
    sensor: [usize; 3],
    nt: usize,
    dt: f64,
) -> (Vec<f64>, Vec<f64>) {
    std::thread::scope(|scope| {
        let gpu = scope.spawn(|| run_gpu_pstd_trace(grid, medium, source, sensor, nt, dt));
        let cpu = run_cpu_pstd(grid, medium, source, sensor, nt, dt).trace;
        let gpu = gpu.join().expect("GPU PSTD parity worker completes");
        (cpu, gpu)
    })
}

/// Run the public GPU PSTD path with the same grid source and sensor.
fn run_gpu_pstd_trace<M: Medium>(
    grid: &Grid,
    medium: &M,
    source: &GridSource,
    sensor: [usize; 3],
    nt: usize,
    dt: f64,
) -> Vec<f64> {
    let mut sensor_mask = Array3::from_elem((grid.nx, grid.ny, grid.nz), false);
    sensor_mask[[sensor[0], sensor[1], sensor[2]]] = true;
    let output = run_gpu_pstd(
        grid,
        medium,
        source,
        &sensor_mask,
        GpuPstdRunConfig {
            time_steps: nt,
            dt,
            alpha_coeff_db: 0.0,
            alpha_power: 1.0,
            pml_size: None,
            pml_size_xyz: None,
            pml_inside: false,
            pml_alpha_xyz: None,
        },
    )
    .expect("GPU PSTD trace");
    output.iter().copied().collect()
}

/// max |cpu − gpu| / (max|cpu| + 1e-30)
fn max_rel_err(cpu: &[f64], gpu: &[f64]) -> f64 {
    let peak = cpu.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let denom = peak + 1e-30;
    cpu.iter()
        .zip(gpu.iter())
        .map(|(&c, &g)| (c - g).abs() / denom)
        .fold(0.0_f64, f64::max)
}

/// Relative full-field error normalized by the CPU field's maximum amplitude.
fn max_field_rel_err(cpu: &[f64], gpu: &[f32]) -> f64 {
    let peak = max_abs(cpu);
    cpu.iter()
        .zip(gpu)
        .map(|(&cpu_value, &gpu_value)| (cpu_value - f64::from(gpu_value)).abs() / (peak + 1e-30))
        .fold(0.0_f64, f64::max)
}

/// Largest absolute amplitude in one trace.
fn max_abs(trace: &[f64]) -> f64 {
    trace
        .iter()
        .fold(0.0_f64, |peak, &value| peak.max(value.abs()))
}

/// Largest absolute value in a single-precision field.
fn max_abs_field(field: &[f32]) -> f32 {
    field.iter().copied().map(f32::abs).fold(0.0_f32, f32::max)
}

/// The complete filtered source field agrees before its first propagation
/// update.  This verifies the full three-axis complex FFT and cosine source
/// correction rather than only the source-cell value.
#[test]
#[ignore = "requires GPU device"]
fn pstd_cpu_gpu_additive_source_filter_contract() {
    let n = 64usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let nt = 2usize;
    let dt = 0.3 * dx / c0;
    let grid = Grid::new(n, n, n, dx, dx, dx).expect("valid grid");
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let source = [n / 2, n / 2, n / 2];
    let grid_source = point_pressure_burst(n, nt, source);
    let mut sensor_mask = Array3::from_elem((n, n, n), false);
    sensor_mask[[source[0], source[1], source[2]]] = true;
    let gpu_output = run_gpu_pstd_with_outputs(
        &grid,
        &medium,
        &grid_source,
        &sensor_mask,
        GpuPstdRunConfig {
            time_steps: nt,
            dt,
            alpha_coeff_db: 0.0,
            alpha_power: 1.0,
            pml_size: None,
            pml_size_xyz: None,
            pml_inside: false,
            pml_alpha_xyz: None,
        },
        PstdOutputRequest::with_final_fields(),
    )
    .expect("GPU PSTD source-filter output");
    let gpu_pressure = &gpu_output
        .final_fields
        .expect("source-filter contract requires final fields")
        .pressure;
    let cpu_output = run_cpu_pstd(&grid, &medium, &grid_source, source, nt, dt);

    assert_eq!(cpu_output.pressure.len(), gpu_pressure.len());
    let rel_err = max_field_rel_err(&cpu_output.pressure, gpu_pressure);
    assert!(
        rel_err < 1e-4,
        "GPU/CPU additive source-filter contract failed: max_rel_err = {rel_err:.2e}, CPU max |p| = {:.2e}, GPU max |p| = {:.2e}",
        max_abs(&cpu_output.pressure),
        max_abs_field(gpu_pressure),
    );
}

/// The linear additive-source conversion and first leapfrog updates agree before a
/// long integration can amplify any spectral-stability defect.  A 64-point
/// radix-2 transform has six stages per axis; three forward/inverse transforms
/// and the local updates contribute fewer than 1_000 f32 roundings.  Higham's
/// `gamma_1000 < 6e-5` therefore leaves the `1e-4` relative bound for the
/// independent f64 CPU reference.
#[test]
#[ignore = "requires GPU device"]
fn pstd_cpu_gpu_source_and_early_leapfrog_contract() {
    let n = 64usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let nt = 3usize;
    let dt = 0.3 * dx / c0;
    let grid = Grid::new(n, n, n, dx, dx, dx).expect("valid grid");
    let mut medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    medium.set_nonlinearity(0.0);
    let source = [n / 2, n / 2, n / 2];
    let grid_source = point_pressure_burst(n, nt, source);
    let mut sensor_mask = Array3::from_elem((n, n, n), false);
    sensor_mask[[source[0], source[1], source[2]]] = true;
    let gpu_output = run_gpu_pstd_with_outputs(
        &grid,
        &medium,
        &grid_source,
        &sensor_mask,
        GpuPstdRunConfig {
            time_steps: nt,
            dt,
            alpha_coeff_db: 0.0,
            alpha_power: 1.0,
            pml_size: None,
            pml_size_xyz: None,
            pml_inside: false,
            pml_alpha_xyz: None,
        },
        PstdOutputRequest::with_final_fields(),
    )
    .expect("GPU PSTD early-leapfrog output");
    let gpu_trace = gpu_output
        .sensor_data
        .iter()
        .copied()
        .map(f64::from)
        .collect::<Vec<_>>();
    let gpu_fields = gpu_output
        .final_fields
        .expect("early-leapfrog contract requires final fields");
    let cpu_output = run_cpu_pstd(&grid, &medium, &grid_source, source, nt, dt);
    let velocity_x_err = max_field_rel_err(&cpu_output.velocity_x, &gpu_fields.velocity_x);
    let velocity_y_err = max_field_rel_err(&cpu_output.velocity_y, &gpu_fields.velocity_y);
    let velocity_z_err = max_field_rel_err(&cpu_output.velocity_z, &gpu_fields.velocity_z);
    assert!(
        velocity_x_err < 1e-4 && velocity_y_err < 1e-4 && velocity_z_err < 1e-4,
        "GPU/CPU first-propagation velocity contract failed: max_rel_err ux={velocity_x_err:.2e}, uy={velocity_y_err:.2e}, uz={velocity_z_err:.2e}"
    );
    let cpu_trace = cpu_output.trace;

    assert_eq!(cpu_trace.len(), nt);
    assert_eq!(gpu_trace.len(), nt);
    let rel_err = max_rel_err(&cpu_trace, &gpu_trace);
    assert!(
        rel_err < 1e-4,
        "GPU/CPU PSTD source and early leapfrog contract failed: max_rel_err = {rel_err:.2e}, CPU trace = {cpu_trace:?}, GPU trace = {gpu_trace:?}, GPU max |p| = {:.2e}, max |ux| = {:.2e}, max |uy| = {:.2e}, max |uz| = {:.2e}",
        max_abs_field(&gpu_fields.pressure),
        max_abs_field(&gpu_fields.velocity_x),
        max_abs_field(&gpu_fields.velocity_y),
        max_abs_field(&gpu_fields.velocity_z),
    );
}

/// CPU and GPU execute the same finite additive pressure burst and sample the
/// same sensor after each of 100 PSTD steps on a 64³ homogeneous water medium.
///
/// For the 128-point case, three seven-stage radix-2 transforms with at most
/// ten scalar roundings per butterfly plus six local physics operations give
/// at most 216 f32 roundings per step. Higham's `γ_(100×216) < 1.3e-3` bound
/// leaves the `2e-3` assertion margin for the independently scheduled provider
/// reductions without masking a real drift.
#[test]
#[ignore = "requires GPU device"]
fn pstd_cpu_gpu_parity_homogeneous() {
    let n = 64usize;
    let dx = 1e-3_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let nt = 100usize;
    let dt = 0.3 * dx / c0;
    let grid = Grid::new(n, n, n, dx, dx, dx).expect("valid grid");
    let medium = HomogeneousMedium::new(rho0, c0, 0.0, 0.0, &grid);
    let source = [n / 2 - 4, n / 2, n / 2];
    let sensor = [n / 2 + 4, n / 2, n / 2];
    let grid_source = point_pressure_burst(n, nt, source);
    let (cpu_trace, gpu_trace) = run_cpu_gpu_traces(&grid, &medium, &grid_source, sensor, nt, dt);

    assert_eq!(cpu_trace.len(), nt);
    assert_eq!(gpu_trace.len(), nt);
    let rel_err = max_rel_err(&cpu_trace, &gpu_trace);
    let cpu_peak = max_abs(&cpu_trace);
    let gpu_peak = max_abs(&gpu_trace);
    assert!(
        rel_err < 2e-3,
        "64³ GPU/CPU PSTD trace parity failed: max_rel_err = {rel_err:.2e}, CPU peak = {cpu_peak:.2e}, GPU peak = {gpu_peak:.2e} (threshold 2e-3)"
    );
}

/// The same finite burst crosses a two-region 128³ medium. Both providers use
/// the same per-voxel density and sound-speed fields and sample one trace.
#[test]
#[ignore = "requires GPU device"]
fn pstd_cpu_gpu_parity_heterogeneous() {
    let n = 128usize;
    let dx = 5e-4_f64;
    let c0 = 1500.0_f64;
    let rho0 = 1000.0_f64;
    let nt = 100usize;
    let dt = 0.3 * dx / c0;
    let mut sound_speed = Array3::from_elem((n, n, n), c0);
    let mut density = Array3::from_elem((n, n, n), rho0);
    for i in n / 2..n {
        for j in 0..n {
            for k in 0..n {
                sound_speed[[i, j, k]] = 1560.0;
                density[[i, j, k]] = 1050.0;
            }
        }
    }
    let grid = Grid::new(n, n, n, dx, dx, dx).expect("valid grid");
    let mut medium = HeterogeneousMedium::new(n, n, n, true);
    medium.sound_speed = sound_speed;
    medium.density = density;
    let source = [n / 2 - 8, n / 2, n / 2];
    let sensor = [n / 2 + 8, n / 2, n / 2];
    let grid_source = point_pressure_burst(n, nt, source);
    let (cpu_trace, gpu_trace) = run_cpu_gpu_traces(&grid, &medium, &grid_source, sensor, nt, dt);

    assert_eq!(cpu_trace.len(), nt);
    assert_eq!(gpu_trace.len(), nt);
    let rel_err = max_rel_err(&cpu_trace, &gpu_trace);
    let cpu_peak = max_abs(&cpu_trace);
    let gpu_peak = max_abs(&gpu_trace);
    assert!(
        rel_err < 2e-3,
        "128³ heterogeneous GPU/CPU PSTD trace parity failed: max_rel_err = {rel_err:.2e}, CPU peak = {cpu_peak:.2e}, GPU peak = {gpu_peak:.2e} (threshold 2e-3)"
    );
}
