//! Shared CPU/GPU PSTD differential-test harness.
//!
//! Both parity targets (pressure bursts in `pstd_parity`, the x-velocity
//! burst in `pstd_velocity_parity`) inject one identical finite burst into
//! the independent f64 CPU PSTD execution and the public provider-owned GPU
//! path, then compare the same per-step sensor trace. The harness owns the
//! reference run, the provider run, and the error metrics so the two targets
//! cannot drift.

use kwavers_gpu::pstd_gpu::{run_gpu_pstd, GpuPstdRunConfig};
use kwavers_grid::Grid;
use kwavers_medium::Medium;
use kwavers_physics::acoustics::mechanics::absorption::AbsorptionMode;
use kwavers_solver::forward::pstd::config::{
    BoundaryConfig, KSpaceMethod, PSTDConfig, PowerLawDispersion,
};
use kwavers_solver::forward::pstd::implementation::core::orchestrator::PSTDSolver;
use kwavers_source::GridSource;
use leto::Array3;

/// Power-law absorption applied identically by both providers; a zero
/// coefficient [dB/(MHz^y·cm)] is lossless.
#[derive(Clone, Copy)]
pub struct PowerLaw {
    pub coefficient: f64,
    pub exponent: f64,
    pub dispersion: PowerLawDispersion,
}

pub const LOSSLESS: PowerLaw = PowerLaw {
    coefficient: 0.0,
    exponent: 1.0,
    dispersion: PowerLawDispersion::Causal,
};

/// CPU-side outputs used as the independent double-precision reference.
pub struct CpuPstdOutput {
    pub trace: Vec<f64>,
    pub pressure: Vec<f64>,
    pub velocity_x: Vec<f64>,
    pub velocity_y: Vec<f64>,
    pub velocity_z: Vec<f64>,
}

/// Run CPU PSTD and retain the sensor trace and final pressure field.
pub fn run_cpu_pstd<M: Medium>(
    grid: &Grid,
    medium: &M,
    source: &GridSource,
    sensor: [usize; 3],
    nt: usize,
    dt: f64,
    absorption: PowerLaw,
) -> CpuPstdOutput {
    let config = PSTDConfig {
        dt,
        nt,
        absorption_mode: if absorption.coefficient > 0.0 {
            AbsorptionMode::PowerLaw {
                alpha_coeff: Some(absorption.coefficient),
                alpha_power: absorption.exponent,
            }
        } else {
            AbsorptionMode::Lossless
        },
        power_law_dispersion: absorption.dispersion,
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
pub fn run_cpu_gpu_traces<M: Medium + Sync>(
    grid: &Grid,
    medium: &M,
    source: &GridSource,
    sensor: [usize; 3],
    nt: usize,
    dt: f64,
    absorption: PowerLaw,
) -> (Vec<f64>, Vec<f64>) {
    std::thread::scope(|scope| {
        let gpu =
            scope.spawn(|| run_gpu_pstd_trace(grid, medium, source, sensor, nt, dt, absorption));
        let cpu = run_cpu_pstd(grid, medium, source, sensor, nt, dt, absorption).trace;
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
    absorption: PowerLaw,
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
            nonlinear: false,
            alpha_coeff_db: absorption.coefficient,
            alpha_power: absorption.exponent,
            power_law_dispersion: absorption.dispersion,
            cpml: None,
            pml_inside: false,
        },
    )
    .expect("GPU PSTD trace");
    output.iter().copied().collect()
}

/// max |cpu − gpu| / (max|cpu| + 1e-30)
pub fn max_rel_err(cpu: &[f64], gpu: &[f64]) -> f64 {
    let peak = cpu.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let denom = peak + 1e-30;
    cpu.iter()
        .zip(gpu.iter())
        .map(|(&c, &g)| (c - g).abs() / denom)
        .fold(0.0_f64, f64::max)
}

/// Relative full-field error normalized by the CPU field's maximum amplitude.
pub fn max_field_rel_err(cpu: &[f64], gpu: &[f32]) -> f64 {
    let peak = max_abs(cpu);
    cpu.iter()
        .zip(gpu)
        .map(|(&cpu_value, &gpu_value)| (cpu_value - f64::from(gpu_value)).abs() / (peak + 1e-30))
        .fold(0.0_f64, f64::max)
}

/// Largest absolute amplitude in one trace.
pub fn max_abs(trace: &[f64]) -> f64 {
    trace
        .iter()
        .fold(0.0_f64, |peak, &value| peak.max(value.abs()))
}

/// Largest absolute value in a single-precision field.
pub fn max_abs_field(field: &[f32]) -> f32 {
    field.iter().copied().map(f32::abs).fold(0.0_f32, f32::max)
}
