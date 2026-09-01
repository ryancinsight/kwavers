//! Consolidated harness for benchmarks outside the merge-critical and
//! feature-isolated instruments.

use criterion::criterion_main;

#[path = "amr_criteria_benchmark.rs"]
mod amr_criteria_benchmark;
#[path = "art_benchmark.rs"]
mod art_benchmark;
#[path = "clinical_sound_speed_shift_openpros.rs"]
mod clinical_sound_speed_shift_openpros;
#[path = "conservative_interpolation_comparison.rs"]
mod conservative_interpolation_comparison;
#[path = "cpml_benchmark.rs"]
mod cpml_benchmark;
#[path = "fdtd_propagation_benchmark.rs"]
mod fdtd_propagation_benchmark;
#[path = "fnm_performance_benchmark.rs"]
mod fnm_performance_benchmark;
#[path = "gpu_beamforming_benchmark.rs"]
mod gpu_beamforming_benchmark;
#[path = "grid_benchmarks.rs"]
mod grid_benchmarks;
#[path = "logging_benchmark.rs"]
mod logging_benchmark;
#[path = "narrowband_beamforming.rs"]
mod narrowband_beamforming;
#[path = "nl_swe_performance.rs"]
mod nl_swe_performance;
#[path = "osem_benchmark.rs"]
mod osem_benchmark;
#[path = "physics_benchmarks.rs"]
mod physics_benchmarks;
#[path = "testing_infrastructure.rs"]
mod testing_infrastructure;
#[path = "ultrasound_benchmarks.rs"]
mod ultrasound_benchmarks;
#[path = "validation_benchmarks.rs"]
mod validation_benchmarks;

#[cfg(feature = "pinn")]
fn gpu_benches() {
    gpu_beamforming_benchmark::gpu_benches();
}

#[cfg(not(feature = "pinn"))]
fn gpu_benches() {}

criterion_main!(
    amr_criteria_benchmark::benches,
    art_benchmark::benches,
    clinical_sound_speed_shift_openpros::benches,
    conservative_interpolation_comparison::benches,
    cpml_benchmark::benches,
    fdtd_propagation_benchmark::benches,
    fnm_performance_benchmark::benches,
    gpu_beamforming_benchmark::benches,
    gpu_benches,
    grid_benchmarks::benches,
    logging_benchmark::benches,
    narrowband_beamforming::benches,
    nl_swe_performance::benches,
    osem_benchmark::benches,
    physics_benchmarks::benches,
    testing_infrastructure::benches,
    ultrasound_benchmarks::benches,
    validation_benchmarks::benches,
);
