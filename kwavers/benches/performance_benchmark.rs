//! Comprehensive Performance Benchmark Suite
//!
//! ## STATUS: BENCHMARKS DISABLED (Sprint 209 Phase 2)
//!
//! **Date**: 2025-01-14
//! **Reason**: Benchmark stubs measured placeholder operations instead of real physics
//! **Decision**: Remove stubs per Dev rules (Correctness > Functionality, No Placeholders)
//! **Future**: Re-enable after physics implementations complete (Sprint 211-213)
//!
//! See:
//! - `BENCHMARK_STUB_REMEDIATION_PLAN.md` for detailed remediation plan
//! - `TODO_AUDIT_PHASE6_SUMMARY.md` Section 1.1 for audit findings
//! - `backlog.md` Sprint 211-213 for implementation roadmap (189-263 hours)
//!
//! ---
//!
//! ## Original Benchmark Categories (DISABLED)
//!
//! ### 1. Wave Propagation Benchmarks (DISABLED - Sprint 211)
//! - **FDTD Acoustic**: Requires staggered grid velocity update (20-28h)
//! - **PSTD**: Requires FFT-based k-space operators (15-20h)
//! - **HAS**: Requires angular spectrum method (12-16h)
//! - **Nonlinear Westervelt**: Requires β/ρc⁴ nonlinear term (10-12h)
//!
//! ### 2. Advanced Physics Benchmarks (DISABLED - Sprint 212)
//! - **SWE**: Requires elastic wave solver + inverse problem (24-32h)
//! - **CEUS**: Requires Rayleigh-Plesset + perfusion model (16-22h)
//! - **Transcranial FUS**: Requires Rayleigh integral + aberration (20-26h)
//!
//! ### 3. Uncertainty Quantification Benchmarks (DISABLED - Sprint 213)
//! - **Monte Carlo Dropout**: Requires variance calculation (44-63h)
//! - **Ensemble Methods**: Requires statistical aggregation
//! - **Conformal Prediction**: Requires conformity scoring
//!
//! ## Why Benchmarks Were Removed
//!
//! Phase 6 TODO audit identified 18 stub helper methods that measured placeholder
//! operations instead of real physics, violating Dev rules:
//! - "Absolute Prohibition: TODOs, stubs, dummy data, zero-filled placeholders"
//! - "Correctness > Functionality"
//! - "No Potemkin Villages"
//!
//! Invalid benchmarks produce misleading performance data and waste optimization
//! effort on incorrect baselines.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::error::KwaversResult;
use kwavers::grid::Grid;
use kwavers::medium::HomogeneousMedium;
use ndarray::Array3;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct BenchmarkConfig {
    grid_sizes: Vec<(usize, usize, usize)>,
    simulation_times: Vec<f64>,
    repetitions: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            grid_sizes: vec![
                (64, 64, 32),    // Small (diagnostic)
                (128, 128, 64),  // Medium (research)
                (256, 256, 128), // Large (clinical)
                (512, 512, 256), // Very large (advanced)
            ],
            simulation_times: vec![1e-5, 1e-4, 1e-3], // 10μs, 100μs, 1ms
            repetitions: 10,
        }
    }
}

/// Performance benchmark results
#[derive(Debug)]
#[allow(dead_code)]
struct BenchmarkResult {
    simulation_type: String,
    grid_size: (usize, usize, usize),
    simulation_time: f64,
    execution_time: Duration,
    memory_usage: usize,
    throughput: f64, // simulations per second
    accuracy_metric: f64,
}

/// Comprehensive benchmark suite
struct PerformanceBenchmarkSuite {
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

#[allow(
    dead_code,
    clippy::too_many_arguments,
    non_snake_case,
    clippy::same_item_push
)]
impl PerformanceBenchmarkSuite {
    fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
            results: Vec::new(),
        }
    }

    /// DISABLED - Run complete benchmark suite (Sprint 209 Phase 2)
    ///
    /// All benchmarks disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md
    #[allow(dead_code)]
    fn run_full_suite_disabled(&mut self) -> KwaversResult<()> {
        panic!("Benchmarks disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Run wave propagation benchmarks (Sprint 209 Phase 2)
    ///
    /// Benchmarks disabled because they use stub implementations.
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md
    #[allow(dead_code)]
    fn run_wave_propagation_benchmarks_disabled(&mut self) -> KwaversResult<()> {
        panic!("Benchmarks disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Benchmark FDTD acoustic wave propagation (Sprint 209 Phase 2)
    ///
    /// Uses stub implementations: update_velocity_fdtd, update_pressure_fdtd
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section "Phase 2A"
    #[allow(dead_code)]
    fn benchmark_fdtd_wave_disabled(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        sim_time: f64,
    ) -> KwaversResult<BenchmarkResult> {
        let grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)?;
        let _medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let mut total_time = Duration::new(0, 0);
        let mut memory_usage = 0;

        for _ in 0..self.config.repetitions {
            let start = Instant::now();

            // Initialize fields
            let mut pressure = Array3::<f32>::zeros(grid.dimensions());
            let _velocity_x = Array3::<f32>::zeros(grid.dimensions());
            let _velocity_y = Array3::<f32>::zeros(grid.dimensions());
            let _velocity_z = Array3::<f32>::zeros(grid.dimensions());

            // TODO: SIMPLIFIED BENCHMARK - using stub update functions, not real physics
            let dt = grid.dx / 1500.0 * 0.5; // CFL condition
            let n_steps = (sim_time / dt) as usize;

            for _ in 0..n_steps.min(1000) {
                // Limit for benchmarking
                // DISABLED - stub implementations removed
                pressure[[nx / 2, ny / 2, nz / 2]] += 1e5;
            }

            total_time += start.elapsed();
            memory_usage = pressure.len() * std::mem::size_of::<f32>() * 4; // 4 arrays
        }

        let avg_time = total_time / self.config.repetitions as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        Ok(BenchmarkResult {
            simulation_type: "FDTD Acoustic".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: sim_time,
            execution_time: avg_time,
            memory_usage,
            throughput,
            accuracy_metric: 0.99, // TODO: SIMPLIFIED - hardcoded accuracy, should compute from validation
        })
    }

    /// DISABLED - Benchmark PSTD wave propagation (Sprint 209 Phase 2)
    ///
    /// Uses stub implementations: simulate_fft_operations
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section "Phase 2A"
    #[allow(dead_code)]
    fn benchmark_pstd_wave_disabled(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        sim_time: f64,
    ) -> KwaversResult<BenchmarkResult> {
        let grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)?;
        let _medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let mut total_time = Duration::new(0, 0);
        let mut memory_usage = 0;

        for _ in 0..self.config.repetitions {
            let start = Instant::now();

            // TODO: SIMPLIFIED BENCHMARK - timing FFT overhead only, not real PSTD physics
            let mut field = Array3::<f32>::zeros(grid.dimensions());

            // Simulate FFT-based time stepping
            let dt = grid.dx / 1500.0 * 0.8; // More relaxed CFL for PSTD
            let n_steps = (sim_time / dt) as usize;

            for step in 0..n_steps.min(500) {
                // Add source with temporal modulation
                let time = step as f64 * dt;
                let source_amplitude = 1e5
                    * (2.0 * std::f64::consts::PI * 5e6 * time).sin()
                    * (-0.5 * (time * 5e6 * 2.0 * std::f64::consts::PI / 2.0).powi(2)).exp();

                field[[nx / 2, ny / 2, nz / 2]] += source_amplitude as f32;

                // DISABLED - stub implementation removed
                let _ = &field;
            }

            total_time += start.elapsed();
            memory_usage = field.len() * std::mem::size_of::<f32>() * 2; // Complex field + workspace
        }

        let avg_time = total_time / self.config.repetitions as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        Ok(BenchmarkResult {
            simulation_type: "PSTD Acoustic".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: sim_time,
            execution_time: avg_time,
            memory_usage,
            throughput,
            accuracy_metric: 0.995, // Higher accuracy for spectral methods
        })
    }

    /// DISABLED - Benchmark HAS wave propagation (Sprint 209 Phase 2)
    ///
    /// Uses stub implementations: simulate_angular_spectrum_propagation
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section "Phase 2A"
    #[allow(dead_code)]
    fn benchmark_has_wave_disabled(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        sim_time: f64,
    ) -> KwaversResult<BenchmarkResult> {
        let grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)?;
        let _medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let mut total_time = Duration::new(0, 0);
        let mut memory_usage = 0;

        for _ in 0..self.config.repetitions {
            let start = Instant::now();

            // TODO: SIMPLIFIED BENCHMARK - timing angular spectrum overhead only, not real HAS physics
            let mut field = Array3::<f32>::zeros(grid.dimensions());

            let dt = grid.dx / 1500.0 * 0.9;
            let n_steps = (sim_time / dt) as usize;

            for step in 0..n_steps.min(300) {
                let time = step as f64 * dt;

                // Add directional source (plane wave approximation)
                for i in 0..nx {
                    for j in 0..ny {
                        let x = i as f64 * grid.dx;
                        let y = j as f64 * grid.dy;
                        let phase = 2.0 * std::f64::consts::PI * 5e6 * (x * 0.7 + y * 0.3) / 1500.0; // Angled wave
                        field[[i, j, nz / 2]] +=
                            1e4 * (phase + 2.0 * std::f64::consts::PI * 5e6 * time).cos() as f32;
                    }
                }

                // DISABLED - stub implementation removed
                let _ = &field;
                let _ = dt;
            }

            total_time += start.elapsed();
            memory_usage =
                field.len() * std::mem::size_of::<f32>() + nx * ny * std::mem::size_of::<f32>() * 2;
            // Angular spectrum workspace
        }

        let avg_time = total_time / self.config.repetitions as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        Ok(BenchmarkResult {
            simulation_type: "HAS Propagation".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: sim_time,
            execution_time: avg_time,
            memory_usage,
            throughput,
            accuracy_metric: 0.98, // Good accuracy for smooth media
        })
    }

    /// DISABLED - Benchmark nonlinear Westervelt equation (Sprint 209 Phase 2)
    ///
    /// Uses stub implementations: update_velocity_fdtd, update_pressure_fdtd, update_pressure_nonlinear
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section "Phase 2A"
    #[allow(dead_code)]
    fn benchmark_westervelt_wave_DISABLED(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
        sim_time: f64,
    ) -> KwaversResult<BenchmarkResult> {
        let grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)?;
        let _medium = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

        let mut total_time = Duration::new(0, 0);
        let mut memory_usage = 0;

        for _ in 0..self.config.repetitions {
            let start = Instant::now();

            let mut pressure = Array3::<f32>::zeros(grid.dimensions());
            let _velocity_x = Array3::<f32>::zeros(grid.dimensions());
            let _velocity_y = Array3::<f32>::zeros(grid.dimensions());
            let _velocity_z = Array3::<f32>::zeros(grid.dimensions());

            // Nonlinear parameters
            let _beta = 3.5; // Nonlinearity parameter
            let _absorption = 0.5; // dB/MHz/cm

            let dt = grid.dx / 1500.0 * 0.3; // Stricter CFL for nonlinear
            let n_steps = (sim_time / dt) as usize;

            for step in 0..n_steps.min(200) {
                // Shorter for nonlinear stability
                let time = step as f64 * dt;

                // Add high-amplitude source for nonlinear effects
                pressure[[nx / 2, ny / 2, nz / 2]] += 5e5
                    * (2.0 * std::f64::consts::PI * 1e6 * time).sin() as f32
                    * (-0.5 * (time * 1e6).powi(2)).exp() as f32;

                // DISABLED - stub implementations removed
            }

            total_time += start.elapsed();
            memory_usage = pressure.len() * std::mem::size_of::<f32>() * 5; // More arrays for nonlinear
        }

        let avg_time = total_time / self.config.repetitions as u32;
        let throughput = 1.0 / avg_time.as_secs_f64();

        Ok(BenchmarkResult {
            simulation_type: "Nonlinear Westervelt".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: sim_time,
            execution_time: avg_time,
            memory_usage,
            throughput,
            accuracy_metric: 0.95, // Nonlinear methods have higher computational error
        })
    }

    /// DISABLED - Run advanced physics benchmarks (Sprint 209 Phase 2)
    ///
    /// All benchmarks disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md
    #[allow(dead_code)]
    fn run_advanced_physics_benchmarks_DISABLED(&mut self) -> KwaversResult<()> {
        panic!("Benchmarks disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Benchmark shear wave elastography (Sprint 209 Phase 2)
    ///
    /// Uses stub implementations: simulate_elastic_wave_step, simulate_displacement_tracking, simulate_stiffness_estimation
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section "Phase 2A"
    #[allow(dead_code)]
    fn benchmark_swe_DISABLED(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<BenchmarkResult> {
        let start = Instant::now();

        // TODO: SIMPLIFIED BENCHMARK - using stub functions, not real elastography physics

        let grid = Grid::new(nx, ny, nz, 2e-4, 2e-4, 2e-4)?; // 0.2mm for SWE
        let displacement_field = Array3::<f32>::zeros((nx, ny, nz));

        // Simulate ARFI push and shear wave propagation
        let n_time_steps = 500; // Typical for SWE

        for step in 0..n_time_steps {
            // DISABLED - stub implementations removed
            let _ = step;
            let _ = &displacement_field;
            let _ = &grid;
        }

        // DISABLED - stub implementation removed

        let execution_time = start.elapsed();
        let memory_usage = displacement_field.len() * std::mem::size_of::<f32>() * 3; // Multiple components

        Ok(BenchmarkResult {
            simulation_type: "Shear Wave Elastography".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: n_time_steps as f64 * 1e-7, // Approximate
            execution_time,
            memory_usage,
            throughput: 1.0 / execution_time.as_secs_f64(),
            accuracy_metric: 0.92, // Clinical accuracy for SWE
        })
    }

    /// DISABLED - Benchmark contrast-enhanced ultrasound (Sprint 209 Phase 2)
    ///
    /// Uses stub implementations: simulate_microbubble_scattering, simulate_tissue_perfusion, simulate_perfusion_analysis
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section "Phase 2A"
    #[allow(dead_code)]
    fn benchmark_ceus_DISABLED(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<BenchmarkResult> {
        let start = Instant::now();

        // CEUS involves microbubble dynamics and perfusion analysis
        let _grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)?;
        let contrast_signal = Array3::<f32>::zeros((nx, ny, nz));

        let n_time_points = 1000; // 10 seconds at 100 fps
        let n_microbubbles = 10000;

        for time_point in 0..n_time_points {
            let time = time_point as f64 * 0.01; // 10ms intervals

            // Simulate microbubble scattering
            for _ in 0..n_microbubbles.min(100) {
                // DISABLED - stub implementation removed
                let _ = &contrast_signal;
                let _ = time;
            }

            // DISABLED - stub implementation removed
            let _ = &contrast_signal;
            let _ = time;
        }

        // DISABLED - stub implementation removed
        let _ = &contrast_signal;

        let execution_time = start.elapsed();
        let memory_usage = contrast_signal.len() * std::mem::size_of::<f32>() * 2;

        Ok(BenchmarkResult {
            simulation_type: "Contrast-Enhanced Ultrasound".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: n_time_points as f64 * 0.01,
            execution_time,
            memory_usage,
            throughput: 1.0 / execution_time.as_secs_f64(),
            accuracy_metric: 0.88, // Clinical accuracy for CEUS
        })
    }

    /// DISABLED - Benchmark transcranial focused ultrasound (Sprint 209 Phase 2)
    ///
    /// Uses stub implementations: simulate_transducer_element, simulate_skull_transmission, simulate_thermal_monitoring
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section "Phase 2A"
    #[allow(dead_code)]
    fn benchmark_transcranial_fus_DISABLED(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<BenchmarkResult> {
        let start = Instant::now();

        // Transcranial FUS involves aberration correction and safety monitoring
        let grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)?;
        let acoustic_field = Array3::<f32>::zeros((nx, ny, nz));

        // Simulate phased array focusing through skull
        let n_elements = 1024; // Typical hemispherical array
        let n_time_steps = 1000;

        for time_step in 0..n_time_steps {
            // Simulate aberration correction
            for elem in 0..n_elements.min(64) {
                // DISABLED - stub implementation removed
                let _ = &acoustic_field;
                let _ = elem;
                let _ = time_step;
                let _ = &grid;
            }

            // DISABLED - stub implementation removed
            let _ = &acoustic_field;
            let _ = &grid;

            // DISABLED - stub implementation removed
            let _ = &acoustic_field;
        }

        let execution_time = start.elapsed();
        let memory_usage = acoustic_field.len() * std::mem::size_of::<f32>()
            + n_elements * std::mem::size_of::<f32>();

        Ok(BenchmarkResult {
            simulation_type: "Transcranial Focused Ultrasound".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: n_time_steps as f64 * 1e-8,
            execution_time,
            memory_usage,
            throughput: 1.0 / execution_time.as_secs_f64(),
            accuracy_metric: 0.94, // Clinical accuracy for tFUS
        })
    }

    /// DISABLED - Run GPU acceleration benchmarks (Sprint 209 Phase 2)
    #[allow(dead_code)]
    #[cfg(feature = "gpu")]
    fn run_gpu_acceleration_benchmarks_DISABLED(&mut self) -> KwaversResult<()> {
        println!("\n--- GPU Acceleration Benchmarks ---");

        // Test different GPU configurations
        for &(nx, ny, nz) in &self.config.grid_sizes {
            if nx * ny * nz > 512 * 512 * 256 {
                continue; // Skip very large grids for GPU tests
            }

            // Single GPU benchmark
            let result = self.benchmark_single_gpu(nx, ny, nz)?;
            self.results.push(result);

            // Multi-GPU benchmark (if available)
            if let Ok(result) = self.benchmark_multi_gpu(nx, ny, nz) {
                self.results.push(result);
            }
        }

        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    fn run_gpu_acceleration_benchmarks(&mut self) -> KwaversResult<()> {
        println!("\n--- GPU Acceleration Benchmarks ---");
        println!("GPU benchmarks skipped (compile with --features gpu)");
        Ok(())
    }

    /// Benchmark single GPU performance
    #[cfg(feature = "gpu")]
    fn benchmark_single_gpu(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<BenchmarkResult> {
        // GPU-accelerated FDTD benchmark
        let start = Instant::now();

        // TODO: SIMPLIFIED - GPU context created but not used for actual computation
        let _gpu_context = pollster::block_on(kwavers::gpu::GpuContext::new())?;

        let grid = Grid::new(nx, ny, nz, 1e-4, 1e-4, 1e-4)?;
        let n_steps = 1000;

        // GPU memory allocation and transfer (simplified timing)
        let memory_transfer_time = Duration::from_micros((nx * ny * nz * 4) as u64 / 10000); // Estimate
        std::thread::sleep(memory_transfer_time);

        // TODO: SIMPLIFIED - simulated timing with sleep, not real GPU kernels
        let compute_time = Duration::from_micros(n_steps as u64 * 10); // Estimate
        std::thread::sleep(compute_time);

        let execution_time = start.elapsed();
        let memory_usage = nx * ny * nz * 4 * 3; // GPU memory for fields

        Ok(BenchmarkResult {
            simulation_type: "Single GPU FDTD".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: n_steps as f64 * 1e-8,
            execution_time,
            memory_usage,
            throughput: 1.0 / execution_time.as_secs_f64(),
            accuracy_metric: 0.99,
        })
    }

    /// Benchmark multi-GPU performance
    #[cfg(feature = "gpu")]
    fn benchmark_multi_gpu(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<BenchmarkResult> {
        // Multi-GPU domain decomposition benchmark
        let start = Instant::now();

        let n_gpus = 2; // Assume 2 GPUs available
        let domains_per_gpu = 2; // Split into domains

        let _sub_nx = nx / (n_gpus * domains_per_gpu);

        for _gpu in 0..n_gpus {
            for domain in 0..domains_per_gpu {
                // TODO: SIMPLIFIED - simulated with sleep, not real multi-GPU work
                let domain_start = Instant::now();
                std::thread::sleep(Duration::from_micros(5000)); // Simulate GPU work
                let _domain_time = domain_start.elapsed();

                // Simulate inter-GPU communication
                if domain < domains_per_gpu - 1 {
                    std::thread::sleep(Duration::from_micros(100)); // Communication overhead
                }
            }
        }

        let execution_time = start.elapsed();
        let memory_usage = nx * ny * nz * 4 * 3 / n_gpus; // Distributed memory

        Ok(BenchmarkResult {
            simulation_type: "Multi-GPU FDTD".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: 1000.0 * 1e-8,
            execution_time,
            memory_usage,
            throughput: 1.0 / execution_time.as_secs_f64(),
            accuracy_metric: 0.98, // Slight accuracy loss from domain decomposition
        })
    }

    /// DISABLED - Run uncertainty quantification benchmarks (Sprint 209 Phase 2)
    ///
    /// Uses stub implementations for all UQ computations
    /// See: BENCHMARK_STUB_REMEDIATION_PLAN.md Section "Phase 2A"
    #[allow(dead_code)]
    fn run_uncertainty_benchmarks_disabled(&mut self) -> KwaversResult<()> {
        println!("\n--- Uncertainty Quantification Benchmarks ---");

        let (nx, ny, nz) = (64, 64, 32); // Small grid for uncertainty tests

        // Monte Carlo dropout benchmark
        let result = self.benchmark_monte_carlo_dropout(nx, ny, nz)?;
        self.results.push(result);

        // Ensemble methods benchmark
        let result = self.benchmark_ensemble_methods(nx, ny, nz)?;
        self.results.push(result);

        // Conformal prediction benchmark
        let result = self.benchmark_conformal_prediction(nx, ny, nz)?;
        self.results.push(result);

        Ok(())
    }

    /// Benchmark Monte Carlo dropout uncertainty
    fn benchmark_monte_carlo_dropout(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<BenchmarkResult> {
        let start = Instant::now();

        let n_samples = 50; // Monte Carlo samples
        let mut predictions = Vec::new();

        for _ in 0..n_samples {
            // Simulate PINN prediction with dropout
            let mut prediction = Array3::<f32>::from_elem((nx, ny, nz), 1.0);

            // TODO: SIMPLIFIED - basic random dropout, not real PINN uncertainty quantification
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        if rand::random::<f32>() < 0.1 {
                            // 10% dropout
                            prediction[[i, j, k]] = 0.0;
                        }
                    }
                }
            }

            predictions.push(prediction);
        }

        // DISABLED - stub implementation removed
        let _ = &predictions;

        let execution_time = start.elapsed();
        let memory_usage = nx * ny * nz * 4 * n_samples; // All samples in memory

        Ok(BenchmarkResult {
            simulation_type: "Monte Carlo Dropout".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: 0.0, // Not time-dependent
            execution_time,
            memory_usage,
            throughput: n_samples as f64 / execution_time.as_secs_f64(),
            accuracy_metric: 0.90, // Uncertainty quantification accuracy
        })
    }

    /// Benchmark ensemble methods
    fn benchmark_ensemble_methods(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<BenchmarkResult> {
        let start = Instant::now();

        let n_models = 10; // Ensemble size
        let mut ensemble_predictions = Vec::new();

        for model in 0..n_models {
            // Simulate different ensemble model predictions
            let mut prediction = Array3::<f32>::from_elem((nx, ny, nz), 1.0);

            // Add model-specific variation
            for i in 0..nx {
                for j in 0..ny {
                    for k in 0..nz {
                        prediction[[i, j, k]] += (model as f32 - 5.0) * 0.1; // Model bias
                    }
                }
            }

            ensemble_predictions.push(prediction);
        }

        // DISABLED - stub implementation removed
        let _ = &ensemble_predictions;
        let _ensemble_mean = Array3::<f32>::zeros((nx, ny, nz));
        let _ensemble_variance = Array3::<f32>::zeros((nx, ny, nz));

        let execution_time = start.elapsed();
        let memory_usage = nx * ny * nz * 4 * n_models;

        Ok(BenchmarkResult {
            simulation_type: "Ensemble Methods".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: 0.0,
            execution_time,
            memory_usage,
            throughput: n_models as f64 / execution_time.as_secs_f64(),
            accuracy_metric: 0.85, // Ensemble uncertainty accuracy
        })
    }

    /// Benchmark conformal prediction
    fn benchmark_conformal_prediction(
        &self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> KwaversResult<BenchmarkResult> {
        let start = Instant::now();

        let n_calibration = 100; // Calibration set size
        let mut conformity_scores = Vec::new();

        // Generate calibration data
        for _ in 0..n_calibration {
            let prediction = Array3::<f32>::from_elem((nx, ny, nz), rand::random::<f32>());
            let target = Array3::<f32>::from_elem((nx, ny, nz), rand::random::<f32>());

            // DISABLED - stub implementation removed
            let _ = &prediction;
            let _ = &target;
            let _score = 0.0_f64;
            conformity_scores.push(_score);
        }

        // Sort for quantile computation
        conformity_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute prediction intervals for test data
        let n_test = 50;
        for _ in 0..n_test {
            let test_prediction = Array3::<f32>::from_elem((nx, ny, nz), rand::random::<f32>());
            // DISABLED - stub implementation removed
            let _ = &test_prediction;
            let _ = &conformity_scores;
        }

        let execution_time = start.elapsed();
        let memory_usage = nx * ny * nz * 4 * (n_calibration + n_test);

        Ok(BenchmarkResult {
            simulation_type: "Conformal Prediction".to_string(),
            grid_size: (nx, ny, nz),
            simulation_time: 0.0,
            execution_time,
            memory_usage,
            throughput: (n_calibration + n_test) as f64 / execution_time.as_secs_f64(),
            accuracy_metric: 0.95, // Conformal prediction coverage accuracy
        })
    }

    /// Generate comprehensive performance report
    fn generate_performance_report(&self) {
        println!("\n=== PERFORMANCE BENCHMARK REPORT ===");
        println!("=====================================");

        // Group results by simulation type
        let mut type_results: std::collections::HashMap<String, Vec<&BenchmarkResult>> =
            std::collections::HashMap::new();

        for result in &self.results {
            type_results
                .entry(result.simulation_type.clone())
                .or_default()
                .push(result);
        }

        for (sim_type, results) in &type_results {
            println!("\n--- {} ---", sim_type);

            let avg_throughput: f64 =
                results.iter().map(|r| r.throughput).sum::<f64>() / results.len() as f64;
            let max_throughput = results.iter().map(|r| r.throughput).fold(0.0, f64::max);
            let avg_memory: f64 =
                results.iter().map(|r| r.memory_usage as f64).sum::<f64>() / results.len() as f64;
            let avg_accuracy: f64 =
                results.iter().map(|r| r.accuracy_metric).sum::<f64>() / results.len() as f64;

            println!("  Average Throughput: {:.2} sim/s", avg_throughput);
            println!("  Peak Throughput: {:.2} sim/s", max_throughput);
            println!("  Average Memory: {:.1} MB", avg_memory / (1024.0 * 1024.0));
            println!("  Average Accuracy: {:.1}%", avg_accuracy * 100.0);

            // Show scaling with grid size
            println!("  Scaling Analysis:");
            for result in results.iter().filter(|r| r.grid_size.1 == r.grid_size.0) {
                // Square grids
                let grid_points = result.grid_size.0 * result.grid_size.1 * result.grid_size.2;
                println!(
                    "    {:>8} points: {:.2} sim/s ({:.1}ms)",
                    grid_points,
                    result.throughput,
                    result.execution_time.as_millis()
                );
            }
        }

        println!("\n=== PERFORMANCE SUMMARY ===");
        println!("Total benchmarks run: {}", self.results.len());
        println!(
            "Best performing simulation: {}",
            self.results
                .iter()
                .max_by(|a, b| a.throughput.partial_cmp(&b.throughput).unwrap())
                .map(|r| r.simulation_type.as_str())
                .unwrap_or("None")
        );
        println!(
            "Most memory efficient: {}",
            self.results
                .iter()
                .min_by(|a, b| a.memory_usage.cmp(&b.memory_usage))
                .map(|r| r.simulation_type.as_str())
                .unwrap_or("None")
        );
    }

    // ========================================================================
    // BENCHMARK STUBS DISABLED - Sprint 209 Phase 2 (2025-01-14)
    // ========================================================================
    //
    // The following helper methods were REMOVED because they measured placeholder
    // operations instead of real physics, violating Dev rules:
    // - "Absolute Prohibition: TODOs, stubs, dummy data, zero-filled placeholders"
    // - "Correctness > Functionality"
    //
    // See: BENCHMARK_STUB_REMEDIATION_PLAN.md for implementation roadmap
    // Implementation effort: 189-263 hours (Sprint 211-213)
    //
    // ========================================================================

    /// DISABLED - Awaiting Real Implementation (Sprint 211 - 20-28h)
    ///
    /// This benchmark stub was removed because it measured placeholder operations
    /// instead of real FDTD physics.
    ///
    /// Required implementation:
    /// - Staggered grid velocity update: v^(n+1/2) = v^(n-1/2) - (dt/ρ) * ∇p^n
    /// - Spatial derivatives: (p[i+1,j,k] - p[i,j,k])/dx
    /// - Boundary handling for grid edges
    /// - CFL stability: dt ≤ dx/(c√3)
    ///
    /// Validation:
    /// - Plane wave test: p(x,t) = A sin(kx - ωt), v(x,t) = (A/ρc) sin(kx - ωt)
    /// - Energy conservation: E = ∫(p²/2ρc² + ρ|v|²/2)dV = const
    ///
    /// Related:
    /// - TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #1
    /// - backlog.md Sprint 211 "FDTD Benchmarks"
    /// - Production implementation: src/solver/fdtd/
    #[allow(dead_code)]
    fn update_velocity_fdtd_disabled(
        &self,
        _vx: &mut Array3<f32>,
        _vy: &mut Array3<f32>,
        _vz: &mut Array3<f32>,
        _p: &Array3<f32>,
        _dt: f64,
        _grid: &Grid,
        _medium: &HomogeneousMedium,
    ) {
        // DISABLED - Benchmark removed, awaiting real FDTD implementation
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 211 - 20-28h)
    ///
    /// Required implementation:
    /// - Pressure update: p^(n+1) = p^n - (ρc²) * dt * ∇·v^(n+1/2)
    /// - Divergence: ∇·v = (vx[i,j,k]-vx[i-1,j,k])/dx + (vy[i,j,k]-vy[i,j-1,k])/dy + (vz[i,j,k]-vz[i,j,k-1])/dz
    /// - Staggered grid indexing
    ///
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #2
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn update_pressure_fdtd_disabled(
        &self,
        _p: &mut Array3<f32>,
        _vx: &Array3<f32>,
        _vy: &Array3<f32>,
        _vz: &Array3<f32>,
        _dt: f64,
        _grid: &Grid,
        _medium: &HomogeneousMedium,
    ) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 211 - 10-12h)
    ///
    /// Required implementation:
    /// - Westervelt equation: ∂²p/∂t² = c²∇²p + (β/ρc⁴)∂²(p²)/∂t² + (δ/ρc²)∂³p/∂t³
    /// - Nonlinear term: N = (β/ρc⁴)∂²(p²)/∂t²
    /// - Shock-capturing scheme
    ///
    /// Validation: Fubini solution p(x,σ) = p₀ sin(σ)/(1 + Γσ cos(σ))
    ///
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #3
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn update_pressure_nonlinear_disabled(
        &self,
        _p: &mut Array3<f32>,
        _vx: &Array3<f32>,
        _vy: &Array3<f32>,
        _vz: &Array3<f32>,
        _dt: f64,
        _grid: &Grid,
        _medium: &HomogeneousMedium,
        _beta: f64,
        _absorption: f64,
    ) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 211 - 15-20h)
    ///
    /// Required: rustfft integration, k-space operators
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #4
    #[allow(dead_code)]
    fn simulate_fft_operations_disabled(&self, _field: &mut Array3<f32>) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 211 - 12-16h)
    ///
    /// Required: Angular spectrum method implementation
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #5
    #[allow(dead_code)]
    fn simulate_angular_spectrum_propagation_disabled(&self, _field: &mut Array3<f32>, _dt: f64) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 12-16h)
    ///
    /// Required: Elastic wave equation solver
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #6
    #[allow(dead_code)]
    fn simulate_elastic_wave_step_disabled(
        &self,
        _displacement: &mut Array3<f32>,
        _step: usize,
        _grid: &Grid,
    ) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 6-8h)
    ///
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #7
    #[allow(dead_code)]
    fn simulate_displacement_tracking_disabled(&self, _displacement: &Array3<f32>, _step: usize) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 6-8h)
    ///
    /// Required: Inverse problem solver (time-of-flight, direct inversion, FEM)
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #8
    #[allow(dead_code)]
    fn simulate_stiffness_estimation_disabled(&self, _displacement: &Array3<f32>) -> Array3<f32> {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 8-12h)
    ///
    /// Required: Rayleigh-Plesset dynamics, scattering
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #9
    #[allow(dead_code)]
    fn simulate_microbubble_scattering_disabled(&self, _signal: &mut Array3<f32>, _time: f64) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 4-6h)
    ///
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #10
    #[allow(dead_code)]
    fn simulate_tissue_perfusion_disabled(&self, _signal: &mut Array3<f32>, _time: f64) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 4-4h)
    ///
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #11
    #[allow(dead_code)]
    fn simulate_perfusion_analysis_disabled(&self, _signal: &Array3<f32>) -> Array3<f32> {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 8-10h)
    ///
    /// Required: Rayleigh integral implementation
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #12
    #[allow(dead_code)]
    fn simulate_transducer_element_disabled(
        &self,
        _field: &mut Array3<f32>,
        _elem: usize,
        _time_step: usize,
        _grid: &Grid,
    ) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 8-12h)
    ///
    /// Required: Skull aberration physics
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #13
    #[allow(dead_code)]
    fn simulate_skull_transmission_disabled(&self, _field: &mut Array3<f32>, _grid: &Grid) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 212 - 4-4h)
    ///
    /// Required: CEM43 thermal dose calculation
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #14
    #[allow(dead_code)]
    fn simulate_thermal_monitoring_disabled(&self, _field: &Array3<f32>) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 213 - 8-12h)
    ///
    /// Required: Variance/confidence interval calculation
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #15
    #[allow(dead_code)]
    fn compute_uncertainty_statistics_disabled(&self, _predictions: &[Array3<f32>]) -> Array3<f32> {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 213 - 4-6h)
    ///
    /// Required: Element-wise averaging
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #16
    #[allow(dead_code)]
    fn compute_ensemble_mean_disabled(&self, _predictions: &[Array3<f32>]) -> Array3<f32> {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 213 - 6-8h)
    ///
    /// Required: Variance calculation
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #17
    #[allow(dead_code)]
    fn compute_ensemble_variance_disabled(
        &self,
        _predictions: &[Array3<f32>],
        _mean: &Array3<f32>,
    ) -> Array3<f32> {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 213 - 10-14h)
    ///
    /// Required: Conformal prediction scoring
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #18
    #[allow(dead_code)]
    fn compute_conformity_score_disabled(
        &self,
        _prediction: &Array3<f32>,
        _target: &Array3<f32>,
    ) -> f64 {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }

    /// DISABLED - Awaiting Real Implementation (Sprint 213 - 16-23h)
    ///
    /// Required: Quantile-based interval calculation
    /// Related: TODO_AUDIT_PHASE6_SUMMARY.md Section 1.1 item #19
    #[allow(dead_code)]
    fn compute_prediction_interval_disabled(
        &self,
        _prediction: &Array3<f32>,
        _scores: &[f64],
        _confidence: f64,
    ) -> (Array3<f32>, Array3<f32>) {
        panic!("Benchmark disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
    }
}

// ============================================================================
// BENCHMARKS DISABLED - Sprint 209 Phase 2 (2025-01-14)
// ============================================================================
//
// All benchmarks in this file have been disabled because they measured
// placeholder operations instead of real physics, violating Dev rules.
//
// See: BENCHMARK_STUB_REMEDIATION_PLAN.md for implementation roadmap
// Estimated effort: 189-263 hours (Sprint 211-213)
//
// ============================================================================

#[allow(dead_code)]
fn benchmark_wave_propagation_disabled(_c: &mut Criterion) {
    panic!("Benchmarks disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
}

#[allow(dead_code)]
fn benchmark_advanced_physics_disabled(_c: &mut Criterion) {
    panic!("Benchmarks disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md");
}

// Criterion registration commented out - no benchmarks to run
// All benchmarks disabled - see BENCHMARK_STUB_REMEDIATION_PLAN.md
//
// Providing a dummy benchmark to satisfy criterion_group! macro requirements

fn dummy_benchmark(c: &mut Criterion) {
    c.bench_function("disabled_placeholder", |b| {
        b.iter(|| {
            // All real benchmarks disabled - this is a placeholder to allow compilation
            // See BENCHMARK_STUB_REMEDIATION_PLAN.md for implementation roadmap
            black_box(1 + 1);
        });
    });
}

criterion_group!(benches, dummy_benchmark);
criterion_main!(benches);
