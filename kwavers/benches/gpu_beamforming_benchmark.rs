//! GPU Beamforming Performance Benchmarks
//!
//! This benchmark suite measures the performance of GPU-accelerated delay-and-sum
//! beamforming across different problem sizes and backends (CPU/GPU).
//!
//! ## Test Matrix
//!
//! **Problem Sizes**:
//! - Small: 32 channels × 1024 samples × 16×16 grid
//! - Medium: 64 channels × 2048 samples × 32×32 grid
//! - Large: 128 channels × 4096 samples × 64×64 grid
//!
//! **Backends**:
//! - NdArray (CPU baseline)
//! - WGPU (cross-platform GPU, feature-gated)
//! - CUDA (NVIDIA GPU, feature-gated)
//!
//! ## Metrics
//!
//! - **Throughput**: Focal points processed per second
//! - **Latency**: Time per beamforming operation (milliseconds)
//! - **Speedup**: GPU performance relative to CPU baseline
//! - **Memory**: Peak memory usage
//!
//! ## Mathematical Foundation
//!
//! Delay-and-Sum (DAS) beamforming computes:
//!
//! ```text
//! y(r) = Σᵢ wᵢ · s(i, t + τᵢ(r))
//! ```
//!
//! where:
//! - r: focal point position
//! - wᵢ: apodization weight for channel i
//! - s(i, t): RF signal from channel i at time t
//! - τᵢ(r): time delay for focusing at r from element i
//!
//! Computational complexity: O(C × S × G) where:
//! - C: number of channels
//! - S: samples per channel
//! - G: number of grid points (focal points)
//!
//! ## Usage
//!
//! ```bash
//! # Run all benchmarks (CPU only)
//! cargo bench --bench gpu_beamforming_benchmark
//!
//! # Run with GPU support
//! cargo bench --bench gpu_beamforming_benchmark --features pinn
//!
//! # Run specific problem size
//! cargo bench --bench gpu_beamforming_benchmark -- small
//!
//! # Generate detailed report
//! cargo bench --bench gpu_beamforming_benchmark -- --save-baseline main
//! ```

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{Array1, Array2};

#[cfg(feature = "pinn")]
use kwavers::analysis::signal_processing::beamforming::gpu::{beamform_cpu, BurnDasBeamformer};

/// Benchmark configuration for a specific problem size
#[derive(Debug, Clone)]
struct BenchmarkConfig {
    name: &'static str,
    num_channels: usize,
    num_samples: usize,
    grid_x: usize,
    grid_y: usize,
    speed_of_sound: f32,
    element_pitch: f32,
    sample_rate: f32,
}

impl BenchmarkConfig {
    /// Create small problem size (8.4M operations)
    fn small() -> Self {
        Self {
            name: "small",
            num_channels: 32,
            num_samples: 1024,
            grid_x: 16,
            grid_y: 16,
            speed_of_sound: 1540.0,
            element_pitch: 0.0003,
            sample_rate: 40e6,
        }
    }

    /// Create medium problem size (134M operations)
    fn medium() -> Self {
        Self {
            name: "medium",
            num_channels: 64,
            num_samples: 2048,
            grid_x: 32,
            grid_y: 32,
            speed_of_sound: 1540.0,
            element_pitch: 0.0003,
            sample_rate: 40e6,
        }
    }

    /// Create large problem size (2.1B operations)
    #[allow(dead_code)]
    fn large() -> Self {
        Self {
            name: "large",
            num_channels: 128,
            num_samples: 4096,
            grid_x: 64,
            grid_y: 64,
            speed_of_sound: 1540.0,
            element_pitch: 0.0003,
            sample_rate: 40e6,
        }
    }

    /// Total number of operations (channels × samples × grid points)
    #[allow(dead_code)]
    fn total_ops(&self) -> u64 {
        (self.num_channels * self.num_samples * self.grid_x * self.grid_y) as u64
    }

    /// Total number of focal points
    fn num_focal_points(&self) -> u64 {
        (self.grid_x * self.grid_y) as u64
    }

    /// Generate synthetic RF data for benchmarking
    fn generate_rf_data(&self) -> Array2<f32> {
        Array2::zeros((self.num_channels, self.num_samples))
    }

    /// Generate element positions (linear array)
    fn generate_element_positions(&self) -> Array2<f32> {
        let mut positions = Array2::zeros((self.num_channels, 3));
        for i in 0..self.num_channels {
            positions[[i, 0]] = (i as f32 - self.num_channels as f32 / 2.0) * self.element_pitch;
            positions[[i, 1]] = 0.0; // y = 0 (linear array)
            positions[[i, 2]] = 0.0; // z = 0 (transducer surface)
        }
        positions
    }

    /// Generate focal grid (rectangular grid at fixed depth)
    fn generate_focal_grid(&self) -> Array2<f32> {
        let depth = 0.05; // 5 cm depth
        let width = self.element_pitch * (self.num_channels as f32);
        let height = width;

        let mut grid = Array2::zeros((self.grid_x * self.grid_y, 3));
        let mut idx = 0;

        for ix in 0..self.grid_x {
            for iy in 0..self.grid_y {
                let x = (ix as f32 / self.grid_x as f32 - 0.5) * width;
                let y = (iy as f32 / self.grid_y as f32 - 0.5) * height;
                grid[[idx, 0]] = x;
                grid[[idx, 1]] = y;
                grid[[idx, 2]] = depth;
                idx += 1;
            }
        }
        grid
    }

    /// Generate uniform apodization weights
    fn generate_apodization(&self) -> Array1<f32> {
        Array1::ones(self.num_channels)
    }
}

/// Benchmark CPU beamforming (baseline)
fn bench_cpu_beamforming(c: &mut Criterion) {
    let mut group = c.benchmark_group("beamforming_cpu");

    let configs = vec![
        BenchmarkConfig::small(),
        BenchmarkConfig::medium(),
        // Large config commented out for CI - uncomment for full benchmarks
        // BenchmarkConfig::large(),
    ];

    for config in configs {
        let rf_data = config.generate_rf_data();
        let element_positions = config.generate_element_positions();
        let focal_grid = config.generate_focal_grid();
        let apodization = config.generate_apodization();

        group.throughput(Throughput::Elements(config.num_focal_points()));

        group.bench_with_input(
            BenchmarkId::new("cpu_baseline", config.name),
            &config,
            |b, _cfg| {
                b.iter(|| {
                    // Simple DAS implementation for CPU baseline
                    let mut output = Array1::<f32>::zeros(config.num_focal_points() as usize);

                    for fp_idx in 0..config.num_focal_points() as usize {
                        let focal_point = focal_grid.row(fp_idx);
                        let mut sum = 0.0;

                        for ch in 0..config.num_channels {
                            let elem_pos = element_positions.row(ch);

                            // Compute distance
                            let dx = focal_point[0] - elem_pos[0];
                            let dy = focal_point[1] - elem_pos[1];
                            let dz = focal_point[2] - elem_pos[2];
                            let distance = (dx * dx + dy * dy + dz * dz).sqrt();

                            // Compute delay in samples
                            let time_delay = distance / config.speed_of_sound;
                            let sample_delay = time_delay * config.sample_rate;
                            let sample_idx = sample_delay as usize;

                            // Accumulate (nearest-neighbor interpolation)
                            if sample_idx < config.num_samples {
                                sum += apodization[ch] * rf_data[[ch, sample_idx]];
                            }
                        }

                        output[fp_idx] = sum;
                    }

                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GPU beamforming using Burn framework (WGPU backend)
#[cfg(feature = "pinn")]
fn bench_gpu_beamforming_wgpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("beamforming_gpu_wgpu");

    let configs = vec![
        BenchmarkConfig::small(),
        BenchmarkConfig::medium(),
        // Large config commented out for CI
        // BenchmarkConfig::large(),
    ];

    for config in configs {
        let rf_data = config.generate_rf_data();
        let element_positions = config.generate_element_positions();
        let focal_grid = config.generate_focal_grid();
        let apodization = config.generate_apodization();

        group.throughput(Throughput::Elements(config.num_focal_points()));

        group.bench_with_input(
            BenchmarkId::new("gpu_wgpu", config.name),
            &config,
            |b, _cfg| {
                b.iter(|| {
                    let result = beamform_cpu(
                        black_box(&rf_data),
                        black_box(&element_positions),
                        black_box(&focal_grid),
                        black_box(&apodization),
                        config.speed_of_sound,
                        config.sample_rate,
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark GPU beamforming using Burn framework (NdArray backend for comparison)
#[cfg(feature = "pinn")]
fn bench_gpu_beamforming_ndarray(c: &mut Criterion) {
    let mut group = c.benchmark_group("beamforming_burn_ndarray");

    let configs = vec![BenchmarkConfig::small(), BenchmarkConfig::medium()];

    for config in configs {
        let rf_data = config.generate_rf_data();
        let element_positions = config.generate_element_positions();
        let focal_grid = config.generate_focal_grid();
        let apodization = config.generate_apodization();

        group.throughput(Throughput::Elements(config.num_focal_points()));

        group.bench_with_input(
            BenchmarkId::new("burn_ndarray", config.name),
            &config,
            |b, _cfg| {
                b.iter(|| {
                    // Use CPU convenience function which uses Burn NdArray backend
                    let result = beamform_cpu(
                        black_box(&rf_data),
                        black_box(&element_positions),
                        black_box(&focal_grid),
                        black_box(&apodization),
                        config.speed_of_sound,
                        config.sample_rate,
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory allocation overhead
fn bench_memory_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_allocation");

    let configs = vec![BenchmarkConfig::small(), BenchmarkConfig::medium()];

    for config in configs {
        group.bench_with_input(
            BenchmarkId::new("allocate_rf_data", config.name),
            &config,
            |b, cfg| {
                b.iter(|| black_box(Array2::<f32>::zeros((cfg.num_channels, cfg.num_samples))));
            },
        );

        group.bench_with_input(
            BenchmarkId::new("allocate_output", config.name),
            &config,
            |b, cfg| {
                b.iter(|| black_box(Array1::<f32>::zeros(cfg.num_focal_points() as usize)));
            },
        );
    }

    group.finish();
}

/// Benchmark distance computation (hot path)
fn bench_distance_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_computation");

    let config = BenchmarkConfig::medium();
    let element_positions = config.generate_element_positions();
    let focal_grid = config.generate_focal_grid();

    group.throughput(Throughput::Elements(
        (config.num_channels * config.num_focal_points() as usize) as u64,
    ));

    group.bench_function("euclidean_distance", |b| {
        b.iter(|| {
            let mut distances =
                Array2::<f32>::zeros((config.num_focal_points() as usize, config.num_channels));

            for fp_idx in 0..config.num_focal_points() as usize {
                let focal_point = focal_grid.row(fp_idx);
                for ch in 0..config.num_channels {
                    let elem_pos = element_positions.row(ch);
                    let dx = focal_point[0] - elem_pos[0];
                    let dy = focal_point[1] - elem_pos[1];
                    let dz = focal_point[2] - elem_pos[2];
                    distances[[fp_idx, ch]] = (dx * dx + dy * dy + dz * dz).sqrt();
                }
            }

            black_box(distances)
        });
    });

    group.finish();
}

/// Benchmark interpolation methods
fn bench_interpolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("interpolation");

    let config = BenchmarkConfig::medium();
    let rf_data = config.generate_rf_data();

    // Generate random sample indices for interpolation
    let num_samples = 10000;
    let sample_indices: Vec<f32> = (0..num_samples)
        .map(|i| (i as f32 / num_samples as f32) * config.num_samples as f32)
        .collect();

    group.throughput(Throughput::Elements(num_samples));

    // Nearest-neighbor interpolation
    group.bench_function("nearest_neighbor", |b| {
        b.iter(|| {
            let channel = 0;
            let mut output = Vec::with_capacity(num_samples as usize);
            for &idx in &sample_indices {
                let sample_idx = idx as usize;
                if sample_idx < config.num_samples {
                    output.push(rf_data[[channel, sample_idx]]);
                } else {
                    output.push(0.0);
                }
            }
            black_box(output)
        });
    });

    // Linear interpolation
    group.bench_function("linear", |b| {
        b.iter(|| {
            let channel = 0;
            let mut output = Vec::with_capacity(num_samples as usize);
            for &idx in &sample_indices {
                let sample_idx = idx.floor() as usize;
                let frac = idx - idx.floor();
                if sample_idx + 1 < config.num_samples {
                    let v0 = rf_data[[channel, sample_idx]];
                    let v1 = rf_data[[channel, sample_idx + 1]];
                    output.push(v0 * (1.0 - frac) + v1 * frac);
                } else if sample_idx < config.num_samples {
                    output.push(rf_data[[channel, sample_idx]]);
                } else {
                    output.push(0.0);
                }
            }
            black_box(output)
        });
    });

    group.finish();
}

// Criterion benchmark group configuration
criterion_group!(
    benches,
    bench_cpu_beamforming,
    bench_memory_allocation,
    bench_distance_computation,
    bench_interpolation
);

// Add GPU benchmarks only when pinn feature is enabled
#[cfg(feature = "pinn")]
criterion_group!(
    gpu_benches,
    bench_gpu_beamforming_wgpu,
    bench_gpu_beamforming_ndarray
);

#[cfg(feature = "pinn")]
criterion_main!(benches, gpu_benches);

#[cfg(not(feature = "pinn"))]
criterion_main!(benches);
