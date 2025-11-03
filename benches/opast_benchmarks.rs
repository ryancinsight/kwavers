//! Performance benchmarks for OPAST (Orthonormal PAST) algorithm
//!
//! These benchmarks validate the performance improvements from SIMD acceleration
//! and parallel processing optimizations in the adaptive beamforming module.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kwavers::sensor::adaptive_beamforming::opast::OrthonormalSubspaceTracker;
use ndarray::Array1;
use num_complex::Complex64;
use num_traits::Zero;
use std::f64::consts::PI;

/// Generate synthetic sensor data for benchmarking
fn generate_test_snapshot(n: usize, num_signals: usize) -> Array1<Complex64> {
    let mut snapshot = Array1::zeros(n);

    // Generate multiple signal components with different frequencies
    for i in 0..n {
        let mut signal = Complex64::zero();
        for s in 0..num_signals {
            let freq = 0.1 * (s + 1) as f64;
            let phase = 2.0 * PI * freq * (i as f64) / (n as f64);
            signal += Complex64::new(phase.cos(), phase.sin()) * Complex64::new(1.0 / (s + 1) as f64, 0.0);
        }
        snapshot[i] = signal;
    }

    snapshot
}

/// Benchmark OPAST update performance across different array sizes
fn bench_opast_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("opast_updates");

    // Test different array sizes (sensor counts)
    let array_sizes = [16, 32, 64, 128, 256];
    let subspace_dims = [2, 4, 8];

    for &n in &array_sizes {
        for &p in &subspace_dims {
            if p >= n {
                continue; // Skip invalid configurations
            }

            group.bench_function(format!("n_{}_p_{}", n, p), |b| {
                let mut tracker = OrthonormalSubspaceTracker::new(n, p, 0.98);
                let snapshot = generate_test_snapshot(n, p);

                b.iter(|| {
                    tracker.update(snapshot.as_slice().unwrap());
                });
            });
        }
    }

    group.finish();
}

/// Benchmark OPAST initialization performance
fn bench_opast_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("opast_initialization");

    let array_sizes = [16, 32, 64, 128, 256];
    let subspace_dims = [2, 4, 8];

    for &n in &array_sizes {
        for &p in &subspace_dims {
            if p >= n {
                continue;
            }

            group.bench_function(format!("init_n_{}_p_{}", n, p), |b| {
                b.iter(|| {
                    black_box(OrthonormalSubspaceTracker::new(n, p, 0.98));
                });
            });
        }
    }

    group.finish();
}

/// Benchmark orthonormality maintenance over multiple updates
fn bench_opast_convergence(c: &mut Criterion) {
    let mut group = c.benchmark_group("opast_convergence");

    // Test convergence over different numbers of iterations
    let iterations = [10, 50, 100, 500];
    let n = 64;
    let p = 4;

    for &num_iter in &iterations {
        group.bench_function(format!("convergence_{}_iterations", num_iter), |b| {
            b.iter(|| {
                let mut tracker = OrthonormalSubspaceTracker::new(n, p, 0.98);
                let snapshot = generate_test_snapshot(n, p);

                for _ in 0..num_iter {
                    tracker.update(snapshot.as_slice().unwrap());
                }
                black_box(&tracker);
            });
        });
    }

    group.finish();
}

/// Benchmark orthonormality error computation (validation overhead)
fn bench_orthonormality_check(c: &mut Criterion) {
    let mut group = c.benchmark_group("orthonormality_check");

    let array_sizes = [32, 64, 128];
    let subspace_dims = [2, 4, 6];

    for &n in &array_sizes {
        for &p in &subspace_dims {
            if p >= n {
                continue;
            }

            group.bench_function(format!("ortho_check_n_{}_p_{}", n, p), |b| {
                let tracker = OrthonormalSubspaceTracker::new(n, p, 0.98);
                let subspace = tracker.get_subspace();

                b.iter(|| {
                    let mut error = 0.0;
                    // Check orthonormality: W^H W should be identity
                    for i in 0..p {
                        for j in 0..p {
                            let mut dot = Complex64::zero();
                            for k in 0..n {
                                dot += subspace[(k, i)].conj() * subspace[(k, j)];
                            }
                            if i == j {
                                error += (dot - Complex64::new(1.0, 0.0)).norm();
                            } else {
                                error += dot.norm();
                            }
                        }
                    }
                    black_box(error);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_opast_updates,
    bench_opast_initialization,
    bench_opast_convergence,
    bench_orthonormality_check
);
criterion_main!(benches);
