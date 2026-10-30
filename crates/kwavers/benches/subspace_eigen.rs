//! Subspace eigensolver bench (KW-SUBSPACE-JACOBI-EIGEN).
//!
//! Compares the classical Jacobi Hermitian eigensolver against the
//! Wilkinson-shift QR path (`leto_ops::application::linalg`) at the covariance
//! sizes the MUSIC and ESMV beamformers operate at, and measures the full
//! `MUSIC::pseudospectrum` / `EigenspaceMV::compute_weights` calls so the
//! eigensolve share falls out as solver-time over call-time. The measurement
//! decides whether the callers adopt the QR path; see the backlog item for the
//! acceptance bound (eigenvalues/subspace agreement, unchanged peak positions).
//!
//! # Measured outcome (2026-10-30, `HermitianEigenConfig::default`)
//!
//! Direct solver time on Gram covariances `R = A Aᴴ + 0.5I`:
//!
//! - n = 8: Jacobi 30.6 µs, QR 31.0 µs (QR delegates to Jacobi at n ≤ 32).
//! - n = 32: Jacobi 2.99 ms, QR 3.44 ms (delegation + double validation).
//! - n = 120: Jacobi 168.6 ms (6 sweeps, off-diagonal 5e-14). QR does **not**
//!   converge: it exhausts all 1000 iterations with off-diagonal norm ≈ 6e3
//!   (≈ 57 s per call) and returns eigenvalues wrong by 80% relative on two
//!   independent covariance fixtures (Gram and Toeplitz-like). The MUSIC, ESMV,
//!   localization, and PAM callers therefore keep the Jacobi path; adopting
//!   the upstream QR implementation would be a correctness regression, not
//!   just a slowdown. The non-convergence is an upstream leto defect to report
//!   there, not a local copy to work around.
//!
//! The `qr/120` case is kept so the comparison stays reproducible, but the
//! group measurement budget is capped below so a full run stays in minutes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use eunomia::Complex64;
use kwavers_analysis::signal_processing::beamforming::adaptive::subspace::{EigenspaceMV, MUSIC};
use leto::{Array1, Array2};
use leto_ops::application::linalg::{
    hermitian_eigen_jacobi, hermitian_eigen_qr, HermitianEigenConfig,
};
use std::time::Duration;

/// Covariance sizes at the beamformer operating points.
const SIZES: [usize; 3] = [8, 32, 120];

/// Deterministic Hermitian covariance `R = A Aᴴ + σ²I`.
///
/// The fill is a fixed sinusoidal pattern stepped in `f64` so the fixture is
/// reproducible without an RNG dependency; all counters stay in their native
/// widths (no representation conversions).
fn hermitian_covariance(n: usize) -> Array2<Complex64> {
    let mut a = Array2::<Complex64>::zeros((n, n));
    let mut row_phase = 0.0_f64;
    for i in 0..n {
        let mut col_phase = 0.0_f64;
        for j in 0..n {
            let re = (row_phase * 0.37 + col_phase * 0.11).sin();
            let im = (row_phase * 0.13 - col_phase * 0.29).cos();
            a[[i, j]] = Complex64::new(re, im);
            col_phase += 1.0;
        }
        row_phase += 1.0;
    }
    let mut r = Array2::<Complex64>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let mut acc = Complex64::new(0.0, 0.0);
            for k in 0..n {
                acc += a[[i, k]] * a[[j, k]].conj();
            }
            r[[i, j]] = acc;
        }
        r[[i, i]] += Complex64::new(0.5, 0.0);
    }
    r
}

/// Deterministic unit-magnitude steering vector.
fn steering_vector(n: usize) -> Array1<Complex64> {
    let mut a = Array1::<Complex64>::zeros(n);
    let mut phase = 0.0_f64;
    for i in 0..n {
        a[i] = Complex64::new(phase.cos(), phase.sin());
        phase += 0.31;
    }
    a
}

fn bench_eigensolvers(c: &mut Criterion) {
    let mut group = c.benchmark_group("subspace_eigensolver");
    // The non-convergent `qr/120` case costs ~1 min per iteration; cap the
    // budget so a full run stays in minutes while the fast cases keep
    // enough samples for tight intervals.
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(60));
    for n in SIZES {
        let r = hermitian_covariance(n);
        let config = HermitianEigenConfig::default();
        group.bench_with_input(BenchmarkId::new("jacobi", n), &n, |b, _| {
            b.iter(|| hermitian_eigen_jacobi(black_box(&r), config).expect("jacobi converges"));
        });
        group.bench_with_input(BenchmarkId::new("qr", n), &n, |b, _| {
            b.iter(|| hermitian_eigen_qr(black_box(&r), config).expect("qr converges"));
        });
    }
    group.finish();
}

fn bench_music_pseudospectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("music_pseudospectrum");
    for n in SIZES {
        let music = MUSIC::new(2);
        let r = hermitian_covariance(n);
        let steering = steering_vector(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                music
                    .pseudospectrum(black_box(&r), black_box(&steering))
                    .expect("pseudospectrum finite")
            });
        });
    }
    group.finish();
}

fn bench_esmv_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("esmv_weights");
    for n in SIZES {
        let esmv = EigenspaceMV::new(2);
        let r = hermitian_covariance(n);
        let steering = steering_vector(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                esmv
                    .compute_weights(black_box(&r), black_box(&steering))
                    .expect("weights finite")
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_eigensolvers,
    bench_music_pseudospectrum,
    bench_esmv_weights,
);
criterion_main!(benches);
