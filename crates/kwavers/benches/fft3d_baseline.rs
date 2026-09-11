//! Baseline for the 3-D complex FFT, the kernel a PSTD timestep is built on.
//!
//! PSTD advances a field by transforming it, applying a k-space operator and
//! transforming back, so the spectral pair is the step's dominant cost — and it
//! was, until this bench, the only part of the critical path with no timing at
//! the shapes the solver actually plans. `critical_path_benchmarks` measures
//! `compute_kx`/`compute_ky`, which build the k-space vectors once per plan,
//! not the transform that runs every step.
//!
//! Shapes are the ones kwavers plans rather than a sweep: 32³ and 64³ are what
//! `fdtd_propagation_benchmark` fixes and what the PSTD paths construct through
//! `get_fft_for_grid`, and 16³ anchors the small end so the scaling is visible
//! rather than inferred from two points.
//!
//! # Reading the numbers
//!
//! The timed operation is a forward followed by a normalized inverse — one
//! round trip, not one transform — so halve it for a per-transform figure. The
//! round trip is what keeps the instrument honest at these sizes: apollo's
//! inverse is FFTW-normalized, so the buffer returns to its input and no
//! per-iteration reseed is needed. Copying a 64³ volume (16 MB) into place
//! before each iteration would have charged that memcpy to the transform.
//!
//! What a round trip cannot show is a forward/inverse asymmetry, since it
//! reports their sum. That is a limit on the quantity's name, not on its use
//! here: the PSTD step runs both directions once each, so the pair is the
//! quantity the solver actually pays.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use kwavers_math::fft::{get_fft_for_grid, Complex64, Fft3dInOutExt};
use leto::Array3;
use std::time::Duration;

/// Grid extents the solver plans, plus a small anchor for the scaling.
const EXTENTS: [usize; 3] = [16, 32, 64];

/// Deterministic, non-degenerate input.
///
/// A zero or constant field would let the transform's own arithmetic collapse
/// and is not what the solver transforms.
fn volume(n: usize) -> Array3<Complex64> {
    Array3::from_shape_fn([n, n, n], |index| {
        let x = (index[0] + n * index[1] + n * n * index[2]) as f64;
        Complex64::new((0.017 * x).sin(), 0.25 * (0.031 * x).cos())
    })
}

fn fft3d_round_trip(c: &mut Criterion) {
    let mut group = c.benchmark_group("fft3d_round_trip");
    // Budgeted rather than left at criterion's defaults: three shapes at the
    // default 3 s warm-up plus 5 s measurement is 24 s of this suite's
    // committed wall-clock bound for one group. The 64³ iteration is
    // milliseconds, so twenty samples over a second still resolves it well
    // inside the noise this host produces.
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);

    for n in EXTENTS {
        let plan = get_fft_for_grid(n, n, n);
        let mut work = volume(n);

        // Plan construction and first-call scratch allocation are warm-plan
        // costs the solver pays once, not per step; run one round trip outside
        // the timer so they are not charged to it.
        plan.forward_complex_inplace(&mut work);
        plan.inverse_complex_inplace(&mut work);

        group.throughput(criterion::Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                plan.forward_complex_inplace(black_box(&mut work));
                plan.inverse_complex_inplace(black_box(&mut work));
            });
        });
    }

    group.finish();

    // The path a PSTD step takes: a real field through the half-spectrum pair
    // and back, the inverse consuming the spectrum as the solver lets it. Beside
    // the complex pair above, the difference is what the real field saves by
    // never being widened to complex.
    let mut group = c.benchmark_group("fft3d_r2c_round_trip");
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);
    for n in EXTENTS {
        let plan = get_fft_for_grid(n, n, n);
        let real = volume(n).mapv(|c| c.re);
        let mut half = Array3::from_elem([n, n, n / 2 + 1], Complex64::default());
        let mut back = Array3::from_elem([n, n, n], 0.0_f64);
        plan.forward_r2c_into(&real, &mut half);
        plan.inverse_c2r_into(&mut half, &mut back);
        group.throughput(criterion::Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                plan.forward_r2c_into(black_box(&real), &mut half);
                plan.inverse_c2r_into(black_box(&mut half), &mut back);
                black_box(back[[0, 0, 0]])
            });
        });
    }
    group.finish();
}

criterion_group!(benches, fft3d_round_trip);
criterion_main!(benches);
