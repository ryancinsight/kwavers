use criterion::{criterion_group, criterion_main, Criterion};
use kwavers::solver::amr::ErrorEstimator;
use ndarray::Array3;

fn gradient_error_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_error");

    let nx = 64; // Use 64 to keep it reasonably fast but significant
    let ny = 64;
    let nz = 64;
    let mut field = Array3::<f64>::zeros((nx, ny, nz));

    // Fill with some data
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                field[[i, j, k]] = ((i as f64) * 0.1).sin() * ((j as f64) * 0.1).cos() + (k as f64);
            }
        }
    }

    let estimator = ErrorEstimator::new();

    group.bench_function("gradient_error_64", |b| {
        b.iter(|| estimator.estimate_error(&field).unwrap());
    });

    group.finish();
}

criterion_group!(benches, gradient_error_benchmark);
criterion_main!(benches);
