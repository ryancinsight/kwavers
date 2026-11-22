//! Physical derivative throughput from borrowed SI samples to owned results.
use aequitas::systems::si::quantities::{Dimensionless, Frequency, Pressure, Time};
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use eunomia::Complex;
use kwavers_medium::absorption::FractionalDerivative;
use std::hint::black_box;

fn fractional_derivative(c: &mut Criterion) {
    let derivative =
        FractionalDerivative::new(Dimensionless::from_base(0.5), 32).expect("finite half order");
    let samples: Vec<_> = (0..4096).map(|i| f64::from(i).sin()).collect();
    let bins: Vec<_> = samples.iter().map(|&v| Complex::new(v, -v)).collect();
    let frequencies: Vec<_> = (-2048..2048)
        .map(|i| Frequency::from_base(f64::from(i) * 10.0))
        .collect();
    let mut group = c.benchmark_group("fractional_derivative");
    group.throughput(Throughput::Elements(4096));
    group.bench_function("causal", |b| {
        b.iter(|| {
            derivative
                .apply(
                    Pressure::from_base(black_box(samples.as_slice())),
                    Time::from_base(0.001),
                )
                .expect("finite causal derivative")
        })
    });
    group.bench_function("spectral", |b| {
        b.iter(|| {
            derivative
                .apply_frequency_domain(
                    Pressure::from_base(black_box(bins.as_slice())),
                    black_box(&frequencies),
                )
                .expect("matched signed frequencies")
        })
    });
    group.finish();
}

criterion_group!(benches, fractional_derivative);
criterion_main!(benches);
