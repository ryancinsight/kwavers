//! Time one RF-line correction and one complex spectrum update, excluding setup.
use aequitas::systems::si::quantities::{Frequency, Length, Velocity};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion, Throughput};
use eunomia::Complex64;
use kwavers_analysis::signal_processing::b_mode::TgcConfig;
use kwavers_medium::absorption::{PowerLawAbsorption, PowerLawModel};
use leto::{Array1, Array3};
use std::hint::black_box;

fn attenuation(c: &mut Criterion) {
    let config = TgcConfig::new(
        kwavers_medium::absorption::power_law::reference_loss_from_decibels_per_centimetre(0.5)
            / Frequency::from_base(1e6),
        Frequency::from_base(5e6),
        Velocity::from_base(1540.0),
        Frequency::from_base(40e6),
    )
    .expect("valid RF calibration");
    let mut group = c.benchmark_group("acoustic_attenuation");
    let line = Array1::from_shape_fn(8192, |[i]| (i as f64).sin());
    group.throughput(Throughput::Elements(line.len() as u64));
    group.bench_function("rf_line", |b| {
        b.iter(|| black_box(config.apply(black_box(&line)).expect("valid RF line")))
    });

    let model = PowerLawModel::new(PowerLawAbsorption::soft_tissue());
    let frequencies: Vec<_> = (0..64)
        .map(|i| Frequency::from_base(i as f64 * 1e5))
        .collect();
    let spectrum = Array3::from_shape_fn([64, 64, 64], |[i, j, k]| {
        Complex64::new((i + j) as f64, -(k as f64))
    });
    group.throughput(Throughput::Elements(spectrum.len() as u64));
    group.bench_function("spectrum", |b| {
        b.iter_batched_ref(
            || spectrum.clone(),
            |wave| {
                model
                    .apply_frequency_domain(black_box(wave), &frequencies, Length::from_base(0.02))
                    .expect("matching frequency axis")
            },
            BatchSize::LargeInput,
        )
    });
    group.finish();
}

criterion_group!(benches, attenuation);
criterion_main!(benches);
