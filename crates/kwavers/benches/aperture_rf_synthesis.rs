//! Baseline for finite-aperture RF synthesis, the Field II path of ADR 113.
//!
//! `ScattererCloud::synthesize_rf_with_aperture` convolves each echo with the
//! element's round-trip spatial impulse response (SIR). Its cost is set per
//! element–scatterer pair, so the instrument fixes one array and one pulse and
//! scales the scatterer count: the per-pair figure is the throughput readout,
//! and the two scatterer counts show whether anything outside the pair loop
//! (per-element work, allocation) is visible at this scale.
//!
//! The scene is a small linear array of circular pistons over a deterministic
//! scatterer cloud spanning 10–40 mm, at 100 MHz sampling — the two-way kernel
//! is then a median 10 samples wide and up to about 54 across the cloud, so a
//! per-pair cost proportional to the trace length (6000 samples) shows against
//! one proportional to the kernel alone (Rivera, Demené & Tanter 2026, the
//! sparse-delta cost model).
//!
//! # Reading the numbers
//!
//! Throughput is element–scatterer pairs per second. Halving the scatterer
//! count should halve the time; a residual that does not scale is the
//! per-element cost.

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, SamplingMode, Throughput,
};
use kwavers_phantom::scatterers::{ApertureElement, RfSynthesisConfig, ScattererCloud};
use kwavers_physics::analytical::transducer::spatial_impulse_response::CircularPistonSir;
use std::time::Duration;

const SOUND_SPEED: f64 = 1540.0;
const SAMPLING_FREQUENCY: f64 = 100.0e6;
const ELEMENT_RADIUS: f64 = 0.5e-3;
const ELEMENT_PITCH: f64 = 1.0e-3;
const ELEMENT_COUNT: usize = 8;
/// Deepest scatterer at 40 mm, laterally up to 4 mm off the array end; the
/// window outruns the farthest round trip so no echo is clipped.
const NUM_SAMPLES: usize = 6000;
const SCATTERER_COUNTS: [usize; 2] = [16, 64];

/// A two-cycle Gaussian-windowed 3 MHz pulse on the sampling grid.
fn pulse() -> Vec<f64> {
    let f0 = 3.0e6;
    let cycles = 2.0;
    let n = (cycles / f0 * SAMPLING_FREQUENCY).ceil() as usize;
    (0..n)
        .map(|k| {
            let t = k as f64 / SAMPLING_FREQUENCY;
            let centre = 0.5 * cycles / f0;
            let sigma = 0.25 * cycles / f0;
            let window = (-0.5 * ((t - centre) / sigma).powi(2)).exp();
            window * (2.0 * std::f64::consts::PI * f0 * t).sin()
        })
        .collect()
}

fn elements() -> Vec<ApertureElement> {
    (0..ELEMENT_COUNT)
        .map(|i| {
            let x = (i as f64 - 0.5 * (ELEMENT_COUNT as f64 - 1.0)) * ELEMENT_PITCH;
            ApertureElement::new([x, 0.0, 0.0], [0.0, 0.0, 1.0]).expect("finite unit normal")
        })
        .collect()
}

/// Deterministic scatterers from a linear congruential generator, so the
/// workload is pinned across runs and revisions.
fn cloud(count: usize) -> ScattererCloud {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut unit = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut cloud = ScattererCloud::new();
    for _ in 0..count {
        let x = (unit() - 0.5) * 8.0e-3;
        let y = (unit() - 0.5) * 2.0e-3;
        let z = 10.0e-3 + unit() * 30.0e-3;
        let amplitude = if unit() < 0.5 { 1.0 } else { -1.0 };
        cloud.push([x, y, z], amplitude);
    }
    cloud
}

fn aperture_rf_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("aperture_rf_synthesis");
    // Budgeted rather than left at criterion's defaults: an iteration was
    // hundreds of milliseconds on the code this instrument was built against,
    // so flat sampling at the criterion floor keeps the group inside a few
    // seconds of the suite's committed bound whatever the code under test.
    group.sampling_mode(SamplingMode::Flat);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(10);

    let config = RfSynthesisConfig {
        sound_speed: SOUND_SPEED,
        sampling_frequency: SAMPLING_FREQUENCY,
        num_samples: NUM_SAMPLES,
        min_distance: 1.0e-3,
        attenuation_db_cm_mhz: 0.5,
        center_frequency_hz: 3.0e6,
    };
    let piston = CircularPistonSir::new(ELEMENT_RADIUS, SOUND_SPEED).expect("valid piston");
    let kernel = |r: f64, z: f64, dt: f64| piston.round_trip_response(r, z, dt).samples;
    let pulse = pulse();
    let elements = elements();

    for count in SCATTERER_COUNTS {
        let cloud = cloud(count);
        group.throughput(Throughput::Elements((ELEMENT_COUNT * count) as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                cloud
                    .synthesize_rf_with_aperture(
                        black_box(&elements),
                        black_box(&pulse),
                        &config,
                        &kernel,
                    )
                    .expect("synthesis succeeds on a valid scene")
            });
        });
    }

    group.finish();
}

criterion_group!(benches, aperture_rf_synthesis);
criterion_main!(benches);
