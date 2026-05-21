//! Reduced-grid frequency-domain FWI spectral CBS performance validation.
//!
//! # Contract
//!
//! This benchmark validates the production public path for the Ali et al. 2025
//! frequency-domain 3-D FWI architecture on a reduced grid:
//!
//! 1. `MultiRowRingArray` supplies multi-row cylindrical transmissions.
//! 2. `simulate_frequency_observation` owns the solver entrypoint.
//! 3. `SpectralConvergentBorn` routes through the absorbed pseudospectral CBS
//!    Green operator.
//! 4. The validation prelude asserts finite pressure and sound-speed
//!    sensitivity before timing any kernel.

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use kwavers::physics::acoustics::imaging::modalities::ultrasound::frequency_domain_fwi::MultiRowRingArray;
use kwavers::solver::inverse::fwi::frequency_domain::{
    simulate_frequency_observation, AbsorbingBoundary, Config, PropagationModel,
};
use ndarray::Array3;
use std::time::Duration;

const GRID: (usize, usize, usize) = (24, 24, 18);
const SPACING_M: f64 = 0.002;
const FREQUENCY_HZ: f64 = 200_000.0;

fn spectral_cbs_reduced_grid(c: &mut Criterion) {
    let array = reduced_multi_row_array();
    let model = reduced_sound_speed_volume();
    let config = spectral_config();
    assert_reduced_grid_response(&model, &array, config);

    let mut group = c.benchmark_group("frequency_domain_fwi_spectral_cbs");
    group.throughput(Throughput::Elements(
        (GRID.0 * GRID.1 * GRID.2 * array.circumferential_elements()) as u64,
    ));
    group.bench_function("reduced_grid_prediction", |b| {
        b.iter(|| {
            let pressure = simulate_frequency_observation(
                black_box(&model),
                black_box(&array),
                black_box(FREQUENCY_HZ),
                black_box(config),
            )
            .expect("spectral CBS reduced-grid prediction");
            black_box(pressure[[0, 0]])
        });
    });
    group.finish();
}

fn reduced_multi_row_array() -> MultiRowRingArray {
    MultiRowRingArray::new(4, 3, 0.018, 0.0024).expect("reduced multi-row ring geometry")
}

fn reduced_sound_speed_volume() -> Array3<f64> {
    let mut model = Array3::from_elem(GRID, 1500.0);
    model[[GRID.0 / 2, GRID.1 / 2, GRID.2 / 2]] = 1515.0;
    model[[GRID.0 / 2 + 2, GRID.1 / 2 - 1, GRID.2 / 2]] = 1490.0;
    model
}

fn spectral_config() -> Config {
    Config {
        spacing_m: SPACING_M,
        iterations: 1,
        estimate_source_scaling: false,
        propagation_model: PropagationModel::SpectralConvergentBorn {
            iterations: 6,
            relative_tolerance: 1.0e-8,
            absorbing_boundary: AbsorbingBoundary::polynomial(2, 1.5, 2)
                .expect("valid reduced-grid absorbing boundary"),
        },
        ..Config::default()
    }
}

fn assert_reduced_grid_response(model: &Array3<f64>, array: &MultiRowRingArray, config: Config) {
    let baseline = simulate_frequency_observation(model, array, FREQUENCY_HZ, config)
        .expect("baseline reduced-grid spectral CBS prediction");
    let mut perturbed = model.clone();
    perturbed[[GRID.0 / 2, GRID.1 / 2, GRID.2 / 2]] += 5.0;
    let shifted = simulate_frequency_observation(&perturbed, array, FREQUENCY_HZ, config)
        .expect("perturbed reduced-grid spectral CBS prediction");
    let baseline_energy = baseline.iter().map(|value| value.norm_sqr()).sum::<f64>();
    let difference_energy = baseline
        .iter()
        .zip(shifted.iter())
        .map(|(&lhs, &rhs)| (lhs - rhs).norm_sqr())
        .sum::<f64>();

    assert!(baseline_energy.is_finite() && baseline_energy > 0.0);
    assert!(
        difference_energy > f64::EPSILON * baseline_energy.max(1.0),
        "difference_energy={difference_energy}, baseline_energy={baseline_energy}"
    );
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_millis(250))
        .measurement_time(Duration::from_secs(1));
    targets = spectral_cbs_reduced_grid
}
criterion_main!(benches);
