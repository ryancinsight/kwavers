use criterion::{criterion_group, criterion_main, Criterion};
use kwavers::solver::inverse::reconstruction::photoacoustic::{
    IterativeAlgorithm, PhotoacousticAlgorithm, PhotoacousticConfig, PhotoacousticReconstructor,
};
use ndarray::Array2;

fn osem_reconstruction_benchmark(c: &mut Criterion) {
    let grid_size = [16, 16, 16]; // 4096 voxels
    let n_sensors = 64;

    // Create random sensor data
    let sensor_data = Array2::zeros((n_sensors, 100)); // 100 samples
    let sensor_positions: Vec<[f64; 3]> = (0..n_sensors)
        .map(|i| [i as f64 * 0.001, 0.0, 0.0])
        .collect();

    // Create OSEM config
    let config = PhotoacousticConfig {
        algorithm: PhotoacousticAlgorithm::Iterative {
            algorithm: IterativeAlgorithm::OSEM { subsets: 4 },
            iterations: 5,
            relaxation_factor: 1.0,
        },
        sensor_positions: sensor_positions.clone(),
        grid_size,
        sound_speed: 1500.0,
        sampling_frequency: 10e6,
        envelope_detection: false,
        bandpass_filter: None,
        regularization_parameter: 0.0,
    };

    let reconstructor = PhotoacousticReconstructor::new(config);

    c.bench_function("osem_reconstruction", |b| {
        b.iter(|| {
            reconstructor.iterative_reconstruction(sensor_data.view(), &sensor_positions, grid_size)
        })
    });
}

criterion_group!(benches, osem_reconstruction_benchmark);
criterion_main!(benches);
