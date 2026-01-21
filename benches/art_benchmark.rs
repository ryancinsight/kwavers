use criterion::{criterion_group, criterion_main, Criterion};
use kwavers::solver::inverse::reconstruction::photoacoustic::{
    PhotoacousticConfig, PhotoacousticAlgorithm, PhotoacousticReconstructor, IterativeAlgorithm
};
use kwavers::solver::reconstruction::{Reconstructor, ReconstructionConfig};
use kwavers::domain::grid::Grid;
use ndarray::Array2;

fn art_reconstruction_benchmark(c: &mut Criterion) {
    let grid_size = [32, 32, 32]; // 32768 voxels
    let n_sensors = 64;

    // Create random sensor data
    let sensor_data = Array2::zeros((n_sensors, 100)); // 100 samples
    let sensor_positions: Vec<[f64; 3]> = (0..n_sensors).map(|i| [i as f64 * 0.001, 0.0, 0.0]).collect();

    // Create ART config
    let config = PhotoacousticConfig {
        algorithm: PhotoacousticAlgorithm::Iterative {
            algorithm: IterativeAlgorithm::ART,
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
    let grid = Grid::default();
    let recon_config = ReconstructionConfig::default();

    c.bench_function("art_reconstruction", |b| {
        b.iter(|| {
            reconstructor.reconstruct(
                &sensor_data,
                &sensor_positions,
                &grid,
                &recon_config,
            )
        })
    });
}

criterion_group!(benches, art_reconstruction_benchmark);
criterion_main!(benches);
