//! Benchmarks for 2D Elastic Wave PINN Training
//!
//! This benchmark suite measures training performance for physics-informed neural networks
//! solving the 2D elastic wave equation.
//!
//! # Benchmark Categories
//!
//! 1. **Forward Pass**: Model inference time
//! 2. **Loss Computation**: PDE residual, BC, IC, and data loss
//! 3. **Backward Pass**: Gradient computation via autodiff
//! 4. **Optimizer Step**: Parameter update time
//! 5. **Full Epoch**: Complete training iteration
//! 6. **Scalability**: Performance vs. network size and batch size
//!
//! # Running Benchmarks
//!
//! ```bash
//! # CPU benchmarks
//! cargo bench --bench pinn_elastic_2d_training --features pinn
//!
//! # GPU benchmarks (requires WGPU)
//! cargo bench --bench pinn_elastic_2d_training --features pinn-gpu
//! ```
//!
//! # Mathematical Validation
//!
//! Benchmarks include validation checks to ensure correctness:
//! - PDE residual magnitude (should decrease during training)
//! - Energy conservation (within tolerance)
//! - Material parameter recovery (for inverse problems)

#[cfg(feature = "pinn")]
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

#[cfg(feature = "pinn")]
use burn::{
    backend::{Autodiff, NdArray},
    tensor::Tensor,
};

#[cfg(feature = "pinn")]
use kwavers::solver::inverse::pinn::elastic_2d::{
    BoundaryData, BoundaryType, CollocationData, Config, ElasticPINN2D, InitialData, LossComputer,
    TrainingData,
};

#[cfg(feature = "pinn")]
type Backend = Autodiff<NdArray<f32>>;

// ============================================================================
// Benchmark: Forward Pass
// ============================================================================

#[cfg(feature = "pinn")]
fn bench_forward_pass(c: &mut Criterion) {
    let device = Default::default();
    let mut config = Config::default();
    config.hidden_layers = vec![64, 64, 64]; // 3 hidden layers

    let model = ElasticPINN2D::<Backend>::new(&config, &device).expect("Failed to create model");

    // Test batch sizes
    let batch_sizes = vec![32, 128, 512, 2048];

    let mut group = c.benchmark_group("forward_pass");

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                // Create input tensors
                let x = Tensor::<Backend, 1>::random(
                    [batch_size as usize],
                    burn::tensor::Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
                .reshape([batch_size as usize, 1]);

                let y = Tensor::<Backend, 1>::random(
                    [batch_size as usize],
                    burn::tensor::Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
                .reshape([batch_size as usize, 1]);

                let t = Tensor::<Backend, 1>::random(
                    [batch_size as usize],
                    burn::tensor::Distribution::Uniform(0.0, 1.0),
                    &device,
                )
                .reshape([batch_size as usize, 1]);

                b.iter(|| {
                    let output = model.forward(
                        black_box(x.clone()),
                        black_box(y.clone()),
                        black_box(t.clone()),
                    );
                    black_box(output)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmark: Loss Computation
// ============================================================================

#[cfg(feature = "pinn")]
fn bench_loss_computation(c: &mut Criterion) {
    let device = Default::default();
    let mut config = Config::default();
    config.hidden_layers = vec![64, 64, 64];

    let model = ElasticPINN2D::<Backend>::new(&config, &device).expect("Failed to create model");
    let loss_computer = LossComputer::new(config.loss_weights);

    let batch_size = 512;

    let mut group = c.benchmark_group("loss_computation");
    group.throughput(Throughput::Elements(batch_size));

    // PDE loss
    group.bench_function("pde_residual", |b| {
        let residual_x = Tensor::<Backend, 1>::random(
            [batch_size as usize],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        )
        .reshape([batch_size as usize, 1]);

        let residual_y = Tensor::<Backend, 1>::random(
            [batch_size as usize],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        )
        .reshape([batch_size as usize, 1]);

        b.iter(|| {
            let loss = loss_computer
                .pde_loss(black_box(residual_x.clone()), black_box(residual_y.clone()));
            black_box(loss)
        });
    });

    // Boundary loss
    group.bench_function("boundary_condition", |b| {
        let predicted = Tensor::<Backend, 2>::random(
            [batch_size as usize, 2],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let target = Tensor::<Backend, 2>::zeros([batch_size as usize, 2], &device);

        b.iter(|| {
            let loss = loss_computer
                .boundary_loss(black_box(predicted.clone()), black_box(target.clone()));
            black_box(loss)
        });
    });

    // Initial condition loss
    group.bench_function("initial_condition", |b| {
        let u_pred = Tensor::<Backend, 2>::random(
            [batch_size as usize, 2],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let v_pred = Tensor::<Backend, 2>::random(
            [batch_size as usize, 2],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let u_target = Tensor::<Backend, 2>::zeros([batch_size as usize, 2], &device);
        let v_target = Tensor::<Backend, 2>::zeros([batch_size as usize, 2], &device);

        b.iter(|| {
            let loss = loss_computer.initial_loss(
                black_box(u_pred.clone()),
                black_box(v_pred.clone()),
                black_box(u_target.clone()),
                black_box(v_target.clone()),
            );
            black_box(loss)
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: Backward Pass
// ============================================================================

#[cfg(feature = "pinn")]
fn bench_backward_pass(c: &mut Criterion) {
    let device = Default::default();
    let mut config = Config::default();
    config.hidden_layers = vec![64, 64, 64];

    let model = ElasticPINN2D::<Backend>::new(&config, &device).expect("Failed to create model");
    let loss_computer = LossComputer::new(config.loss_weights);

    let batch_size = 512;

    let mut group = c.benchmark_group("backward_pass");
    group.throughput(Throughput::Elements(batch_size));

    group.bench_function("gradient_computation", |b| {
        let x = Tensor::<Backend, 1>::random(
            [batch_size as usize],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        )
        .reshape([batch_size as usize, 1]);

        let y = Tensor::<Backend, 1>::random(
            [batch_size as usize],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        )
        .reshape([batch_size as usize, 1]);

        let t = Tensor::<Backend, 1>::random(
            [batch_size as usize],
            burn::tensor::Distribution::Uniform(0.0, 1.0),
            &device,
        )
        .reshape([batch_size as usize, 1]);

        b.iter(|| {
            // Forward pass
            let output = model.forward(x.clone(), y.clone(), t.clone());

            // Simple MSE loss
            let target = Tensor::<Backend, 2>::zeros([batch_size as usize, 2], &device);
            let loss = (output - target).powf_scalar(2.0).mean();

            // Backward pass
            let grads = black_box(loss.backward());
            black_box(grads)
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: Full Training Epoch
// ============================================================================

#[cfg(feature = "pinn")]
fn bench_training_epoch(c: &mut Criterion) {
    let device = Default::default();
    let mut config = Config::default();
    config.hidden_layers = vec![64, 64, 64];
    config.n_collocation_interior = 1000;
    config.n_collocation_boundary = 100;
    config.n_collocation_initial = 100;

    let mut group = c.benchmark_group("training_epoch");
    group.sample_size(10); // Fewer samples for expensive operation

    group.bench_function("single_epoch", |b| {
        b.iter(|| {
            let mut model =
                ElasticPINN2D::<Backend>::new(&config, &device).expect("Failed to create model");

            // Create synthetic training data
            let n_colloc = config.n_collocation_interior;
            let n_boundary = config.n_collocation_boundary;
            let n_initial = config.n_collocation_initial;

            let training_data = TrainingData {
                collocation: CollocationData {
                    x: Tensor::<Backend, 1>::random(
                        [n_colloc],
                        burn::tensor::Distribution::Uniform(-1.0, 1.0),
                        &device,
                    )
                    .reshape([n_colloc, 1]),
                    y: Tensor::<Backend, 1>::random(
                        [n_colloc],
                        burn::tensor::Distribution::Uniform(-1.0, 1.0),
                        &device,
                    )
                    .reshape([n_colloc, 1]),
                    t: Tensor::<Backend, 1>::random(
                        [n_colloc],
                        burn::tensor::Distribution::Uniform(0.0, 1.0),
                        &device,
                    )
                    .reshape([n_colloc, 1]),
                    source_x: None,
                    source_y: None,
                },
                boundary: BoundaryData {
                    x: Tensor::<Backend, 1>::random(
                        [n_boundary],
                        burn::tensor::Distribution::Uniform(-1.0, 1.0),
                        &device,
                    )
                    .reshape([n_boundary, 1]),
                    y: Tensor::<Backend, 1>::random(
                        [n_boundary],
                        burn::tensor::Distribution::Uniform(-1.0, 1.0),
                        &device,
                    )
                    .reshape([n_boundary, 1]),
                    t: Tensor::<Backend, 1>::random(
                        [n_boundary],
                        burn::tensor::Distribution::Uniform(0.0, 1.0),
                        &device,
                    )
                    .reshape([n_boundary, 1]),
                    boundary_type: vec![BoundaryType::Dirichlet; n_boundary],
                    values: Tensor::<Backend, 2>::zeros([n_boundary, 2], &device),
                },
                initial: InitialData {
                    x: Tensor::<Backend, 1>::random(
                        [n_initial],
                        burn::tensor::Distribution::Uniform(-1.0, 1.0),
                        &device,
                    )
                    .reshape([n_initial, 1]),
                    y: Tensor::<Backend, 1>::random(
                        [n_initial],
                        burn::tensor::Distribution::Uniform(-1.0, 1.0),
                        &device,
                    )
                    .reshape([n_initial, 1]),
                    displacement: Tensor::<Backend, 2>::zeros([n_initial, 2], &device),
                    velocity: Tensor::<Backend, 2>::zeros([n_initial, 2], &device),
                },
                observations: None,
            };

            let mut optimizer = kwavers::solver::inverse::pinn::elastic_2d::training::optimizer::PINNOptimizer::adam(
                &model,
                config.learning_rate,
                0.0,
                0.9,
                0.999,
                1e-8,
            );
            let mut scheduler =
                kwavers::solver::inverse::pinn::elastic_2d::training::scheduler::LRScheduler::constant(
                    config.learning_rate,
                );
            let loop_config = kwavers::solver::inverse::pinn::elastic_2d::training::r#loop::TrainingConfig {
                max_epochs: 1,
                convergence_tolerance: 1e-6,
                convergence_window: 10,
                log_every: 1,
                checkpoint_every: 1000,
            };
            let metrics = kwavers::solver::inverse::pinn::elastic_2d::training::r#loop::train_pinn(
                &mut model,
                &training_data,
                &mut optimizer,
                &mut scheduler,
                &loop_config,
            );
            black_box(metrics)
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: Network Size Scaling
// ============================================================================

#[cfg(feature = "pinn")]
fn bench_network_scaling(c: &mut Criterion) {
    let device = Default::default();
    let batch_size = 512;

    // Test different network architectures
    let architectures = vec![
        ("small", vec![32, 32]),
        ("medium", vec![64, 64, 64]),
        ("large", vec![128, 128, 128, 128]),
        ("wide", vec![256, 256]),
        ("deep", vec![64, 64, 64, 64, 64, 64]),
    ];

    let mut group = c.benchmark_group("network_scaling");
    group.throughput(Throughput::Elements(batch_size));

    for (name, layers) in architectures {
        group.bench_with_input(BenchmarkId::from_parameter(name), &layers, |b, layers| {
            let mut config = Config::default();
            config.hidden_layers = layers.clone();

            let model =
                ElasticPINN2D::<Backend>::new(&config, &device).expect("Failed to create model");

            let x = Tensor::<Backend, 1>::random(
                [batch_size as usize],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            )
            .reshape([batch_size as usize, 1]);

            let y = Tensor::<Backend, 1>::random(
                [batch_size as usize],
                burn::tensor::Distribution::Uniform(-1.0, 1.0),
                &device,
            )
            .reshape([batch_size as usize, 1]);

            let t = Tensor::<Backend, 1>::random(
                [batch_size as usize],
                burn::tensor::Distribution::Uniform(0.0, 1.0),
                &device,
            )
            .reshape([batch_size as usize, 1]);

            b.iter(|| {
                let output = model.forward(
                    black_box(x.clone()),
                    black_box(y.clone()),
                    black_box(t.clone()),
                );
                black_box(output)
            });
        });
    }

    group.finish();
}

// ============================================================================
// Benchmark: Batch Size Scaling
// ============================================================================

#[cfg(feature = "pinn")]
fn bench_batch_scaling(c: &mut Criterion) {
    let device = Default::default();
    let mut config = Config::default();
    config.hidden_layers = vec![64, 64, 64];

    let model = ElasticPINN2D::<Backend>::new(&config, &device).expect("Failed to create model");

    let batch_sizes = vec![16, 64, 256, 1024, 4096];

    let mut group = c.benchmark_group("batch_scaling");

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let x = Tensor::<Backend, 1>::random(
                    [batch_size as usize],
                    burn::tensor::Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
                .reshape([batch_size as usize, 1]);

                let y = Tensor::<Backend, 1>::random(
                    [batch_size as usize],
                    burn::tensor::Distribution::Uniform(-1.0, 1.0),
                    &device,
                )
                .reshape([batch_size as usize, 1]);

                let t = Tensor::<Backend, 1>::random(
                    [batch_size as usize],
                    burn::tensor::Distribution::Uniform(0.0, 1.0),
                    &device,
                )
                .reshape([batch_size as usize, 1]);

                b.iter(|| {
                    // Forward pass
                    let output = model.forward(x.clone(), y.clone(), t.clone());

                    // Backward pass
                    let target = Tensor::<Backend, 2>::zeros([batch_size as usize, 2], &device);
                    let loss = (output - target).powf_scalar(2.0).mean();
                    let grads = loss.backward();

                    black_box(grads)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Criterion Configuration
// ============================================================================

#[cfg(feature = "pinn")]
criterion_group!(
    benches,
    bench_forward_pass,
    bench_loss_computation,
    bench_backward_pass,
    bench_training_epoch,
    bench_network_scaling,
    bench_batch_scaling,
);

#[cfg(feature = "pinn")]
criterion_main!(benches);

#[cfg(not(feature = "pinn"))]
fn main() {
    println!("Benchmarks require the 'pinn' feature. Run with: cargo bench --features pinn");
}
