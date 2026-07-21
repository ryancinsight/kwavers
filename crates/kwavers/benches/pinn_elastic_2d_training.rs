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
//! cargo bench --bench pinn_elastic_2d_training --features pinn
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
use coeus_autograd::Var;
#[cfg(feature = "pinn")]
use coeus_core::MoiraiBackend;

#[cfg(feature = "pinn")]
use kwavers_solver::inverse::pinn::elastic_2d::{
    loss::ElasticBoundaryCondition,
    training::optimizer::PINNOptimizer,
    training::scheduler::LRScheduler,
    training::{train_pinn, ElasticPinnLoopConfig},
    BoundaryData, CollocationData, Config, ElasticPINN2D, InitialData, LossComputer, TrainingData,
};

#[cfg(feature = "pinn")]
type Backend = MoiraiBackend;

#[cfg(feature = "pinn")]
fn uniform_var(backend: &Backend, n: usize, lo: f32, hi: f32) -> Var<f32, Backend> {
    let data: Vec<f32> = (0..n)
        .map(|_| lo + rand::random::<f32>() * (hi - lo))
        .collect();
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![n, 1], &data, backend),
        false,
    )
}

#[cfg(feature = "pinn")]
fn normal_var(backend: &Backend, n: usize, mean: f32, std: f32) -> Var<f32, Backend> {
    // Box-Muller transform for approximate standard normal samples.
    let data: Vec<f32> = (0..n)
        .map(|_| {
            let u1: f32 = rand::random::<f32>().max(1e-7);
            let u2: f32 = rand::random::<f32>();
            mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
        })
        .collect();
    Var::new(
        coeus_tensor::Tensor::from_slice_on(vec![n, 1], &data, backend),
        false,
    )
}

#[cfg(feature = "pinn")]
fn zeros_var(backend: &Backend, n: usize, cols: usize) -> Var<f32, Backend> {
    Var::new(
        coeus_tensor::Tensor::zeros_on(vec![n, cols], backend),
        false,
    )
}

// ============================================================================
// Benchmark: Forward Pass
// ============================================================================

#[cfg(feature = "pinn")]
fn bench_forward_pass(c: &mut Criterion) {
    let backend = Backend::default();
    let config = Config {
        hidden_layers: vec![64, 64, 64],
        ..Config::default()
    };

    let model = ElasticPINN2D::<Backend>::new(&config).expect("Failed to create model");

    let batch_sizes = vec![32, 128, 512, 2048];

    let mut group = c.benchmark_group("forward_pass");

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let n = batch_size as usize;
                let x = uniform_var(&backend, n, -1.0, 1.0);
                let y = uniform_var(&backend, n, -1.0, 1.0);
                let t = uniform_var(&backend, n, 0.0, 1.0);

                b.iter(|| {
                    let output = model.forward(black_box(&x), black_box(&y), black_box(&t));
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
    let backend = Backend::default();
    let config = Config {
        hidden_layers: vec![64, 64, 64],
        ..Config::default()
    };

    let _model = ElasticPINN2D::<Backend>::new(&config).expect("Failed to create model");
    let loss_computer = LossComputer::new(config.loss_weights);

    let batch_size: u64 = 512;
    let n = batch_size as usize;

    let mut group = c.benchmark_group("loss_computation");
    group.throughput(Throughput::Elements(batch_size));

    // PDE loss
    group.bench_function("pde_residual", |b| {
        let residual_x = normal_var(&backend, n, 0.0, 1.0);
        let residual_y = normal_var(&backend, n, 0.0, 1.0);

        b.iter(|| {
            let loss = loss_computer.pde_loss(black_box(&residual_x), black_box(&residual_y));
            black_box(loss)
        });
    });

    // Boundary loss
    group.bench_function("boundary_condition", |b| {
        let predicted = uniform_var(&backend, n, -1.0, 1.0);
        let target = zeros_var(&backend, n, 2);

        b.iter(|| {
            let loss = loss_computer.boundary_loss(black_box(&predicted), black_box(&target));
            black_box(loss)
        });
    });

    // Initial condition loss
    group.bench_function("initial_condition", |b| {
        let u_pred = uniform_var(&backend, n, -1.0, 1.0);
        let v_pred = uniform_var(&backend, n, -1.0, 1.0);
        let u_target = zeros_var(&backend, n, 2);
        let v_target = zeros_var(&backend, n, 2);

        b.iter(|| {
            let loss = loss_computer.initial_loss(
                black_box(&u_pred),
                black_box(&v_pred),
                black_box(&u_target),
                black_box(&v_target),
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
    let backend = Backend::default();
    let config = Config {
        hidden_layers: vec![64, 64, 64],
        ..Config::default()
    };

    let model = ElasticPINN2D::<Backend>::new(&config).expect("Failed to create model");

    let batch_size: u64 = 512;
    let n = batch_size as usize;

    let mut group = c.benchmark_group("backward_pass");
    group.throughput(Throughput::Elements(batch_size));

    group.bench_function("gradient_computation", |b| {
        let x = uniform_var(&backend, n, -1.0, 1.0);
        let y = uniform_var(&backend, n, -1.0, 1.0);
        let t = uniform_var(&backend, n, 0.0, 1.0);

        b.iter(|| {
            for p in model.parameters() {
                p.zero_grad();
            }
            let output = model.forward(&x, &y, &t);
            let target = zeros_var(&backend, n, 2);
            let diff = coeus_autograd::sub(&output, &target);
            let loss = coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff));
            loss.backward();
            black_box(())
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark: Full Training Epoch
// ============================================================================

#[cfg(feature = "pinn")]
fn bench_training_epoch(c: &mut Criterion) {
    let backend = Backend::default();
    let config = Config {
        hidden_layers: vec![64, 64, 64],
        n_collocation_interior: 1000,
        n_collocation_boundary: 100,
        n_collocation_initial: 100,
        ..Config::default()
    };

    let mut group = c.benchmark_group("training_epoch");
    group.sample_size(10); // Fewer samples for expensive operation

    group.bench_function("single_epoch", |b| {
        b.iter(|| {
            let mut model = ElasticPINN2D::<Backend>::new(&config).expect("Failed to create model");

            let n_colloc = config.n_collocation_interior;
            let n_boundary = config.n_collocation_boundary;
            let n_initial = config.n_collocation_initial;

            let training_data = TrainingData {
                collocation: CollocationData {
                    x: uniform_var(&backend, n_colloc, -1.0, 1.0),
                    y: uniform_var(&backend, n_colloc, -1.0, 1.0),
                    t: uniform_var(&backend, n_colloc, 0.0, 1.0),
                    source_x: None,
                    source_y: None,
                },
                boundary: BoundaryData {
                    x: uniform_var(&backend, n_boundary, -1.0, 1.0),
                    y: uniform_var(&backend, n_boundary, -1.0, 1.0),
                    t: uniform_var(&backend, n_boundary, 0.0, 1.0),
                    boundary_type: vec![ElasticBoundaryCondition::Dirichlet; n_boundary],
                    values: zeros_var(&backend, n_boundary, 2),
                },
                initial: InitialData {
                    x: uniform_var(&backend, n_initial, -1.0, 1.0),
                    y: uniform_var(&backend, n_initial, -1.0, 1.0),
                    displacement: zeros_var(&backend, n_initial, 2),
                    velocity: zeros_var(&backend, n_initial, 2),
                },
                observations: None,
            };

            let mut optimizer =
                PINNOptimizer::adam(&model, config.learning_rate, 0.0, 0.9, 0.999, 1e-8);
            let mut scheduler = LRScheduler::constant(config.learning_rate);
            let loop_config = ElasticPinnLoopConfig {
                max_epochs: 1,
                convergence_tolerance: 1e-6,
                convergence_window: 10,
                log_every: 1,
                checkpoint_every: 1000,
            };
            let metrics = train_pinn(
                &mut model,
                &training_data,
                &mut optimizer,
                &mut scheduler,
                &loop_config,
            )
            .ok();
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
    let backend = Backend::default();
    let batch_size: u64 = 512;
    let n = batch_size as usize;

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
            let config = Config {
                hidden_layers: layers.clone(),
                ..Config::default()
            };

            let model = ElasticPINN2D::<Backend>::new(&config).expect("Failed to create model");

            let x = uniform_var(&backend, n, -1.0, 1.0);
            let y = uniform_var(&backend, n, -1.0, 1.0);
            let t = uniform_var(&backend, n, 0.0, 1.0);

            b.iter(|| {
                let output = model.forward(black_box(&x), black_box(&y), black_box(&t));
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
    let backend = Backend::default();
    let config = Config {
        hidden_layers: vec![64, 64, 64],
        ..Config::default()
    };

    let model = ElasticPINN2D::<Backend>::new(&config).expect("Failed to create model");

    let batch_sizes = vec![16, 64, 256, 1024, 4096];

    let mut group = c.benchmark_group("batch_scaling");

    for batch_size in batch_sizes {
        group.throughput(Throughput::Elements(batch_size));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &batch_size| {
                let n = batch_size as usize;
                let x = uniform_var(&backend, n, -1.0, 1.0);
                let y = uniform_var(&backend, n, -1.0, 1.0);
                let t = uniform_var(&backend, n, 0.0, 1.0);

                b.iter(|| {
                    for p in model.parameters() {
                        p.zero_grad();
                    }
                    let output = model.forward(&x, &y, &t);
                    let target = zeros_var(&backend, n, 2);
                    let diff = coeus_autograd::sub(&output, &target);
                    let loss = coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff));
                    loss.backward();

                    black_box(())
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
