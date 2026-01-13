# PINN Training Quick Start Guide

**Quick reference for training 2D Elastic Wave PINNs in kwavers**

---

## Basic Usage

### 1. Minimal Training Example

```rust
use burn::backend::{Autodiff, NdArray};
use kwavers::solver::inverse::pinn::elastic_2d::{
    Config, ElasticPINN2D, Trainer, TrainingData,
    CollocationData, BoundaryData, InitialData,
};

type Backend = Autodiff<NdArray<f32>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configuration
    let config = Config::default();
    
    // Model
    let device = Default::default();
    let model = ElasticPINN2D::<Backend>::new(&config, &device)?;
    
    // Trainer
    let mut trainer = Trainer::<Backend>::new(model, config);
    
    // Training data
    let training_data = create_training_data(&device);
    
    // Train
    let metrics = trainer.train(&training_data)?;
    
    println!("Final loss: {:.6e}", metrics.final_loss().unwrap());
    
    Ok(())
}
```

---

## Configuration Presets

### Forward Problem (Wave Propagation)

```rust
let config = Config::forward_problem(
    vec![64, 64, 64, 64],  // Network architecture
    1e-3,                   // Learning rate
    5000,                   // Epochs
);
```

### Inverse Problem (Material Identification)

```rust
let config = Config::inverse_problem(
    vec![128, 128, 128, 128],  // Larger network
    5e-4,                       // Lower LR
    10000,                      // More epochs
);
```

### Custom Configuration

```rust
use kwavers::solver::inverse::pinn::elastic_2d::{
    Config, OptimizerType, LearningRateScheduler, LossWeights,
};

let mut config = Config::default();

// Network architecture
config.hidden_layers = vec![64, 64, 64];
config.activation = ActivationFunction::Tanh;

// Optimization
config.learning_rate = 1e-3;
config.n_epochs = 5000;
config.optimizer = OptimizerType::Adam {
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
};

// Learning rate schedule
config.scheduler = LearningRateScheduler::ReduceOnPlateau {
    factor: 0.5,
    patience: 100,
    threshold: 1e-6,
};

// Loss weights
config.loss_weights = LossWeights {
    pde: 1.0,
    boundary: 100.0,
    initial: 100.0,
    data: 10.0,
    interface: 10.0,
};

// Regularization
config.weight_decay = 0.0;  // No regularization
config.gradient_clip = None;  // No gradient clipping

// Logging and checkpointing
config.log_interval = 100;
config.checkpoint_interval = 1000;
config.checkpoint_dir = Some("checkpoints".to_string());
```

---

## Creating Training Data

### Collocation Points (Interior PDE)

```rust
use burn::tensor::Tensor;

// Uniform grid
let n = 1000;
let x = Tensor::<Backend, 1>::random(
    [n],
    burn::tensor::Distribution::Uniform(-1.0, 1.0),
    &device,
).reshape([n, 1]);

let y = Tensor::<Backend, 1>::random(
    [n],
    burn::tensor::Distribution::Uniform(-1.0, 1.0),
    &device,
).reshape([n, 1]);

let t = Tensor::<Backend, 1>::random(
    [n],
    burn::tensor::Distribution::Uniform(0.0, 1.0),
    &device,
).reshape([n, 1]);

let collocation = CollocationData {
    x,
    y,
    t,
    source_x: None,  // No body forces
    source_y: None,
};
```

### Boundary Conditions

```rust
use kwavers::solver::inverse::pinn::elastic_2d::BoundaryType;

let n_bc = 100;

// Points on boundary (e.g., x = -1)
let x_bc = Tensor::<Backend, 1>::ones([n_bc], &device)
    .mul_scalar(-1.0)
    .reshape([n_bc, 1]);

let y_bc = Tensor::<Backend, 1>::random(
    [n_bc],
    burn::tensor::Distribution::Uniform(-1.0, 1.0),
    &device,
).reshape([n_bc, 1]);

let t_bc = Tensor::<Backend, 1>::random(
    [n_bc],
    burn::tensor::Distribution::Uniform(0.0, 1.0),
    &device,
).reshape([n_bc, 1]);

// Zero displacement (Dirichlet BC)
let values_bc = Tensor::<Backend, 2>::zeros([n_bc, 2], &device);

let boundary = BoundaryData {
    x: x_bc,
    y: y_bc,
    t: t_bc,
    boundary_type: vec![BoundaryType::Dirichlet; n_bc],
    values: values_bc,
};
```

### Initial Conditions

```rust
let n_ic = 100;

let x_ic = Tensor::<Backend, 1>::random(
    [n_ic],
    burn::tensor::Distribution::Uniform(-1.0, 1.0),
    &device,
).reshape([n_ic, 1]);

let y_ic = Tensor::<Backend, 1>::random(
    [n_ic],
    burn::tensor::Distribution::Uniform(-1.0, 1.0),
    &device,
).reshape([n_ic, 1]);

// Zero initial displacement and velocity
let u_ic = Tensor::<Backend, 2>::zeros([n_ic, 2], &device);
let v_ic = Tensor::<Backend, 2>::zeros([n_ic, 2], &device);

let initial = InitialData {
    x: x_ic,
    y: y_ic,
    displacement: u_ic,
    velocity: v_ic,
};
```

### Observations (Inverse Problems)

```rust
use kwavers::solver::inverse::pinn::elastic_2d::ObservationData;

// Load synthetic or real measurements
let n_obs = 50;
let x_obs = // ... load observation points
let y_obs = // ...
let t_obs = // ...
let u_obs = // ... measured displacement

let observations = Some(ObservationData {
    x: x_obs,
    y: y_obs,
    t: t_obs,
    displacement: u_obs,
});
```

---

## Training Loop

### Basic Training

```rust
let metrics = trainer.train(&training_data)?;

println!("Training complete!");
println!("  Epochs: {}", metrics.epochs_completed);
println!("  Final loss: {:.6e}", metrics.final_loss().unwrap());
println!("  Time: {:.2}s", metrics.total_time);
```

### With Progress Monitoring

```rust
// Configure logging
config.log_interval = 100;  // Log every 100 epochs

let metrics = trainer.train(&training_data)?;

// Plot loss history
for (epoch, loss) in metrics.total_loss.iter().enumerate() {
    println!("Epoch {}: {:.6e}", epoch, loss);
}
```

### With Convergence Check

```rust
// Configure convergence
let tolerance = 1e-6;
let window = 50;

let metrics = trainer.train(&training_data)?;

if trainer.has_converged(tolerance, window) {
    println!("Converged at epoch {}", metrics.epochs_completed);
} else {
    println!("Did not converge - may need more epochs");
}
```

---

## Optimizer Selection

### Adam (Default - Recommended)

```rust
config.optimizer = OptimizerType::Adam {
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
};
```

**Use when**: General purpose, works well for most problems

### AdamW (Better Regularization)

```rust
config.optimizer = OptimizerType::AdamW {
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
};
config.weight_decay = 1e-5;
```

**Use when**: Overfitting concerns, need regularization

### SGD with Momentum

```rust
config.optimizer = OptimizerType::SGD {
    momentum: 0.9,
};
```

**Use when**: Fine-tuning after Adam, need deterministic behavior

---

## Learning Rate Schedules

### Constant (Simple)

```rust
config.scheduler = LearningRateScheduler::Constant;
```

### Exponential Decay

```rust
config.scheduler = LearningRateScheduler::Exponential {
    decay_rate: 0.95,  // LR *= 0.95 each epoch
};
```

### Step Decay

```rust
config.scheduler = LearningRateScheduler::Step {
    factor: 0.5,       // Multiply by 0.5
    step_size: 1000,   // Every 1000 epochs
};
```

### Cosine Annealing (Smooth)

```rust
config.scheduler = LearningRateScheduler::CosineAnnealing {
    lr_min: 1e-6,  // Final learning rate
};
```

### ReduceOnPlateau (Adaptive)

```rust
config.scheduler = LearningRateScheduler::ReduceOnPlateau {
    factor: 0.5,       // Reduce by 50%
    patience: 100,     // After 100 epochs without improvement
    threshold: 1e-6,   // Improvement threshold
};
```

**Recommended**: ReduceOnPlateau for automatic adaptation

---

## Loss Weight Tuning

### Default (Balanced)

```rust
config.loss_weights = LossWeights {
    pde: 1.0,
    boundary: 100.0,
    initial: 100.0,
    data: 10.0,
    interface: 10.0,
};
```

### PDE-Dominant (Forward Problem)

```rust
config.loss_weights = LossWeights {
    pde: 10.0,      // High PDE weight
    boundary: 100.0,
    initial: 100.0,
    data: 0.0,      // No data
    interface: 0.0,
};
```

### Data-Dominant (Inverse Problem)

```rust
config.loss_weights = LossWeights {
    pde: 0.1,       // Lower PDE weight
    boundary: 100.0,
    initial: 100.0,
    data: 100.0,    // High data weight
    interface: 10.0,
};
```

**Tuning Strategy**:
1. Start with default weights
2. If PDE residual not decreasing: increase `pde` weight
3. If BCs not satisfied: increase `boundary` weight
4. If ICs not satisfied: increase `initial` weight
5. Balance is problem-dependent!

---

## Model Inference (After Training)

### Extract Trained Model

```rust
// Get model without autodiff overhead
let inference_model = trainer.valid_model();

// Or keep autodiff version
let autodiff_model = trainer.model();
```

### Predict Displacement

```rust
use burn::tensor::Tensor;

// Query points
let x = Tensor::<Backend::InnerBackend, 1>::from_floats([0.5], &device)
    .reshape([1, 1]);
let y = Tensor::<Backend::InnerBackend, 1>::from_floats([0.5], &device)
    .reshape([1, 1]);
let t = Tensor::<Backend::InnerBackend, 1>::from_floats([0.1], &device)
    .reshape([1, 1]);

// Forward pass
let u = inference_model.forward(x, y, t);

// Extract values
let u_data = u.into_data();
let u_x = u_data.as_slice::<f32>().unwrap()[0];
let u_y = u_data.as_slice::<f32>().unwrap()[1];

println!("Displacement: u_x={}, u_y={}", u_x, u_y);
```

---

## Troubleshooting

### Loss Not Decreasing

**Symptoms**: Loss stays constant or increases

**Solutions**:
1. Reduce learning rate: `config.learning_rate = 1e-4;`
2. Increase network size: `config.hidden_layers = vec![128, 128, 128];`
3. Check data scaling (inputs should be ~[-1, 1])
4. Increase PDE weight: `config.loss_weights.pde = 10.0;`

### Loss Oscillating

**Symptoms**: Loss jumps up and down

**Solutions**:
1. Reduce learning rate
2. Enable gradient clipping: `config.gradient_clip = Some(1.0);`
3. Use AdamW with weight decay
4. Switch to cosine annealing LR schedule

### Overfitting

**Symptoms**: Training loss decreases but validation loss increases

**Solutions**:
1. Add weight decay: `config.weight_decay = 1e-5;`
2. Reduce network size
3. Increase training data
4. Add dropout (future feature)

### Slow Training

**Symptoms**: Each epoch takes too long

**Solutions**:
1. Reduce batch size (fewer collocation points)
2. Reduce network size
3. Use smaller hidden layers
4. Check benchmark results to identify bottleneck

---

## Performance Tips

### Optimal Batch Sizes

- **CPU**: 256-1024 collocation points
- **GPU**: 2048-8192 collocation points

### Network Architecture

- **Start small**: [32, 32] for testing
- **Production**: [64, 64, 64] or [128, 128, 128]
- **Deep networks**: Rarely better than wide networks for PINNs

### Convergence Acceleration

1. **Warm start**: Train small network, then expand
2. **Curriculum learning**: Start with easy constraints (IC/BC), then add PDE
3. **Adaptive sampling**: Resample high-residual regions (future)

---

## Running Benchmarks

### All Benchmarks

```bash
cargo bench --bench pinn_elastic_2d_training --features pinn
```

### Specific Benchmark

```bash
cargo bench --bench pinn_elastic_2d_training --features pinn -- forward_pass
```

### Save Baseline

```bash
cargo bench --bench pinn_elastic_2d_training --features pinn -- --save-baseline main
```

### Compare to Baseline

```bash
cargo bench --bench pinn_elastic_2d_training --features pinn -- --baseline main
```

---

## Example: Complete Training Script

```rust
use burn::backend::{Autodiff, NdArray};
use kwavers::solver::inverse::pinn::elastic_2d::*;

type Backend = Autodiff<NdArray<f32>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configuration
    let mut config = Config::default();
    config.hidden_layers = vec![64, 64, 64, 64];
    config.learning_rate = 1e-3;
    config.n_epochs = 5000;
    config.optimizer = OptimizerType::Adam {
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
    };
    config.scheduler = LearningRateScheduler::ReduceOnPlateau {
        factor: 0.5,
        patience: 100,
        threshold: 1e-6,
    };
    config.log_interval = 100;
    
    // 2. Initialize
    let device = Default::default();
    let model = ElasticPINN2D::<Backend>::new(&config, &device)?;
    let mut trainer = Trainer::<Backend>::new(model, config.clone());
    
    // 3. Training data
    let n_colloc = 1000;
    let n_boundary = 100;
    let n_initial = 100;
    
    let training_data = TrainingData {
        collocation: create_collocation_points(n_colloc, &device),
        boundary: create_boundary_conditions(n_boundary, &device),
        initial: create_initial_conditions(n_initial, &device),
        observations: None,
    };
    
    // 4. Train
    println!("Starting training...");
    let metrics = trainer.train(&training_data)?;
    
    // 5. Results
    println!("\nTraining complete!");
    println!("  Epochs: {}", metrics.epochs_completed);
    println!("  Total time: {:.2}s", metrics.total_time);
    println!("  Avg epoch time: {:.3}s", metrics.average_epoch_time());
    println!("  Final loss: {:.6e}", metrics.final_loss().unwrap());
    
    // 6. Validation
    if metrics.final_loss().unwrap() > 1e-3 {
        eprintln!("Warning: Training did not converge well");
    }
    
    // 7. Save model (placeholder)
    // trainer.save_model("trained_model.bin")?;
    
    Ok(())
}

fn create_collocation_points(n: usize, device: &<Backend as burn::tensor::backend::Backend>::Device) -> CollocationData<Backend> {
    use burn::tensor::Tensor;
    
    CollocationData {
        x: Tensor::<Backend, 1>::random([n], burn::tensor::Distribution::Uniform(-1.0, 1.0), device).reshape([n, 1]),
        y: Tensor::<Backend, 1>::random([n], burn::tensor::Distribution::Uniform(-1.0, 1.0), device).reshape([n, 1]),
        t: Tensor::<Backend, 1>::random([n], burn::tensor::Distribution::Uniform(0.0, 1.0), device).reshape([n, 1]),
        source_x: None,
        source_y: None,
    }
}

fn create_boundary_conditions(n: usize, device: &<Backend as burn::tensor::backend::Backend>::Device) -> BoundaryData<Backend> {
    use burn::tensor::Tensor;
    
    BoundaryData {
        x: Tensor::<Backend, 1>::ones([n], device).mul_scalar(-1.0).reshape([n, 1]),
        y: Tensor::<Backend, 1>::random([n], burn::tensor::Distribution::Uniform(-1.0, 1.0), device).reshape([n, 1]),
        t: Tensor::<Backend, 1>::random([n], burn::tensor::Distribution::Uniform(0.0, 1.0), device).reshape([n, 1]),
        boundary_type: vec![BoundaryType::Dirichlet; n],
        values: Tensor::<Backend, 2>::zeros([n, 2], device),
    }
}

fn create_initial_conditions(n: usize, device: &<Backend as burn::tensor::backend::Backend>::Device) -> InitialData<Backend> {
    use burn::tensor::Tensor;
    
    InitialData {
        x: Tensor::<Backend, 1>::random([n], burn::tensor::Distribution::Uniform(-1.0, 1.0), device).reshape([n, 1]),
        y: Tensor::<Backend, 1>::random([n], burn::tensor::Distribution::Uniform(-1.0, 1.0), device).reshape([n, 1]),
        displacement: Tensor::<Backend, 2>::zeros([n, 2], device),
        velocity: Tensor::<Backend, 2>::zeros([n, 2], device),
    }
}
```

---

## References

- Full documentation: `docs/phase4_tasks5_6_complete.md`
- Autodiff guide: `docs/autodiff_stress_gradients_quick_reference.md`
- Phase 4 plan: `docs/phase4_action_plan.md`
- Benchmarks: `benches/pinn_elastic_2d_training.rs`
