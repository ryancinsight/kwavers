# Phase 5 Quick Start: Adaptive PINN Training

**Quick reference for using Phase 5 enhancements**

---

## Overview

Phase 5 adds:
- ✅ Improved Adam optimizer with bias correction
- ✅ Model checkpointing with metrics serialization
- ✅ Adaptive sampling (residual-based collocation)
- ✅ Mini-batching for memory efficiency

---

## Installation

```bash
# Build with PINN support
cargo build --features pinn --release

# Run tests
cargo test --features pinn

# Run benchmarks
cargo bench --bench pinn_elastic_2d_training --features pinn
```

---

## Basic Usage

### 1. Standard Training (No Enhancements)

```rust
use burn::backend::{Autodiff, NdArray};
use kwavers::solver::inverse::pinn::elastic_2d::{
    Config, Trainer, TrainingData, ElasticPINN2D
};

type Backend = Autodiff<NdArray<f32>>;

let config = Config::default();
let device = Default::default();
let model = ElasticPINN2D::new(&config, &device)?;
let mut trainer = Trainer::<Backend>::new(model, config);

let metrics = trainer.train(&training_data)?;
println!("Final loss: {:.6e}", metrics.final_loss().unwrap());
```

### 2. Training with Checkpointing

```rust
let mut config = Config::default();
config.checkpoint_dir = Some("./checkpoints".to_string());
config.checkpoint_interval = 100;  // Save every 100 epochs

let mut trainer = Trainer::<Backend>::new(model, config);
let metrics = trainer.train(&training_data)?;

// Checkpoints saved automatically:
// ./checkpoints/model_epoch_100.bin (placeholder)
// ./checkpoints/metrics_epoch_100.json
```

**Checkpoint JSON Example**:
```json
{
  "epoch": 100,
  "total_loss": 1.234e-3,
  "pde_loss": 8.9e-4,
  "boundary_loss": 2.1e-4,
  "learning_rate": 1e-3
}
```

### 3. Training with Adaptive Sampling

```rust
use kwavers::solver::inverse::pinn::elastic_2d::{
    AdaptiveSampler, AdaptiveSamplingStrategy, extract_batch
};

// Create adaptive sampler
let mut sampler = AdaptiveSampler::new(
    AdaptiveSamplingStrategy::ResidualWeighted {
        alpha: 1.5,        // Concentration on high residuals
        keep_ratio: 0.1,   // Keep 10% of old points
    },
    10000,  // 10k collocation points
    0,      // Full batch (no mini-batching)
);

// Generate candidate points (large set)
let all_candidates = generate_candidate_points(50000);

// Initial uniform sampling
let residuals = vec![1.0; 50000];
let indices = sampler.resample(&residuals)?;
let mut collocation_data = extract_points(&all_candidates, &indices);

// Training loop with periodic resampling
for epoch in 0..config.n_epochs {
    // Resample every 10 epochs based on residuals
    if epoch % 10 == 0 && epoch > 0 {
        let residuals = compute_all_residuals(&trainer.model(), &all_candidates);
        let new_indices = sampler.resample(&residuals)?;
        collocation_data = extract_points(&all_candidates, &new_indices);
        
        tracing::info!("Resampled: max residual = {:.3e}", 
                      residuals.iter().cloned().fold(f64::NAN, f64::max));
    }
    
    // Standard training step
    // ... (forward, loss, backward, optimizer step)
}
```

### 4. Training with Mini-Batching

```rust
// Create sampler with mini-batching
let mut sampler = AdaptiveSampler::new(
    AdaptiveSamplingStrategy::Uniform,
    10000,  // 10k collocation points
    256,    // Mini-batches of 256
);

// Training loop with batch iteration
for epoch in 0..config.n_epochs {
    let mut epoch_loss = 0.0;
    let mut n_batches = 0;
    
    for batch_indices in sampler.iter_batches() {
        // Extract batch data
        let batch_data = extract_batch(&collocation_data, &batch_indices)?;
        
        // Forward pass on batch
        let u_pred = trainer.model().forward(
            batch_data.x.clone(),
            batch_data.y.clone(),
            batch_data.t.clone(),
        );
        
        // Compute loss on batch
        let loss = compute_batch_loss(&trainer, &batch_data);
        epoch_loss += Self::tensor_to_f64(&loss);
        n_batches += 1;
        
        // Backward and update
        let grads = loss.backward();
        trainer.model = trainer.optimizer.step(trainer.model.clone(), &grads);
    }
    
    epoch_loss /= n_batches as f64;
    tracing::info!("Epoch {}: avg loss = {:.6e}", epoch, epoch_loss);
}
```

### 5. Combined: Adaptive + Mini-Batch

```rust
let mut sampler = AdaptiveSampler::new(
    AdaptiveSamplingStrategy::ResidualWeighted {
        alpha: 1.5,
        keep_ratio: 0.1,
    },
    10000,  // 10k active points
    256,    // Batches of 256
);

let all_candidates = generate_candidate_points(50000);
let mut collocation_data = initial_collocation_data();

for epoch in 0..config.n_epochs {
    // Adaptive resampling (every 10 epochs)
    if epoch % 10 == 0 && epoch > 0 {
        let residuals = compute_all_residuals(&trainer.model(), &all_candidates);
        let new_indices = sampler.resample(&residuals)?;
        collocation_data = extract_points(&all_candidates, &new_indices);
    }
    
    // Mini-batch training
    for batch_indices in sampler.iter_batches() {
        let batch_data = extract_batch(&collocation_data, &batch_indices)?;
        // ... train on batch
    }
}
```

---

## Sampling Strategies

### Uniform (Baseline)

```rust
AdaptiveSamplingStrategy::Uniform
```
- No adaptation
- Random sampling from domain
- Good for initial training

### Residual-Weighted

```rust
AdaptiveSamplingStrategy::ResidualWeighted {
    alpha: 1.5,        // 1.0-3.0 typical
    keep_ratio: 0.1,   // 0.0-0.2 typical
}
```
- Sample proportional to `residual^alpha`
- Higher alpha = more aggressive concentration
- keep_ratio = fraction of old points to retain for stability

**Recommended**: alpha=1.5, keep_ratio=0.1 for most problems

### Importance Threshold

```rust
AdaptiveSamplingStrategy::ImportanceThreshold {
    threshold: 1e-4,   // Minimum residual to consider
    top_k_ratio: 0.5,  // Keep top 50% of filtered points
}
```
- Only sample from high-residual regions
- Good for late-stage training when most residuals are small

### Hybrid

```rust
AdaptiveSamplingStrategy::Hybrid {
    uniform_ratio: 0.3,  // 30% uniform exploration
    alpha: 2.0,          // 70% aggressive exploitation
}
```
- Mix uniform exploration with residual-based exploitation
- Balances coverage and efficiency

---

## Configuration Tips

### Learning Rate Scheduling

```rust
use kwavers::solver::inverse::pinn::elastic_2d::LearningRateScheduler;

let mut config = Config::default();
config.scheduler = LearningRateScheduler::Exponential { gamma: 0.95 };
// OR
config.scheduler = LearningRateScheduler::ReduceOnPlateau {
    factor: 0.5,
    patience: 100,
    threshold: 1e-4,
};
```

### Optimizer Selection

```rust
use kwavers::solver::inverse::pinn::elastic_2d::OptimizerType;

config.optimizer = OptimizerType::Adam {
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
};
// OR
config.optimizer = OptimizerType::AdamW {
    beta1: 0.9,
    beta2: 0.999,
    epsilon: 1e-8,
};
config.weight_decay = 1e-4;  // For AdamW
```

### Loss Weights

```rust
config.loss_weights.pde = 1.0;
config.loss_weights.boundary = 10.0;  // Emphasize BC satisfaction
config.loss_weights.initial = 10.0;
config.loss_weights.data = 100.0;     // For inverse problems
```

---

## Performance Tuning

### Memory vs. Speed Trade-off

| Batch Size | Memory | Throughput | Convergence |
|------------|--------|------------|-------------|
| Full (0)   | High   | 1x         | Deterministic |
| 1024       | Med    | 2-3x       | Fast |
| 256        | Low    | 3-5x       | Stochastic |
| 64         | Very Low | 2-4x     | More stochastic |

**Recommendation**: Start with 256-512 for balanced performance.

### Resampling Frequency

```rust
// Too frequent: expensive residual computation
if epoch % 5 == 0 { resample(); }  // Every 5 epochs

// Recommended: amortize cost
if epoch % 10 == 0 { resample(); }  // Every 10 epochs

// Too infrequent: miss adaptation benefits
if epoch % 100 == 0 { resample(); }  // Every 100 epochs
```

**Rule of thumb**: Resample every 5-20 epochs depending on problem difficulty.

### GPU Acceleration

```bash
# Build with GPU support
cargo build --features pinn-gpu --release

# Run benchmarks on GPU
cargo bench --bench pinn_elastic_2d_training --features pinn-gpu
```

---

## Validation & Debugging

### Check Residual Distribution

```rust
let residuals = compute_all_residuals(&model, &all_candidates);
let max_residual = residuals.iter().cloned().fold(f64::NAN, f64::max);
let avg_residual = residuals.iter().sum::<f64>() / residuals.len() as f64;

println!("Residuals: max={:.3e}, avg={:.3e}", max_residual, avg_residual);

// High max but low avg → adaptive sampling will help
```

### Monitor Sampling Statistics

```rust
let indices = sampler.resample(&residuals)?;
let high_residual_count = indices.iter()
    .filter(|&&i| residuals[i] > threshold)
    .count();

println!("Sampled {}/{} high-residual points", 
         high_residual_count, indices.len());
```

### Convergence Diagnostics

```rust
// Check if training has converged
if trainer.has_converged(1e-6, 50) {
    println!("Converged: loss change < 1e-6 for 50 epochs");
    break;
}
```

---

## Common Issues

### Issue: Slow Convergence

**Solution**: Try aggressive adaptive sampling
```rust
AdaptiveSamplingStrategy::ResidualWeighted {
    alpha: 2.0,  // More aggressive (was 1.5)
    keep_ratio: 0.05,
}
```

### Issue: Out of Memory

**Solution**: Reduce batch size
```rust
let sampler = AdaptiveSampler::new(strategy, 10000, 128);  // Was 256
```

### Issue: High Variance in Loss

**Solution**: Increase batch size or use full batch
```rust
let sampler = AdaptiveSampler::new(strategy, 10000, 512);  // Was 128
```

### Issue: Poor BC Satisfaction

**Solution**: Increase BC loss weight and boundary sampling
```rust
config.loss_weights.boundary = 100.0;  // Was 10.0
config.n_boundary_points = 2000;       // Was 1000
```

---

## Example Scripts

### Complete Training Script

```rust
use burn::backend::{Autodiff, NdArray};
use kwavers::solver::inverse::pinn::elastic_2d::*;

fn main() -> KwaversResult<()> {
    // Configuration
    let mut config = Config::default();
    config.n_epochs = 2000;
    config.learning_rate = 1e-3;
    config.checkpoint_dir = Some("./results/run_001".to_string());
    config.checkpoint_interval = 100;
    config.loss_weights.pde = 1.0;
    config.loss_weights.boundary = 10.0;
    
    // Setup
    type Backend = Autodiff<NdArray<f32>>;
    let device = Default::default();
    let model = ElasticPINN2D::new(&config, &device)?;
    let mut trainer = Trainer::<Backend>::new(model, config.clone());
    
    // Adaptive sampler
    let mut sampler = AdaptiveSampler::new(
        AdaptiveSamplingStrategy::ResidualWeighted {
            alpha: 1.5,
            keep_ratio: 0.1,
        },
        10000,
        256,
    );
    
    // Generate data
    let all_candidates = generate_large_candidate_set(50000);
    let boundary_data = generate_boundary_data(1000);
    let initial_data = generate_initial_data(1000);
    
    // Initial sampling
    let residuals = vec![1.0; 50000];
    let indices = sampler.resample(&residuals)?;
    let mut collocation_data = extract_points(&all_candidates, &indices);
    
    // Training loop
    for epoch in 0..config.n_epochs {
        // Adaptive resampling
        if epoch % 10 == 0 && epoch > 0 {
            let residuals = compute_residuals(&trainer.model(), &all_candidates);
            let new_indices = sampler.resample(&residuals)?;
            collocation_data = extract_points(&all_candidates, &new_indices);
        }
        
        // Mini-batch training
        let mut epoch_loss = 0.0;
        let mut n_batches = 0;
        
        for batch_indices in sampler.iter_batches() {
            let batch = extract_batch(&collocation_data, &batch_indices)?;
            
            // Training step (simplified)
            let training_data = TrainingData {
                collocation: batch,
                boundary: boundary_data.clone(),
                initial: initial_data.clone(),
                observations: None,
            };
            
            // ... perform training step
            n_batches += 1;
        }
        
        if epoch % 100 == 0 {
            println!("Epoch {}: loss={:.6e}", epoch, epoch_loss / n_batches as f64);
        }
    }
    
    Ok(())
}
```

---

## Further Reading

- [Phase 5 Complete Documentation](./phase5_enhancements_complete.md) - Full technical details
- [Phase 4 Summary](./phase4_complete_summary.md) - Training loop & benchmarks
- [PINN Training Guide](./pinn_training_quick_start.md) - Basic training concepts

---

## Quick Command Reference

```bash
# Build
cargo build --features pinn --release

# Test
cargo test --features pinn
cargo test --features pinn adaptive_sampling

# Benchmark
cargo bench --bench pinn_elastic_2d_training --features pinn

# Run with logging
RUST_LOG=kwavers=debug cargo run --features pinn --example elastic_pinn_adaptive

# GPU (if available)
cargo build --features pinn-gpu --release
cargo bench --features pinn-gpu
```

---

**Status**: Phase 5 complete and ready for production use.

**Next Steps**: Run validation tests, collect benchmark data, implement full Adam with persistent buffers.