# PINN Training Guide for Kwavers

**Version**: 3.0.0  
**Last Updated**: 2026-02-04  
**Audience**: Researchers and engineers using Physics-Informed Neural Networks for inverse problems

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Hyperparameter Tuning](#hyperparameter-tuning)
4. [Training Diagnostics](#training-diagnostics)
5. [Troubleshooting](#troubleshooting)
6. [Advanced Topics](#advanced-topics)
7. [Best Practices](#best-practices)
8. [References](#references)

---

## Introduction

Physics-Informed Neural Networks (PINNs) enable solving partial differential equations (PDEs) by encoding physical laws directly into the loss function. Kwavers implements PINNs for the 3D wave equation, allowing you to:

- Solve forward problems (given medium properties, predict wave field)
- Solve inverse problems (given observations, infer medium properties)
- Enforce boundary and initial conditions automatically
- Train on GPU for acceleration (via burn-wgpu backend)

### Architecture Overview

```
Wave Equation PINN (3D)
├── Network: (x,y,z,t) → u(x,y,z,t)
├── Loss Components:
│   ├── Data Loss: ||u_pred - u_obs||²
│   ├── PDE Loss: ||∂²u/∂t² - c²∇²u||²
│   ├── BC Loss: ||u(boundary) - u_BC||²
│   └── IC Loss: ||u(t=0) - u₀||² + ||∂u/∂t(t=0) - v₀||²
└── Training: Adaptive LR + EMA Loss Normalization
```

### Prerequisites

- Rust 1.70+ with Cargo
- Kwavers with `pinn` feature enabled
- Optional: GPU with WGPU support for acceleration

---

## Quick Start

### 5-Minute First Training

```rust
use kwavers::solver::inverse::pinn::ml::burn_wave_equation_3d::{
    BurnPINN3DConfig, BurnPINN3DWave, Geometry3D,
};
use burn::backend::NdArray;

type Backend = NdArray<f32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure the network
    let config = BurnPINN3DConfig::default()
        .with_hidden_layers(vec![64, 64, 64])
        .with_learning_rate(1e-4)
        .with_epochs(1000);

    // 2. Define the domain geometry
    let geometry = Geometry3D::rectangular(
        -1.0, 1.0,  // x: [-1, 1]
        -1.0, 1.0,  // y: [-1, 1]
        -1.0, 1.0,  // z: [-1, 1]
    );

    // 3. Create the solver
    let device = Default::default();
    let mut solver = BurnPINN3DWave::<Backend>::new(
        config,
        geometry,
        None,  // Homogeneous wave speed (c=1)
        &device,
    )?;

    // 4. Prepare training data
    let x_data = vec![0.0, 0.5, -0.5];
    let y_data = vec![0.0, 0.0, 0.0];
    let z_data = vec![0.0, 0.0, 0.0];
    let t_data = vec![0.1, 0.2, 0.3];
    let u_data = vec![1.0, 0.8, 0.6];  // Observed wave amplitudes

    // 5. Train the PINN
    let metrics = solver.train(
        &x_data, &y_data, &z_data, &t_data, &u_data,
        None,  // No initial velocity data
        &device,
    )?;

    // 6. Inspect results
    println!("Training completed in {:.2}s", metrics.training_time_secs);
    println!("Final total loss: {:.6e}", metrics.total_loss.last().unwrap());

    // 7. Make predictions
    let x_test = vec![0.25];
    let y_test = vec![0.0];
    let z_test = vec![0.0];
    let t_test = vec![0.15];
    let predictions = solver.predict(&x_test, &y_test, &z_test, &t_test, &device)?;
    println!("Predicted u(0.25, 0, 0, 0.15) = {:.4}", predictions[0]);

    Ok(())
}
```

**Expected Output**:
```
Training completed in 12.45s
Final total loss: 2.345e-04
Predicted u(0.25, 0, 0, 0.15) = 0.8912
```

---

## Hyperparameter Tuning

### Network Architecture

**Hidden Layers**:
```rust
// Small problems (< 100 observations)
.with_hidden_layers(vec![32, 32])

// Medium problems (100-1000 observations)
.with_hidden_layers(vec![64, 64, 64])  // DEFAULT

// Large problems (> 1000 observations)
.with_hidden_layers(vec![128, 128, 128, 128])

// High-frequency problems (complex wave patterns)
.with_hidden_layers(vec![256, 256, 256])
```

**Guidelines**:
- **Depth**: 3-4 layers for most problems; 5+ for complex geometries
- **Width**: 64-128 neurons per layer; increase for high-frequency content
- **Rule of thumb**: Total parameters ≈ 10× number of training points

### Learning Rate

**Default**: `1e-4` (safe starting point)

```rust
// Conservative (stable but slow)
.with_learning_rate(1e-5)

// Standard (recommended for most problems)
.with_learning_rate(1e-4)  // DEFAULT

// Aggressive (faster but may diverge)
.with_learning_rate(5e-4)

// Very aggressive (use with caution)
.with_learning_rate(1e-3)
```

**Adaptive LR Schedule**:
- Automatic decay by factor 0.95 when no improvement for 10 epochs
- Minimum LR: `initial_lr * 1e-3`
- No manual tuning needed (built-in)

**Symptoms**:
- **LR too high**: Loss oscillates or explodes, NaN/Inf values
- **LR too low**: Slow convergence, loss plateaus early

### Loss Weights

**Default Weights**:
```rust
BurnLossWeights3D {
    data_weight: 1.0,   // Observation fitting
    pde_weight: 1.0,    // Physics enforcement
    bc_weight: 10.0,    // Boundary conditions (higher priority)
    ic_weight: 1.0,     // Initial conditions
}
```

**Tuning Strategy**:

1. **Data-Rich Regime** (many observations):
   ```rust
   data_weight: 1.0,   // Trust the data
   pde_weight: 0.1,    // Physics as regularizer
   bc_weight: 5.0,     // Moderate BC enforcement
   ic_weight: 0.5,     // Moderate IC enforcement
   ```

2. **Data-Scarce Regime** (few observations):
   ```rust
   data_weight: 0.5,   // Limited observations
   pde_weight: 1.0,    // Strong physics enforcement
   bc_weight: 10.0,    // Strong BC enforcement
   ic_weight: 2.0,     // Strong IC enforcement
   ```

3. **Boundary-Critical Problems** (focused beams, reflections):
   ```rust
   data_weight: 1.0,
   pde_weight: 1.0,
   bc_weight: 50.0,    // Very high BC weight
   ic_weight: 1.0,
   ```

**Guidelines**:
- Start with defaults, then adjust based on loss component magnitudes
- If BC loss >> data loss, increase `bc_weight`
- If PDE loss remains high, increase `pde_weight` or network capacity
- Use logarithmic search: 0.1, 1.0, 10.0, 100.0

### Training Epochs

```rust
// Quick prototyping (rough solution)
.with_epochs(500)

// Standard training (production quality)
.with_epochs(5000)  // DEFAULT

// High-accuracy training (research)
.with_epochs(20000)

// Convergence study (find optimal)
.with_epochs(50000)
```

**Early Stopping**:
- Training auto-stops on NaN/Inf detection
- No built-in early stopping on loss plateau (planned feature)
- Manually monitor loss curves for convergence

### Collocation Points

**Default**: `n_colloc = 1000` random points in domain

```rust
// Sparse sampling (fast training)
config.n_colloc = 500;

// Standard sampling
config.n_colloc = 1000;  // DEFAULT

// Dense sampling (better PDE enforcement)
config.n_colloc = 5000;

// Very dense (high-accuracy physics)
config.n_colloc = 10000;
```

**Guidelines**:
- More collocation points → better PDE satisfaction
- Diminishing returns beyond 5000 for smooth problems
- Increase for high-frequency or discontinuous solutions

---

## Training Diagnostics

### Loss Curves

**What to Monitor**:
```rust
// After training
let metrics = solver.train(...)?;

println!("Total loss: {:?}", metrics.total_loss);
println!("Data loss: {:?}", metrics.data_loss);
println!("PDE loss: {:?}", metrics.pde_loss);
println!("BC loss: {:?}", metrics.bc_loss);
println!("IC loss: {:?}", metrics.ic_loss);
```

**Healthy Training**:
```
Epoch 0:    total=1.23e+00, data=5.67e-01, pde=3.45e-01, bc=2.11e-01, ic=1.00e-01
Epoch 1000: total=3.45e-02, data=1.23e-02, pde=8.90e-03, bc=5.67e-03, ic=4.56e-03
Epoch 5000: total=2.34e-04, data=8.90e-05, pde=6.78e-05, bc=3.45e-05, ic=2.34e-05
```

**Interpretation**:
- **Monotonic decrease**: Convergence is smooth (ideal)
- **Oscillations**: Learning rate may be too high
- **Plateau**: May need more epochs, higher LR, or better initialization
- **Component imbalance**: Adjust loss weights (see above)

### Logging Levels

```rust
// Set logging level in your code
env_logger::Builder::new()
    .filter_level(log::LevelFilter::Info)  // Standard logging
    .init();

// Or via environment variable
// RUST_LOG=debug cargo run
```

**Logging Levels**:
- `ERROR`: Numerical instability, NaN/Inf detection
- `WARN`: Large gradients, potential issues
- `INFO`: Training progress every 100 epochs (default)
- `DEBUG`: Detailed diagnostics (gradient norms, parameter updates)
- `TRACE`: Full tensor shapes and values

### Real-Time Monitoring

**Console Output** (every 100 epochs):
```
INFO: Epoch 100/5000: total=1.234e-02, data=4.567e-03, pde=3.456e-03, bc=2.345e-03, ic=1.234e-03, lr=1.000e-04
INFO: Epoch 200/5000: total=5.678e-03, data=2.345e-03, pde=1.456e-03, bc=1.234e-03, ic=6.432e-04, lr=1.000e-04
```

**Adaptive LR Messages**:
```
INFO: Learning rate decayed: 1.000e-04 → 9.500e-05 (no improvement for 10 epochs)
```

**Numerical Instability**:
```
ERROR: Numerical instability detected at epoch 1234: total=NaN, data=Inf, pde=1.234e+10, bc=5.678e+08, ic=1.234e+05
```

---

## Troubleshooting

### Problem: Training Diverges (NaN/Inf)

**Symptoms**:
- Loss becomes NaN or Inf
- Error message: "Training diverged at epoch X"
- BC loss explodes (e.g., 1e+31)

**Solutions**:

1. **Reduce Learning Rate**:
   ```rust
   config.learning_rate = 1e-5;  // Was 1e-4
   ```

2. **Increase BC Weight** (if BC loss is culprit):
   ```rust
   config.loss_weights.bc_weight = 50.0;  // Was 10.0
   ```

3. **Check Data Normalization**:
   ```rust
   // Ensure data is scaled appropriately
   // Coordinates should be O(1), not O(1000)
   let x_normalized = x_data.iter().map(|&x| x / x_max).collect();
   ```

4. **Reduce Network Size** (for stability):
   ```rust
   config.hidden_layers = vec![32, 32];  // Smaller network
   ```

5. **Check Initial Conditions** (if IC loss diverges):
   ```rust
   // Ensure IC data is consistent with observations
   // u(t=0) should match u_data at t=0
   ```

### Problem: Slow Convergence

**Symptoms**:
- Loss decreases very slowly
- Training takes many epochs (>20,000)
- Final loss still high (>1e-3)

**Solutions**:

1. **Increase Learning Rate**:
   ```rust
   config.learning_rate = 5e-4;  // Was 1e-4
   ```

2. **Increase Network Capacity**:
   ```rust
   config.hidden_layers = vec![128, 128, 128, 128];
   ```

3. **Add More Collocation Points**:
   ```rust
   config.n_colloc = 5000;  // Was 1000
   ```

4. **Check Wave Speed Function**:
   ```rust
   // Ensure wave speed is physically reasonable (not too large/small)
   // c = 1500 m/s for water, not 1.5e6 or 0.0015
   ```

5. **Better Initialization** (planned feature):
   - Pre-train on coarse grid, refine on fine grid
   - Transfer learning from similar problem

### Problem: BC Loss Remains High

**Symptoms**:
- BC loss > 1e-2 after many epochs
- Other loss components decrease normally
- Wave doesn't satisfy boundary conditions

**Solutions**:

1. **Increase BC Weight**:
   ```rust
   config.loss_weights.bc_weight = 100.0;  // Was 10.0
   ```

2. **Check Boundary Implementation**:
   ```rust
   // Verify geometry boundaries are correct
   let geometry = Geometry3D::rectangular(
       x_min, x_max,
       y_min, y_max,
       z_min, z_max,
   );
   ```

3. **Add More Boundary Collocation Points**:
   - Current: 6 faces × 100 points = 600 total
   - Future enhancement: Configurable boundary sampling

4. **Verify BC Type**:
   - Current implementation: Dirichlet (u=0 on boundaries)
   - For Neumann BC: Custom implementation needed (planned)

### Problem: PDE Loss Remains High

**Symptoms**:
- PDE loss > 1e-2 after convergence
- Wave equation not satisfied in domain interior
- Data and BC losses are low

**Solutions**:

1. **Increase PDE Weight**:
   ```rust
   config.loss_weights.pde_weight = 10.0;  // Was 1.0
   ```

2. **Add More Collocation Points**:
   ```rust
   config.n_colloc = 10000;  // Was 1000
   ```

3. **Check Wave Speed Consistency**:
   ```rust
   // Ensure wave_speed_fn matches the actual medium
   let wave_speed_fn = |x, y, z| {
       1500.0  // Constant for homogeneous medium
   };
   ```

4. **Increase Network Capacity**:
   ```rust
   // Deeper/wider network for complex wave patterns
   config.hidden_layers = vec![256, 256, 256];
   ```

### Problem: Oscillating Loss

**Symptoms**:
- Loss goes up and down, not monotonic
- Oscillation amplitude may decrease over time
- Learning is unstable

**Solutions**:

1. **Reduce Learning Rate**:
   ```rust
   config.learning_rate = 5e-5;  // Was 1e-4
   ```

2. **Increase Adaptive LR Patience**:
   - Currently hardcoded: patience=10
   - Future enhancement: Configurable via config

3. **Enable Gradient Clipping** (when Burn API supports):
   - Infrastructure ready but disabled
   - Monitor for Burn 0.20+ updates

4. **Check Data Quality**:
   ```rust
   // Ensure no duplicate or contradictory observations
   // No NaN/Inf in input data
   ```

### Problem: Memory Exhaustion (GPU)

**Symptoms**:
- Out-of-memory error during training
- GPU utilization 100%, then crash
- Works on CPU, fails on GPU

**Solutions**:

1. **Reduce Batch Size** (for future batched training):
   - Current: All data in one batch
   - Future: Mini-batch training

2. **Reduce Collocation Points**:
   ```rust
   config.n_colloc = 500;  // Was 1000 or more
   ```

3. **Reduce Network Size**:
   ```rust
   config.hidden_layers = vec![64, 64];  // Was 128+ neurons
   ```

4. **Use Mixed Precision** (future enhancement):
   - FP16 training for memory efficiency
   - Burn backend support needed

5. **Gradient Checkpointing** (future enhancement):
   - Trade compute for memory
   - Recompute activations during backward pass

---

## Advanced Topics

### GPU Acceleration

**Enable GPU Backend**:
```toml
# Cargo.toml
[dependencies]
kwavers = { version = "3.0", features = ["pinn-gpu"] }
```

```rust
use burn::backend::Wgpu;

type Backend = Wgpu<f32, i32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // GPU device selection (automatic)
    let device = Default::default();
    
    // Rest of training code same as CPU
    let mut solver = BurnPINN3DWave::<Backend>::new(
        config, geometry, wave_speed_fn, &device
    )?;
    
    // Training on GPU (10-50× faster than CPU)
    let metrics = solver.train(..., &device)?;
    
    Ok(())
}
```

**Expected Speedups**:
- Small problems (< 1000 points): 2-5×
- Medium problems (1000-10000 points): 10-20×
- Large problems (> 10000 points): 20-50×

**GPU Requirements**:
- Vulkan-compatible GPU (NVIDIA, AMD, Intel)
- 4GB+ VRAM for typical problems
- 8GB+ VRAM for large networks (256+ neurons/layer)

### Custom Wave Speed Functions

**Heterogeneous Medium**:
```rust
use std::sync::Arc;
use burn::tensor::Tensor;

// Define spatially-varying wave speed
let wave_speed_fn = Arc::new(|x: Tensor<B, 2>, y: Tensor<B, 2>, z: Tensor<B, 2>| {
    // Example: c(x,y,z) = 1500 + 100*sin(x)
    let c0 = Tensor::full([x.shape().dims[0], 1], 1500.0, &x.device());
    let perturbation = x.clone().sin().mul_scalar(100.0);
    c0.add(perturbation)
});

let mut solver = BurnPINN3DWave::<Backend>::new(
    config,
    geometry,
    Some(wave_speed_fn),
    &device,
)?;
```

**Layered Medium**:
```rust
let wave_speed_fn = Arc::new(|x: Tensor<B, 2>, y: Tensor<B, 2>, z: Tensor<B, 2>| {
    // Layer 1: z < 0 → c = 1500 m/s
    // Layer 2: z > 0 → c = 1800 m/s
    let c_layer1 = Tensor::full([z.shape().dims[0], 1], 1500.0, &z.device());
    let c_layer2 = Tensor::full([z.shape().dims[0], 1], 1800.0, &z.device());
    
    // Mask: z < 0
    let mask = z.clone().lower_scalar(0.0);
    
    // Blend layers: c = mask*c1 + (1-mask)*c2
    mask.mul(c_layer1).add((Tensor::ones_like(&mask).sub(mask)).mul(c_layer2))
});
```

### Initial Velocity Specification

**Complete IC (Displacement + Velocity)**:
```rust
// Gaussian pulse with zero initial velocity
let x_ic = vec![0.0];
let y_ic = vec![0.0];
let z_ic = vec![0.0];
let u_ic = vec![1.0];  // Initial displacement
let v_ic = vec![0.0];  // Initial velocity (∂u/∂t at t=0)

let metrics = solver.train(
    &x_data, &y_data, &z_data, &t_data, &u_data,
    Some(&v_ic),  // Provide velocity data
    &device,
)?;
```

**Traveling Wave IC**:
```rust
// u(x,0) = sin(kx), v(x,0) = -c*k*cos(kx)
let k = std::f32::consts::PI;
let c = 1500.0;

let x_ic: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
let u_ic: Vec<f32> = x_ic.iter().map(|&x| (k*x).sin()).collect();
let v_ic: Vec<f32> = x_ic.iter().map(|&x| -c*k*(k*x).cos()).collect();

// Train with traveling wave IC
let metrics = solver.train(
    &x_data, &y_data, &z_data, &t_data, &u_data,
    Some(&v_ic),
    &device,
)?;
```

### Multi-GPU Training (Future)

**Planned Feature** (not yet implemented):
```rust
// Pseudo-code for future API
use burn::backend::WgpuMulti;

type Backend = WgpuMulti<f32, i32>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Automatic multi-GPU detection and load balancing
    let devices = burn::backend::wgpu::devices()?;
    println!("Found {} GPUs", devices.len());
    
    // Data-parallel training across GPUs
    let mut solver = BurnPINN3DWave::<Backend>::new_multi_gpu(
        config, geometry, wave_speed_fn, &devices
    )?;
    
    // Training automatically distributed
    let metrics = solver.train(...)?;
    
    Ok(())
}
```

**Expected Speedups**:
- 2 GPUs: 1.8× vs single GPU (communication overhead)
- 4 GPUs: 3.2× vs single GPU
- 8 GPUs: 5.6× vs single GPU (diminishing returns)

### Transfer Learning (Future)

**Planned Feature**:
```rust
// Train on coarse problem, refine on fine problem
let coarse_config = BurnPINN3DConfig::default()
    .with_hidden_layers(vec![64, 64])
    .with_epochs(5000);

let mut coarse_solver = BurnPINN3DWave::<Backend>::new(
    coarse_config, geometry, wave_speed_fn, &device
)?;

// Coarse training
let _ = coarse_solver.train(&x_coarse, &y_coarse, &z_coarse, &t_coarse, &u_coarse, None, &device)?;

// Fine-tuning on high-resolution data
let fine_config = BurnPINN3DConfig::default()
    .with_hidden_layers(vec![64, 64])
    .with_learning_rate(1e-5)  // Lower LR for fine-tuning
    .with_epochs(2000);

let mut fine_solver = BurnPINN3DWave::<Backend>::new(
    fine_config, geometry, wave_speed_fn, &device
)?;

// Transfer weights (API not yet exposed)
// fine_solver.load_weights_from(&coarse_solver)?;

// Fine-tuning
let metrics = fine_solver.train(&x_fine, &y_fine, &z_fine, &t_fine, &u_fine, None, &device)?;
```

---

## Best Practices

### Data Preparation

1. **Normalization**: Scale coordinates to [-1, 1] or [0, 1]
   ```rust
   let x_norm: Vec<f32> = x_data.iter().map(|&x| (x - x_min) / (x_max - x_min)).collect();
   ```

2. **Units**: Use consistent units (SI preferred)
   - Length: meters (m)
   - Time: seconds (s)
   - Wave speed: m/s
   - Pressure/displacement: Pa or m

3. **Noise Handling**: Add small noise to training data for robustness
   ```rust
   use rand::Rng;
   let mut rng = rand::thread_rng();
   let u_noisy: Vec<f32> = u_data.iter()
       .map(|&u| u + rng.gen_range(-0.01..0.01))
       .collect();
   ```

4. **Outlier Removal**: Filter out obviously wrong measurements

### Network Architecture

1. **Start Small**: Begin with 2-3 layers of 64 neurons
2. **Increase Gradually**: Double capacity if underfitting
3. **Monitor Overfitting**: If training loss << validation loss, reduce capacity
4. **Activation Function**: Tanh (hardcoded, mathematically justified)

### Training Strategy

1. **Warm Start**: Train for 1000 epochs, inspect loss, then continue
2. **Loss Balancing**: Adjust weights so all components are O(1e-2) at convergence
3. **Learning Rate**: Use adaptive schedule (built-in), don't override unless needed
4. **Early Stopping**: Manually stop if loss plateaus for 5000+ epochs

### Validation

1. **Held-Out Data**: Reserve 20% of observations for validation
2. **Physics Check**: Verify PDE residual is small on test points
3. **Boundary Check**: Verify BC satisfied on boundaries
4. **Convergence Study**: Train with increasing collocation points, ensure convergence

### Reproducibility

1. **Seed Setting** (future enhancement):
   ```rust
   // Currently: Random initialization on each run
   // Future: config.seed = Some(42)
   ```

2. **Save Checkpoints** (future enhancement):
   ```rust
   // Pseudo-code for future API
   solver.save_checkpoint("checkpoint_epoch5000.pth")?;
   solver.load_checkpoint("checkpoint_epoch5000.pth")?;
   ```

3. **Log Hyperparameters**: Always document your config in logs

### Performance

1. **Use GPU**: 10-50× speedup for medium/large problems
2. **Profile First**: Use `cargo bench` to identify bottlenecks
3. **Batch Training** (future): Process multiple problems in parallel
4. **Mixed Precision** (future): FP16 for 2× memory, slight accuracy loss

---

## References

### Theory

1. **PINNs**:
   - Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations". *Journal of Computational Physics*, 378, 686-707.

2. **Wave Equation PINNs**:
   - Rasht-Behesht, M., Huber, C., Shukla, K., & Karniadakis, G.E. (2022). "Physics-Informed Neural Networks (PINNs) for Wave Propagation and Full Waveform Inversions". *Journal of Geophysical Research: Solid Earth*, 127(5).

3. **Training Stabilization**:
   - Wang, S., Teng, Y., & Perdikaris, P. (2021). "Understanding and mitigating gradient flow pathologies in physics-informed neural networks". *SIAM Journal on Scientific Computing*, 43(5), A3055-A3081.

### Implementation

4. **Burn Framework**:
   - Burn Documentation: https://burn.dev/
   - GitHub: https://github.com/tracel-ai/burn

5. **Kwavers Documentation**:
   - API Reference: https://docs.rs/kwavers
   - Architecture: `docs/ARCHITECTURE.md`
   - Sprint Reports: `docs/sprints/`

### Related Guides

- **GPU Acceleration Guide**: `docs/guides/gpu_acceleration_guide.md` (planned)
- **Performance Tuning**: `docs/guides/performance_tuning.md` (planned)
- **Getting Started Tutorial**: `docs/tutorials/getting_started.md` (planned)

---

## Appendix: Checklist

### Pre-Training Checklist

- [ ] Data normalized to [-1, 1] or [0, 1]
- [ ] Units are consistent (SI recommended)
- [ ] No NaN/Inf in input data
- [ ] Geometry bounds match data extent
- [ ] Wave speed function is physically reasonable
- [ ] Initial conditions provided (u₀ and optionally v₀)
- [ ] Boundary conditions match problem type (Dirichlet currently)

### During Training Checklist

- [ ] Monitor loss curves every 100 epochs
- [ ] Check for NaN/Inf (auto-detected)
- [ ] Verify all loss components decreasing
- [ ] Watch for adaptive LR decay messages
- [ ] Log hyperparameters for reproducibility

### Post-Training Checklist

- [ ] Final loss < 1e-3 (or acceptable threshold)
- [ ] Validate on held-out test data
- [ ] Verify PDE residual on test points
- [ ] Check boundary condition satisfaction
- [ ] Save predictions and metrics
- [ ] Document any issues or unusual behavior

---

## Support & Contributions

**Issues**: Report bugs or request features at [GitHub Issues](https://github.com/ryancinsight/kwavers/issues)  
**Discussions**: Ask questions at [GitHub Discussions](https://github.com/ryancinsight/kwavers/discussions)  
**Email**: ryanclanton@outlook.com  
**Citation**: See `README.md` for BibTeX

**Contributions Welcome**: Pull requests for improvements, additional examples, or bug fixes are appreciated!

---

**Last Updated**: 2026-02-04  
**Guide Version**: 1.0  
**Kwavers Version**: 3.0.0  
**Author**: Ryan Clanton PhD (@ryancinsight)