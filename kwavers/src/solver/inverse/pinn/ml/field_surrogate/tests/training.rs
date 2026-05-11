//! Trainer step, convergence, and Helmholtz-path tests (Phase C-2).

use burn::tensor::{Tensor, TensorData};

use super::super::config::ParamFieldPINNConfig;
use super::super::network::ParamFieldPINNNetwork;
use super::super::training::{ParamFieldPINNTrainer, TrainingBatch, TrainingConfig};
use super::AB;

// ─────────────────────────────────────────────────────────────────────────────
// Synthetic data helper
// ─────────────────────────────────────────────────────────────────────────────

/// Build a synthetic training batch from a separable Gaussian envelope:
/// `p_max(x, y, z) = exp(-(x²+y²+z²)/(2σ²))`, independent of `(f0, pnp)`.
///
/// Targets are normalised to `[0, 1]` so they live within the network's
/// `tanh`-friendly range.
fn make_synthetic_batch(
    device: &<AB as burn::tensor::backend::Backend>::Device,
    rng_seed: u64,
    n: usize,
) -> TrainingBatch<AB> {
    use burn::tensor::Distribution;
    // Inputs: uniform in [-1, 1]^5
    let inputs = Tensor::<AB, 2>::random(
        [n, 5],
        Distribution::Uniform(-1.0, 1.0),
        device,
    );
    // Pull data back to host to compute targets analytically.
    let host_data: Vec<f32> = inputs
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec()
        .unwrap();
    let mut targets = vec![0.0_f32; n * 3];
    let mut f0_vec = vec![0.0_f32; n];
    let _ = rng_seed; // kept for future deterministic seeding
    for i in 0..n {
        let x = host_data[i * 5];
        let y = host_data[i * 5 + 1];
        let z = host_data[i * 5 + 2];
        let f0_norm = host_data[i * 5 + 3];
        let r2 = x * x + y * y + z * z;
        let env = (-r2 / 0.5).exp(); // sigma² = 0.25 in normalised units
        // (p_min_norm, p_max_norm, p_rms_norm) — same shape, different scales
        targets[i * 3] = -env * 0.95;
        targets[i * 3 + 1] = env * 0.95;
        targets[i * 3 + 2] = env * 0.7;
        // Map f0_norm in [-1, 1] back to physical Hz in [0.5, 1.0] MHz
        f0_vec[i] = 0.75e6 + 0.25e6 * f0_norm;
    }
    let targets =
        Tensor::<AB, 2>::from_data(TensorData::new(targets, [n, 3]), device);
    let f0_phys =
        Tensor::<AB, 1>::from_data(TensorData::new(f0_vec, [n]), device);
    TrainingBatch {
        inputs,
        targets,
        f0_phys_hz: f0_phys,
        coord_half_m: (10.0e-3, 10.0e-3, 10.0e-3),
        p_max_scale_pa: 30.0e6,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_trainer_step_returns_finite_metrics() {
    let cfg = ParamFieldPINNConfig {
        hidden_layers: vec![32, 32],
        ..ParamFieldPINNConfig::default()
    };
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<AB>::new(&cfg, &device).unwrap();
    let train_cfg = TrainingConfig {
        helmholtz_weight: 0.0, // pure-data step for this smoke test
        ..TrainingConfig::default()
    };
    let mut trainer = ParamFieldPINNTrainer::<AB>::new(net, train_cfg).unwrap();
    let batch = make_synthetic_batch(&device, 0, 64);
    let m = trainer.step(batch);
    assert!(m.data.is_finite() && m.data >= 0.0, "data loss = {}", m.data);
    assert!(m.helmholtz == 0.0, "weight-0 helmholtz must be exactly 0");
    assert!(m.total.is_finite() && m.total >= 0.0);
}

#[test]
fn test_trainer_data_loss_decreases_over_50_steps() {
    let cfg = ParamFieldPINNConfig {
        hidden_layers: vec![32, 32],
        ..ParamFieldPINNConfig::default()
    };
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<AB>::new(&cfg, &device).unwrap();
    let train_cfg = TrainingConfig {
        learning_rate: 5.0e-2, // larger LR for the small synthetic problem
        helmholtz_weight: 0.0,
        ..TrainingConfig::default()
    };
    let mut trainer = ParamFieldPINNTrainer::<AB>::new(net, train_cfg).unwrap();

    // Average over the first 5 / last 5 steps to suppress per-batch
    // stochastic variance from the random input sampling.
    let n_steps = 100usize;
    let mut history = Vec::with_capacity(n_steps);
    for step in 0..n_steps {
        let batch = make_synthetic_batch(&device, step as u64, 128);
        history.push(trainer.step(batch).data);
    }
    let first_avg: f32 = history[..5].iter().sum::<f32>() / 5.0;
    let last_avg: f32 = history[n_steps - 5..].iter().sum::<f32>() / 5.0;
    // Plain SGD on a smooth Gaussian regression converges slowly;
    // require a 20% drop on the smoothed loss as the regression-
    // is-progressing signal.
    assert!(
        last_avg < first_avg * 0.8,
        "data loss did not decrease: first_avg={first_avg}, last_avg={last_avg}"
    );
}

// The long-running demo training has been moved to the standalone
// example binary at `kwavers/examples/field_surrogate_demo.rs`. See
// the doc-comment near the top of this file for the rationale.

#[test]
fn test_trainer_with_helmholtz_weight_runs_finite() {
    // The Helmholtz loss path executes 7 forward passes per step.
    // This test exercises that path on a tiny network and verifies
    // both loss components stay finite over 5 steps.
    let cfg = ParamFieldPINNConfig {
        hidden_layers: vec![16, 16],
        ..ParamFieldPINNConfig::default()
    };
    let device = Default::default();
    let net = ParamFieldPINNNetwork::<AB>::new(&cfg, &device).unwrap();
    // The Helmholtz residual is dimensionless O(1); weight 1.0
    // makes it co-dominant with the data loss. Smaller weights
    // (0.01–0.1) are typical when both must coexist.
    let train_cfg = TrainingConfig {
        learning_rate: 1.0e-3,
        helmholtz_weight: 0.1,
        ..TrainingConfig::default()
    };
    let mut trainer = ParamFieldPINNTrainer::<AB>::new(net, train_cfg).unwrap();
    for step in 0..5 {
        let batch = make_synthetic_batch(&device, step as u64, 32);
        let m = trainer.step(batch);
        assert!(m.data.is_finite(), "data loss not finite at step {step}");
        assert!(m.helmholtz.is_finite(), "helm loss not finite at step {step}");
        assert!(m.total.is_finite(), "total loss not finite at step {step}");
    }
}
