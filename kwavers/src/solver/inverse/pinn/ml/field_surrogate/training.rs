//! Phase C-2 supervised + physics-residual training loop.
//!
//! Combines two loss terms:
//!
//! * **Data loss** — supervised MSE on `(p_min, p_max, p_rms)` voxel
//!   tuples sampled from the [`KernelCube`](
//!   crate::physics::field_surrogate::KernelCube) ground truth.
//! * **Helmholtz residual loss** — finite-difference
//!   `R = ∇²p + k²p` on the predicted `p_max` channel evaluated at a
//!   set of collocation points (typically the same batch as the data
//!   loss, possibly augmented with samples drawn at random `(f0, pnp)`
//!   for outside-cube generalization).
//!
//! Both losses are computed in the same autodiff graph; one
//! `backward()` produces the combined gradient.
//!
//! ## API surface
//!
//! * [`TrainingConfig`] — hyperparameters (learning rate, loss
//!   weights, FD epsilon, sound speed).
//! * [`TrainingBatch`] — one mini-batch's worth of network inputs +
//!   data targets + per-sample physical `f0` for the Helmholtz term.
//! * [`train_step`] — one forward/backward/update step, returns the
//!   per-step loss components.
//! * [`TrainingMetrics`] — running per-epoch loss aggregates.

use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::lr_scheduler::cosine::{
    CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig,
};
use burn::optim::lr_scheduler::LrScheduler;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::AutodiffBackend, ElementConversion, Tensor, TensorData};

use crate::core::error::{KwaversError, KwaversResult};
use super::network::ParamFieldPINNNetwork;

/// Hyperparameters for the field-surrogate trainer.
///
/// Defaults are calibrated for the 4-corner kernel cube
/// (`{0.5, 1.0} MHz × {15, 30} MPa`) on a CPU `NdArray` autodiff
/// backend; GPU or larger sweeps may want larger batches and
/// smaller `learning_rate`.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// SGD learning rate. Default 1e-3.
    pub learning_rate: f32,
    /// Weight on the supervised data MSE term. Default 1.0.
    pub data_weight: f32,
    /// Weight on the (dimensionless) Helmholtz residual term.
    /// Default 0 (off — the C-1 unit tests run pure-data MSE).
    /// Typical training uses ~1.0 since the residual is now
    /// dimensionless O(1) — see `helmholtz_residual_tensor` doc.
    pub helmholtz_weight: f32,
    /// Finite-difference step (m) for the spatial Laplacian. Default
    /// 5 × 10⁻⁴ — half a typical kernel `dx`. Tradeoff: smaller is
    /// more accurate at the cost of catastrophic cancellation under
    /// f32 precision.
    pub helmholtz_eps_m: f32,
    /// Medium sound speed (m/s) for the wavenumber `k = 2π·f0/c0`.
    /// Default 1500 (water).
    pub c0_m_per_s: f32,
    /// Optional cosine-annealing LR schedule. When `Some((max_iter,
    /// min_lr))` the per-step LR follows
    /// `min_lr + 0.5·(initial − min)·(1 + cos(π · t / max_iter))`.
    /// `None` keeps `learning_rate` constant. Cosine annealing closes
    /// the peak-RMSE gap by letting the network do high-LR exploration
    /// in the early phase and low-LR fine-tuning near convergence —
    /// critical for the residual focal-peak fitting under importance
    /// sampling.
    pub cosine_schedule: Option<(usize, f64)>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0e-3,
            data_weight: 1.0,
            helmholtz_weight: 0.0,
            helmholtz_eps_m: 5.0e-4,
            c0_m_per_s: 1500.0,
            cosine_schedule: None,
        }
    }
}

impl TrainingConfig {
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when any field is
    /// non-positive (or negative loss weights).
    pub fn validate(&self) -> KwaversResult<()> {
        if self.learning_rate <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "learning_rate must be > 0".into(),
            ));
        }
        if self.data_weight < 0.0 || self.helmholtz_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "loss weights must be ≥ 0".into(),
            ));
        }
        if self.helmholtz_eps_m <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "helmholtz_eps_m must be > 0".into(),
            ));
        }
        if self.c0_m_per_s <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "c0_m_per_s must be > 0".into(),
            ));
        }
        Ok(())
    }
}

/// One mini-batch for [`train_step`].
///
/// All inputs are pre-normalised to the network's `[-1, 1]` input
/// space; targets are pre-normalised against the per-channel scales
/// stored in the surrounding training context.
#[derive(Debug)]
pub struct TrainingBatch<B: AutodiffBackend> {
    /// Network inputs `[batch, 5]`.
    pub inputs: Tensor<B, 2>,
    /// Per-channel targets `[batch, 3]`.
    pub targets: Tensor<B, 2>,
    /// Per-sample physical `f0` (Hz) `[batch]` — used to compute the
    /// per-sample wavenumber for the Helmholtz residual.
    pub f0_phys_hz: Tensor<B, 1>,
    /// Spatial half-extent `(hx, hy, hz)` (m) used to denormalise the
    /// finite-difference perturbations from input-space units back
    /// into physical metres. Must match the normalisation applied at
    /// data-loading time.
    pub coord_half_m: (f32, f32, f32),
    /// Output denormalisation scale for `p_max` (Pa). Maps the
    /// network's `[-1, 1]` `p_max` channel back to physical Pa for
    /// the Helmholtz residual.
    pub p_max_scale_pa: f32,
}

/// Per-step loss values returned from [`train_step`]. All values are
/// post-weighting so `total = data + helmholtz`.
#[derive(Debug, Clone, Copy)]
pub struct StepMetrics {
    pub data: f32,
    pub helmholtz: f32,
    pub total: f32,
}

/// Running per-epoch metrics.
#[derive(Debug, Default, Clone, Copy)]
pub struct TrainingMetrics {
    pub steps: usize,
    pub data_sum: f32,
    pub helmholtz_sum: f32,
    pub total_sum: f32,
}

impl TrainingMetrics {
    pub fn accumulate(&mut self, step: StepMetrics) {
        self.steps += 1;
        self.data_sum += step.data;
        self.helmholtz_sum += step.helmholtz;
        self.total_sum += step.total;
    }

    pub fn averages(&self) -> StepMetrics {
        let n = self.steps.max(1) as f32;
        StepMetrics {
            data: self.data_sum / n,
            helmholtz: self.helmholtz_sum / n,
            total: self.total_sum / n,
        }
    }
}

/// Compute the **dimensionless** Helmholtz residual on a batch via
/// central finite differences. All seven forward passes share the
/// same autodiff graph so one `backward()` propagates gradients
/// through the network.
///
/// The physical residual `R_phys = ∇²p + k²·p` has units of
/// [Pa/m²] and magnitudes ~10¹⁴ for histotripsy fields — squaring it
/// overflows f32 precision and produces NaN under autodiff. We
/// instead return the dimensionless ratio
///
/// ```text
///   R̂ = R_phys / (k² · p_max_scale)
///      = [∇²p̂ + (k·eps_m)⁻² · (k_eps_m)² · p̂] / (k·eps_m)²
///      = (∑ p̂(±ε̂) − 6·p̂) / (k·eps_m)² + p̂
/// ```
///
/// Both terms are O(1) for Helmholtz-consistent predictions and
/// stay in f32-safe range under squaring + autodiff.
fn helmholtz_residual_tensor<B: AutodiffBackend>(
    network: &ParamFieldPINNNetwork<B>,
    batch: &TrainingBatch<B>,
    eps_m: f32,
    c0: f32,
) -> Tensor<B, 1> {
    let device = batch.inputs.device();
    let n = batch.inputs.dims()[0];
    let center_out = network.forward(batch.inputs.clone());
    let p_center = center_out.slice([0..n, 1..2]).reshape([n]);

    // Per-axis perturbation tensors. Input-space step is eps_m / h_axis.
    let inv_hx = 1.0 / batch.coord_half_m.0;
    let inv_hy = 1.0 / batch.coord_half_m.1;
    let inv_hz = 1.0 / batch.coord_half_m.2;
    let one_hot = |axis: usize, sign: f32, inv_h: f32| -> Tensor<B, 2> {
        let mut data = vec![0.0_f32; n * 5];
        let perturb = sign * eps_m * inv_h;
        for i in 0..n {
            data[i * 5 + axis] = perturb;
        }
        Tensor::<B, 2>::from_data(TensorData::new(data, [n, 5]), &device)
    };

    let plus_x = batch.inputs.clone() + one_hot(0, 1.0, inv_hx);
    let minus_x = batch.inputs.clone() + one_hot(0, -1.0, inv_hx);
    let plus_y = batch.inputs.clone() + one_hot(1, 1.0, inv_hy);
    let minus_y = batch.inputs.clone() + one_hot(1, -1.0, inv_hy);
    let plus_z = batch.inputs.clone() + one_hot(2, 1.0, inv_hz);
    let minus_z = batch.inputs.clone() + one_hot(2, -1.0, inv_hz);

    let pick_pmax = |t: Tensor<B, 2>| -> Tensor<B, 1> {
        t.slice([0..n, 1..2]).reshape([n])
    };
    let p_xp = pick_pmax(network.forward(plus_x));
    let p_xm = pick_pmax(network.forward(minus_x));
    let p_yp = pick_pmax(network.forward(plus_y));
    let p_ym = pick_pmax(network.forward(minus_y));
    let p_zp = pick_pmax(network.forward(plus_z));
    let p_zm = pick_pmax(network.forward(minus_z));

    // Sum of finite-difference second-difference contributions across
    // all three axes (still dimensionless, equal to eps_m² · ∇²p̂).
    let lap_sum = p_xp + p_xm + p_yp + p_ym + p_zp + p_zm
        - p_center.clone().mul_scalar(6.0);

    // Per-sample (k·eps_m)². Tensor of shape [n].
    let k_eps = batch
        .f0_phys_hz
        .clone()
        .mul_scalar(2.0 * std::f32::consts::PI * eps_m / c0);
    let k_eps_sq = k_eps.clone() * k_eps;

    // Dimensionless residual: lap_sum / (k·eps_m)² + p̂.
    // Use a small floor on (k·eps_m)² to avoid divide-by-zero when
    // f0 happens to be supplied as zero in degenerate test cases.
    let safe_k_eps_sq = k_eps_sq.add_scalar(1.0e-12);
    lap_sum / safe_k_eps_sq + p_center
}

/// Training context bundling network + optimiser + config.
///
/// Uses Burn's built-in `Adam` optimizer (β₁=0.9, β₂=0.999, ε=1e-5)
/// — empirically converges 5–10× faster than plain SGD on the
/// field-surrogate regression. The legacy [`ParamFieldOptimizer`]
/// (plain SGD) remains available for tests that need an
/// allocation-free optimizer step.
pub struct ParamFieldPINNTrainer<B: AutodiffBackend> {
    pub network: ParamFieldPINNNetwork<B>,
    pub optimizer: OptimizerAdaptor<Adam, ParamFieldPINNNetwork<B>, B>,
    pub config: TrainingConfig,
    /// Optional cosine-annealing scheduler. `None` keeps the LR fixed.
    pub lr_scheduler: Option<CosineAnnealingLrScheduler>,
}

impl<B: AutodiffBackend> std::fmt::Debug for ParamFieldPINNTrainer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParamFieldPINNTrainer")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<B: AutodiffBackend> ParamFieldPINNTrainer<B> {
    /// Construct a trainer from a fresh network + config.
    /// # Errors
    /// Propagates [`TrainingConfig::validate`] errors.
    pub fn new(
        network: ParamFieldPINNNetwork<B>,
        config: TrainingConfig,
    ) -> KwaversResult<Self> {
        config.validate()?;
        let optimizer = AdamConfig::new().init();
        let lr_scheduler = match config.cosine_schedule {
            Some((max_iter, min_lr)) if max_iter > 0 => {
                let sched_cfg = CosineAnnealingLrSchedulerConfig::new(
                    config.learning_rate as f64,
                    max_iter,
                )
                .with_min_lr(min_lr);
                Some(sched_cfg.init().map_err(KwaversError::InvalidInput)?)
            }
            _ => None,
        };
        Ok(Self {
            network,
            optimizer,
            config,
            lr_scheduler,
        })
    }

    /// One forward / backward / update cycle.
    ///
    /// Returns the data, Helmholtz, and total loss for the batch.
    /// Mutates `self.network` in place by consuming and replacing it
    /// with the updated network from the optimiser.
    pub fn step(&mut self, batch: TrainingBatch<B>) -> StepMetrics
    where
        B: AutodiffBackend,
        ParamFieldPINNNetwork<B>: AutodiffModule<B>,
    {
        // Forward + data MSE
        let pred = self.network.forward(batch.inputs.clone());
        let diff = pred - batch.targets.clone();
        let data_loss = diff.powf_scalar(2.0).mean();
        let data_loss_w = data_loss.clone().mul_scalar(self.config.data_weight);

        // Helmholtz residual (same autodiff graph)
        let mut total = data_loss_w;
        let helm_value: f32 = if self.config.helmholtz_weight > 0.0 {
            let r = helmholtz_residual_tensor(
                &self.network,
                &batch,
                self.config.helmholtz_eps_m,
                self.config.c0_m_per_s,
            );
            let helm = r.powf_scalar(2.0).mean();
            let helm_w = helm.clone().mul_scalar(self.config.helmholtz_weight);
            total = total + helm_w;
            helm.into_scalar().elem::<f32>() * self.config.helmholtz_weight
        } else {
            0.0
        };
        let data_value: f32 =
            data_loss.into_scalar().elem::<f32>() * self.config.data_weight;
        let total_value: f32 = total.clone().into_scalar().elem::<f32>();

        // Backward + Adam parameter update. Use the cosine-scheduled
        // learning rate when a scheduler is configured; otherwise use
        // the constant rate from the config.
        let grads = total.backward();
        let grads_params = GradientsParams::from_grads(grads, &self.network);
        let lr = if let Some(scheduler) = self.lr_scheduler.as_mut() {
            scheduler.step()
        } else {
            self.config.learning_rate as f64
        };
        let placeholder = self.network.clone();
        let net_taken = std::mem::replace(&mut self.network, placeholder);
        self.network = self.optimizer.step(lr, net_taken, grads_params);

        StepMetrics {
            data: data_value,
            helmholtz: helm_value,
            total: total_value,
        }
    }

    /// Run `n_steps` training iterations, calling `make_batch(step)`
    /// each iteration to produce the batch. Returns the running
    /// per-epoch metrics across the entire run.
    pub fn run<F>(&mut self, n_steps: usize, mut make_batch: F) -> TrainingMetrics
    where
        F: FnMut(usize) -> TrainingBatch<B>,
        ParamFieldPINNNetwork<B>: AutodiffModule<B>,
    {
        let mut metrics = TrainingMetrics::default();
        for step in 0..n_steps {
            let batch = make_batch(step);
            let m = self.step(batch);
            metrics.accumulate(m);
        }
        metrics
    }
}

// `AutodiffModule` is required so the optimiser's `step` consumes the
// network and the gradient mapper can traverse parameters; this trait
// is auto-derived for any type marked `#[derive(Module)]`.
impl<B: AutodiffBackend> ParamFieldPINNTrainer<B>
where
    ParamFieldPINNNetwork<B>: AutodiffModule<B,
        InnerModule = ParamFieldPINNNetwork<<B as AutodiffBackend>::InnerBackend>>,
{
    /// Move the network into a non-autodiff backend `B::InnerBackend`
    /// after training is complete — typically used to detach the
    /// autodiff graph before serialising or doing pure inference.
    pub fn into_inference_network(self) -> ParamFieldPINNNetwork<B::InnerBackend> {
        self.network.valid()
    }
}
