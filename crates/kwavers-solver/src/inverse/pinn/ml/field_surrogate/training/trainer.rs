use burn::module::AutodiffModule;
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::lr_scheduler::cosine::{
    CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig,
};
use burn::optim::lr_scheduler::LrScheduler;
use burn::optim::{Adam, AdamConfig, GradientsParams, Optimizer};
use burn::tensor::{backend::AutodiffBackend, ElementConversion};

use super::super::network::ParamFieldPINNNetwork;
use super::helmholtz::helmholtz_residual_tensor;
use super::types::{
    FieldSurrogateTrainingConfig, StepMetrics, SurrogateTrainingMetrics, TrainingBatch,
};
use kwavers_core::error::{KwaversError, KwaversResult};

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
    pub config: FieldSurrogateTrainingConfig,
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
    /// Propagates [`FieldSurrogateTrainingConfig::validate`] errors.
    pub fn new(
        network: ParamFieldPINNNetwork<B>,
        config: FieldSurrogateTrainingConfig,
    ) -> KwaversResult<Self> {
        config.validate()?;
        let optimizer = AdamConfig::new().init();
        let lr_scheduler = match config.cosine_schedule {
            Some((max_iter, min_lr)) if max_iter > 0 => {
                let sched_cfg =
                    CosineAnnealingLrSchedulerConfig::new(config.learning_rate as f64, max_iter)
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
        let pred = self.network.forward(batch.inputs.clone());
        let diff = pred.clone() - batch.targets.clone();
        let data_loss = diff.powf_scalar(2.0).mean();
        let data_loss_w = data_loss.clone().mul_scalar(self.config.data_weight);

        let mut total = data_loss_w;
        // Per-kernel-scoped peak-prominence loss (Phase C-10).
        //
        // Phase C-9 demonstrated that a single batch-wide
        // `(max(pred) − max(target))²` term fragments per-f0 fits
        // because the batch mixes voxels from every kernel and a
        // single `max(target)` aggregates over an ambiguous
        // `(f0, pnp)`. The C-10 fix groups batch rows by
        // `group_ids[i] = source-kernel index`, computes the
        // max-pair *per group* via boolean masking, and accumulates
        // the squared gaps. Empty groups contribute 0. Masking uses
        // a large negative constant on out-of-group rows so the
        // per-group max correctly picks the in-group maximum;
        // gradient flows only through in-group rows because the
        // mask is a constant in the autodiff graph.
        let prom_value: f32 = if self.config.peak_prominence_weight > 0.0 {
            let n = batch.inputs.dims()[0];
            let pred_pmax = pred.clone().slice([0..n, 1..2]).reshape([n]);
            let tgt_pmax = batch.targets.clone().slice([0..n, 1..2]).reshape([n]);
            // OUT_FILL is well below any plausible network output in
            // `[-1, 1]`. Out-of-group rows are forced to this value
            // so the per-group `.max()` always selects an in-group
            // row. For groups with zero representatives in this
            // batch, *all* rows are OUT_FILL after masking; the
            // resulting `(OUT_FILL − OUT_FILL)² = 0` and gradient
            // through `pred * 0` is zero, so empty groups
            // contribute neither loss nor gradient — exactly the
            // desired behaviour.
            const OUT_FILL: f32 = -1.0e6;
            // Start the accumulator with an autodiff-connected zero
            // (pred * 0 still carries the autodiff lineage).
            let mut prom_acc = pred_pmax.clone().mul_scalar(0.0).sum();
            for g in 0..batch.num_groups {
                let mask_bool = batch.group_ids.clone().equal_elem(g as f32);
                let mask = mask_bool.float();
                let inv_mask = mask.clone().mul_scalar(-1.0).add_scalar(1.0);
                let pred_masked =
                    pred_pmax.clone() * mask.clone() + inv_mask.clone().mul_scalar(OUT_FILL);
                let tgt_masked = tgt_pmax.clone() * mask + inv_mask.mul_scalar(OUT_FILL);
                let gap_g = pred_masked.max() - tgt_masked.max();
                prom_acc = prom_acc + gap_g.powf_scalar(2.0);
            }
            let denom = batch.num_groups.max(1) as f32;
            let prom = prom_acc.div_scalar(denom);
            let prom_w = prom.clone().mul_scalar(self.config.peak_prominence_weight);
            total = total + prom_w;
            prom.into_scalar().elem::<f32>() * self.config.peak_prominence_weight
        } else {
            0.0
        };
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
        let data_value: f32 = data_loss.into_scalar().elem::<f32>() * self.config.data_weight;
        let total_value: f32 = total.clone().into_scalar().elem::<f32>();

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
            peak_prominence: prom_value,
            total: total_value,
        }
    }

    /// Run `n_steps` training iterations, calling `make_batch(step)`
    /// each iteration to produce the batch. Returns the running
    /// per-epoch metrics across the entire run.
    pub fn run<F>(&mut self, n_steps: usize, mut make_batch: F) -> SurrogateTrainingMetrics
    where
        F: FnMut(usize) -> TrainingBatch<B>,
        ParamFieldPINNNetwork<B>: AutodiffModule<B>,
    {
        let mut metrics = SurrogateTrainingMetrics::default();
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
// is auto-derived for any type marked ``.
impl<B: AutodiffBackend> ParamFieldPINNTrainer<B>
where
    ParamFieldPINNNetwork<B>: AutodiffModule<
        B,
        InnerModule = ParamFieldPINNNetwork<<B as AutodiffBackend>::InnerBackend>,
    >,
{
    /// Move the network into a non-autodiff backend `B::InnerBackend`
    /// after training is complete — typically used to detach the
    /// autodiff graph before serialising or doing pure inference.
    pub fn into_inference_network(self) -> ParamFieldPINNNetwork<B::InnerBackend> {
        self.network.valid()
    }
}
