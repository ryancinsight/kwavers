use coeus_optim::scheduler::{CosineAnneal, SchedulerStrategy};
use coeus_optim::{Adam, Optimizer as CoeusOptimizer};

use super::super::network::ParamFieldPINNNetwork;
use super::helmholtz::helmholtz_residual_tensor;
use super::types::{
    FieldSurrogateTrainingConfig, StepMetrics, SurrogateTrainingMetrics, TrainingBatch,
};
use kwavers_core::error::KwaversResult;

/// Training context bundling network + optimiser + config.
///
/// Uses `coeus_optim::Adam` (β₁=0.9, β₂=0.999, ε=1e-5) — empirically
/// converges 5–10× faster than plain SGD on the field-surrogate
/// regression. The legacy [`super::super::ParamFieldOptimizer`]
/// (plain SGD) remains available for tests that need an
/// allocation-free optimizer step.
pub struct ParamFieldPINNTrainer<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    pub network: ParamFieldPINNNetwork<B>,
    pub optimizer: Adam<f32, B>,
    pub config: FieldSurrogateTrainingConfig,
    /// Optional cosine-annealing schedule `(t_max, eta_min)`. `None`
    /// keeps the LR fixed at `config.learning_rate`.
    pub lr_schedule: Option<CosineAnneal>,
    /// Step counter driving the cosine schedule.
    step_count: usize,
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> std::fmt::Debug
    for ParamFieldPINNTrainer<B>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParamFieldPINNTrainer")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> ParamFieldPINNTrainer<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Construct a trainer from a fresh network + config.
    /// # Errors
    /// Propagates [`FieldSurrogateTrainingConfig::validate`] errors.
    pub fn new(
        network: ParamFieldPINNNetwork<B>,
        config: FieldSurrogateTrainingConfig,
    ) -> KwaversResult<Self> {
        config.validate()?;
        let optimizer = Adam::new(network.parameters(), config.learning_rate, 0.9, 0.999, 1e-5);
        let lr_schedule = match config.cosine_schedule {
            Some((max_iter, min_lr)) if max_iter > 0 => Some(CosineAnneal {
                t_max: max_iter,
                eta_min: min_lr,
            }),
            _ => None,
        };
        Ok(Self {
            network,
            optimizer,
            config,
            lr_schedule,
            step_count: 0,
        })
    }

    /// One forward / backward / update cycle.
    ///
    /// Returns the data, Helmholtz, and total loss for the batch.
    /// Mutates `self.network` in place by consuming and replacing it
    /// with the updated network from the optimiser.
    pub fn step(&mut self, batch: TrainingBatch<B>) -> StepMetrics {
        for p in self.network.parameters() {
            p.zero_grad();
        }

        let pred = self.network.forward(&batch.inputs);
        let diff = coeus_autograd::sub(&pred, &batch.targets);
        let data_loss = coeus_autograd::mean(&coeus_autograd::mul(&diff, &diff));
        let data_loss_w = coeus_autograd::scalar_mul(&data_loss, self.config.data_weight);

        let mut total = data_loss_w;

        // Per-kernel-scoped peak-prominence loss (Phase C-10).
        //
        // Phase C-9 demonstrated that a single batch-wide
        // `(max(pred) − max(target))²` term fragments per-f0 fits
        // because the batch mixes voxels from every kernel and a
        // single `max(target)` aggregates over an ambiguous
        // `(f0, pnp)`. This groups batch rows by
        // `group_ids[i] = source-kernel index`, finds the argmax row
        // *per group* from the raw (untracked) prediction values, and
        // re-selects that exact row through `coeus_autograd::slice` so
        // the gradient flows only into the argmax voxel of each
        // group — the same selectivity `Tensor::max()`'s backward
        // gives in the original formulation.
        let prom_value: f32 = if self.config.peak_prominence_weight > 0.0 {
            let n = batch.inputs.tensor.shape()[0];
            let pred_pmax_vals = pred.tensor.as_slice();
            let group_ids = batch.group_ids.tensor.as_slice();
            let target_pmax_vals = batch.targets.tensor.as_slice();

            let mut prom_acc: Option<coeus_autograd::Var<f32, B>> = None;
            for g in 0..batch.num_groups {
                let mut best_idx = None;
                let mut best_val = f32::NEG_INFINITY;
                for i in 0..n {
                    if group_ids[i] as usize == g {
                        let v = pred_pmax_vals[i * 3 + 1];
                        if v > best_val {
                            best_val = v;
                            best_idx = Some(i);
                        }
                    }
                }
                let Some(idx) = best_idx else { continue };
                let pred_sel = coeus_autograd::slice(&pred, &[(idx, idx + 1), (1, 2)]);
                let target_best_idx = (0..n)
                    .filter(|&i| group_ids[i] as usize == g)
                    .max_by(|&a, &b| {
                        target_pmax_vals[a * 3 + 1].total_cmp(&target_pmax_vals[b * 3 + 1])
                    })
                    .unwrap_or(idx);
                let target_sel = coeus_autograd::slice(
                    &batch.targets,
                    &[(target_best_idx, target_best_idx + 1), (1, 2)],
                );
                let gap_g = coeus_autograd::sub(&pred_sel, &target_sel);
                let gap_sq = coeus_autograd::mul(&gap_g, &gap_g);
                prom_acc = Some(match prom_acc {
                    Some(acc) => coeus_autograd::add(&acc, &gap_sq),
                    None => gap_sq,
                });
            }
            let denom = batch.num_groups.max(1) as f32;
            match prom_acc {
                Some(acc) => {
                    let prom = coeus_autograd::scalar_mul(&acc, 1.0 / denom);
                    let prom_w =
                        coeus_autograd::scalar_mul(&prom, self.config.peak_prominence_weight);
                    total = coeus_autograd::add(&total, &prom_w);
                    prom.tensor.as_slice()[0] * self.config.peak_prominence_weight
                }
                None => 0.0,
            }
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
            let helm = coeus_autograd::mean(&coeus_autograd::mul(&r, &r));
            let helm_w = coeus_autograd::scalar_mul(&helm, self.config.helmholtz_weight);
            total = coeus_autograd::add(&total, &helm_w);
            helm.tensor.as_slice()[0] * self.config.helmholtz_weight
        } else {
            0.0
        };

        let data_value: f32 = data_loss.tensor.as_slice()[0] * self.config.data_weight;
        let total_value: f32 = total.tensor.as_slice()[0];

        total.backward();

        let lr = if let Some(schedule) = self.lr_schedule.as_ref() {
            let lr = schedule.lr(self.config.learning_rate as f64, self.step_count);
            self.step_count += 1;
            lr as f32
        } else {
            self.config.learning_rate
        };
        self.optimizer.lr = lr;
        self.optimizer.step();
        self.network.load_parameters(&self.optimizer.params);

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
    {
        let mut metrics = SurrogateTrainingMetrics::default();
        for step in 0..n_steps {
            let batch = make_batch(step);
            let m = self.step(batch);
            metrics.accumulate(m);
        }
        metrics
    }

    /// Detach the network from any further training state — coeus has
    /// no separate autodiff/inference backend split, so this simply
    /// returns a clone of the current network for use in inference-only
    /// contexts.
    pub fn into_inference_network(self) -> ParamFieldPINNNetwork<B> {
        self.network
    }
}
