use coeus_autograd::Var;

use kwavers_core::constants::SOUND_SPEED_WATER_SIM;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Hyperparameters for the field-surrogate trainer.
///
/// Defaults are calibrated for the 4-corner kernel cube
/// (`{0.5, 1.0} MHz × {15, 30} MPa`) on a CPU `NdArray` autodiff
/// backend; GPU or larger sweeps may want larger batches and
/// smaller `learning_rate`.
#[derive(Debug, Clone)]
pub struct FieldSurrogateTrainingConfig {
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
    /// Weight on the **peak-prominence** loss term `(max(pred_pmax)
    /// − max(target_pmax))²` aggregated over the batch. The
    /// volumetric data MSE under-fits the focal peak because the
    /// argmax voxel contributes only `1/batch_size` of the total
    /// loss; this term gives the argmax voxel a direct, dedicated
    /// gradient channel so peak prediction tracks target peak.
    /// Default 0 (off) preserves pre-C-9 behaviour. Typical training
    /// uses 0.1–1.0 — large enough to bias the argmax voxel without
    /// overwhelming the volumetric fit.
    pub peak_prominence_weight: f32,
}

impl Default for FieldSurrogateTrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1.0e-3,
            data_weight: 1.0,
            helmholtz_weight: 0.0,
            helmholtz_eps_m: 5.0e-4,
            c0_m_per_s: SOUND_SPEED_WATER_SIM as f32,
            cosine_schedule: None,
            peak_prominence_weight: 0.0,
        }
    }
}

impl FieldSurrogateTrainingConfig {
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
            return Err(KwaversError::InvalidInput("c0_m_per_s must be > 0".into()));
        }
        if self.peak_prominence_weight < 0.0 {
            return Err(KwaversError::InvalidInput(
                "peak_prominence_weight must be ≥ 0".into(),
            ));
        }
        Ok(())
    }
}

/// One mini-batch for the training step.
///
/// All inputs are pre-normalised to the network's `[-1, 1]` input
/// space; targets are pre-normalised against the per-channel scales
/// stored in the surrounding training context.
#[derive(Debug)]
pub struct TrainingBatch<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> {
    /// Network inputs `[batch, 5]`.
    pub inputs: Var<f32, B>,
    /// Per-channel targets `[batch, 3]`.
    pub targets: Var<f32, B>,
    /// Per-sample physical `f0` (Hz) `[batch]` — used to compute the
    /// per-sample wavenumber for the Helmholtz residual.
    pub f0_phys_hz: Var<f32, B>,
    /// Per-sample source-kernel index `[batch]` as f32 (storage-only
    /// — not used as a network input). Voxels from kernel `k` carry
    /// `group_id == k`. Consumed by the per-kernel-scoped
    /// peak-prominence loss (Phase C-10) so the `(max(pred) −
    /// max(target))²` aggregation is per-kernel rather than batch-
    /// wide, eliminating the cross-kernel `max(target)` ambiguity
    /// that fragmented per-f0 fits in Phase C-9.
    pub group_ids: Var<f32, B>,
    /// Total number of distinct groups across the source dataset.
    /// Used to bound the per-group prominence loop without scanning
    /// every batch for unique IDs.
    pub num_groups: usize,
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

/// Per-step loss values returned from a training step.
/// All values are post-weighting so `total = data + helmholtz + peak_prominence`.
#[derive(Debug, Clone, Copy)]
pub struct StepMetrics {
    pub data: f32,
    pub helmholtz: f32,
    pub peak_prominence: f32,
    pub total: f32,
}

/// Running per-epoch metrics.
#[derive(Debug, Default, Clone, Copy)]
pub struct SurrogateTrainingMetrics {
    pub steps: usize,
    pub data_sum: f32,
    pub helmholtz_sum: f32,
    pub peak_prominence_sum: f32,
    pub total_sum: f32,
}

impl SurrogateTrainingMetrics {
    pub fn accumulate(&mut self, step: StepMetrics) {
        self.steps += 1;
        self.data_sum += step.data;
        self.helmholtz_sum += step.helmholtz;
        self.peak_prominence_sum += step.peak_prominence;
        self.total_sum += step.total;
    }

    pub fn averages(&self) -> StepMetrics {
        let n = self.steps.max(1) as f32;
        StepMetrics {
            data: self.data_sum / n,
            helmholtz: self.helmholtz_sum / n,
            peak_prominence: self.peak_prominence_sum / n,
            total: self.total_sum / n,
        }
    }
}
