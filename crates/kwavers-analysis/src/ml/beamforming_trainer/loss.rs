//! Loss computation methods for `BeamformingTrainer`.
//!
//! - `compute_epoch_loss`: iterates mini-batches, combines data + physics loss.
//! - `compute_batch_data_loss`: null-model MSE (target-column variance).
//! - `compute_batch_physics_loss`: coherence, sparsity, and reciprocity terms
//!   from [`PhysicsLoss`].

use super::BeamformingTrainer;
use crate::ml::training::{PhysicsLoss, TrainingDataset};
use kwavers_core::error::KwaversResult;
use leto::SliceArg;

impl BeamformingTrainer {
    /// Compute loss for entire epoch (simplified implementation)
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub(super) fn compute_epoch_loss(&self, dataset: &TrainingDataset) -> KwaversResult<f64> {
        let mut total_loss = 0.0;
        let mut num_batches = 0;

        let mut offset = 0;
        while offset < dataset.len() {
            let batch = dataset.batch(offset, self.config.batch_size)?;

            let data_loss = self.compute_batch_data_loss(&batch)?;
            let physics_loss = self.compute_batch_physics_loss(&batch)?;

            let batch_loss = self
                .config
                .lambda_data
                .mul_add(data_loss, self.config.lambda_physics * physics_loss);

            total_loss += batch_loss;
            num_batches += 1;
            offset += self.config.batch_size;
        }

        Ok(if num_batches > 0 {
            total_loss / num_batches as f64
        } else {
            0.0
        })
    }

    /// Compute data loss (MSE) for batch.
    ///
    /// Without a concrete neural network model, this computes the *null-model*
    /// loss: MSE between the batch-mean prediction (constant baseline) and the
    /// true targets.  This equals the variance of the targets and provides a
    /// meaningful starting loss that the training loop can reduce.
    ///
    /// When a Burn model is integrated, replace the `y_pred` calculation with
    /// a forward pass: `y_pred = model.forward(batch.inputs)`.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_batch_data_loss(&self, batch: &TrainingDataset) -> KwaversResult<f64> {
        let n = batch.targets.len();
        if n == 0 {
            return Ok(0.0);
        }

        let n_rows = batch.targets.shape()[0];
        let n_cols = batch.targets.shape()[1];
        let inv_n = 1.0 / n_rows as f64;

        let mut mse = 0.0;
        for col in 0..n_cols {
            let col_mean: f64 = batch
                .targets
                .index_axis::<1>(1, col)
                .expect("invariant: column index in bounds")
                .iter()
                .sum::<f64>()
                * inv_n;
            mse += batch
                .targets
                .index_axis::<1>(1, col)
                .expect("invariant: column index in bounds")
                .iter()
                .map(|&y| (y - col_mean).powi(2))
                .sum::<f64>();
        }

        Ok(mse / n as f64)
    }

    /// Compute physics-informed loss for a training batch.
    ///
    /// Evaluates three constraint terms from [`PhysicsLoss`]:
    ///
    /// 1. **Coherence** — adjacent input channels should have smooth phase
    ///    relationships (continuous wavefront assumption).
    /// 2. **Sparsity** — L1 penalty on target amplitudes encourages sparse
    ///    reconstructions (noise suppression).
    /// 3. **Reciprocity** — first half vs. second half of input features
    ///    (proxy for forward/reverse path symmetry).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_batch_physics_loss(&self, batch: &TrainingDataset) -> KwaversResult<f64> {
        let mut loss = 0.0;

        if self.physics_loss.coherence_weight > 0.0 && !batch.inputs.is_empty() {
            loss += self.physics_loss.coherence_weight
                * PhysicsLoss::coherence_violation(&batch.inputs);
        }

        if self.physics_loss.sparsity_weight > 0.0 && !batch.targets.is_empty() {
            loss +=
                self.physics_loss.sparsity_weight * PhysicsLoss::sparsity_violation(&batch.targets);
        }

        if self.physics_loss.reciprocity_weight > 0.0 {
            let n_cols = batch.inputs.shape()[1];
            if n_cols >= 2 {
                let half = n_cols / 2;
                let forward = batch
                    .inputs
                    .slice_with::<2>(&[
                        SliceArg::All,
                        SliceArg::Range {
                            start: Some(0),
                            end: Some(half as isize),
                            step: 1,
                        },
                    ])
                    .expect("invariant: column range in bounds")
                    .to_contiguous();
                let reverse = batch
                    .inputs
                    .slice_with::<2>(&[
                        SliceArg::All,
                        SliceArg::Range {
                            start: Some(half as isize),
                            end: Some((half * 2) as isize),
                            step: 1,
                        },
                    ])
                    .expect("invariant: column range in bounds")
                    .to_contiguous();
                loss += self.physics_loss.reciprocity_weight
                    * PhysicsLoss::reciprocity_violation(&forward, &reverse);
            }
        }

        Ok(loss)
    }
}