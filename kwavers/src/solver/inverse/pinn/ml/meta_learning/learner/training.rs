use super::MetaLearner;
use crate::core::error::KwaversResult;
use crate::solver::inverse::pinn::ml::burn_wave_equation_2d::SimpleOptimizer2D;
use crate::solver::inverse::pinn::ml::meta_learning::gradient::{
    GradientApplicator, GradientExtractor,
};
use crate::solver::inverse::pinn::ml::meta_learning::metrics::MetaLoss;
use crate::solver::inverse::pinn::ml::meta_learning::types::PhysicsTask;
use burn::module::Module;
use burn::tensor::{backend::AutodiffBackend, Tensor};

impl<B: AutodiffBackend> MetaLearner<B> {
    /// Perform one meta-training step
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn meta_train_step(&mut self) -> KwaversResult<MetaLoss> {
        let tasks = self
            .task_sampler
            .sample_batch(self.config.meta_batch_size)?;

        let mut total_loss = 0.0;
        let mut task_losses = Vec::new();
        let mut physics_losses = Vec::new();
        let mut aggregated_grads: Vec<B::Gradients> = Vec::new();

        for task in &tasks {
            let (task_loss, physics_loss, adapted_model) = self.adapt_to_task_internal(task)?;

            total_loss += task_loss;
            task_losses.push(task_loss);
            physics_losses.push(physics_loss);

            let valid_data = self.generate_task_data(task)?;
            let (_val_loss, val_grads) =
                self.compute_gradients_and_loss(&adapted_model, &valid_data, task)?;

            aggregated_grads.push(val_grads);
        }

        let num_tasks = tasks.len();
        let mut accumulated_grads: Vec<Option<Tensor<B::InnerBackend, 1>>> = Vec::new();

        for (i, task_grad_set) in aggregated_grads.iter().enumerate() {
            let mut extractor = GradientExtractor::new(task_grad_set);
            self.base_model.clone().map(&mut extractor);
            let task_grads_flat = extractor.into_gradients();

            if i == 0 {
                accumulated_grads = task_grads_flat;
            } else {
                for (acc, new) in accumulated_grads.iter_mut().zip(task_grads_flat.iter()) {
                    if let (Some(a), Some(b)) = (acc.as_mut(), new.as_ref()) {
                        *a = a.clone() + b.clone();
                    } else if acc.is_none() && new.is_some() {
                        *acc = new.clone();
                    }
                }
            }
        }

        let averaged_grads: Vec<Option<Tensor<B::InnerBackend, 1>>> = accumulated_grads
            .into_iter()
            .map(|opt| opt.map(|g| g.div_scalar(num_tasks as f64)))
            .collect();

        let mut applicator = GradientApplicator::new(averaged_grads, self.config.outer_lr);
        self.base_model = self.base_model.clone().map(&mut applicator);

        let average_physics_loss = if !physics_losses.is_empty() {
            physics_losses.iter().sum::<f64>() / physics_losses.len() as f64
        } else {
            0.0
        };

        let generalization_score = self.compute_generalization_score(&task_losses);

        let meta_loss = MetaLoss {
            total_loss: if !tasks.is_empty() {
                total_loss / tasks.len() as f64
            } else {
                0.0
            },
            task_losses,
            physics_loss: average_physics_loss,
            generalization_score,
        };

        self.stats.meta_epochs_completed += 1;
        self.stats.total_tasks_processed += tasks.len();
        self.stats.average_meta_loss = (self.stats.average_meta_loss + meta_loss.total_loss) / 2.0;
        self.stats.best_generalization_score = self
            .stats
            .best_generalization_score
            .max(generalization_score);

        Ok(meta_loss)
    }

    /// Adapt to a specific physics task using learned meta-parameters
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn adapt_to_task(&self, task: &PhysicsTask) -> KwaversResult<(f64, f64)> {
        let (loss, physics_loss, _) = self.adapt_to_task_internal(task)?;
        Ok((loss, physics_loss))
    }

    /// Internal adaptation method returning the adapted model
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn adapt_to_task_internal(
        &self,
        task: &PhysicsTask,
    ) -> KwaversResult<(
        f64,
        f64,
        crate::solver::inverse::pinn::ml::burn_wave_equation_2d::BurnPINN2DWave<B>,
    )> {
        let mut task_model = self.base_model.clone();
        let task_data = self.generate_task_data(task)?;
        let optimizer = SimpleOptimizer2D::new(self.config.inner_lr as f32);

        let mut task_loss = 0.0;
        let mut physics_loss = 0.0;

        for _ in 0..self.config.adaptation_steps {
            let (loss, grads) = self.compute_gradients_and_loss(&task_model, &task_data, task)?;
            task_model = optimizer.step(task_model, &grads);
            task_loss = loss.total_loss;
            physics_loss = loss.physics_loss;
        }

        Ok((task_loss, physics_loss, task_model))
    }

    /// Compute generalization score across tasks
    pub(super) fn compute_generalization_score(&self, task_losses: &[f64]) -> f64 {
        let mean = task_losses.iter().sum::<f64>() / task_losses.len() as f64;
        let variance = task_losses
            .iter()
            .map(|loss| (loss - mean).powi(2))
            .sum::<f64>()
            / task_losses.len() as f64;
        1.0 / (1.0 + variance.sqrt())
    }

    /// Compute meta-gradients using MAML-style second-order derivatives
    pub(super) fn _compute_meta_gradients(
        &self,
        task_gradients: &[Vec<Tensor<B, 2>>],
        task_losses: &[f64],
    ) -> Vec<Option<Tensor<B, 2>>> {
        let num_params = self.base_model.parameters().len();
        if task_gradients.is_empty() || task_losses.is_empty() {
            return vec![None; num_params];
        }

        let num_tasks = task_gradients.len();
        let mut meta_gradients = Vec::with_capacity(num_params);

        for param_idx in 0..num_params {
            let mut param_meta_grad: Option<Tensor<B, 2>> = None;

            for task_idx in 0..num_tasks {
                if param_idx >= task_gradients[task_idx].len() {
                    continue;
                }

                let task_grad = &task_gradients[task_idx][param_idx];
                let task_loss = task_losses[task_idx];

                let weighted_grad = task_grad.clone().mul_scalar(task_loss as f32);

                if let Some(acc) = param_meta_grad.as_mut() {
                    *acc = acc.clone().add(weighted_grad);
                } else {
                    param_meta_grad = Some(weighted_grad);
                }
            }

            if let Some(grad) = param_meta_grad {
                meta_gradients.push(Some(grad.div_scalar(num_tasks as f32)));
            } else {
                meta_gradients.push(None);
            }
        }

        meta_gradients
    }
}
