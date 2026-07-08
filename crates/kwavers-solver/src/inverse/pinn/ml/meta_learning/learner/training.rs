use super::MetaLearner;
use crate::inverse::pinn::ml::meta_learning::gradient::utils::add_gradients;
use crate::inverse::pinn::ml::meta_learning::metrics::MetaLoss;
use crate::inverse::pinn::ml::meta_learning::types::PhysicsTask;
use crate::inverse::pinn::ml::wave_equation_2d::SimpleOptimizer2D;
use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> MetaLearner<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
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
        let num_params = self.base_model.parameters().len();
        let mut accumulated_grads: Vec<Option<Vec<f32>>> = vec![None; num_params];

        for task in &tasks {
            let (task_loss, physics_loss, adapted_model) = self.adapt_to_task_internal(task)?;

            total_loss += task_loss;
            task_losses.push(task_loss);
            physics_losses.push(physics_loss);

            let valid_data = self.generate_task_data(task)?;
            let (_val_loss, val_grads) =
                self.compute_gradients_and_loss(&adapted_model, &valid_data, task)?;

            accumulated_grads = add_gradients(accumulated_grads, &val_grads);
        }

        let num_tasks = tasks.len().max(1) as f32;
        let averaged_grads: Vec<Option<Vec<f32>>> = accumulated_grads
            .into_iter()
            .map(|opt| opt.map(|g| g.into_iter().map(|v| v / num_tasks).collect()))
            .collect();

        let updated_params: Vec<Var<f32, B>> = self
            .base_model
            .parameters()
            .into_iter()
            .zip(averaged_grads.iter())
            .map(|(param, grad)| match grad {
                Some(g) => {
                    let backend = B::default();
                    let old = param.tensor.as_slice();
                    let updated: Vec<f32> = old
                        .iter()
                        .zip(g.iter())
                        .map(|(&p, &gr)| p - self.config.outer_lr as f32 * gr)
                        .collect();
                    Var::new(
                        coeus_tensor::Tensor::from_slice_on(
                            param.tensor.shape().to_vec(),
                            &updated,
                            &backend,
                        ),
                        param.grad.is_some(),
                    )
                }
                None => param,
            })
            .collect();
        self.base_model.load_parameters(&updated_params);

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
        crate::inverse::pinn::ml::wave_equation_2d::PinnWave2D<B>,
    )> {
        let mut task_model = self.base_model.clone();
        let task_data = self.generate_task_data(task)?;
        let optimizer = SimpleOptimizer2D::new(self.config.inner_lr as f32);

        let mut task_loss = 0.0;
        let mut physics_loss = 0.0;

        for _ in 0..self.config.adaptation_steps {
            let (loss, _grads) = self.compute_gradients_and_loss(&task_model, &task_data, task)?;
            task_model = optimizer.step(task_model);
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
}
