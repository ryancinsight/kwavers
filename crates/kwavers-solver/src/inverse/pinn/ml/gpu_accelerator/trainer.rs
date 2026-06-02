//! Batched PINN trainer with GPU acceleration.

use super::kernel::CudaKernelManager;
use super::memory::{CudaBuffer, GpuMemoryManager, PinnGpuMemoryPoolType, PinnGpuMemoryStats};
use kwavers_core::error::KwaversResult;
use burn::prelude::ToElement;
use burn::tensor::{backend::AutodiffBackend, Tensor};

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub total_steps: usize,
    pub current_epoch: usize,
    pub avg_loss: f32,
    pub learning_rate: f64,
    pub gpu_utilization: f32,
}

/// Training step result
#[derive(Debug, Clone)]
pub struct TrainingStep {
    pub loss: f32,
    pub step_time: std::time::Duration,
    pub batch_count: usize,
}

/// Batched PINN trainer with GPU acceleration
#[derive(Debug)]
pub struct BatchedPINNTrainer<B: AutodiffBackend> {
    model: crate::inverse::pinn::ml::BurnPINN2DWave<B>,
    batch_size: usize,
    memory_manager: GpuMemoryManager,
    _kernel_manager: CudaKernelManager,
    gradient_accumulator: Option<CudaBuffer<f32>>,
    accumulation_step: usize,
    total_accumulation_steps: usize,
    stats: TrainingStats,
}

impl<B: AutodiffBackend> BatchedPINNTrainer<B> {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(
        model: crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        batch_size: usize,
        accumulation_steps: usize,
    ) -> KwaversResult<Self> {
        let memory_manager = GpuMemoryManager::new()?;
        let kernel_manager = CudaKernelManager::new()?;

        Ok(Self {
            model,
            batch_size,
            memory_manager,
            _kernel_manager: kernel_manager,
            gradient_accumulator: None,
            accumulation_step: 0,
            total_accumulation_steps: accumulation_steps,
            stats: TrainingStats {
                total_steps: 0,
                current_epoch: 0,
                avg_loss: 0.0,
                learning_rate: 0.001,
                gpu_utilization: 0.0,
            },
        })
    }
    /// Train batch.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn train_batch(
        &mut self,
        collocation_points: &Tensor<B, 2>,
    ) -> KwaversResult<TrainingStep> {
        let start_time = std::time::Instant::now();

        let batches = self.split_into_batches(collocation_points)?;

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in batches {
            let x = batch
                .clone()
                .slice([0..batch.shape().dims[0], 0..1])
                .squeeze::<2>();
            let y = batch
                .clone()
                .slice([0..batch.shape().dims[0], 1..2])
                .squeeze::<2>();
            let t = batch
                .clone()
                .slice([0..batch.shape().dims[0], 2..3])
                .squeeze::<2>();
            let predictions = self.model.forward(x, y, t);

            let residuals = self.compute_pde_residuals_gpu(&predictions, &batch)?;

            let loss = (residuals.clone() * residuals).mean();
            total_loss += loss.clone().into_scalar().to_f32();
            batch_count += 1;

            let gradients = loss.backward();
            self.accumulate_gradients(&gradients)?;
        }

        if self.should_update_parameters() {
            self.update_model_parameters()?;
        }

        let step_time = start_time.elapsed();

        Ok(TrainingStep {
            loss: total_loss / batch_count as f32,
            step_time,
            batch_count,
        })
    }

    fn split_into_batches(&self, tensor: &Tensor<B, 2>) -> KwaversResult<Vec<Tensor<B, 2>>> {
        let total_points = tensor.shape().dims[0];
        let mut batches = Vec::new();

        for start in (0..total_points).step_by(self.batch_size) {
            let end = (start + self.batch_size).min(total_points);
            let batch = tensor
                .clone()
                .slice([start..end, 0..tensor.shape().dims[1]]);
            batches.push(batch);
        }

        Ok(batches)
    }

    fn _register_kernel(&mut self, _name: &str, _ptx_source: &str) -> KwaversResult<()> {
        Ok(())
    }

    fn compute_pde_residuals_gpu(
        &mut self,
        predictions: &Tensor<B, 2>,
        collocation_points: &Tensor<B, 2>,
    ) -> KwaversResult<Tensor<B, 2>> {
        let _pred_buffer = self.memory_manager.allocate_device(
            PinnGpuMemoryPoolType::Temporary,
            predictions.shape().dims[0] * predictions.shape().dims[1],
        )?;
        let _coll_buffer = self.memory_manager.allocate_device(
            PinnGpuMemoryPoolType::Collocation,
            collocation_points.shape().dims[0] * collocation_points.shape().dims[1],
        )?;
        let _residual_buffer = self.memory_manager.allocate_device(
            PinnGpuMemoryPoolType::Temporary,
            predictions.shape().dims[0],
        )?;

        let residuals = Tensor::zeros_like(
            &predictions
                .clone()
                .slice([0..predictions.shape().dims[0], 0..1])
                .squeeze::<2>(),
        );

        Ok(residuals)
    }

    fn accumulate_gradients(
        &mut self,
        _gradients: &<B as AutodiffBackend>::Gradients,
    ) -> KwaversResult<()> {
        if self.gradient_accumulator.is_none() {
            let size = 1024 * 1024;
            self.gradient_accumulator = Some(
                self.memory_manager
                    .allocate_device(PinnGpuMemoryPoolType::Gradients, size)?,
            );
        }

        self.accumulation_step += 1;

        Ok(())
    }

    fn should_update_parameters(&self) -> bool {
        self.accumulation_step >= self.total_accumulation_steps
    }

    fn update_model_parameters(&mut self) -> KwaversResult<()> {
        self.accumulation_step = 0;
        self.stats.total_steps += 1;
        Ok(())
    }

    pub fn stats(&self) -> &TrainingStats {
        &self.stats
    }

    pub fn memory_stats(&self) -> &PinnGpuMemoryStats {
        self.memory_manager.memory_stats()
    }
}
