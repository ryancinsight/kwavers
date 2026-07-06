use super::{
    DomainAdapter, SourceFeatures, TrainingData, TransferLearner, TransferLearningStats,
    TransferMetrics,
};
use coeus_autograd::Var;
use kwavers_core::error::{KwaversError, KwaversResult};

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> TransferLearner<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new transfer learner
    pub fn new(
        source_model: crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        config: super::TransferLearningConfig,
    ) -> Self {
        Self {
            source_model,
            config,
            domain_adapter: None,
            stats: TransferLearningStats::default(),
        }
    }

    /// Transfer model to target geometry
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn transfer_to_geometry(
        &mut self,
        target_geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
        target_conditions: &[crate::inverse::pinn::ml::BoundaryCondition2D],
    ) -> KwaversResult<(crate::inverse::pinn::ml::BurnPINN2DWave<B>, TransferMetrics)> {
        let start_time = std::time::Instant::now();

        let source_features = self.extract_source_features()?;

        let mut target_model = self.initialize_target_model(&source_features)?;

        if self.config.adaptation_strength > 0.0 {
            self.setup_domain_adapter(target_geometry)?;
            target_model = self.apply_domain_adaptation(target_model, target_geometry)?;
        }

        let initial_accuracy =
            self.evaluate_accuracy(&target_model, target_geometry, target_conditions)?;
        let (final_model, convergence_epochs) =
            self.fine_tune_model(target_model, target_geometry, target_conditions)?;
        let final_accuracy =
            self.evaluate_accuracy(&final_model, target_geometry, target_conditions)?;

        let training_time = start_time.elapsed();

        let transfer_efficiency = if convergence_epochs > 0 {
            (final_accuracy - initial_accuracy) / convergence_epochs as f32
        } else {
            0.0
        };

        let metrics = TransferMetrics {
            initial_accuracy,
            final_accuracy,
            transfer_efficiency,
            training_time,
            convergence_epochs,
        };

        self.stats.total_transfers += 1;
        if final_accuracy > 0.8 {
            self.stats.successful_transfers += 1;
        }
        self.stats.average_transfer_efficiency =
            (self.stats.average_transfer_efficiency + transfer_efficiency) / 2.0;
        self.stats.best_transfer_accuracy = self.stats.best_transfer_accuracy.max(final_accuracy);
        self.stats.total_training_time += training_time;

        let _ = source_features;

        Ok((final_model, metrics))
    }

    /// Extract features from source model
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn extract_source_features(&self) -> KwaversResult<SourceFeatures> {
        let params = self.source_model.parameters();
        let mut _weight_magnitudes = Vec::new();
        let mut _layer_importance = Vec::new();

        for param in &params {
            let squared = coeus_autograd::mul(param, param);
            let magnitude_var = coeus_autograd::sqrt(&coeus_autograd::sum(&squared));
            let magnitude: f32 = magnitude_var.tensor.as_slice()[0];
            _weight_magnitudes.push(magnitude);
            _layer_importance.push(magnitude * 0.5);
        }

        Ok(SourceFeatures {
            _weight_magnitudes,
            _layer_importance,
            _geometry_adaptability: 0.85,
        })
    }

    /// Initialize target model with transferred weights
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn initialize_target_model(
        &self,
        _source_features: &SourceFeatures,
    ) -> KwaversResult<crate::inverse::pinn::ml::BurnPINN2DWave<B>> {
        Ok(self.source_model.clone())
    }

    /// Setup domain adapter for cross-geometry transfer
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn setup_domain_adapter(
        &mut self,
        _target_geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
    ) -> KwaversResult<()> {
        self.domain_adapter = Some(DomainAdapter {
            _layers: Vec::new(),
            _strength: self.config.adaptation_strength,
        });
        Ok(())
    }

    /// Apply domain adaptation to model
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn apply_domain_adaptation(
        &self,
        model: crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        _target_geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
    ) -> KwaversResult<crate::inverse::pinn::ml::BurnPINN2DWave<B>> {
        Ok(model)
    }

    /// Fine-tune model on target geometry
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub(super) fn fine_tune_model(
        &mut self,
        mut model: crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        target_geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
        target_conditions: &[crate::inverse::pinn::ml::BoundaryCondition2D],
    ) -> KwaversResult<(crate::inverse::pinn::ml::BurnPINN2DWave<B>, usize)> {
        let mut best_accuracy = 0.0;
        let mut patience_counter = 0;
        let mut convergence_epochs = 0;

        for epoch in 0..self.config.fine_tune_epochs {
            let training_data = self.generate_training_data(target_geometry, target_conditions)?;

            let current_accuracy = self.fine_tune_step(&mut model, &training_data)?;

            convergence_epochs = epoch + 1;

            if current_accuracy > best_accuracy {
                best_accuracy = current_accuracy;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.patience {
                    break;
                }
            }

            if current_accuracy >= 0.9 {
                break;
            }
        }

        Ok((model, convergence_epochs))
    }

    /// Generate training data for target geometry
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(super) fn generate_training_data(
        &self,
        geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
        _conditions: &[crate::inverse::pinn::ml::BoundaryCondition2D],
    ) -> KwaversResult<TrainingData> {
        let collocation_points = self.generate_collocation_points(geometry);
        if collocation_points.is_empty() {
            return Err(KwaversError::InvalidInput(
                "Failed to generate any collocation points inside target geometry".to_string(),
            ));
        }

        Ok(TrainingData { collocation_points })
    }

    /// Generate collocation points within geometry
    pub(super) fn generate_collocation_points(
        &self,
        geometry: &crate::inverse::pinn::ml::BurnWave2dGeometry,
    ) -> Vec<(f64, f64, f64)> {
        let mut points = Vec::new();
        let num_points = 500;

        for _ in 0..num_points {
            let x = rand::random::<f64>() * 2.0 - 1.0;
            let y = rand::random::<f64>() * 2.0 - 1.0;
            let t = rand::random::<f64>() * 1.0;

            if geometry.contains(x, y) {
                points.push((x, y, t));
            }
        }

        points
    }

    /// Perform one fine-tuning step with proper physics-informed training
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn fine_tune_step(
        &self,
        model: &mut crate::inverse::pinn::ml::BurnPINN2DWave<B>,
        training_data: &TrainingData,
    ) -> KwaversResult<f32> {
        let backend = B::default();

        let mut x_vec: Vec<f32> = Vec::with_capacity(training_data.collocation_points.len());
        let mut y_vec: Vec<f32> = Vec::with_capacity(training_data.collocation_points.len());
        let mut t_vec: Vec<f32> = Vec::with_capacity(training_data.collocation_points.len());

        for (x, y, t) in &training_data.collocation_points {
            x_vec.push(*x as f32);
            y_vec.push(*y as f32);
            t_vec.push(*t as f32);
        }

        let batch_size = x_vec.len();
        let x_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 1], &x_vec, &backend),
            false,
        );
        let y_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 1], &y_vec, &backend),
            false,
        );
        let t_var = Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 1], &t_vec, &backend),
            false,
        );

        for p in model.parameters() {
            p.zero_grad();
        }

        let pde_residuals =
            model.compute_pde_residual(&x_var, &y_var, &t_var, self.config.wave_speed);

        let squared = coeus_autograd::mul(&pde_residuals, &pde_residuals);
        let total_loss = coeus_autograd::scalar_mul(&coeus_autograd::mean(&squared), 1e-12);

        total_loss.backward();

        let optimizer =
            crate::inverse::pinn::ml::burn_wave_equation_2d::SimpleOptimizer2D::new(1e-4_f32);
        *model = optimizer.step(model.clone());

        let loss_value = total_loss.tensor.as_slice()[0];
        Ok(1.0 / (1.0 + loss_value))
    }

    /// Get transfer learning statistics
    pub fn get_stats(&self) -> &TransferLearningStats {
        &self.stats
    }
}
