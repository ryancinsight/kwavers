use super::super::{DistributedPinnTrainer, TrainingState};
use crate::core::error::KwaversResult;
use crate::solver::inverse::pinn::ml::{BurnTrainingMetrics2D, BurnTrainingMetrics2D as Metrics};
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;

impl<B: AutodiffBackend> DistributedPinnTrainer<B> {
    /// Single-epoch training step across all model replicas.
    ///
    /// ## Algorithm
    ///
    /// Each replica receives the full point set (data-parallel replication).
    /// For each replica:
    /// 1. Convert (x, y, t) point arrays to Burn tensors on the replica device.
    /// 2. Call `BurnPINN2DWave::compute_physics_loss` (5-tuple output).
    /// 3. Backward pass → gradient tensors.
    /// 4. `SimpleOptimizer2D::step` updates replica parameters.
    /// 5. Return per-replica `BurnTrainingMetrics2D` and serialised gradients.
    ///
    /// True multi-GPU collective gradient aggregation (NCCL/MPI) is not yet
    /// available; `aggregate_gradients_and_update` averages loss scalars from
    /// each replica instead.  The gradient vectors returned are empty because
    /// no cross-device parameter synchronisation occurs in this fallback.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub(super) async fn train_epoch_distributed(
        &mut self,
        collocation_points: &[(f64, f64, f64)],
        boundary_points: &[(f64, f64, f64)],
        initial_points: &[(f64, f64, f64)],
        target_values: &[f64],
    ) -> KwaversResult<Vec<(BurnTrainingMetrics2D, Vec<f32>)>> {
        use crate::solver::inverse::pinn::ml::BurnLossWeights2D;
        use crate::solver::inverse::pinn::ml::burn_wave_equation_2d::SimpleOptimizer2D;

        // Build tensor helper: &[(f64,f64,f64)] → three [N,1] tensors.
        fn to_xyz_tensors<B: AutodiffBackend>(
            pts: &[(f64, f64, f64)],
            device: &B::Device,
        ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
            let n = pts.len().max(1);
            let xv: Vec<f32> = pts.iter().map(|p| p.0 as f32).collect();
            let yv: Vec<f32> = pts.iter().map(|p| p.1 as f32).collect();
            let tv: Vec<f32> = pts.iter().map(|p| p.2 as f32).collect();
            let pad = |v: Vec<f32>| {
                if v.is_empty() { vec![0.0_f32] } else { v }
            };
            let x = Tensor::<B, 1>::from_floats(pad(xv).as_slice(), device).reshape([n, 1]);
            let y = Tensor::<B, 1>::from_floats(pad(yv).as_slice(), device).reshape([n, 1]);
            let t = Tensor::<B, 1>::from_floats(pad(tv).as_slice(), device).reshape([n, 1]);
            (x, y, t)
        }

        let n_colloc = collocation_points.len().max(1);
        let n_bc = boundary_points.len().max(1);
        let n_ic = initial_points.len().max(1);

        let mut results: Vec<(BurnTrainingMetrics2D, Vec<f32>)> = Vec::new();

        let n_replicas = self.coordinator.model_replicas.len();
        for replica_idx in 0..n_replicas {
            let device = B::Device::default();

            let (x_colloc, y_colloc, t_colloc) =
                to_xyz_tensors::<B>(collocation_points, &device);
            let (x_bc, y_bc, t_bc) = to_xyz_tensors::<B>(boundary_points, &device);
            let (x_ic, y_ic, t_ic) = to_xyz_tensors::<B>(initial_points, &device);

            let u_colloc_vals: Vec<f32> = (0..n_colloc)
                .map(|i| target_values.get(i).copied().unwrap_or(0.0) as f32)
                .collect();
            let u_colloc = Tensor::<B, 1>::from_floats(u_colloc_vals.as_slice(), &device)
                .reshape([n_colloc, 1]);

            let u_bc = Tensor::<B, 2>::zeros([n_bc, 1], &device);
            let u_ic = Tensor::<B, 2>::zeros([n_ic, 1], &device);

            let loss_weights = BurnLossWeights2D::default();
            let wave_speed = 1500.0_f64;

            let (total_loss, data_loss, pde_loss, bc_loss, ic_loss) =
                self.coordinator.model_replicas[replica_idx].compute_physics_loss(
                    x_colloc.clone(),
                    y_colloc.clone(),
                    t_colloc.clone(),
                    u_colloc,
                    x_colloc,
                    y_colloc,
                    t_colloc,
                    x_bc.clone(),
                    y_bc.clone(),
                    t_bc.clone(),
                    u_bc,
                    x_ic.clone(),
                    y_ic.clone(),
                    t_ic.clone(),
                    u_ic,
                    wave_speed,
                    loss_weights,
                );

            let total_val =
                total_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let data_val =
                data_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let pde_val =
                pde_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let bc_val =
                bc_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;
            let ic_val =
                ic_loss.clone().into_data().as_slice::<f32>().unwrap()[0] as f64;

            if total_val.is_finite() {
                let grads = total_loss.backward();
                let optimizer = SimpleOptimizer2D::new(1e-3_f32);
                self.coordinator.model_replicas[replica_idx] =
                    optimizer.step(self.coordinator.model_replicas[replica_idx].clone(), &grads);
            }

            results.push((
                Metrics {
                    total_loss: vec![total_val],
                    data_loss: vec![data_val],
                    pde_loss: vec![pde_val],
                    bc_loss: vec![bc_val],
                    ic_loss: vec![ic_val],
                    training_time_secs: 0.0,
                    epochs_completed: 1,
                },
                Vec::new(),
            ));
        }

        Ok(results)
    }

    pub(super) async fn aggregate_gradients_and_update(
        &mut self,
        gpu_results: &[(BurnTrainingMetrics2D, Vec<f32>)],
    ) -> KwaversResult<()> {
        let n_gpus = gpu_results.len();

        let mut total_loss = 0.0_f64;
        let mut data_loss = 0.0_f64;
        let mut pde_loss = 0.0_f64;
        let mut bc_loss = 0.0_f64;
        let mut ic_loss = 0.0_f64;

        for (metrics, _) in gpu_results {
            if let Some(tl) = metrics.total_loss.last() { total_loss += tl; }
            if let Some(dl) = metrics.data_loss.last()  { data_loss  += dl; }
            if let Some(pl) = metrics.pde_loss.last()   { pde_loss   += pl; }
            if let Some(bl) = metrics.bc_loss.last()    { bc_loss    += bl; }
            if let Some(il) = metrics.ic_loss.last()    { ic_loss    += il; }
        }

        let n = n_gpus as f64;
        let g = &mut self.coordinator.training_state.global_metrics;
        g.total_loss.push(total_loss / n);
        g.data_loss.push(data_loss  / n);
        g.pde_loss.push(pde_loss   / n);
        g.bc_loss.push(bc_loss    / n);
        g.ic_loss.push(ic_loss    / n);

        Ok(())
    }
}

// Suppress unused-import for TrainingState referenced only by the parent impl.
const _: fn() = || { let _: TrainingState; };
