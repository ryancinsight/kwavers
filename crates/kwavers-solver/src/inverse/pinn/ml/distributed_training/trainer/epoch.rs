use super::super::{DistributedPinnTrainer, TrainingState};
use crate::inverse::pinn::ml::{TrainingMetrics2D, TrainingMetrics2D as Metrics};
use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> DistributedPinnTrainer<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Single-epoch training step across all model replicas.
    ///
    /// ## Algorithm
    ///
    /// Each replica receives the full point set (data-parallel replication).
    /// For each replica:
    /// 1. Convert (x, y, t) point arrays to leaf `Var`s on the replica backend.
    /// 2. Call `PinnWave2D::compute_physics_loss` (5-tuple output).
    /// 3. Backward pass → per-parameter gradients (accumulated in-place).
    /// 4. `SimpleOptimizer2D::step` updates replica parameters.
    /// 5. Return per-replica `TrainingMetrics2D` and serialised gradients.
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
    pub(super) fn train_epoch_distributed(
        &mut self,
        collocation_points: &[(f64, f64, f64)],
        boundary_points: &[(f64, f64, f64)],
        initial_points: &[(f64, f64, f64)],
        target_values: &[f64],
    ) -> KwaversResult<Vec<(TrainingMetrics2D, Vec<f32>)>> {
        use crate::inverse::pinn::ml::wave_equation_2d::SimpleOptimizer2D;
        use crate::inverse::pinn::ml::LossWeights2D;

        // Build Var helper: &[(f64,f64,f64)] → three [N,1] leaf Vars.
        fn to_xyz_vars<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>(
            pts: &[(f64, f64, f64)],
            backend: &B,
        ) -> (Var<f32, B>, Var<f32, B>, Var<f32, B>) {
            let n = pts.len().max(1);
            let xv: Vec<f32> = pts.iter().map(|p| p.0 as f32).collect();
            let yv: Vec<f32> = pts.iter().map(|p| p.1 as f32).collect();
            let tv: Vec<f32> = pts.iter().map(|p| p.2 as f32).collect();
            let pad = |v: Vec<f32>| {
                if v.is_empty() {
                    vec![0.0_f32]
                } else {
                    v
                }
            };
            let mk = |v: Vec<f32>| {
                Var::new(
                    coeus_tensor::Tensor::from_slice_on(vec![n, 1], &pad(v), backend),
                    false,
                )
            };
            (mk(xv), mk(yv), mk(tv))
        }

        let n_colloc = collocation_points.len().max(1);
        let n_bc = boundary_points.len().max(1);
        let n_ic = initial_points.len().max(1);

        let mut results: Vec<(TrainingMetrics2D, Vec<f32>)> = Vec::new();

        let n_replicas = self.coordinator.model_replicas.len();
        for replica_idx in 0..n_replicas {
            let backend = B::default();

            let (x_colloc, y_colloc, t_colloc) = to_xyz_vars::<B>(collocation_points, &backend);
            let (x_bc, y_bc, t_bc) = to_xyz_vars::<B>(boundary_points, &backend);
            let (x_ic, y_ic, t_ic) = to_xyz_vars::<B>(initial_points, &backend);

            let u_colloc_vals: Vec<f32> = (0..n_colloc)
                .map(|i| target_values.get(i).copied().unwrap_or(0.0) as f32)
                .collect();
            let u_colloc = Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n_colloc, 1], &u_colloc_vals, &backend),
                false,
            );

            let u_bc = Var::new(
                coeus_tensor::Tensor::zeros_on(vec![n_bc, 1], &backend),
                false,
            );
            let u_ic = Var::new(
                coeus_tensor::Tensor::zeros_on(vec![n_ic, 1], &backend),
                false,
            );

            let loss_weights = LossWeights2D::default();
            let wave_speed = kwavers_core::constants::fundamental::SOUND_SPEED_WATER_SIM;

            for p in self.coordinator.model_replicas[replica_idx].parameters() {
                p.zero_grad();
            }

            let (total_loss, data_loss, pde_loss, bc_loss, ic_loss) =
                self.coordinator.model_replicas[replica_idx].compute_physics_loss(
                    &x_colloc,
                    &y_colloc,
                    &t_colloc,
                    &u_colloc,
                    &x_colloc,
                    &y_colloc,
                    &t_colloc,
                    &x_bc,
                    &y_bc,
                    &t_bc,
                    &u_bc,
                    &x_ic,
                    &y_ic,
                    &t_ic,
                    &u_ic,
                    wave_speed,
                    loss_weights,
                );

            let total_val = total_loss.tensor.as_slice()[0] as f64;
            let data_val = data_loss.tensor.as_slice()[0] as f64;
            let pde_val = pde_loss.tensor.as_slice()[0] as f64;
            let bc_val = bc_loss.tensor.as_slice()[0] as f64;
            let ic_val = ic_loss.tensor.as_slice()[0] as f64;

            if total_val.is_finite() {
                total_loss.backward();
                let optimizer = SimpleOptimizer2D::new(1e-3_f32);
                self.coordinator.model_replicas[replica_idx] =
                    optimizer.step(self.coordinator.model_replicas[replica_idx].clone());
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

    pub(super) fn aggregate_gradients_and_update(
        &mut self,
        gpu_results: &[(TrainingMetrics2D, Vec<f32>)],
    ) -> KwaversResult<()> {
        let n_gpus = gpu_results.len();

        let mut total_loss = 0.0_f64;
        let mut data_loss = 0.0_f64;
        let mut pde_loss = 0.0_f64;
        let mut bc_loss = 0.0_f64;
        let mut ic_loss = 0.0_f64;

        for (metrics, _) in gpu_results {
            if let Some(tl) = metrics.total_loss.last() {
                total_loss += tl;
            }
            if let Some(dl) = metrics.data_loss.last() {
                data_loss += dl;
            }
            if let Some(pl) = metrics.pde_loss.last() {
                pde_loss += pl;
            }
            if let Some(bl) = metrics.bc_loss.last() {
                bc_loss += bl;
            }
            if let Some(il) = metrics.ic_loss.last() {
                ic_loss += il;
            }
        }

        let n = n_gpus as f64;
        let g = &mut self.coordinator.training_state.global_metrics;
        g.total_loss.push(total_loss / n);
        g.data_loss.push(data_loss / n);
        g.pde_loss.push(pde_loss / n);
        g.bc_loss.push(bc_loss / n);
        g.ic_loss.push(ic_loss / n);

        Ok(())
    }
}

// Suppress unused-import for TrainingState referenced only by the parent impl.
const _: fn() = || {
    let _: TrainingState;
};
