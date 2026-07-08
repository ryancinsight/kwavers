use super::MetaLearner;
use crate::inverse::pinn::ml::meta_learning::metrics::MetaLoss;
use crate::inverse::pinn::ml::meta_learning::types::{PhysicsTask, TaskData};
use crate::inverse::pinn::ml::wave_equation_2d::{LossWeights2D, PinnWave2D};
use coeus_autograd::Var;
use kwavers_core::error::KwaversResult;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> MetaLearner<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Compute the per-parameter gradient snapshot and loss.
    ///
    /// Returns one flat `Vec<f32>` gradient per entry of `model.parameters()`
    /// (in that order), or `None` where a parameter has no gradient (e.g. it
    /// did not participate in the loss computation).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub(super) fn compute_gradients_and_loss(
        &self,
        model: &PinnWave2D<B>,
        data: &TaskData,
        task: &PhysicsTask,
    ) -> KwaversResult<(MetaLoss, Vec<Option<Vec<f32>>>)> {
        let backend = B::default();

        let (x_colloc, y_colloc, t_colloc) = self.to_vars_3(&data.collocation_points, &backend);
        let (x_bc, y_bc, t_bc, u_bc) = self.to_vars_4(&data.boundary_data, &backend);
        let (x_ic, y_ic, t_ic, u_ic) = self.to_vars_5_to_4(&data.initial_data, &backend);

        for p in model.parameters() {
            p.zero_grad();
        }

        let (total_loss, _data_loss, pde_loss, _bc_loss, _ic_loss) = model.compute_physics_loss(
            &x_bc,
            &y_bc,
            &t_bc,
            &u_bc,
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
            task.physics_params.wave_speed,
            LossWeights2D::default(),
        );

        total_loss.backward();

        let grads: Vec<Option<Vec<f32>>> = model
            .parameters()
            .iter()
            .map(|p| p.grad().map(|g| g.as_slice().to_vec()))
            .collect();

        let meta_loss = MetaLoss {
            total_loss: total_loss.tensor.as_slice()[0] as f64,
            task_losses: vec![],
            physics_loss: pde_loss.tensor.as_slice()[0] as f64,
            generalization_score: 0.0,
        };

        Ok((meta_loss, grads))
    }

    pub(super) fn to_vars_3(
        &self,
        data: &[(f64, f64, f64)],
        backend: &B,
    ) -> (Var<f32, B>, Var<f32, B>, Var<f32, B>) {
        let n = data.len().max(1);
        if data.is_empty() {
            let dummy = || Var::new(coeus_tensor::Tensor::zeros_on(vec![1, 1], backend), false);
            return (dummy(), dummy(), dummy());
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();

        (
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &x, backend),
                false,
            ),
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &y, backend),
                false,
            ),
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &t, backend),
                false,
            ),
        )
    }

    #[allow(clippy::type_complexity)] // (x, y, t, u) tensors, no cohesive grouping
    pub(super) fn to_vars_4(
        &self,
        data: &[(f64, f64, f64, f64)],
        backend: &B,
    ) -> (Var<f32, B>, Var<f32, B>, Var<f32, B>, Var<f32, B>) {
        let n = data.len().max(1);
        if data.is_empty() {
            let dummy = || Var::new(coeus_tensor::Tensor::zeros_on(vec![1, 1], backend), false);
            return (dummy(), dummy(), dummy(), dummy());
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();
        let u: Vec<f32> = data.iter().map(|p| p.3 as f32).collect();

        (
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &x, backend),
                false,
            ),
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &y, backend),
                false,
            ),
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &t, backend),
                false,
            ),
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &u, backend),
                false,
            ),
        )
    }

    #[allow(clippy::type_complexity)] // (x, y, t, u) tensors, no cohesive grouping
    pub(super) fn to_vars_5_to_4(
        &self,
        data: &[(f64, f64, f64, f64, f64)],
        backend: &B,
    ) -> (Var<f32, B>, Var<f32, B>, Var<f32, B>, Var<f32, B>) {
        let n = data.len().max(1);
        if data.is_empty() {
            let dummy = || Var::new(coeus_tensor::Tensor::zeros_on(vec![1, 1], backend), false);
            return (dummy(), dummy(), dummy(), dummy());
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();
        let u: Vec<f32> = data.iter().map(|p| p.3 as f32).collect();

        (
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &x, backend),
                false,
            ),
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &y, backend),
                false,
            ),
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &t, backend),
                false,
            ),
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n, 1], &u, backend),
                false,
            ),
        )
    }
}
