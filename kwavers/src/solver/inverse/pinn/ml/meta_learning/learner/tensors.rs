use super::MetaLearner;
use crate::core::error::KwaversResult;
use crate::solver::inverse::pinn::ml::burn_wave_equation_2d::{BurnLossWeights2D, BurnPINN2DWave};
use crate::solver::inverse::pinn::ml::meta_learning::metrics::MetaLoss;
use crate::solver::inverse::pinn::ml::meta_learning::types::{PhysicsTask, TaskData};
use burn::prelude::ToElement;
use burn::tensor::{backend::AutodiffBackend, Tensor};

impl<B: AutodiffBackend> MetaLearner<B> {
    pub(super) fn compute_gradients_and_loss(
        &self,
        model: &BurnPINN2DWave<B>,
        data: &TaskData,
        task: &PhysicsTask,
    ) -> KwaversResult<(MetaLoss, B::Gradients)> {
        let device = model.device();

        let (x_colloc, y_colloc, t_colloc) = self.to_tensors_3(&data.collocation_points, &device);
        let (x_bc, y_bc, t_bc, u_bc) = self.to_tensors_4(&data.boundary_data, &device);
        let (x_ic, y_ic, t_ic, u_ic) = self.to_tensors_5_to_4(&data.initial_data, &device);

        let x_data = x_bc.clone();
        let y_data = y_bc.clone();
        let t_data = t_bc.clone();
        let u_data = u_bc.clone();

        let (total_loss, _data_loss, pde_loss, _bc_loss, _ic_loss) = model.compute_physics_loss(
            x_data,
            y_data,
            t_data,
            u_data,
            x_colloc,
            y_colloc,
            t_colloc,
            x_bc,
            y_bc,
            t_bc,
            u_bc,
            x_ic,
            y_ic,
            t_ic,
            u_ic,
            task.physics_params.wave_speed,
            BurnLossWeights2D::default(),
        );

        let grads = total_loss.backward();

        let meta_loss = MetaLoss {
            total_loss: total_loss.into_scalar().to_f64(),
            task_losses: vec![],
            physics_loss: pde_loss.into_scalar().to_f64(),
            generalization_score: 0.0,
        };

        Ok((meta_loss, grads))
    }

    pub(super) fn to_tensors_3(
        &self,
        data: &[(f64, f64, f64)],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let n = data.len();
        if n == 0 {
            let dummy = Tensor::zeros([1, 1], device);
            return (dummy.clone(), dummy.clone(), dummy);
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();

        (
            Tensor::<B, 1>::from_floats(x.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(y.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(t.as_slice(), device).reshape([n, 1]),
        )
    }

    pub(super) fn to_tensors_4(
        &self,
        data: &[(f64, f64, f64, f64)],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let n = data.len();
        if n == 0 {
            let dummy = Tensor::zeros([1, 1], device);
            return (dummy.clone(), dummy.clone(), dummy.clone(), dummy);
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();
        let u: Vec<f32> = data.iter().map(|p| p.3 as f32).collect();

        (
            Tensor::<B, 1>::from_floats(x.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(y.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(t.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(u.as_slice(), device).reshape([n, 1]),
        )
    }

    pub(super) fn to_tensors_5_to_4(
        &self,
        data: &[(f64, f64, f64, f64, f64)],
        device: &B::Device,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 2>) {
        let n = data.len();
        if n == 0 {
            let dummy = Tensor::zeros([1, 1], device);
            return (dummy.clone(), dummy.clone(), dummy.clone(), dummy);
        }

        let x: Vec<f32> = data.iter().map(|p| p.0 as f32).collect();
        let y: Vec<f32> = data.iter().map(|p| p.1 as f32).collect();
        let t: Vec<f32> = data.iter().map(|p| p.2 as f32).collect();
        let u: Vec<f32> = data.iter().map(|p| p.3 as f32).collect();

        (
            Tensor::<B, 1>::from_floats(x.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(y.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(t.as_slice(), device).reshape([n, 1]),
            Tensor::<B, 1>::from_floats(u.as_slice(), device).reshape([n, 1]),
        )
    }
}
