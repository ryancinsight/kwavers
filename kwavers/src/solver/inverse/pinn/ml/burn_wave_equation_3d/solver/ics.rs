use super::core::BurnPINN3DWave;
use crate::core::error::{KwaversError, KwaversResult};
use burn::tensor::{backend::Backend, Tensor, TensorData};

impl<B: Backend> BurnPINN3DWave<B> {
    /// Compute temporal derivative ∂u/∂t at t=0 via forward finite difference
    ///
    /// Uses forward difference: ∂u/∂t(0) ≈ (u(ε) - u(0)) / ε
    ///
    /// # Arguments
    ///
    /// * `x` - X-coordinates at t=0
    /// * `y` - Y-coordinates at t=0
    /// * `z` - Z-coordinates at t=0
    /// * `t` - Time coordinates (should be t=0)
    ///
    /// # Returns
    ///
    /// Tensor containing ∂u/∂t values at the specified points
    pub(crate) fn compute_temporal_derivative_at_t0(
        &self,
        x: Tensor<B, 2>,
        y: Tensor<B, 2>,
        z: Tensor<B, 2>,
        t: Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let eps = 1e-3_f32;

        // u(t=0)
        let u_t0 = self
            .pinn
            .forward(x.clone(), y.clone(), z.clone(), t.clone());

        // u(t=ε)
        let t_eps = t.add_scalar(eps);
        let u_t_eps = self.pinn.forward(x, y, z, t_eps);

        // Forward difference: ∂u/∂t ≈ (u(ε) - u(0)) / ε
        u_t_eps.sub(u_t0).div_scalar(eps)
    }

    /// Extract velocity initial condition tensor from training data
    ///
    /// Finds all points at t=0 and extracts their velocity values.
    ///
    /// # Arguments
    ///
    /// * `x_data` - X-coordinates of training data
    /// * `y_data` - Y-coordinates of training data
    /// * `z_data` - Z-coordinates of training data
    /// * `t_data` - Time coordinates of training data
    /// * `v_data` - Velocity values (∂u/∂t)
    /// * `device` - Target device
    ///
    /// # Returns
    ///
    /// Tensor [n_ic, 1] containing velocity IC values
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(crate) fn extract_velocity_initial_condition_tensor(
        _x_data: &[f32],
        _y_data: &[f32],
        _z_data: &[f32],
        t_data: &[f32],
        v_data: &[f32],
        device: &B::Device,
    ) -> KwaversResult<Tensor<B, 2>> {
        if v_data.len() != t_data.len() {
            return Err(KwaversError::InvalidInput(
                "v_data and t_data must have equal length".into(),
            ));
        }

        let min_t = t_data.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
        if !min_t.is_finite() {
            return Err(KwaversError::InvalidInput(
                "Training time coordinates must be finite".into(),
            ));
        }

        let eps = 1e-6_f32;
        let mut v_ic = Vec::new();

        for i in 0..t_data.len() {
            if (t_data[i] - min_t).abs() <= eps {
                v_ic.push(v_data[i]);
            }
        }

        if v_ic.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No initial-condition velocity samples found in training data".into(),
            ));
        }

        let n_ic = v_ic.len();
        Ok(Tensor::<B, 2>::from_data(
            TensorData::new(v_ic, [n_ic, 1]),
            device,
        ))
    }
    /// Extract initial condition tensors.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub(crate) fn extract_initial_condition_tensors(
        x_data: &[f32],
        y_data: &[f32],
        z_data: &[f32],
        t_data: &[f32],
        u_data: &[f32],
        device: &B::Device,
    ) -> KwaversResult<(
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
        Tensor<B, 2>,
    )> {
        let min_t = t_data.iter().copied().fold(f32::INFINITY, |a, b| a.min(b));
        if !min_t.is_finite() {
            return Err(KwaversError::InvalidInput(
                "Training time coordinates must be finite".into(),
            ));
        }

        let eps = 1e-6_f32;
        let mut x_ic = Vec::new();
        let mut y_ic = Vec::new();
        let mut z_ic = Vec::new();
        let mut u_ic = Vec::new();

        for i in 0..t_data.len() {
            if (t_data[i] - min_t).abs() <= eps {
                x_ic.push(x_data[i]);
                y_ic.push(y_data[i]);
                z_ic.push(z_data[i]);
                u_ic.push(u_data[i]);
            }
        }

        if x_ic.is_empty() {
            return Err(KwaversError::InvalidInput(
                "No initial-condition samples found in training data".into(),
            ));
        }

        let n_ic = x_ic.len();
        let t_ic = vec![min_t; n_ic];

        Ok((
            Tensor::<B, 2>::from_data(TensorData::new(x_ic, [n_ic, 1]), device),
            Tensor::<B, 2>::from_data(TensorData::new(y_ic, [n_ic, 1]), device),
            Tensor::<B, 2>::from_data(TensorData::new(z_ic, [n_ic, 1]), device),
            Tensor::<B, 2>::from_data(TensorData::new(t_ic, [n_ic, 1]), device),
            Tensor::<B, 2>::from_data(TensorData::new(u_ic, [n_ic, 1]), device),
        ))
    }
}
