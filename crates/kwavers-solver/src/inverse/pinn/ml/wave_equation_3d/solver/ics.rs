use super::core::PinnWave3D;
use coeus_autograd::Var;
use kwavers_core::error::{KwaversError, KwaversResult};

/// Initial-condition coordinate/value tensors `(x, y, z, t, u)`, each `[N, 1]`.
#[allow(clippy::type_complexity)] // 5 independent IC tensors, no cohesive grouping
type InitialConditionTensors<B> = (
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
    Var<f32, B>,
);

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> PinnWave3D<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Compute temporal derivative ∂u/∂t at t=0 via forward finite difference
    ///
    /// Uses forward difference: ∂u/∂t(0) ≈ (u(ε) - u(0)) / ε
    pub(crate) fn compute_temporal_derivative_at_t0(
        &self,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        z: &Var<f32, B>,
        t: &Var<f32, B>,
    ) -> Var<f32, B> {
        let eps = 1e-3_f32;

        let u_t0 = self.pinn.forward(x, y, z, t);

        let t_eps = coeus_autograd::scalar_add(t, eps);
        let u_t_eps = self.pinn.forward(x, y, z, &t_eps);

        coeus_autograd::scalar_mul(&coeus_autograd::sub(&u_t_eps, &u_t0), 1.0 / eps)
    }

    /// Extract velocity initial condition tensor from training data
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    pub(crate) fn extract_velocity_initial_condition_tensor(
        _x_data: &[f32],
        _y_data: &[f32],
        _z_data: &[f32],
        t_data: &[f32],
        v_data: &[f32],
    ) -> KwaversResult<Var<f32, B>> {
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
        let backend = B::default();
        Ok(Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![n_ic, 1], &v_ic, &backend),
            false,
        ))
    }

    /// Extract initial condition tensors.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    pub(crate) fn extract_initial_condition_tensors(
        x_data: &[f32],
        y_data: &[f32],
        z_data: &[f32],
        t_data: &[f32],
        u_data: &[f32],
    ) -> KwaversResult<InitialConditionTensors<B>> {
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

        let backend = B::default();
        let mk = |v: &[f32]| {
            Var::new(
                coeus_tensor::Tensor::from_slice_on(vec![n_ic, 1], v, &backend),
                false,
            )
        };

        Ok((mk(&x_ic), mk(&y_ic), mk(&z_ic), mk(&t_ic), mk(&u_ic)))
    }
}
