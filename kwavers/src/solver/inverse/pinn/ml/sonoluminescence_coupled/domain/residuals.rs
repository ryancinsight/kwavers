//! Light-source computation and Maxwell residual with sonoluminescence sources.
//!
//! # Physics references
//!
//! - Jackson (1999) *Classical Electrodynamics*
//! - Putterman (1995) *Sonoluminescence: Sound into Light*

use burn::tensor::{backend::AutodiffBackend, Tensor};

use super::SonoluminescenceCoupledDomain;
use crate::solver::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;

impl<B: AutodiffBackend> SonoluminescenceCoupledDomain<B> {
    /// Interpolate the sonoluminescence emission field at query coordinates
    /// `(x, y, t)` and scale by the coupling efficiency.
    ///
    /// Returns a `[batch, 1]` tensor of light-source current density values.
    pub(super) fn compute_light_sources(
        &self,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        _physics_params: &PinnDomainPhysicsParameters,
    ) -> Tensor<B, 2> {
        let batch_size = x.shape().dims[0];
        let mut source_terms = Vec::with_capacity(batch_size);

        let emission_dims = self.emission_calculator.emission_field.dim();
        let (nx, ny, nz) = (emission_dims.0, emission_dims.1, emission_dims.2);

        let x_coords: Vec<f32> = x.to_data().to_vec().unwrap();
        let y_coords: Vec<f32> = y.to_data().to_vec().unwrap();
        let t_coords: Vec<f32> = t.to_data().to_vec().unwrap();

        for i in 0..batch_size {
            let x_pos = x_coords[i] as f64;
            let y_pos = y_coords[i] as f64;
            let t_pos = t_coords[i] as f64;

            let i_idx = ((x_pos * (nx - 1) as f64).round() as usize).clamp(0, nx - 1);
            let j_idx = ((y_pos * (ny - 1) as f64).round() as usize).clamp(0, ny - 1);
            let k_idx = ((t_pos * (nz - 1) as f64).round() as usize).clamp(0, nz - 1);

            let emission_value = if i_idx < nx && j_idx < ny && k_idx < nz {
                self.emission_calculator.emission_field[[i_idx, j_idx, k_idx]]
            } else {
                0.0
            };

            let source_term = emission_value as f32 * self.config.coupling_efficiency as f32;
            source_terms.push(source_term);
        }

        Tensor::<B, 1>::from_floats(
            source_terms.as_slice(),
            &<B as burn::tensor::backend::Backend>::Device::default(),
        )
        .reshape([batch_size, 1])
    }

    /// Compute the electromagnetic PDE residual augmented with sonoluminescence
    /// current sources.
    ///
    /// Implements Maxwell's equations with sonoluminescence coupling:
    /// - ∇×E + ∂B/∂t + μ₀ J = 0  (Faraday's law)
    /// - ∇×B − μ₀ ε₀ ∂E/∂t − μ₀ J = 0  (Ampère's law)
    pub(super) fn electromagnetic_residual_with_sources(
        &self,
        model: &crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>,
        x: &Tensor<B, 2>,
        y: &Tensor<B, 2>,
        t: &Tensor<B, 2>,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> Tensor<B, 2> {
        let x_grad = x.clone().require_grad();
        let y_grad = y.clone().require_grad();
        let t_grad = t.clone().require_grad();

        let electric_field = model.forward(x_grad.clone(), y_grad.clone(), t_grad.clone());
        let _magnetic_field = model.forward(x_grad.clone(), y_grad.clone(), t_grad.clone());

        let grad_electric = electric_field.backward();
        let e_dx = x_grad
            .grad(&grad_electric)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| x.zeros_like());
        let e_dy = y_grad
            .grad(&grad_electric)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| y.zeros_like());
        let e_dt = t_grad
            .grad(&grad_electric)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| t.zeros_like());

        let x_grad_2 = x.clone().require_grad();
        let y_grad_2 = y.clone().require_grad();
        let t_grad_2 = t.clone().require_grad();

        let magnetic_field_2 = model.forward(x_grad_2.clone(), y_grad_2.clone(), t_grad_2.clone());
        let grad_magnetic = magnetic_field_2.backward();
        let b_dx = x_grad_2
            .grad(&grad_magnetic)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| x.zeros_like());
        let b_dy = y_grad_2
            .grad(&grad_magnetic)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| y.zeros_like());
        let b_dt = t_grad_2
            .grad(&grad_magnetic)
            .map(|g| Tensor::<B, 2>::from_data(g.into_data(), &Default::default()))
            .unwrap_or_else(|| t.zeros_like());

        // Physical constants (SI).
        let mu_0 = 4.0 * std::f64::consts::PI * 1e-7_f64;
        let epsilon_0 = 8.854e-12_f64;
        let mu_0_f32 = mu_0 as f32;
        let epsilon_0_f32 = epsilon_0 as f32;

        let current_density = self.compute_light_sources(x, y, t, physics_params);

        let ampere_residual = e_dy - e_dx + b_dt * mu_0_f32 + current_density.clone() * mu_0_f32;
        let faraday_residual =
            b_dx - b_dy - mu_0_f32 * epsilon_0_f32 * e_dt - current_density * mu_0_f32;

        ampere_residual + faraday_residual
    }
}
