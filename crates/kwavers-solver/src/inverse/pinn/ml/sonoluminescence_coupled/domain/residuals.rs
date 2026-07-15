//! Light-source computation and Maxwell residual with sonoluminescence sources.
//!
//! # Physics references
//!
//! - Jackson (1999) *Classical Electrodynamics*
//! - Putterman (1995) *Sonoluminescence: Sound into Light*

use coeus_autograd::Var;

use super::SonoluminescenceCoupledDomain;
use crate::inverse::pinn::ml::physics::PinnDomainPhysicsParameters;
use kwavers_core::constants::fundamental::{VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default>
    SonoluminescenceCoupledDomain<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Interpolate the sonoluminescence emission field at query coordinates
    /// `(x, y, t)` and scale by the coupling efficiency.
    ///
    /// Returns a `[batch, 1]` tensor of light-source current density values.
    pub(super) fn compute_light_sources(
        &self,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        t: &Var<f32, B>,
        _physics_params: &PinnDomainPhysicsParameters,
    ) -> Var<f32, B> {
        let batch_size = x.tensor.shape()[0];
        let mut source_terms = Vec::with_capacity(batch_size);

        let [nx, ny, nz] = self.emission_calculator.emission_field.shape();

        let x_coords = x.tensor.as_slice();
        let y_coords = y.tensor.as_slice();
        let t_coords = t.tensor.as_slice();

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

        let backend = B::default();
        Var::new(
            coeus_tensor::Tensor::from_slice_on(vec![batch_size, 1], &source_terms, &backend),
            false,
        )
    }

    /// Compute the electromagnetic PDE residual augmented with sonoluminescence
    /// current sources.
    ///
    /// Implements Maxwell's equations with sonoluminescence coupling:
    /// - ∇×E + ∂B/∂t + μ₀ J = 0  (Faraday's law)
    /// - ∇×B − μ₀ ε₀ ∂E/∂t − μ₀ J = 0  (Ampère's law)
    pub(super) fn electromagnetic_residual_with_sources(
        &self,
        model: &crate::inverse::pinn::ml::PinnWave2D<B>,
        x: &Var<f32, B>,
        y: &Var<f32, B>,
        t: &Var<f32, B>,
        physics_params: &PinnDomainPhysicsParameters,
    ) -> Var<f32, B> {
        let backend = B::default();
        let zeros_like = |v: &Var<f32, B>| {
            Var::new(
                coeus_tensor::Tensor::zeros_on(v.tensor.shape(), &backend),
                false,
            )
        };

        let x_grad = Var::new(x.tensor.clone(), true);
        let y_grad = Var::new(y.tensor.clone(), true);
        let t_grad = Var::new(t.tensor.clone(), true);

        let electric_field = model.forward(&x_grad, &y_grad, &t_grad);
        electric_field.backward();
        let e_dx = x_grad
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(x));
        let e_dy = y_grad
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(y));
        let e_dt = t_grad
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(t));

        let x_grad_2 = Var::new(x.tensor.clone(), true);
        let y_grad_2 = Var::new(y.tensor.clone(), true);
        let t_grad_2 = Var::new(t.tensor.clone(), true);

        let magnetic_field_2 = model.forward(&x_grad_2, &y_grad_2, &t_grad_2);
        magnetic_field_2.backward();
        let b_dx = x_grad_2
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(x));
        let b_dy = y_grad_2
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(y));
        let b_dt = t_grad_2
            .grad()
            .map(|g| Var::new(g, false))
            .unwrap_or_else(|| zeros_like(t));

        // Physical constants (SI) from SSOT (cast to f32 for tensor arithmetic).
        let mu_0_f32 = VACUUM_PERMEABILITY as f32;
        let epsilon_0_f32 = VACUUM_PERMITTIVITY as f32;

        let current_density = self.compute_light_sources(x, y, t, physics_params);

        // ampere_residual = e_dy - e_dx + b_dt*mu_0 + current_density*mu_0
        let ampere_residual = coeus_autograd::add(
            &coeus_autograd::sub(&e_dy, &e_dx),
            &coeus_autograd::add(
                &coeus_autograd::scalar_mul(&b_dt, mu_0_f32),
                &coeus_autograd::scalar_mul(&current_density, mu_0_f32),
            ),
        );
        // faraday_residual = b_dx - b_dy - mu_0*epsilon_0*e_dt - current_density*mu_0
        let faraday_residual = coeus_autograd::sub(
            &coeus_autograd::sub(&b_dx, &b_dy),
            &coeus_autograd::add(
                &coeus_autograd::scalar_mul(&e_dt, mu_0_f32 * epsilon_0_f32),
                &coeus_autograd::scalar_mul(&current_density, mu_0_f32),
            ),
        );

        coeus_autograd::add(&ampere_residual, &faraday_residual)
    }
}
