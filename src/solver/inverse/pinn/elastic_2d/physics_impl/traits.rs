//! Trait implementations for PINN physics solver
//!
//! This module contains all the trait implementations that allow the
//! ElasticPINN2DSolver to satisfy the domain-layer physics specifications.

#[cfg(feature = "pinn")]
use super::solver::ElasticPINN2DSolver;
#[cfg(feature = "pinn")]
use crate::physics::foundations::{
    AutodiffElasticWaveEquation, AutodiffWaveEquation, Domain, TimeIntegration,
};

#[cfg(feature = "pinn")]
use ndarray::{ArrayD, IxDyn};

#[cfg(feature = "pinn")]
use burn::tensor::backend::Backend;

#[cfg(feature = "pinn")]
impl<B: Backend> AutodiffWaveEquation for ElasticPINN2DSolver<B> {
    fn domain(&self) -> &Domain {
        &self.domain
    }

    fn time_integration(&self) -> TimeIntegration {
        // PINNs are implicit solvers (no time stepping)
        TimeIntegration::Implicit
    }

    fn cfl_timestep(&self) -> f64 {
        // PINNs don't have CFL limitations (continuous in time)
        // Return a recommended timestep for output sampling based on wave speeds
        let cp = ((self.lambda + 2.0 * self.mu) / self.rho).sqrt();
        let spacing = self.domain.spacing();
        let min_dx = spacing.iter().cloned().fold(f64::INFINITY, f64::min);

        // Nyquist-like sampling: at least 10 points per wavelength
        0.1 * min_dx / cp
    }

    fn spatial_operator(&self, field: &ArrayD<f64>) -> ArrayD<f64> {
        // For PINNs, the spatial operator is implicitly encoded in the network
        // This method evaluates the PDE residual (not typically used in PINN inference)
        //
        // field shape: [nx, ny, 2] (displacement field on grid)
        // returns: [nx, ny, 2] (acceleration field: ∂²u/∂t²)

        let shape = field.shape();
        let nx = shape[0];
        let ny = shape[1];

        // Create coordinate arrays
        let (_x_grid, _y_grid) = self.grid_points();
        let _t = vec![0.0; nx * ny]; // Evaluate at t=0 (stationary assumption)

        // Evaluate PINN to get displacement (not directly useful here)
        // For a proper spatial operator, we'd need to compute PDE residuals
        // which requires autodiff - not available in inference mode

        // Placeholder: return zeros (proper implementation requires autodiff backend)
        ArrayD::zeros(IxDyn(shape))
    }

    fn apply_boundary_conditions(&mut self, _field: &mut ArrayD<f64>) {
        // PINNs enforce boundary conditions during training via loss function
        // No explicit boundary enforcement needed during inference
        // This is a no-op for trained PINNs
    }

    fn check_constraints(&self, field: &ArrayD<f64>) -> Result<(), String> {
        // Verify field has correct shape and finite values
        let shape = field.shape();

        if shape.len() != 3 {
            return Err(format!(
                "Expected 3D field [nx, ny, 2], got {}D",
                shape.len()
            ));
        }

        if shape[2] != 2 {
            return Err(format!(
                "Expected 2 displacement components, got {}",
                shape[2]
            ));
        }

        // Check for NaN or Inf
        for &val in field.iter() {
            if !val.is_finite() {
                return Err("Field contains non-finite values (NaN or Inf)".to_string());
            }
        }

        Ok(())
    }
}

#[cfg(feature = "pinn")]
impl<B: Backend> AutodiffElasticWaveEquation for ElasticPINN2DSolver<B> {
    fn lame_lambda(&self) -> ArrayD<f64> {
        // Get current parameter (learned or fixed)
        let (lambda, _, _) = self.current_parameters();

        // Return homogeneous field (scalar broadcast to grid)
        let nx = self.domain.resolution[0];
        let ny = self.domain.resolution[1];
        ArrayD::from_elem(IxDyn(&[nx, ny]), lambda)
    }

    fn lame_mu(&self) -> ArrayD<f64> {
        let (_, mu, _) = self.current_parameters();
        let nx = self.domain.resolution[0];
        let ny = self.domain.resolution[1];
        ArrayD::from_elem(IxDyn(&[nx, ny]), mu)
    }

    fn density(&self) -> ArrayD<f64> {
        let (_, _, rho) = self.current_parameters();
        let nx = self.domain.resolution[0];
        let ny = self.domain.resolution[1];
        ArrayD::from_elem(IxDyn(&[nx, ny]), rho)
    }

    fn stress_from_displacement(&self, displacement: &ArrayD<f64>) -> ArrayD<f64> {
        // Compute stress tensor from displacement field
        // displacement: [nx, ny, 2] → stress: [nx, ny, 3] (σₓₓ, σᵧᵧ, σₓᵧ)
        //
        // σₓₓ = (λ + 2μ)·∂uₓ/∂x + λ·∂uᵧ/∂y
        // σᵧᵧ = λ·∂uₓ/∂x + (λ + 2μ)·∂uᵧ/∂y
        // σₓᵧ = μ·(∂uₓ/∂y + ∂uᵧ/∂x)

        let shape = displacement.shape();
        let nx = shape[0];
        let ny = shape[1];

        let (lambda, mu, _) = self.current_parameters();
        let lambda_2mu = lambda + 2.0 * mu;

        let spacing = self.domain.spacing();
        let dx = spacing[0];
        let dy = spacing[1];

        let mut stress = ArrayD::zeros(IxDyn(&[nx, ny, 3]));

        // Compute spatial derivatives via finite differences
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                // Central differences
                let ux_dx =
                    (displacement[[i + 1, j, 0]] - displacement[[i - 1, j, 0]]) / (2.0 * dx);
                let ux_dy =
                    (displacement[[i, j + 1, 0]] - displacement[[i, j - 1, 0]]) / (2.0 * dy);
                let uy_dx =
                    (displacement[[i + 1, j, 1]] - displacement[[i - 1, j, 1]]) / (2.0 * dx);
                let uy_dy =
                    (displacement[[i, j + 1, 1]] - displacement[[i, j - 1, 1]]) / (2.0 * dy);

                // Stress components
                stress[[i, j, 0]] = lambda_2mu * ux_dx + lambda * uy_dy; // σₓₓ
                stress[[i, j, 1]] = lambda * ux_dx + lambda_2mu * uy_dy; // σᵧᵧ
                stress[[i, j, 2]] = mu * (ux_dy + uy_dx); // σₓᵧ
            }
        }

        stress
    }

    fn strain_from_displacement(&self, displacement: &ArrayD<f64>) -> ArrayD<f64> {
        // Compute strain tensor from displacement field
        // displacement: [nx, ny, 2] → strain: [nx, ny, 3] (εₓₓ, εᵧᵧ, εₓᵧ)
        //
        // εₓₓ = ∂uₓ/∂x
        // εᵧᵧ = ∂uᵧ/∂y
        // εₓᵧ = ½(∂uₓ/∂y + ∂uᵧ/∂x)

        let shape = displacement.shape();
        let nx = shape[0];
        let ny = shape[1];

        let spacing = self.domain.spacing();
        let dx = spacing[0];
        let dy = spacing[1];

        let mut strain = ArrayD::zeros(IxDyn(&[nx, ny, 3]));

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let ux_dx =
                    (displacement[[i + 1, j, 0]] - displacement[[i - 1, j, 0]]) / (2.0 * dx);
                let ux_dy =
                    (displacement[[i, j + 1, 0]] - displacement[[i, j - 1, 0]]) / (2.0 * dy);
                let uy_dx =
                    (displacement[[i + 1, j, 1]] - displacement[[i - 1, j, 1]]) / (2.0 * dx);
                let uy_dy =
                    (displacement[[i, j + 1, 1]] - displacement[[i, j - 1, 1]]) / (2.0 * dy);

                strain[[i, j, 0]] = ux_dx; // εₓₓ
                strain[[i, j, 1]] = uy_dy; // εᵧᵧ
                strain[[i, j, 2]] = 0.5 * (ux_dy + uy_dx); // εₓᵧ
            }
        }

        strain
    }

    fn elastic_energy(&self, displacement: &ArrayD<f64>, velocity: &ArrayD<f64>) -> f64 {
        // Compute total elastic energy: E = ∫ (½ρ|v|² + ½σ:ε) dV
        //
        // Kinetic energy: Eₖ = ½ρ|v|²
        // Strain energy: Eₛ = ½σ:ε = ½(σₓₓεₓₓ + σᵧᵧεᵧᵧ + 2σₓᵧεₓᵧ)

        let (_, _, rho) = self.current_parameters();

        let stress = self.stress_from_displacement(displacement);
        let strain = self.strain_from_displacement(displacement);

        let mut kinetic_energy = 0.0;
        let mut strain_energy = 0.0;

        let shape = displacement.shape();
        let nx = shape[0];
        let ny = shape[1];

        let spacing = self.domain.spacing();
        let dx = spacing[0];
        let dy = spacing[1];
        let dA = dx * dy;

        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                // Kinetic energy density: ½ρ|v|²
                let vx = velocity[[i, j, 0]];
                let vy = velocity[[i, j, 1]];
                kinetic_energy += 0.5 * rho * (vx * vx + vy * vy) * dA;

                // Strain energy density: ½σ:ε
                let sxx = stress[[i, j, 0]];
                let syy = stress[[i, j, 1]];
                let sxy = stress[[i, j, 2]];
                let exx = strain[[i, j, 0]];
                let eyy = strain[[i, j, 1]];
                let exy = strain[[i, j, 2]];

                strain_energy += 0.5 * (sxx * exx + syy * eyy + 2.0 * sxy * exy) * dA;
            }
        }

        kinetic_energy + strain_energy
    }
}
