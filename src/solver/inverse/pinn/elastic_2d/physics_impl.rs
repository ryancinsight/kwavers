//! Physics Trait Implementation for 2D Elastic PINN
//!
//! This module provides trait implementations that allow the PINN model to satisfy
//! the domain-layer physics specifications defined in `domain::physics::ElasticWaveEquation`.
//!
//! # Architecture
//!
//! The PINN neural network (`ElasticPINN2D`) is a solver implementation, not a physics
//! specification. To enable shared validation, testing, and comparison with other solvers
//! (FD, FEM, spectral), we wrap the PINN in a struct that implements the physics traits.
//!
//! ```text
//! domain::physics::ElasticWaveEquation (trait specification)
//!        ↑
//!        | implements
//!        |
//! ElasticPINN2DSolver (wrapper struct)
//!        |
//!        | contains
//!        |
//!        +-> ElasticPINN2D<B> (neural network)
//!        +-> Domain (spatial domain spec)
//!        +-> Material parameters (λ, μ, ρ)
//! ```
//!
//! # Design Rationale
//!
//! **Separation of Concerns**:
//! - `ElasticPINN2D<B>`: Neural network architecture (Burn tensors, autodiff)
//! - `ElasticPINN2DSolver`: Physics specification (ndarray, domain traits)
//!
//! This separation allows:
//! 1. Neural network to remain backend-agnostic (CPU/GPU, different autodiff engines)
//! 2. Physics traits to remain solver-agnostic (FD/FEM/PINN/analytical)
//! 3. Tensor conversions to be isolated in one place
//!
//! **Tensor Bridge**:
//! The PINN operates on Burn tensors for autodiff, but physics traits expect ndarray.
//! This module handles conversions between representations:
//! - Forward: ndarray → Burn tensor (for inference through trained network)
//! - Backward: Burn tensor → ndarray (for trait method returns)
//!
//! # Usage
//!
//! ```rust,ignore
//! use kwavers::solver::inverse::pinn::elastic_2d::{Config, ElasticPINN2D, ElasticPINN2DSolver};
//! use kwavers::domain::physics::{Domain, ElasticWaveEquation};
//!
//! // Define physics domain
//! let domain = Domain::new_2d(0.0, 1.0, 0.0, 1.0, 101, 101, BoundaryCondition::Absorbing { damping: 0.1 });
//!
//! // Material properties
//! let lambda = 1e9;  // Pa
//! let mu = 0.5e9;    // Pa
//! let rho = 1000.0;  // kg/m³
//!
//! // Create PINN model
//! let config = Config::forward_problem(lambda, mu, rho);
//! let device = Default::default();
//! let pinn = ElasticPINN2D::new(&config, &device)?;
//!
//! // Wrap in solver that implements physics traits
//! let solver = ElasticPINN2DSolver::new(pinn, domain, lambda, mu, rho);
//!
//! // Now can use as ElasticWaveEquation
//! let cp = solver.p_wave_speed();
//! let cs = solver.s_wave_speed();
//! let cfl_dt = solver.cfl_timestep();
//!
//! // Can validate against other solvers via shared trait
//! fn validate_solver<S: ElasticWaveEquation>(solver: &S) {
//!     // Common validation logic...
//! }
//! validate_solver(&solver);
//! ```

#[cfg(feature = "pinn")]
use super::model::ElasticPINN2D;
#[cfg(feature = "pinn")]
use crate::domain::physics::{
    AutodiffElasticWaveEquation, AutodiffWaveEquation, Domain, TimeIntegration,
};

#[cfg(feature = "pinn")]
use ndarray::{Array1, ArrayD, IxDyn};

#[cfg(feature = "pinn")]
use burn::tensor::{backend::Backend, Tensor};

/// PINN solver wrapper implementing ElasticWaveEquation trait
///
/// Combines a trained PINN neural network with domain specification and
/// material properties to provide a complete physics solver that satisfies
/// the trait interface.
///
/// # Type Parameters
///
/// * `B` - Burn backend (NdArray, Wgpu, Cuda, etc.)
///
/// # Fields
///
/// * `model` - Trained PINN neural network
/// * `domain` - Spatial domain specification
/// * `lambda` - Lamé first parameter [Pa] (may be learned or fixed)
/// * `mu` - Shear modulus [Pa] (may be learned or fixed)
/// * `rho` - Density [kg/m³] (may be learned or fixed)
#[cfg(feature = "pinn")]
pub struct ElasticPINN2DSolver<B: Backend> {
    /// Neural network model
    model: ElasticPINN2D<B>,
    /// Spatial domain
    domain: Domain,
    /// Lamé first parameter (Pa)
    lambda: f64,
    /// Shear modulus (Pa)
    mu: f64,
    /// Density (kg/m³)
    rho: f64,
}

#[cfg(feature = "pinn")]
impl<B: Backend> ElasticPINN2DSolver<B> {
    /// Create new solver from trained model and domain specification
    ///
    /// # Arguments
    ///
    /// * `model` - Trained PINN model
    /// * `domain` - Spatial domain specification
    /// * `lambda` - Lamé first parameter [Pa]
    /// * `mu` - Shear modulus [Pa]
    /// * `rho` - Density [kg/m³]
    ///
    /// # Returns
    ///
    /// Solver wrapper ready for physics trait methods
    pub fn new(model: ElasticPINN2D<B>, domain: Domain, lambda: f64, mu: f64, rho: f64) -> Self {
        Self {
            model,
            domain,
            lambda,
            mu,
            rho,
        }
    }

    /// Update material parameters (for tracking learned values during inverse problems)
    ///
    /// # Arguments
    ///
    /// * `lambda` - New Lamé first parameter [Pa]
    /// * `mu` - New shear modulus [Pa]
    /// * `rho` - New density [kg/m³]
    pub fn update_parameters(&mut self, lambda: f64, mu: f64, rho: f64) {
        self.lambda = lambda;
        self.mu = mu;
        self.rho = rho;
    }

    /// Extract current material parameters from the model (if being optimized)
    ///
    /// Returns the learned parameters or the fixed values.
    pub fn current_parameters(&self) -> (f64, f64, f64) {
        let (lambda_opt, mu_opt, rho_opt) = self.model.estimated_parameters();
        (
            lambda_opt.unwrap_or(self.lambda),
            mu_opt.unwrap_or(self.mu),
            rho_opt.unwrap_or(self.rho),
        )
    }

    /// Get reference to underlying PINN model
    pub fn model(&self) -> &ElasticPINN2D<B> {
        &self.model
    }

    /// Evaluate PINN at spatial-temporal points
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinates (flattened)
    /// * `y` - Y coordinates (flattened)
    /// * `t` - Time coordinates (flattened)
    ///
    /// # Returns
    ///
    /// Displacement field [N, 2] where columns are (uₓ, uᵧ)
    fn evaluate_field(&self, x: &[f64], y: &[f64], t: &[f64]) -> ArrayD<f64> {
        assert_eq!(x.len(), y.len());
        assert_eq!(y.len(), t.len());

        let device = self.model.device();
        let n = x.len();

        // Convert to Burn tensors [N, 1]
        let x_data: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let y_data: Vec<f32> = y.iter().map(|&v| v as f32).collect();
        let t_data: Vec<f32> = t.iter().map(|&v| v as f32).collect();

        let x_tensor = Tensor::<B, 2>::from_floats(x_data.as_slice(), &device).reshape([n, 1]);
        let y_tensor = Tensor::<B, 2>::from_floats(y_data.as_slice(), &device).reshape([n, 1]);
        let t_tensor = Tensor::<B, 2>::from_floats(t_data.as_slice(), &device).reshape([n, 1]);

        // Forward pass through PINN
        let u = self.model.forward(x_tensor, y_tensor, t_tensor);

        // Convert back to ndarray [N, 2]
        let u_data = u.to_data();
        let u_vec: Vec<f64> = u_data
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .map(|&v| v as f64)
            .collect();

        ArrayD::from_shape_vec(IxDyn(&[n, 2]), u_vec)
            .expect("Shape mismatch in PINN output conversion")
    }

    /// Compute spatial grid coordinates for field evaluation
    fn grid_points(&self) -> (Vec<f64>, Vec<f64>) {
        let nx = self.domain.resolution[0];
        let ny = self.domain.resolution[1];
        let xmin = self.domain.bounds[0];
        let xmax = self.domain.bounds[1];
        let ymin = self.domain.bounds[2];
        let ymax = self.domain.bounds[3];

        let dx = (xmax - xmin) / (nx - 1) as f64;
        let dy = (ymax - ymin) / (ny - 1) as f64;

        let mut x_coords = Vec::with_capacity(nx * ny);
        let mut y_coords = Vec::with_capacity(nx * ny);

        for j in 0..ny {
            for i in 0..nx {
                x_coords.push(xmin + i as f64 * dx);
                y_coords.push(ymin + j as f64 * dy);
            }
        }

        (x_coords, y_coords)
    }
}

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
        let (x_grid, y_grid) = self.grid_points();
        let t = vec![0.0; nx * ny]; // Evaluate at t=0 (stationary assumption)

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
        let spacing = self.domain.spacing();
        let dx = spacing[0];
        let dy = spacing[1];
        let dv = dx * dy; // Volume element (unit thickness in z)

        let stress = self.stress_from_displacement(displacement);
        let strain = self.strain_from_displacement(displacement);

        let shape = displacement.shape();
        let nx = shape[0];
        let ny = shape[1];

        let mut energy = 0.0;

        for j in 0..ny {
            for i in 0..nx {
                // Kinetic energy density
                let vx = velocity[[i, j, 0]];
                let vy = velocity[[i, j, 1]];
                let kinetic_density = 0.5 * rho * (vx * vx + vy * vy);

                // Strain energy density
                let sigma_xx = stress[[i, j, 0]];
                let sigma_yy = stress[[i, j, 1]];
                let sigma_xy = stress[[i, j, 2]];
                let eps_xx = strain[[i, j, 0]];
                let eps_yy = strain[[i, j, 1]];
                let eps_xy = strain[[i, j, 2]];

                let strain_density =
                    0.5 * (sigma_xx * eps_xx + sigma_yy * eps_yy + 2.0 * sigma_xy * eps_xy);

                energy += (kinetic_density + strain_density) * dv;
            }
        }

        energy
    }
}

#[cfg(all(test, feature = "pinn"))]
mod tests {
    use super::*;
    use crate::solver::inverse::pinn::elastic_2d::Config;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_solver_creation() {
        let domain = Domain::new_2d(
            0.0,
            1.0,
            0.0,
            1.0,
            11,
            11,
            BoundaryCondition::Absorbing { damping: 0.1 },
        );

        let config = Config::forward_problem(1e9, 0.5e9, 1000.0);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        let solver = ElasticPINN2DSolver::new(model, domain, 1e9, 0.5e9, 1000.0);

        assert_eq!(solver.lambda, 1e9);
        assert_eq!(solver.mu, 0.5e9);
        assert_eq!(solver.rho, 1000.0);
    }

    #[test]
    fn test_material_parameter_fields() {
        let domain = Domain::new_2d(0.0, 1.0, 0.0, 1.0, 11, 11, BoundaryCondition::Periodic);

        let lambda = 1e9;
        let mu = 0.5e9;
        let rho = 1000.0;

        let config = Config::forward_problem(lambda, mu, rho);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        let solver = ElasticPINN2DSolver::new(model, domain, lambda, mu, rho);

        let lambda_field = solver.lame_lambda();
        let mu_field = solver.lame_mu();
        let rho_field = solver.density();

        assert_eq!(lambda_field.shape(), &[11, 11]);
        assert_eq!(mu_field.shape(), &[11, 11]);
        assert_eq!(rho_field.shape(), &[11, 11]);

        assert!((lambda_field[[5, 5]] - lambda).abs() < 1e-6);
        assert!((mu_field[[5, 5]] - mu).abs() < 1e-6);
        assert!((rho_field[[5, 5]] - rho).abs() < 1e-6);
    }

    #[test]
    fn test_wave_speeds() {
        let domain = Domain::new_2d(0.0, 1.0, 0.0, 1.0, 11, 11, BoundaryCondition::Periodic);

        let lambda = 1e9;
        let mu = 0.5e9;
        let rho = 1000.0;

        let config = Config::forward_problem(lambda, mu, rho);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        let solver = ElasticPINN2DSolver::new(model, domain, lambda, mu, rho);

        let cp = solver.p_wave_speed();
        let cs = solver.s_wave_speed();

        let expected_cp = ((lambda + 2.0 * mu) / rho).sqrt();
        let expected_cs = (mu / rho).sqrt();

        assert!((cp[[5, 5]] - expected_cp).abs() < 1e-6);
        assert!((cs[[5, 5]] - expected_cs).abs() < 1e-6);
    }

    #[test]
    fn test_cfl_timestep() {
        let domain = Domain::new_2d(0.0, 1.0, 0.0, 1.0, 101, 101, BoundaryCondition::Periodic);

        let lambda = 1e9;
        let mu = 0.5e9;
        let rho = 1000.0;

        let config = Config::forward_problem(lambda, mu, rho);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        let solver = ElasticPINN2DSolver::new(model, domain, lambda, mu, rho);

        let dt = solver.cfl_timestep();

        assert!(dt > 0.0);
        assert!(dt.is_finite());
    }

    #[test]
    fn test_time_integration() {
        let domain = Domain::new_2d(0.0, 1.0, 0.0, 1.0, 11, 11, BoundaryCondition::Periodic);

        let config = Config::forward_problem(1e9, 0.5e9, 1000.0);
        let device = Default::default();
        let model = ElasticPINN2D::<TestBackend>::new(&config, &device).unwrap();

        let solver = ElasticPINN2DSolver::new(model, domain, 1e9, 0.5e9, 1000.0);

        assert_eq!(solver.time_integration(), TimeIntegration::Implicit);
    }
}
