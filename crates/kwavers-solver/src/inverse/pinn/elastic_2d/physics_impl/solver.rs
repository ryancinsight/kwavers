//! Main PINN solver structure and basic methods
//!
//! This module contains the core ElasticPINN2DSolver struct and its
//! basic functionality for evaluating the neural network.

#[cfg(feature = "pinn")]
use super::super::model::ElasticPINN2D;
#[cfg(feature = "pinn")]
use kwavers_physics::foundations::Domain;

#[cfg(feature = "pinn")]
use burn::tensor::backend::Backend;

/// PINN solver wrapper implementing ElasticWaveEquation trait
///
/// Combines a trained PINN neural network with domain specification and
/// material properties to provide a complete physics solver that satisfies
/// the trait interface.
///
/// # Type Parameters
///
/// * `B` - Burn backend for the current CPU PINN implementation.
///
/// # Fields
///
/// * `model` - Trained PINN neural network
/// * `domain` - Spatial domain specification
/// * `lambda` - Lamé first parameter (Pa) (may be learned or fixed)
/// * `mu` - Shear modulus (Pa) (may be learned or fixed)
/// * `rho` - Density [kg/m³] (may be learned or fixed)
#[cfg(feature = "pinn")]
#[derive(Debug)]
pub struct ElasticPINN2DSolver<B: Backend> {
    /// Neural network model
    pub model: ElasticPINN2D<B>,
    /// Spatial domain
    pub domain: Domain,
    /// Lamé first parameter (Pa)
    pub lambda: f64,
    /// Shear modulus (Pa)
    pub mu: f64,
    /// Density (kg/m³)
    pub rho: f64,
}

#[cfg(feature = "pinn")]
impl<B: Backend> ElasticPINN2DSolver<B> {
    /// Create new solver from trained model and domain specification
    ///
    /// # Arguments
    ///
    /// * `model` - Trained PINN model
    /// * `domain` - Spatial domain specification
    /// * `lambda` - Lamé first parameter (Pa)
    /// * `mu` - Shear modulus (Pa)
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
    /// * `lambda` - New Lamé first parameter (Pa)
    /// * `mu` - New shear modulus (Pa)
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

    /// Compute spatial grid coordinates for field evaluation
    pub fn grid_points(&self) -> (Vec<f64>, Vec<f64>) {
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
