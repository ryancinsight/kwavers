//! `UniversalPINNSolver` training methods.
//!
//! SRP: changes when the PINN training loop, collocation sampling,
//! or model initialization logic changes.

use super::solver::UniversalPINNSolver;
use super::types::{
    ConvergenceInfo, DomainInfo, GeometricFeature, Geometry2D, PhysicsSolution,
    UniversalSolverStats, UniversalTrainingConfig,
};
use crate::core::error::{KwaversError, KwaversResult};
use crate::solver::inverse::pinn::ml::physics::{PhysicsDomain, PinnDomainPhysicsParameters};
use burn::prelude::ToElement;
use burn::tensor::{backend::AutodiffBackend, Tensor};
use log::info;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::time::Instant;

impl<B: AutodiffBackend> UniversalPINNSolver<B> {
    /// Solve physics problem for a single domain
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    pub fn solve_physics_domain(
        &mut self,
        domain_name: &str,
        geometry: &Geometry2D,
        physics_params: &PinnDomainPhysicsParameters,
        training_config: Option<&UniversalTrainingConfig>,
    ) -> KwaversResult<PhysicsSolution<B>> {
        let domain = self
            .physics_registry
            .get_domain(domain_name)
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::ResourceUnavailable {
                    resource: format!("physics domain {}", domain_name),
                })
            })?;

        let config = training_config
            .or_else(|| self.configs.get(domain_name))
            .cloned()
            .unwrap_or_default();

        let collocation_points = self.generate_collocation_points(
            geometry,
            domain,
            config.collocation_points,
            Self::seed_from_domain_name(domain_name),
        )?;

        if !self.models.contains_key(domain_name) {
            let model = self.initialize_model(domain)?;
            self.models.insert(domain_name.to_string(), model);
        }

        let model = self.models.get_mut(domain_name).unwrap();

        let training_start = Instant::now();
        let (final_losses, loss_history) =
            Self::train_model(model, domain, &collocation_points, physics_params, &config)?;
        let training_time = training_start.elapsed();

        let physics_metrics = domain.validation_metrics();

        let (best_epoch, best_loss, first_loss, final_loss) =
            Self::summarize_loss_history(&loss_history);
        let convergence_info = ConvergenceInfo {
            converged: best_loss < 1e-4,
            final_epoch: config.epochs,
            best_loss,
            best_epoch,
            loss_reduction_ratio: if best_loss > 0.0 {
                first_loss / best_loss
            } else {
                f64::INFINITY
            },
        };

        let stats = UniversalSolverStats {
            training_time,
            final_loss,
            final_losses,
            loss_history,
            physics_metrics,
            convergence_info,
            memory_stats: None,
        };

        let domain_info = DomainInfo {
            domain_name: domain_name.to_string(),
            physics_params: physics_params.clone(),
            boundary_conditions: domain.boundary_conditions(),
            initial_conditions: domain.initial_conditions(),
        };

        let solution = PhysicsSolution {
            model: model.clone(),
            config,
            stats: stats.clone(),
            domain_info,
        };

        self.stats.insert(domain_name.to_string(), stats);

        Ok(solution)
    }

    /// Generate physics-aware collocation points
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn generate_collocation_points(
        &self,
        geometry: &Geometry2D,
        _domain: &dyn PhysicsDomain<B>,
        num_points: usize,
        seed: u64,
    ) -> KwaversResult<Vec<(f64, f64, f64)>> {
        let mut points = Vec::new();
        let [x_min, x_max, y_min, y_max] = geometry.bounds;
        let mut rng = StdRng::seed_from_u64(seed);

        for _ in 0..num_points {
            let x = x_min + (x_max - x_min) * rng.gen::<f64>();
            let y = y_min + (y_max - y_min) * rng.gen::<f64>();
            let t = rng.gen::<f64>();
            if self.is_point_in_geometry(x, y, geometry) {
                points.push((x, y, t));
            }
        }

        Ok(points)
    }

    /// Check if a point is within the geometric domain
    pub(super) fn is_point_in_geometry(&self, x: f64, y: f64, geometry: &Geometry2D) -> bool {
        let [x_min, x_max, y_min, y_max] = geometry.bounds;
        if x < x_min || x > x_max || y < y_min || y > y_max {
            return false;
        }
        for feature in &geometry.features {
            match feature {
                GeometricFeature::Circle {
                    center: (cx, cy),
                    radius,
                } => {
                    let distance = ((x - cx).powi(2) + (y - cy).powi(2)).sqrt();
                    if distance <= *radius {
                        return false;
                    }
                }
                GeometricFeature::Rectangle {
                    x_min: rx_min,
                    x_max: rx_max,
                    y_min: ry_min,
                    y_max: ry_max,
                } => {
                    if x >= *rx_min && x <= *rx_max && y >= *ry_min && y <= *ry_max {
                        return false;
                    }
                }
                GeometricFeature::Interface { .. } => {}
            }
        }
        true
    }

    /// Initialize a neural network model for a physics domain
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn initialize_model(
        &self,
        _domain: &dyn PhysicsDomain<B>,
    ) -> KwaversResult<crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>> {
        let config = crate::solver::inverse::pinn::ml::BurnPINN2DConfig {
            hidden_layers: vec![64, 64, 64],
            learning_rate: 0.001,
            num_collocation_points: 1000,
            ..Default::default()
        };
        let device = Default::default();
        let model = crate::solver::inverse::pinn::ml::BurnPINN2DWave::new(config, &device)?;
        Ok(model)
    }

    /// Train the neural network model
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    /// # Panics
    /// - Panics if an internal invariant assumed to hold at this call site is violated.
    ///
    fn train_model(
        model: &mut crate::solver::inverse::pinn::ml::BurnPINN2DWave<B>,
        domain: &dyn PhysicsDomain<B>,
        collocation_points: &[(f64, f64, f64)],
        physics_params: &PinnDomainPhysicsParameters,
        config: &UniversalTrainingConfig,
    ) -> KwaversResult<(HashMap<String, f64>, Vec<HashMap<String, f64>>)> {
        let start_time = Instant::now();
        let n_points = collocation_points.len();
        let x_coords: Vec<f32> = collocation_points
            .iter()
            .map(|(x, _, _)| *x as f32)
            .collect();
        let y_coords: Vec<f32> = collocation_points
            .iter()
            .map(|(_, y, _)| *y as f32)
            .collect();
        let t_coords: Vec<f32> = collocation_points
            .iter()
            .map(|(_, _, t)| *t as f32)
            .collect();

        let device = B::Device::default();
        let x_tensor =
            Tensor::<B, 1>::from_floats(x_coords.as_slice(), &device).reshape([n_points, 1]);
        let y_tensor =
            Tensor::<B, 1>::from_floats(y_coords.as_slice(), &device).reshape([n_points, 1]);
        let t_tensor =
            Tensor::<B, 1>::from_floats(t_coords.as_slice(), &device).reshape([n_points, 1]);

        let mut loss_history = Vec::new();
        let optimizer =
            crate::solver::inverse::pinn::ml::burn_wave_equation_2d::SimpleOptimizer2D::new(
                config.learning_rate as f32,
            );

        for epoch in 0..config.epochs {
            let residual =
                domain.pde_residual(model, &x_tensor, &y_tensor, &t_tensor, physics_params);
            let pde_loss = residual.clone().powf_scalar(2.0).mean();
            let grads = pde_loss.backward();
            *model = optimizer.step(model.clone(), &grads);

            let loss_value = pde_loss.clone().into_scalar().to_f32() as f64;
            let mut epoch_losses = HashMap::new();
            epoch_losses.insert("pde".to_string(), loss_value);
            epoch_losses.insert("total".to_string(), loss_value);
            loss_history.push(epoch_losses);

            if epoch % 100 == 0 && epoch > 0 {
                info!("Epoch {}: PDE Loss = {:.6e}", epoch, loss_value);
            }
        }

        let final_losses = if !loss_history.is_empty() {
            loss_history.last().unwrap().clone()
        } else {
            HashMap::from([("pde".to_string(), 0.0), ("total".to_string(), 0.0)])
        };

        info!(
            "PINN training completed in {:.2}s with physics-informed optimization",
            start_time.elapsed().as_secs_f64()
        );

        Ok((final_losses, loss_history))
    }

    pub(super) fn seed_from_domain_name(domain_name: &str) -> u64 {
        let mut hash: u64 = 14695981039346656037;
        for &b in domain_name.as_bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(1099511628211);
        }
        hash
    }

    pub(super) fn summarize_loss_history(
        loss_history: &[HashMap<String, f64>],
    ) -> (usize, f64, f64, f64) {
        let mut best_epoch = 0usize;
        let mut best_loss = f64::INFINITY;
        let mut first_loss = f64::INFINITY;
        let mut final_loss = f64::INFINITY;

        for (i, losses) in loss_history.iter().enumerate() {
            if let Some(&total) = losses.get("total") {
                if i == 0 {
                    first_loss = total;
                }
                final_loss = total;
                if total < best_loss {
                    best_loss = total;
                    best_epoch = i + 1;
                }
            }
        }

        if !first_loss.is_finite() {
            first_loss = 0.0;
        }
        if !final_loss.is_finite() {
            final_loss = 0.0;
        }
        if !best_loss.is_finite() {
            best_loss = 0.0;
        }

        (best_epoch, best_loss, first_loss, final_loss)
    }
}
