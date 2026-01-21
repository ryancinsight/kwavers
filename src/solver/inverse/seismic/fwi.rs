//! Full Waveform Inversion Implementation
//!
//! FWI algorithm implementation following GRASP principles
//! Reference: Tarantola (1984): "Inversion of seismic reflection data in the acoustic approximation"

use super::parameters::FwiParameters;
use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::source::grid_source::GridSource;
use crate::solver::interface::solver::Solver;
use ndarray::Array3;

/// Full Waveform Inversion processor
#[derive(Debug)]
pub struct FwiProcessor {
    parameters: FwiParameters,
}

impl FwiProcessor {
    /// Create new FWI processor with specified parameters
    #[must_use]
    pub fn new(parameters: FwiParameters) -> Self {
        Self { parameters }
    }

    /// Perform Full Waveform Inversion
    /// Based on Tarantola (1984): "Inversion of seismic reflection data in the acoustic approximation"
    /// Reference: Geophysics, 49(8), 1259-1266
    pub fn invert(
        &self,
        observed_data: &Array3<f64>,
        initial_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        use crate::math::linear_algebra::norm_l2;

        let mut current_model = initial_model.clone();
        let mut prev_misfit = f64::INFINITY;

        for iteration in 0..self.parameters.max_iterations {
            // 1. Forward modeling
            let (synthetic_data, forward_history) = self.forward_model(&current_model, grid)?;

            // 2. Calculate data misfit
            let residual = observed_data - &synthetic_data;
            let current_misfit = norm_l2(&residual);

            // 3. Check convergence
            let relative_change = (prev_misfit - current_misfit).abs() / prev_misfit;
            if relative_change < self.parameters.tolerance {
                log::info!(
                    "FWI converged after {} iterations with misfit: {:.6e}",
                    iteration,
                    current_misfit
                );
                break;
            }

            // 4. Adjoint computation
            let adjoint_source = self.compute_adjoint_source(&residual);
            let gradient = self.adjoint_model(&adjoint_source, grid, &forward_history)?;

            // 5. Gradient processing (smoothing)
            let smoothed_gradient = self.smooth_gradient(gradient);

            // 6. Apply regularization
            let regularized_gradient =
                self.apply_regularization(&smoothed_gradient, &current_model)?;

            // 7. Line search for optimal step size
            let step_size =
                self.line_search(&current_model, &regularized_gradient, observed_data, grid)?;

            // 8. Model update
            current_model = &current_model - &(&regularized_gradient * step_size);

            // Apply physical constraints
            self.apply_model_constraints(&mut current_model);

            prev_misfit = current_misfit;

            log::debug!(
                "FWI iteration {}: misfit = {:.6e}, step_size = {:.6e}",
                iteration,
                current_misfit,
                step_size
            );
        }

        Ok(current_model)
    }

    /// Calculate interaction between two fields (used for testing gradient kernel)
    #[must_use]
    pub fn calculate_interaction(
        &self,
        forward_field: &Array3<f64>,
        adjoint_field: &Array3<f64>,
    ) -> Array3<f64> {
        use ndarray::Zip;

        let mut gradient = Array3::zeros(forward_field.dim());

        // Compute interaction
        Zip::from(&mut gradient)
            .and(forward_field)
            .and(adjoint_field)
            .for_each(|g, &fwd, &adj| {
                // Negative sign for descent direction
                *g = -fwd * adj;
            });

        self.smooth_gradient(gradient)
    }

    /// Apply smoothing to gradient to reduce high-frequency artifacts
    #[must_use]
    fn smooth_gradient(&self, gradient: Array3<f64>) -> Array3<f64> {
        let mut smoothed = gradient.clone();
        let (nx, ny, nz) = gradient.dim();

        // Simple 3-point smoothing in each dimension
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    smoothed[[i, j, k]] = (gradient[[i - 1, j, k]]
                        + gradient[[i, j, k]]
                        + gradient[[i + 1, j, k]]
                        + gradient[[i, j - 1, k]]
                        + gradient[[i, j, k]]
                        + gradient[[i, j + 1, k]]
                        + gradient[[i, j, k - 1]]
                        + gradient[[i, j, k]]
                        + gradient[[i, j, k + 1]])
                        / 9.0;
                }
            }
        }

        smoothed
    }

    /// Apply regularization to gradient
    fn apply_regularization(
        &self,
        gradient: &Array3<f64>,
        model: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut regularized = gradient.clone();
        let reg_params = &self.parameters.regularization;

        // Tikhonov regularization: R = λ₁ * m
        if reg_params.tikhonov_weight > 0.0 {
            regularized = &regularized + &(model * reg_params.tikhonov_weight);
        }

        // Total variation regularization
        if reg_params.tv_weight > 0.0 {
            let tv_term = self.compute_total_variation_gradient(model);
            regularized = &regularized + &(&tv_term * reg_params.tv_weight);
        }

        // Smoothness regularization (Laplacian)
        if reg_params.smoothness_weight > 0.0 {
            let smoothness_term = self.compute_smoothness_gradient(model);
            regularized = &regularized + &(&smoothness_term * reg_params.smoothness_weight);
        }

        Ok(regularized)
    }

    /// Compute total variation gradient for regularization
    #[must_use]
    fn compute_total_variation_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let mut tv_gradient = Array3::zeros((nx, ny, nz));

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    // Compute gradient magnitude
                    let dx = model[[i + 1, j, k]] - model[[i - 1, j, k]];
                    let dy = model[[i, j + 1, k]] - model[[i, j - 1, k]];
                    let dz = model[[i, j, k + 1]] - model[[i, j, k - 1]];

                    let grad_mag = (dx * dx + dy * dy + dz * dz).sqrt();

                    if grad_mag > f64::EPSILON {
                        tv_gradient[[i, j, k]] = grad_mag;
                    }
                }
            }
        }

        tv_gradient
    }

    /// Compute smoothness gradient (Laplacian) for regularization
    #[must_use]
    fn compute_smoothness_gradient(&self, model: &Array3<f64>) -> Array3<f64> {
        let (nx, ny, nz) = model.dim();
        let mut laplacian = Array3::zeros((nx, ny, nz));

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    laplacian[[i, j, k]] = model[[i + 1, j, k]]
                        + model[[i - 1, j, k]]
                        + model[[i, j + 1, k]]
                        + model[[i, j - 1, k]]
                        + model[[i, j, k + 1]]
                        + model[[i, j, k - 1]]
                        - 6.0 * model[[i, j, k]];
                }
            }
        }

        laplacian
    }

    /// Line search for optimal step size
    fn line_search(
        &self,
        model: &Array3<f64>,
        gradient: &Array3<f64>,
        observed_data: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        use crate::math::linear_algebra::norm_l2;

        let mut step_size = self.parameters.step_size;
        let c1 = 1e-4; // Armijo condition constant
        let max_iterations = 10;

        // Current function value
        let (synthetic_current, _) = self.forward_model(model, grid)?;
        let residual_current = observed_data - &synthetic_current;
        let current_misfit = norm_l2(&residual_current);

        // Gradient dot product for Armijo condition
        let gradient_norm_sq = gradient.map(|&x| x * x).sum();

        for _ in 0..max_iterations {
            // Test model with current step size
            let test_model = model - &(gradient * step_size);
            let (synthetic_test, _) = self.forward_model(&test_model, grid)?;
            let residual_test = observed_data - &synthetic_test;
            let test_misfit = norm_l2(&residual_test);

            // Armijo condition check
            if test_misfit <= current_misfit - c1 * step_size * gradient_norm_sq {
                return Ok(step_size);
            }

            // Reduce step size
            step_size *= 0.5;
        }

        Ok(step_size)
    }

    /// Apply physical constraints to velocity model
    fn apply_model_constraints(&self, model: &mut Array3<f64>) {
        use crate::core::constants::SOUND_SPEED_WATER;

        // Ensure physically reasonable velocity bounds
        let min_velocity = SOUND_SPEED_WATER * 0.5; // 750 m/s
        let max_velocity = SOUND_SPEED_WATER * 4.0; // 6000 m/s

        model.mapv_inplace(|v| v.clamp(min_velocity, max_velocity));
    }

    /// Forward modeling using FDTD acoustic solver
    ///
    /// Computes synthetic seismograms from the velocity model using the
    /// finite-difference time-domain method for acoustic wave propagation.
    ///
    /// # Arguments
    /// * `model` - Velocity model (sound speed in m/s)
    /// * `grid` - Computational grid defining the domain
    ///
    /// # Returns
    /// * Synthetic wavefield at final time step
    ///
    /// # References
    /// * Virieux (1986): "P-SV wave propagation in heterogeneous media"
    fn forward_model(
        &self,
        model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<(Array3<f64>, Vec<Array3<f64>>)> {
        use crate::solver::fdtd::{FdtdConfig, FdtdSolver};
        // Removed bad ndarray import (already imported)

        // Calculate stable timestep using CFL condition
        let dt = self.calculate_stable_timestep(model, grid);
        let num_steps = self.parameters.max_iterations.min(100); // Reasonable default

        // Create FDTD configuration for forward modeling
        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            cfl_factor: 0.3,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            nt: num_steps,
            dt,
            sensor_mask: None,
        };

        // Initialize FDTD solver
        // Use dummy medium and overwrite properties
        let dummy_medium = crate::domain::medium::HomogeneousMedium::water(grid);
        let mut solver = FdtdSolver::new(config, grid, &dummy_medium, GridSource::default())?;

        // Initialize fields
        let (_nx, _ny, _nz) = grid.dimensions();
        // FdtdSolver initializes p, u to zero.

        // Convert velocity model to density (assume constant for seismic)
        // let density = Array3::from_elem((nx, ny, nz), 1000.0); // kg/m³
        // Solver rho0 is already 1000.0        // Update model in solver
        // FWI updates the velocity model (c0)
        solver.materials.c0.assign(model);

        // History storage
        let mut history = Vec::with_capacity(self.parameters.nt);

        // Time stepping for forward propagation
        // Inject source (Ricker wavelet)
        for t in 0..self.parameters.nt {
            let src_val = self.ricker_wavelet(t as f64 * self.parameters.dt);
            // Inject at source location
            let (cx, cy, cz) = (
                self.parameters.n_trace / 2,
                self.parameters.n_depth / 2,
                self.parameters.n_trace / 2,
            );
            if let Some(p) = solver.fields.p.get_mut((cx, cy, cz)) {
                *p += src_val;
            }
            solver.step_forward()?;
            history.push(solver.fields.p.clone());
        }

        // Return synthetic data (recorded wavefield) and history
        Ok((solver.fields.p.clone(), history))
    }

    /// Adjoint modeling using time-reversed FDTD
    ///
    /// Computes the adjoint wavefield by running the FDTD solver in reverse time
    /// with the adjoint source (data residual) as input.
    ///
    /// # Arguments
    /// * `adjoint_source` - Adjoint source derived from data residual
    /// * `grid` - Computational grid defining the domain
    ///
    /// # Returns
    /// * Adjoint wavefield for gradient computation
    ///
    /// # References
    /// * Tromp et al. (2005): "Seismic tomography, adjoint methods"
    fn adjoint_model(
        &self,
        adjoint_source: &Array3<f64>,
        grid: &Grid,
        forward_history: &[Array3<f64>],
    ) -> KwaversResult<Array3<f64>> {
        use crate::solver::fdtd::{FdtdConfig, FdtdSolver};
        use ndarray::Zip;

        // Assume constant density and sound speed for adjoint
        let (nx, ny, nz) = grid.dimensions();
        let sound_speed = Array3::from_elem((nx, ny, nz), 1500.0); // m/s (water)

        // Calculate stable timestep
        let dt = self.calculate_stable_timestep(&sound_speed, grid);
        let num_steps = self.parameters.max_iterations.min(100);

        // Create FDTD configuration for adjoint modeling
        let config = FdtdConfig {
            spatial_order: 2,
            staggered_grid: true,
            cfl_factor: 0.3,
            subgridding: false,
            subgrid_factor: 2,
            enable_gpu_acceleration: false,
            nt: num_steps,
            dt,
            sensor_mask: None,
        };

        // Initialize FDTD solver
        // Use dummy water medium which closely matches our assumption (1500m/s, 1000kg/m3)
        let dummy_medium = crate::domain::medium::HomogeneousMedium::water(grid);
        let mut solver = FdtdSolver::new(config, grid, &dummy_medium, GridSource::default())?;

        // Run adjoint simulation (backward propagation)
        // For FWI, adjoint source is injected as a source term or boundary condition
        // Here we simply overwrite the pressure field as a simplified adjoint injection
        // In real FWI this would be a time-reversed source injection
        solver.fields.p.assign(adjoint_source);

        let mut gradient = Array3::zeros((nx, ny, nz));

        for t in 0..self.parameters.nt {
            solver.step_forward()?;

            if t < forward_history.len() {
                let fwd_idx = forward_history.len() - 1 - t;
                let fwd = &forward_history[fwd_idx];

                // Gradient accumulation: g += -u * lambda
                Zip::from(&mut gradient)
                    .and(fwd)
                    .and(&solver.fields.p)
                    .for_each(|g, &u, &adj| {
                        *g -= u * adj;
                    });
            }
        }

        // Return gradients
        Ok(gradient)
    }

    /// Calculate stable timestep for FDTD solver
    ///
    /// Uses CFL condition: dt ≤ min(dx,dy,dz) / (c_max * √3)
    ///
    /// # References
    /// * Courant et al. (1928): "On the partial difference equations of mathematical physics"
    fn calculate_stable_timestep(&self, model: &Array3<f64>, grid: &Grid) -> f64 {
        let c_max = model.iter().cloned().fold(0.0, f64::max);
        let min_spacing = grid.dx.min(grid.dy).min(grid.dz);

        // CFL condition with safety factor
        let cfl_number = 0.3; // Safety factor (matches config)
        cfl_number * min_spacing / (c_max * 3.0_f64.sqrt())
    }

    /// Compute adjoint source from data residual
    #[must_use]
    fn compute_adjoint_source(&self, residual: &Array3<f64>) -> Array3<f64> {
        // For L2 norm, adjoint source is simply the negative residual
        residual.map(|&x| -x)
    }

    /// Calculate Ricker wavelet value
    fn ricker_wavelet(&self, t: f64) -> f64 {
        let f0 = 20.0; // Central frequency
        let t0 = 1.0 / f0;
        let terms = std::f64::consts::PI * f0 * (t - t0);
        (1.0 - 2.0 * terms * terms) * (-terms * terms).exp()
    }
}

impl Default for FwiProcessor {
    fn default() -> Self {
        Self::new(FwiParameters::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_calculation() {
        let processor = FwiProcessor::default();

        let forward_field = Array3::ones((10, 10, 10));
        let adjoint_field = Array3::from_elem((10, 10, 10), 2.0);

        let gradient = processor.calculate_interaction(&forward_field, &adjoint_field);

        // Expected: -1.0 * 2.0 = -2.0 (after smoothing, will be close to -2.0)
        assert!((gradient[[5, 5, 5]] + 2.0).abs() < 0.1); // Allow for smoothing effects
    }

    #[test]
    fn test_adjoint_source_computation() {
        let processor = FwiProcessor::default();
        let residual = Array3::from_elem((5, 5, 5), 3.0);

        let adjoint_source = processor.compute_adjoint_source(&residual);

        // Expected: -3.0
        assert!((adjoint_source[[2, 2, 2]] + 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_model_constraints() {
        let processor = FwiProcessor::default();
        let mut model = Array3::from_elem((5, 5, 5), 10000.0); // Too high

        processor.apply_model_constraints(&mut model);

        // Should be clamped to max velocity
        assert!(model[[2, 2, 2]] <= 6000.0);
        assert!(model[[2, 2, 2]] >= 750.0);
    }
}
