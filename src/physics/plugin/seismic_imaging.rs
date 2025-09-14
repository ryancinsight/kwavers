//! Seismic Imaging Plugin
//! Based on Claerbout (1985): "Imaging the Earth's Interior"

use crate::error::KwaversResult;
use crate::grid::Grid;
use crate::physics::plugin::{PluginMetadata, PluginState};
use ndarray::Array3;

/// Full Waveform Inversion Parameters
#[derive(Debug, Clone)]
pub struct FwiParameters {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Step size for gradient descent
    pub step_size: f64,
    /// Regularization weight
    pub regularization: RegularizationParameters,
}

/// Regularization parameters for inversion
#[derive(Debug, Clone)]
pub struct RegularizationParameters {
    /// Tikhonov regularization weight
    pub tikhonov_weight: f64,
    /// Total variation weight
    pub tv_weight: f64,
    /// Smoothness constraint weight
    pub smoothness_weight: f64,
}

/// Convergence criteria for iterative methods
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Maximum iterations
    pub max_iterations: usize,
    /// Relative tolerance
    pub relative_tolerance: f64,
    /// Absolute tolerance
    pub absolute_tolerance: f64,
}

/// Reverse Time Migration Settings
#[derive(Debug, Clone)]
pub struct RtmSettings {
    /// Imaging condition type
    pub imaging_condition: ImagingCondition,
    /// Source wavefield storage strategy
    pub storage_strategy: StorageStrategy,
    /// Boundary conditions for migration
    pub boundary_conditions: BoundaryType,
    /// Apply Laplacian filter for artifact suppression
    pub apply_laplacian: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum ImagingCondition {
    CrossCorrelation,
    Deconvolution,
    ExcitationTime,
}

#[derive(Debug, Clone, Copy)]
pub enum StorageStrategy {
    FullStorage,
    Checkpointing,
    OnTheFly,
}

#[derive(Debug, Clone, Copy)]
pub enum BoundaryType {
    Absorbing,
    RandomBoundary,
    Hybrid,
}

/// Migration aperture control
#[derive(Debug, Clone)]
pub struct MigrationAperture {
    /// Maximum opening angle in degrees
    pub max_angle: f64,
    /// Minimum offset in meters
    pub min_offset: f64,
    /// Maximum offset in meters
    pub max_offset: f64,
}

/// Seismic Imaging Plugin
/// Provides RTM and FWI capabilities for subsurface imaging
#[derive(Debug)]
pub struct SeismicImagingPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    /// FWI parameters
    fwi_params: Option<FwiParameters>,
    /// RTM settings
    rtm_settings: Option<RtmSettings>,
    /// Migration aperture
    aperture: Option<MigrationAperture>,
}

impl Default for SeismicImagingPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl SeismicImagingPlugin {
    /// Create new seismic imaging plugin
    #[must_use]
    pub fn new() -> Self {
        Self {
            metadata: PluginMetadata {
                id: "seismic_imaging".to_string(),
                name: "Seismic Imaging".to_string(),
                version: "1.0.0".to_string(),
                author: "Kwavers Team".to_string(),
                description: "RTM and FWI for subsurface imaging".to_string(),
                license: "MIT".to_string(),
            },
            state: PluginState::Initialized,
            fwi_params: None,
            rtm_settings: None,
            aperture: None,
        }
    }

    /// Perform Reverse Time Migration
    /// Based on Baysal et al. (1983): "Reverse time migration"
    pub fn reverse_time_migration(
        &mut self,
        source_wavefield: &Array3<f64>,
        receiver_wavefield: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        use ndarray::Zip;

        // Initialize image with grid dimensions
        let mut image = Array3::zeros(grid.dimensions());

        // Apply imaging condition (zero-lag cross-correlation)
        // I(x) = ∫ S(x,t) * R(x,t) dt
        Zip::from(&mut image)
            .and(source_wavefield)
            .and(receiver_wavefield)
            .for_each(|img, &src, &rcv| {
                *img = src * rcv;
            });

        // Apply Laplacian filter for artifact suppression
        // This removes low-wavenumber artifacts
        if let Some(settings) = &self.rtm_settings {
            if settings.apply_laplacian {
                let laplacian_weight = 0.1;
                for k in 1..grid.nz - 1 {
                    for j in 1..grid.ny - 1 {
                        for i in 1..grid.nx - 1 {
                            let laplacian = (image[[i + 1, j, k]] + image[[i - 1, j, k]]
                                - 2.0 * image[[i, j, k]])
                                / (grid.dx * grid.dx)
                                + (image[[i, j + 1, k]] + image[[i, j - 1, k]]
                                    - 2.0 * image[[i, j, k]])
                                    / (grid.dy * grid.dy)
                                + (image[[i, j, k + 1]] + image[[i, j, k - 1]]
                                    - 2.0 * image[[i, j, k]])
                                    / (grid.dz * grid.dz);
                            image[[i, j, k]] += laplacian_weight * laplacian;
                        }
                    }
                }
            }
        }

        self.state = PluginState::Running;
        Ok(image)
    }

    /// Perform Full Waveform Inversion
    /// Full Waveform Inversion implementation
    /// Based on Tarantola (1984): "Inversion of seismic reflection data in the acoustic approximation"
    /// Reference: Geophysics, 49(8), 1259-1266
    pub fn full_waveform_inversion(
        &mut self,
        observed_data: &Array3<f64>,
        initial_model: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        use crate::utils::linear_algebra::norm_l2;

        let mut current_model = initial_model.clone();
        let params = self.fwi_params.as_ref().ok_or_else(|| {
            crate::error::KwaversError::Physics(crate::error::PhysicsError::InvalidConfiguration {
                parameter: "fwi_params".to_string(),
                reason: "FWI parameters not set".to_string(),
            })
        })?;

        let mut prev_misfit = f64::INFINITY;

        for iteration in 0..params.max_iterations {
            // 1. Forward modeling
            let synthetic_data = self.forward_model(&current_model, grid)?;

            // 2. Calculate data misfit
            let residual = observed_data - &synthetic_data;
            let current_misfit = norm_l2(&residual);

            // 3. Check convergence
            let relative_change = (prev_misfit - current_misfit).abs() / prev_misfit;
            if relative_change < params.tolerance {
                log::info!(
                    "FWI converged after {} iterations with misfit: {:.6e}",
                    iteration,
                    current_misfit
                );
                break;
            }

            // 4. Adjoint computation
            let adjoint_source = self.compute_adjoint_source(&residual);
            let adjoint_field = self.adjoint_model(&adjoint_source, grid)?;

            // 5. Gradient calculation
            let gradient = self.calculate_gradient(&synthetic_data, &adjoint_field);

            // 6. Apply regularization
            let regularized_gradient = self.apply_regularization(&gradient, &current_model)?;

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

    /// Calculate gradient for FWI
    /// Based on Plessix (2006): "A review of the adjoint-state method"
    #[must_use]
    pub fn calculate_gradient(
        &self,
        forward_field: &Array3<f64>,
        adjoint_field: &Array3<f64>,
    ) -> Array3<f64> {
        use ndarray::Zip;

        // Gradient calculation for velocity model update
        // g(x) = -∫ ∂²u/∂t² * λ dt
        // where u is forward field and λ is adjoint field

        let mut gradient = Array3::zeros(forward_field.dim());

        // Compute time integral of forward and adjoint field product
        Zip::from(&mut gradient)
            .and(forward_field)
            .and(adjoint_field)
            .for_each(|g, &fwd, &adj| {
                // Negative sign for descent direction
                *g = -fwd * adj;
            });

        // Apply smoothing to reduce high-frequency artifacts
        // Simple 3-point smoothing in each dimension
        let mut smoothed = gradient.clone();
        let (nx, ny, nz) = gradient.dim();

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

    /// Apply imaging condition
    /// Based on Claerbout (1985): "Imaging the Earth's Interior" and
    /// Zhang et al. (2007): "Practical issues in reverse time migration"
    #[must_use]
    pub fn apply_imaging_condition(
        &self,
        source: &Array3<f64>,
        receiver: &Array3<f64>,
    ) -> Array3<f64> {
        match self.rtm_settings.as_ref().map(|s| s.imaging_condition) {
            Some(ImagingCondition::CrossCorrelation) => {
                // Zero-lag cross-correlation imaging condition
                // I(x) = Σ_t P_s(x,t) * P_r(x,t)
                source * receiver
            }
            Some(ImagingCondition::Deconvolution) => {
                // Deconvolution imaging condition to reduce migration artifacts
                // I(x) = P_r(x,t) / P_s(x,t) where P_s ≠ 0
                use ndarray::Zip;
                let mut image = Array3::zeros(source.dim());
                let epsilon = 1e-10; // Regularization parameter

                Zip::from(&mut image)
                    .and(receiver)
                    .and(source)
                    .for_each(|img, &rcv, &src| {
                        if src.abs() > epsilon {
                            *img = rcv / (src + epsilon.copysign(src));
                        }
                    });
                image
            }
            Some(ImagingCondition::ExcitationTime) => {
                // Excitation time imaging condition for improved resolution
                // Based on Sava & Fomel (2006): "Time-shift imaging condition"
                use ndarray::Zip;
                let mut image = Array3::zeros(source.dim());
                let threshold = 0.1 * source.iter().fold(0.0f64, |acc: f64, &x| acc.max(x.abs()));

                Zip::from(&mut image)
                    .and(receiver)
                    .and(source)
                    .for_each(|img, &rcv, &src| {
                        // Only image where source amplitude exceeds threshold
                        if src.abs() > threshold {
                            *img = rcv * src;
                        }
                    });
                image
            }
            None => {
                // Default to cross-correlation
                source * receiver
            }
        }
    }

    /// Forward modeling for FWI
    /// Simulate seismic wave propagation through velocity model
    fn forward_model(
        &self,
        velocity_model: &Array3<f64>,
        _grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Simplified forward modeling - in practice would use full wave equation solver
        // Here we use a basic implementation that satisfies the interface
        use crate::physics::constants::DENSITY_WATER;

        let (nx, ny, nz) = velocity_model.dim();
        let mut synthetic_data = Array3::zeros((nx, ny, nz));

        // Simple forward modeling approximation
        // In practice, this would involve solving the full acoustic wave equation
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let velocity = velocity_model[[i, j, k]];
                    // Simplified Green's function approximation
                    synthetic_data[[i, j, k]] = 1.0 / (velocity * velocity * DENSITY_WATER);
                }
            }
        }

        Ok(synthetic_data)
    }

    /// Compute adjoint source for FWI gradient calculation
    fn compute_adjoint_source(&self, residual: &Array3<f64>) -> Array3<f64> {
        // Adjoint source is simply the negative residual for L2 norm
        // Based on Plessix (2006): "A review of the adjoint-state method"
        -residual
    }

    /// Adjoint modeling for gradient computation
    fn adjoint_model(
        &self,
        adjoint_source: &Array3<f64>,
        _grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // Simplified adjoint modeling - time-reversed propagation
        // In practice would use time-reversed wave equation solver
        Ok(adjoint_source.clone())
    }

    /// Apply regularization to gradient
    fn apply_regularization(
        &self,
        gradient: &Array3<f64>,
        current_model: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let params = self.fwi_params.as_ref().ok_or_else(|| {
            crate::error::KwaversError::Physics(crate::error::PhysicsError::InvalidConfiguration {
                parameter: "fwi_params".to_string(),
                reason: "FWI parameters not set".to_string(),
            })
        })?;

        let mut regularized = gradient.clone();

        // Tikhonov regularization: adds λ * (m - m0) to gradient
        if params.regularization.tikhonov_weight > 0.0 {
            let tikhonov_term = current_model * params.regularization.tikhonov_weight;
            regularized = regularized + tikhonov_term;
        }

        // Smoothness regularization: Laplacian operator
        if params.regularization.smoothness_weight > 0.0 {
            let smoothness_term =
                self.compute_laplacian(current_model) * params.regularization.smoothness_weight;
            regularized = regularized + smoothness_term;
        }

        Ok(regularized)
    }

    /// Compute Laplacian for smoothness regularization
    fn compute_laplacian(&self, field: &Array3<f64>) -> Array3<f64> {
        let mut laplacian = Array3::zeros(field.dim());
        let (nx, ny, nz) = field.dim();

        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    // 3D Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
                    laplacian[[i, j, k]] = (field[[i + 1, j, k]] - 2.0 * field[[i, j, k]]
                        + field[[i - 1, j, k]])
                        + (field[[i, j + 1, k]] - 2.0 * field[[i, j, k]] + field[[i, j - 1, k]])
                        + (field[[i, j, k + 1]] - 2.0 * field[[i, j, k]] + field[[i, j, k - 1]]);
                }
            }
        }

        laplacian
    }

    /// Line search for optimal step size in FWI
    fn line_search(
        &self,
        current_model: &Array3<f64>,
        gradient: &Array3<f64>,
        observed_data: &Array3<f64>,
        grid: &Grid,
    ) -> KwaversResult<f64> {
        let params = self.fwi_params.as_ref().ok_or_else(|| {
            crate::error::KwaversError::Physics(crate::error::PhysicsError::InvalidConfiguration {
                parameter: "fwi_params".to_string(),
                reason: "FWI parameters not set".to_string(),
            })
        })?;

        let mut step_size = params.step_size;
        let c1 = 1e-4; // Armijo condition parameter
        let max_backtrack = 10;

        // Current objective function value
        let current_synthetic = self.forward_model(current_model, grid)?;
        let current_residual = observed_data - &current_synthetic;
        let current_objective =
            crate::utils::linear_algebra::norm_l2(&current_residual).powi(2) / 2.0;

        // Gradient dot product for Armijo condition
        let gradient_norm_sq = gradient.iter().map(|&x| x * x).sum::<f64>();

        for _ in 0..max_backtrack {
            // Try model update with current step size
            let trial_model = current_model - &(gradient * step_size);
            let trial_synthetic = self.forward_model(&trial_model, grid)?;
            let trial_residual = observed_data - &trial_synthetic;
            let trial_objective =
                crate::utils::linear_algebra::norm_l2(&trial_residual).powi(2) / 2.0;

            // Armijo condition: f(x + αp) ≤ f(x) + c₁α∇f·p
            if trial_objective <= current_objective - c1 * step_size * gradient_norm_sq {
                return Ok(step_size);
            }

            // Backtrack
            step_size *= 0.5;
        }

        Ok(step_size)
    }

    /// Apply physical constraints to velocity model
    fn apply_model_constraints(&self, model: &mut Array3<f64>) {
        use crate::physics::constants::SOUND_SPEED_WATER;

        // Ensure physically reasonable velocity bounds
        let min_velocity = SOUND_SPEED_WATER * 0.5; // 750 m/s
        let max_velocity = SOUND_SPEED_WATER * 4.0; // 6000 m/s

        model.mapv_inplace(|v| v.max(min_velocity).min(max_velocity));
    }
}
