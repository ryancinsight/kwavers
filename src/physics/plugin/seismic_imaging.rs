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
    /// Based on Tarantola (1984): "Inversion of seismic reflection data"
    pub fn full_waveform_inversion(
        &mut self,
        _observed_data: &Array3<f64>,
        initial_model: &Array3<f64>,
        _grid: &Grid,
    ) -> KwaversResult<Array3<f64>> {
        // TODO: Implement FWI
        // This should include:
        // 1. Forward modeling
        // 2. Adjoint computation
        // 3. Gradient calculation
        // 4. Model update with regularization

        Ok(initial_model.clone())
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
    #[must_use]
    pub fn apply_imaging_condition(
        &self,
        source: &Array3<f64>,
        receiver: &Array3<f64>,
    ) -> Array3<f64> {
        // TODO: Implement various imaging conditions
        match self.rtm_settings.as_ref().map(|s| s.imaging_condition) {
            Some(ImagingCondition::CrossCorrelation) => {
                // Zero-lag cross-correlation
                source * receiver
            }
            Some(ImagingCondition::Deconvolution) => {
                // Deconvolution imaging condition
                Array3::zeros(source.dim())
            }
            Some(ImagingCondition::ExcitationTime) => {
                // Excitation time imaging condition
                Array3::zeros(source.dim())
            }
            None => Array3::zeros(source.dim()),
        }
    }
}
