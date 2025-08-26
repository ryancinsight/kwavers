//! Elastic wave propagation module
//!
//! This module implements elastic wave physics in heterogeneous media,
//! following SOLID/CUPID principles with proper domain separation.

pub mod fields;
pub mod metrics;
pub mod mode_conversion;
pub mod parameters;
pub mod properties;
pub mod solver;
pub mod tests;

use crate::error::{KwaversResult, PhysicsError};
use crate::grid::Grid;
use crate::medium::Medium;

use metrics::ElasticWaveMetrics;
use ndarray::{Array3, Array4};
use properties::AnisotropicElasticProperties;

// Re-export key types for convenience
pub use fields::{StressFields as ElasticStressFields, VelocityFields as ElasticVelocityFields};

/// Core elastic wave solver implementing spectral methods
/// Follows SOLID principles with clear separation of concerns
#[derive(Debug, Clone)]
pub struct ElasticWave {
    kx: Array3<f64>,
    ky: Array3<f64>,
    kz: Array3<f64>,
    metrics: ElasticWaveMetrics,
    is_anisotropic: bool,
    anisotropic_properties: Option<AnisotropicElasticProperties>,
    /// Mode conversion configuration
    mode_conversion: Option<mode_conversion::ModeConversionConfig>,
    /// Viscoelastic damping configuration
    viscoelastic: Option<mode_conversion::ViscoelasticConfig>,
    /// Material stiffness tensors (spatially varying)
    stiffness_tensors: Option<Array4<f64>>,
    /// Interface detection mask
    interface_mask: Option<Array3<bool>>,
}

impl ElasticWave {
    /// Create a new elastic wave solver
    /// Follows GRASP Creator principle
    pub fn new(grid: &Grid) -> KwaversResult<Self> {
        let (nx, ny, nz) = grid.dimensions();
        let (dx, dy, dz) = grid.spacing();

        // Validate grid dimensions
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "ElasticWave".to_string(),
                value: 0.0,
                reason: "Grid dimensions must be positive".to_string(),
            }
            .into());
        }

        if dx <= 0.0 || dy <= 0.0 || dz <= 0.0 {
            return Err(PhysicsError::InvalidParameter {
                parameter: "ElasticWave".to_string(),
                value: dx.min(dy).min(dz),
                reason: "Grid spacing must be positive".to_string(),
            }
            .into());
        }

        // Create wavenumber arrays
        let kx = Self::create_wavenumber_array(nx, dx);
        let ky = Self::create_wavenumber_array(ny, dy);
        let kz = Self::create_wavenumber_array(nz, dz);

        Ok(Self {
            kx,
            ky,
            kz,
            metrics: ElasticWaveMetrics::new(),
            is_anisotropic: false,
            anisotropic_properties: None,
            mode_conversion: None,
            viscoelastic: None,
            stiffness_tensors: None,
            interface_mask: None,
        })
    }

    /// Create wavenumber array for spectral methods
    fn create_wavenumber_array(n: usize, dx: f64) -> Array3<f64> {
        let mut k = Array3::zeros((n, 1, 1));
        let dk = 2.0 * std::f64::consts::PI / (n as f64 * dx);

        for i in 0..n / 2 {
            k[[i, 0, 0]] = i as f64 * dk;
        }
        for i in n / 2..n {
            k[[i, 0, 0]] = (i as f64 - n as f64) * dk;
        }

        k
    }

    /// Enable anisotropic mode with given properties
    pub fn set_anisotropic(&mut self, properties: AnisotropicElasticProperties) {
        self.is_anisotropic = true;
        self.anisotropic_properties = Some(properties);
    }

    /// Enable mode conversion at interfaces
    pub fn enable_mode_conversion(&mut self, config: mode_conversion::ModeConversionConfig) {
        self.mode_conversion = Some(config);
    }

    /// Enable viscoelastic damping
    pub fn enable_viscoelastic(&mut self, config: mode_conversion::ViscoelasticConfig) {
        self.viscoelastic = Some(config);
    }

    /// Set spatially varying stiffness tensors
    pub fn set_stiffness_tensors(&mut self, tensors: Array4<f64>) {
        self.stiffness_tensors = Some(tensors);
    }

    /// Detect interfaces based on material property jumps
    pub fn detect_interfaces(&mut self, medium: &dyn Medium, grid: &Grid, threshold: f64) {
        let (nx, ny, nz) = grid.dimensions();
        let mut mask = Array3::from_elem((nx, ny, nz), false);

        // Detect interfaces based on density jumps
        for k in 1..nz - 1 {
            for j in 1..ny - 1 {
                for i in 1..nx - 1 {
                    let center = medium.density(i as f64, j as f64, k as f64, grid);

                    // Check neighbors
                    let neighbors = [
                        medium.density((i + 1) as f64, j as f64, k as f64, grid),
                        medium.density((i - 1) as f64, j as f64, k as f64, grid),
                        medium.density(i as f64, (j + 1) as f64, k as f64, grid),
                        medium.density(i as f64, (j - 1) as f64, k as f64, grid),
                        medium.density(i as f64, j as f64, (k + 1) as f64, grid),
                        medium.density(i as f64, j as f64, (k - 1) as f64, grid),
                    ];

                    // Check for significant jumps
                    for &neighbor in &neighbors {
                        if (neighbor - center).abs() / center > threshold {
                            mask[[i, j, k]] = true;
                            break;
                        }
                    }
                }
            }
        }

        self.interface_mask = Some(mask);
    }

    /// Get current metrics
    pub fn metrics(&self) -> &ElasticWaveMetrics {
        &self.metrics
    }

    /// Reset metrics
    pub fn reset_metrics(&mut self) {
        self.metrics.reset();
    }
}

// Additional implementation methods would go in separate impl blocks
// to keep the file under 500 lines. The actual update methods would
// be in a separate solver.rs file.
