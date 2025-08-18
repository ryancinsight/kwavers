//! Type-safe physics factory for creating physics models
//!
//! This module provides a strongly-typed configuration system for physics models,
//! eliminating the error-prone string-based HashMap approach.

use crate::error::{KwaversResult, ConfigError};
use crate::physics::plugin::PluginManager;
use crate::grid::Grid;

// Import concrete configuration types from solver modules
// Note: These would need to be made public in their respective modules
// For now, we'll define placeholder versions here

/// Strongly-typed physics model configuration
#[derive(Debug, Clone)]
pub enum PhysicsModel {
    /// PSTD acoustic solver with its specific configuration
    Pstd(PstdConfig),
    /// FDTD acoustic solver with its specific configuration  
    Fdtd(FdtdConfig),
    /// Hybrid PSTD/FDTD solver
    Hybrid(HybridConfig),
    /// Kuznetsov nonlinear wave solver
    Kuznetsov(KuznetsovConfig),
    /// Thermal diffusion solver
    ThermalDiffusion(ThermalConfig),
    /// Elastic wave solver
    ElasticWave(ElasticConfig),
    /// Cavitation dynamics solver
    Cavitation(CavitationConfig),
    /// Light diffusion for photoacoustics
    LightDiffusion(LightConfig),
    /// Chemical kinetics solver
    Chemical(ChemicalConfig),
}

/// Main physics configuration holding a list of models
#[derive(Debug, Clone, Default)]
pub struct PhysicsConfig {
    /// List of physics models to instantiate
    pub models: Vec<PhysicsModel>,
    /// Global simulation frequency (Hz)
    pub frequency: f64,
}

impl PhysicsConfig {
    /// Create a new physics configuration
    pub fn new(frequency: f64) -> Self {
        Self {
            models: Vec::new(),
            frequency,
        }
    }
    
    /// Add a physics model to the configuration
    pub fn add_model(mut self, model: PhysicsModel) -> Self {
        self.models.push(model);
        self
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> KwaversResult<()> {
        if self.frequency <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "frequency".to_string(),
                value: self.frequency.to_string(),
                constraint: "Frequency must be positive".to_string(),
            }.into());
        }
        
        if self.models.is_empty() {
            return Err(ConfigError::InvalidValue {
                parameter: "models".to_string(),
                value: "empty".to_string(),
                constraint: "At least one physics model must be specified".to_string(),
            }.into());
        }
        
        // Validate individual model configurations
        for model in &self.models {
            model.validate()?;
        }
        
        Ok(())
    }
}

impl PhysicsModel {
    /// Validate model-specific configuration
    fn validate(&self) -> KwaversResult<()> {
        match self {
            PhysicsModel::Pstd(config) => config.validate(),
            PhysicsModel::Fdtd(config) => config.validate(),
            PhysicsModel::Hybrid(config) => config.validate(),
            PhysicsModel::Kuznetsov(config) => config.validate(),
            PhysicsModel::ThermalDiffusion(config) => config.validate(),
            PhysicsModel::ElasticWave(config) => config.validate(),
            PhysicsModel::Cavitation(config) => config.validate(),
            PhysicsModel::LightDiffusion(config) => config.validate(),
            PhysicsModel::Chemical(config) => config.validate(),
        }
    }
}

/// PSTD solver configuration
#[derive(Debug, Clone)]
pub struct PstdConfig {
    /// CFL safety factor (0 < cfl <= 1)
    pub cfl_factor: f64,
    /// Number of spatial harmonics
    pub harmonics: usize,
    /// Use k-space correction
    pub k_space_correction: bool,
    /// PML thickness in grid points
    pub pml_thickness: usize,
}

impl Default for PstdConfig {
    fn default() -> Self {
        Self {
            cfl_factor: 0.3,
            harmonics: 128,
            k_space_correction: true,
            pml_thickness: 10,
        }
    }
}

impl PstdConfig {
    fn validate(&self) -> KwaversResult<()> {
        if self.cfl_factor <= 0.0 || self.cfl_factor > 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "cfl_factor".to_string(),
                value: self.cfl_factor.to_string(),
                constraint: "CFL factor must be in (0, 1]".to_string(),
            }.into());
        }
        if self.harmonics == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "harmonics".to_string(),
                value: self.harmonics.to_string(),
                constraint: "Number of harmonics must be > 0".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// FDTD solver configuration
#[derive(Debug, Clone)]
pub struct FdtdConfig {
    /// Spatial accuracy order (2, 4, or 6)
    pub spatial_order: usize,
    /// CFL safety factor (0 < cfl <= 1)
    pub cfl_factor: f64,
    /// Use staggered grid (Yee cell)
    pub staggered_grid: bool,
    /// Boundary condition type
    pub boundary: BoundaryType,
}

#[derive(Debug, Clone)]
pub enum BoundaryType {
    Pml { thickness: usize },
    Cpml { thickness: usize, alpha_max: f64 },
    Periodic,
    Dirichlet,
    Neumann,
}

impl Default for FdtdConfig {
    fn default() -> Self {
        Self {
            spatial_order: 4,
            cfl_factor: 0.5,
            staggered_grid: true,
            boundary: BoundaryType::Cpml { 
                thickness: 10,
                alpha_max: 0.24,
            },
        }
    }
}

impl FdtdConfig {
    fn validate(&self) -> KwaversResult<()> {
        if ![2, 4, 6].contains(&self.spatial_order) {
            return Err(ConfigError::InvalidValue {
                parameter: "spatial_order".to_string(),
                value: self.spatial_order.to_string(),
                constraint: "Spatial order must be 2, 4, or 6".to_string(),
            }.into());
        }
        if self.cfl_factor <= 0.0 || self.cfl_factor > 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "cfl_factor".to_string(),
                value: self.cfl_factor.to_string(),
                constraint: "CFL factor must be in (0, 1]".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// Hybrid PSTD/FDTD configuration
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// PSTD configuration for smooth regions
    pub pstd: PstdConfig,
    /// FDTD configuration for complex boundaries
    pub fdtd: FdtdConfig,
    /// Threshold for method selection
    pub complexity_threshold: f64,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            pstd: PstdConfig::default(),
            fdtd: FdtdConfig::default(),
            complexity_threshold: 0.3,
        }
    }
}

impl HybridConfig {
    fn validate(&self) -> KwaversResult<()> {
        self.pstd.validate()?;
        self.fdtd.validate()?;
        if self.complexity_threshold < 0.0 || self.complexity_threshold > 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "complexity_threshold".to_string(),
                value: self.complexity_threshold.to_string(),
                constraint: "Complexity threshold must be in [0, 1]".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// Kuznetsov nonlinear wave configuration
#[derive(Debug, Clone)]
pub struct KuznetsovConfig {
    /// Nonlinearity parameter B/A
    pub nonlinearity_b_a: f64,
    /// Include thermoviscous losses
    pub thermoviscous: bool,
    /// Diffusivity of sound
    pub diffusivity: f64,
}

impl Default for KuznetsovConfig {
    fn default() -> Self {
        Self {
            nonlinearity_b_a: 5.0,  // Water
            thermoviscous: true,
            diffusivity: 4.5e-6,    // Water at 20°C
        }
    }
}

impl KuznetsovConfig {
    fn validate(&self) -> KwaversResult<()> {
        if self.nonlinearity_b_a < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "nonlinearity_b_a".to_string(),
                value: self.nonlinearity_b_a.to_string(),
                constraint: "B/A parameter must be non-negative".to_string(),
            }.into());
        }
        if self.diffusivity < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "diffusivity".to_string(),
                value: self.diffusivity.to_string(),
                constraint: "Diffusivity must be non-negative".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// Thermal diffusion configuration
#[derive(Debug, Clone)]
pub struct ThermalConfig {
    /// Thermal conductivity (W/(m·K))
    pub conductivity: f64,
    /// Specific heat capacity (J/(kg·K))
    pub specific_heat: f64,
    /// Include perfusion term
    pub perfusion: bool,
    /// Perfusion rate (1/s)
    pub perfusion_rate: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            conductivity: 0.5,      // Soft tissue
            specific_heat: 3600.0,  // Soft tissue
            perfusion: true,
            perfusion_rate: 0.01,   // Typical tissue
        }
    }
}

impl ThermalConfig {
    fn validate(&self) -> KwaversResult<()> {
        if self.conductivity <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "conductivity".to_string(),
                value: self.conductivity.to_string(),
                constraint: "Thermal conductivity must be positive".to_string(),
            }.into());
        }
        if self.specific_heat <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "specific_heat".to_string(),
                value: self.specific_heat.to_string(),
                constraint: "Specific heat must be positive".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// Elastic wave configuration
#[derive(Debug, Clone)]
pub struct ElasticConfig {
    /// Young's modulus (Pa)
    pub youngs_modulus: f64,
    /// Poisson's ratio
    pub poisson_ratio: f64,
    /// Include attenuation
    pub attenuation: bool,
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            youngs_modulus: 1e9,   // Soft tissue
            poisson_ratio: 0.45,   // Nearly incompressible
            attenuation: true,
        }
    }
}

impl ElasticConfig {
    fn validate(&self) -> KwaversResult<()> {
        if self.youngs_modulus <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "youngs_modulus".to_string(),
                value: self.youngs_modulus.to_string(),
                constraint: "Young's modulus must be positive".to_string(),
            }.into());
        }
        if self.poisson_ratio < -1.0 || self.poisson_ratio >= 0.5 {
            return Err(ConfigError::InvalidValue {
                parameter: "poisson_ratio".to_string(),
                value: self.poisson_ratio.to_string(),
                constraint: "Poisson's ratio must be in (-1, 0.5)".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// Cavitation dynamics configuration
#[derive(Debug, Clone)]
pub struct CavitationConfig {
    /// Initial bubble radius (m)
    pub initial_radius: f64,
    /// Surface tension (N/m)
    pub surface_tension: f64,
    /// Vapor pressure (Pa)
    pub vapor_pressure: f64,
}

impl Default for CavitationConfig {
    fn default() -> Self {
        Self {
            initial_radius: 1e-6,   // 1 micron
            surface_tension: 0.072, // Water-air
            vapor_pressure: 2338.0, // Water at 20°C
        }
    }
}

impl CavitationConfig {
    fn validate(&self) -> KwaversResult<()> {
        if self.initial_radius <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "initial_radius".to_string(),
                value: self.initial_radius.to_string(),
                constraint: "Initial radius must be positive".to_string(),
            }.into());
        }
        if self.surface_tension < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "surface_tension".to_string(),
                value: self.surface_tension.to_string(),
                constraint: "Surface tension must be non-negative".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// Light diffusion configuration for photoacoustics
#[derive(Debug, Clone)]
pub struct LightConfig {
    /// Optical absorption coefficient (1/m)
    pub absorption: f64,
    /// Reduced scattering coefficient (1/m)
    pub scattering: f64,
    /// Anisotropy factor
    pub anisotropy: f64,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            absorption: 10.0,   // Typical tissue
            scattering: 1000.0, // Typical tissue
            anisotropy: 0.9,    // Forward scattering
        }
    }
}

impl LightConfig {
    fn validate(&self) -> KwaversResult<()> {
        if self.absorption < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "absorption".to_string(),
                value: self.absorption.to_string(),
                constraint: "Absorption coefficient must be non-negative".to_string(),
            }.into());
        }
        if self.scattering < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "scattering".to_string(),
                value: self.scattering.to_string(),
                constraint: "Scattering coefficient must be non-negative".to_string(),
            }.into());
        }
        if self.anisotropy < -1.0 || self.anisotropy > 1.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "anisotropy".to_string(),
                value: self.anisotropy.to_string(),
                constraint: "Anisotropy factor must be in [-1, 1]".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// Chemical kinetics configuration
#[derive(Debug, Clone)]
pub struct ChemicalConfig {
    /// Reaction rate constant (1/s)
    pub rate_constant: f64,
    /// Diffusion coefficient (m²/s)
    pub diffusion: f64,
    /// Initial concentration (mol/m³)
    pub initial_concentration: f64,
}

impl Default for ChemicalConfig {
    fn default() -> Self {
        Self {
            rate_constant: 0.1,
            diffusion: 1e-9,
            initial_concentration: 1.0,
        }
    }
}

impl ChemicalConfig {
    fn validate(&self) -> KwaversResult<()> {
        if self.rate_constant < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "rate_constant".to_string(),
                value: self.rate_constant.to_string(),
                constraint: "Rate constant must be non-negative".to_string(),
            }.into());
        }
        if self.diffusion < 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "diffusion".to_string(),
                value: self.diffusion.to_string(),
                constraint: "Diffusion coefficient must be non-negative".to_string(),
            }.into());
        }
        Ok(())
    }
}

/// Type-safe physics factory
pub struct PhysicsFactory;

impl PhysicsFactory {
    /// Create physics models from strongly-typed configuration
    pub fn create_physics(config: &PhysicsConfig, grid: &Grid) -> KwaversResult<PluginManager> {
        config.validate()?;
        
        let mut manager = PluginManager::new();
        
        for model in &config.models {
            match model {
                PhysicsModel::Pstd(pstd_config) => {
                    // Create PSTD plugin with specific configuration
                    // let plugin = crate::solver::pstd::PstdPlugin::new(pstd_config.clone(), grid)?;
                    // manager.register(Box::new(plugin))?;
                    
                    // Placeholder until solver modules are refactored
                    log::info!("Would create PSTD solver with config: {:?}", pstd_config);
                }
                
                PhysicsModel::Fdtd(fdtd_config) => {
                    // Create FDTD plugin with specific configuration
                    // let plugin = crate::solver::fdtd::FdtdPlugin::new(fdtd_config.clone(), grid)?;
                    // manager.register(Box::new(plugin))?;
                    
                    log::info!("Would create FDTD solver with config: {:?}", fdtd_config);
                }
                
                PhysicsModel::Hybrid(hybrid_config) => {
                    // Create Hybrid plugin
                    // let plugin = crate::solver::hybrid::HybridPlugin::new(hybrid_config.clone(), grid)?;
                    // manager.register(Box::new(plugin))?;
                    
                    log::info!("Would create Hybrid solver with config: {:?}", hybrid_config);
                }
                
                PhysicsModel::Kuznetsov(kuznetsov_config) => {
                    // Create Kuznetsov plugin
                    log::info!("Would create Kuznetsov solver with config: {:?}", kuznetsov_config);
                }
                
                PhysicsModel::ThermalDiffusion(thermal_config) => {
                    // Create thermal diffusion plugin
                    log::info!("Would create Thermal solver with config: {:?}", thermal_config);
                }
                
                PhysicsModel::ElasticWave(elastic_config) => {
                    // Create elastic wave plugin
                    log::info!("Would create Elastic solver with config: {:?}", elastic_config);
                }
                
                PhysicsModel::Cavitation(cavitation_config) => {
                    // Create cavitation plugin
                    log::info!("Would create Cavitation solver with config: {:?}", cavitation_config);
                }
                
                PhysicsModel::LightDiffusion(light_config) => {
                    // Create light diffusion plugin
                    log::info!("Would create Light diffusion solver with config: {:?}", light_config);
                }
                
                PhysicsModel::Chemical(chemical_config) => {
                    // Create chemical kinetics plugin
                    log::info!("Would create Chemical solver with config: {:?}", chemical_config);
                }
            }
        }
        
        Ok(manager)
    }
}

/// Builder pattern for convenient configuration
impl PhysicsConfig {
    /// Add a PSTD solver to the configuration
    pub fn with_pstd(mut self, config: PstdConfig) -> Self {
        self.models.push(PhysicsModel::Pstd(config));
        self
    }
    
    /// Add an FDTD solver to the configuration
    pub fn with_fdtd(mut self, config: FdtdConfig) -> Self {
        self.models.push(PhysicsModel::Fdtd(config));
        self
    }
    
    /// Add a hybrid solver to the configuration
    pub fn with_hybrid(mut self, config: HybridConfig) -> Self {
        self.models.push(PhysicsModel::Hybrid(config));
        self
    }
    
    /// Add a Kuznetsov solver to the configuration
    pub fn with_kuznetsov(mut self, config: KuznetsovConfig) -> Self {
        self.models.push(PhysicsModel::Kuznetsov(config));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_type_safe_config() {
        // Create a configuration with specific solver types
        let config = PhysicsConfig::new(1e6)
            .with_pstd(PstdConfig {
                cfl_factor: 0.3,
                harmonics: 256,
                k_space_correction: true,
                pml_thickness: 15,
            })
            .with_fdtd(FdtdConfig {
                spatial_order: 6,
                cfl_factor: 0.4,
                staggered_grid: true,
                boundary: BoundaryType::Cpml {
                    thickness: 20,
                    alpha_max: 0.3,
                },
            });
        
        assert_eq!(config.models.len(), 2);
        assert!(config.validate().is_ok());
        
        // Type safety: these would be compile-time errors
        // config.models[0].spatial_order; // Error: PSTD doesn't have spatial_order
        // config.models[1].harmonics;     // Error: FDTD doesn't have harmonics
    }
    
    #[test]
    fn test_validation() {
        // Invalid CFL factor
        let pstd = PstdConfig {
            cfl_factor: 1.5, // Invalid: > 1.0
            ..Default::default()
        };
        assert!(pstd.validate().is_err());
        
        // Invalid spatial order
        let fdtd = FdtdConfig {
            spatial_order: 3, // Invalid: not 2, 4, or 6
            ..Default::default()
        };
        assert!(fdtd.validate().is_err());
    }
}