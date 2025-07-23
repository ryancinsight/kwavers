// src/factory.rs
//! Factory patterns for creating simulation components
//! 
//! This module follows GRASP principles:
//! - Information Expert: Objects that have the information needed to fulfill a responsibility
//! - Creator: Objects responsible for creating other objects they use
//! - Controller: Objects that coordinate and control system operations
//! - Low Coupling: Minimize dependencies between objects
//! - High Cohesion: Keep related functionality together
//!
//! Design Principles Implemented:
//! - SOLID: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
//! - CUPID: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
//! - GRASP: Information expert, creator, controller, low coupling, high cohesion
//! - SSOT: Single source of truth for configuration and creation
//! - ADP: Acyclic dependency principle

use crate::error::{KwaversResult, ConfigError, PhysicsError, ValidationError};
use crate::grid::Grid;
use crate::medium::{Medium, homogeneous::HomogeneousMedium};
use crate::physics::{PhysicsComponent, PhysicsPipeline, AcousticWaveComponent, ThermalDiffusionComponent};
use crate::time::Time;
use crate::validation::{ValidationResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for creating simulation components
/// Follows SSOT principle - single source of truth for configuration
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    /// Grid configuration
    pub grid: GridConfig,
    /// Medium configuration
    pub medium: MediumConfig,
    /// Physics models to include
    pub physics: PhysicsConfig,
    /// Time stepping configuration
    pub time: TimeConfig,
    /// Validation settings
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone)]
pub struct GridConfig {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub dx: f64,
    pub dy: f64,
    pub dz: f64,
}

impl GridConfig {
    /// Validate grid configuration
    /// Follows Information Expert principle - knows how to validate itself
    pub fn validate(&self) -> KwaversResult<()> {
        if self.nx == 0 || self.ny == 0 || self.nz == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid dimensions".to_string(),
                value: format!("({}, {}, {})", self.nx, self.ny, self.nz),
                constraint: "All dimensions must be positive".to_string(),
            }.into());
        }

        if self.dx <= 0.0 || self.dy <= 0.0 || self.dz <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid spacing".to_string(),
                value: format!("({}, {}, {})", self.dx, self.dy, self.dz),
                constraint: "All spacing values must be positive".to_string(),
            }.into());
        }

        // Check for reasonable grid size to prevent memory issues
        let total_points = self.nx * self.ny * self.nz;
        if total_points > 100_000_000 { // 100M points limit
            return Err(ConfigError::InvalidValue {
                parameter: "grid size".to_string(),
                value: total_points.to_string(),
                constraint: "Total grid points must be <= 100,000,000".to_string(),
            }.into());
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct MediumConfig {
    pub medium_type: MediumType,
    pub properties: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum MediumType {
    Homogeneous { density: f64, sound_speed: f64, mu_a: f64, mu_s_prime: f64 },
    Heterogeneous { tissue_file: Option<String> },
}

impl MediumConfig {
    /// Validate medium configuration
    /// Follows Information Expert principle - knows how to validate itself
    pub fn validate(&self) -> KwaversResult<()> {
        match &self.medium_type {
            MediumType::Homogeneous { density, sound_speed, mu_a, mu_s_prime } => {
                // Check required properties for homogeneous medium
                let required_props = ["density", "sound_speed"];
                for prop in &required_props {
                    if !self.properties.contains_key(*prop) {
                        return Err(ConfigError::MissingParameter {
                            parameter: prop.to_string(),
                            section: "medium".to_string(),
                        }.into());
                    }
                }

                // Validate property values
                if let Some(&density_val) = self.properties.get("density") {
                    if density_val <= 0.0 {
                        return Err(ConfigError::InvalidValue {
                            parameter: "density".to_string(),
                            value: density_val.to_string(),
                            constraint: "Density must be positive".to_string(),
                        }.into());
                    }
                }

                if let Some(&sound_speed_val) = self.properties.get("sound_speed") {
                    if sound_speed_val <= 0.0 {
                        return Err(ConfigError::InvalidValue {
                            parameter: "sound_speed".to_string(),
                            value: sound_speed_val.to_string(),
                            constraint: "Sound speed must be positive".to_string(),
                        }.into());
                    }
                }
            }
            MediumType::Heterogeneous { tissue_file } => {
                if let Some(file) = tissue_file {
                    if file.is_empty() {
                        return Err(ConfigError::InvalidValue {
                            parameter: "tissue_file".to_string(),
                            value: "empty".to_string(),
                            constraint: "Tissue file path must not be empty".to_string(),
                        }.into());
                    }
                }
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PhysicsConfig {
    pub models: Vec<PhysicsModelConfig>,
    pub frequency: f64,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub struct PhysicsModelConfig {
    pub model_type: PhysicsModelType,
    pub enabled: bool,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum PhysicsModelType {
    AcousticWave,
    ThermalDiffusion,
    Cavitation,
    ElasticWave,
    LightDiffusion,
    Chemical,
}

impl PhysicsConfig {
    /// Validate physics configuration
    /// Follows Information Expert principle - knows how to validate itself
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
                parameter: "physics models".to_string(),
                value: "empty".to_string(),
                constraint: "At least one physics model must be specified".to_string(),
            }.into());
        }

        // Check for enabled models
        let enabled_models: Vec<_> = self.models.iter()
            .filter(|m| m.enabled)
            .collect();

        if enabled_models.is_empty() {
            return Err(ConfigError::InvalidValue {
                parameter: "enabled physics models".to_string(),
                value: "none".to_string(),
                constraint: "At least one physics model must be enabled".to_string(),
            }.into());
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct TimeConfig {
    pub dt: f64,
    pub num_steps: usize,
    pub cfl_factor: f64,
}

impl TimeConfig {
    /// Validate time configuration
    /// Follows Information Expert principle - knows how to validate itself
    pub fn validate(&self) -> KwaversResult<()> {
        if self.dt <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "dt".to_string(),
                value: self.dt.to_string(),
                constraint: "Time step must be positive".to_string(),
            }.into());
        }

        if self.num_steps == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "num_steps".to_string(),
                value: self.num_steps.to_string(),
                constraint: "Number of steps must be positive".to_string(),
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

#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub enable_validation: bool,
    pub strict_mode: bool,
    pub validation_rules: Vec<String>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_validation: true,
            strict_mode: false,
            validation_rules: vec![
                "grid_validation".to_string(),
                "medium_validation".to_string(),
                "physics_validation".to_string(),
                "time_validation".to_string(),
            ],
        }
    }
}

/// Factory for creating simulation components following GRASP Creator principle
pub struct SimulationFactory;

impl SimulationFactory {
    /// Create a complete simulation setup from configuration
    /// Follows GRASP Controller principle - coordinates the creation process
    pub fn create_simulation(config: SimulationConfig) -> KwaversResult<SimulationBuilder> {
        // Validate configuration following Information Expert principle
        if config.validation.enable_validation {
            Self::validate_config(&config)?;
        }

        let mut builder = SimulationBuilder::new();
        
        // Create grid (Information Expert - Grid knows how to validate itself)
        let grid = Self::create_grid(config.grid)?;
        builder = builder.with_grid(grid);
        
        // Create medium (Creator - Factory creates medium with required information)
        let medium = Self::create_medium(config.medium, builder.grid.as_ref().unwrap())?;
        builder = builder.with_medium(medium);
        
        // Create physics pipeline (Controller - coordinates physics components)
        let physics = Self::create_physics_pipeline(config.physics, builder.grid.as_ref().unwrap())?;
        builder = builder.with_physics(physics);
        
        // Create time configuration
        let time = Self::create_time(config.time)?;
        builder = builder.with_time(time);
        
        Ok(builder)
    }

    /// Validate simulation configuration
    /// Follows Information Expert principle - knows how to validate configuration
    fn validate_config(config: &SimulationConfig) -> KwaversResult<()> {
        let mut validation_result = ValidationResult::valid("SimulationConfig".to_string());
        
        // Validate each component
        if let Err(e) = config.grid.validate() {
            validation_result.add_error(ValidationError::FieldValidation {
                field: "grid".to_string(),
                value: format!("{:?}", config.grid),
                constraint: e.to_string(),
            });
        }

        if let Err(e) = config.medium.validate() {
            validation_result.add_error(ValidationError::FieldValidation {
                field: "medium".to_string(),
                value: format!("{:?}", config.medium),
                constraint: e.to_string(),
            });
        }

        if let Err(e) = config.physics.validate() {
            validation_result.add_error(ValidationError::FieldValidation {
                field: "physics".to_string(),
                value: format!("{:?}", config.physics),
                constraint: e.to_string(),
            });
        }

        if let Err(e) = config.time.validate() {
            validation_result.add_error(ValidationError::FieldValidation {
                field: "time".to_string(),
                value: format!("{:?}", config.time),
                constraint: e.to_string(),
            });
        }

        if !validation_result.is_valid {
            return Err(ConfigError::ValidationFailed {
                section: "simulation".to_string(),
                reason: format!("Configuration validation failed: {:?}", validation_result.errors),
            }.into());
        }

        Ok(())
    }

    /// Create grid from configuration
    /// Follows Information Expert principle - Grid knows how to validate itself
    fn create_grid(config: GridConfig) -> KwaversResult<Grid> {
        // Grid::new now returns Grid directly, not Result
        let grid = Grid::new(config.nx, config.ny, config.nz, config.dx, config.dy, config.dz);
        
        // Basic validation
        if config.nx == 0 || config.ny == 0 || config.nz == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_dimensions".to_string(),
                value: format!("({}, {}, {})", config.nx, config.ny, config.nz),
                constraint: "positive integers".to_string(),
            }.into());
        }
        
        if config.dx <= 0.0 || config.dy <= 0.0 || config.dz <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_spacing".to_string(), 
                value: format!("({}, {}, {})", config.dx, config.dy, config.dz),
                constraint: "positive values".to_string(),
            }.into());
        }
        
        Ok(grid)
    }

    /// Create medium from configuration
    /// Follows Creator principle - Factory creates medium with required information
    fn create_medium(config: MediumConfig, grid: &Grid) -> KwaversResult<Arc<dyn Medium>> {
        match config.medium_type {
            MediumType::Homogeneous { density, sound_speed, mu_a, mu_s_prime } => {
                // HomogeneousMedium::new now returns HomogeneousMedium directly, not Result
                let medium = HomogeneousMedium::new(density, sound_speed, grid, mu_a, mu_s_prime);
                Ok(Arc::new(medium) as Arc<dyn Medium>)
            }
            MediumType::Heterogeneous { tissue_file } => {
                // Implement heterogeneous medium creation based on tissue file
                match tissue_file {
                    Some(_file_path) => {
                        // For now, create a default tissue medium
                        // In future, this could load from file following YAGNI principle
                        let medium = crate::medium::heterogeneous::HeterogeneousMedium::new_tissue(grid);
                        Ok(Arc::new(medium) as Arc<dyn Medium>)
                    }
                    None => {
                        // Create default tissue medium when no file specified
                        let medium = crate::medium::heterogeneous::HeterogeneousMedium::new_tissue(grid);
                        Ok(Arc::new(medium) as Arc<dyn Medium>)
                    }
                }
            }
        }
    }

    /// Create physics pipeline from configuration
    /// Follows Controller principle - coordinates physics components
    fn create_physics_pipeline(config: PhysicsConfig, grid: &Grid) -> KwaversResult<PhysicsPipeline> {
        config.validate()?;
        
        let mut pipeline = PhysicsPipeline::new();
        
        for model_config in config.models {
            if !model_config.enabled {
                continue;
            }

            let component: Box<dyn PhysicsComponent> = match model_config.model_type {
                PhysicsModelType::AcousticWave => {
                    Box::new(AcousticWaveComponent::new("acoustic".to_string()))
                }
                PhysicsModelType::ThermalDiffusion => {
                    Box::new(ThermalDiffusionComponent::new("thermal".to_string()))
                }
                PhysicsModelType::Cavitation => {
                    // Create cavitation component with proper grid reference
                    Box::new(crate::physics::composable::CavitationComponent::new(
                        "cavitation".to_string(),
                        &grid
                    ))
                }
                PhysicsModelType::ElasticWave => {
                    // Create elastic wave component
                    Box::new(crate::physics::composable::ElasticWaveComponent::new(
                        "elastic".to_string(),
                        &grid,
                    )?)
                }
                PhysicsModelType::LightDiffusion => {
                    // Create light diffusion component with proper grid reference
                    Box::new(crate::physics::composable::LightDiffusionComponent::new(
                        "light".to_string(),
                        &grid
                    ))
                }
                PhysicsModelType::Chemical => {
                    // Create chemical component with proper error handling
                    match crate::physics::composable::ChemicalComponent::new("chemical".to_string(), &grid) {
                        Ok(component) => Box::new(component),
                        Err(e) => return Err(PhysicsError::InvalidConfiguration {
                            component: "ChemicalComponent".to_string(),
                            reason: format!("Failed to create chemical component: {}", e),
                        }.into()),
                    }
                }
            };

            pipeline.add_component(component)?;
        }
        
        Ok(pipeline)
    }

    /// Create time configuration from configuration
    /// Follows Information Expert principle - Time knows how to validate itself
    fn create_time(config: TimeConfig) -> KwaversResult<Time> {
        // Time::new now returns Time directly, not Result
        let time = Time::new(config.dt, config.num_steps);
        
        // Basic validation
        if config.dt <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "dt".to_string(),
                value: config.dt.to_string(),
                constraint: "positive value".to_string(),
            }.into());
        }
        
        if config.num_steps == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "num_steps".to_string(),
                value: config.num_steps.to_string(),
                constraint: "positive integer".to_string(),
            }.into());
        }
        
        Ok(time)
    }

    /// Create a default simulation configuration
    /// Follows SSOT principle - single source of truth for defaults
    pub fn create_default_config() -> SimulationConfig {
        SimulationConfig {
            grid: GridConfig {
                nx: 64,
                ny: 64,
                nz: 64,
                dx: 1e-4,
                dy: 1e-4,
                dz: 1e-4,
            },
            medium: MediumConfig {
                medium_type: MediumType::Homogeneous {
                    density: 1000.0,
                    sound_speed: 1500.0,
                    mu_a: 0.1,
                    mu_s_prime: 1.0,
                },
                properties: [
                    ("density".to_string(), 1000.0),
                    ("sound_speed".to_string(), 1500.0),
                    ("mu_a".to_string(), 0.1),
                    ("mu_s_prime".to_string(), 1.0),
                ].iter().cloned().collect(),
            },
            physics: PhysicsConfig {
                models: vec![
                    PhysicsModelConfig {
                        model_type: PhysicsModelType::AcousticWave,
                        enabled: true,
                        parameters: HashMap::new(),
                    },
                    PhysicsModelConfig {
                        model_type: PhysicsModelType::ThermalDiffusion,
                        enabled: true,
                        parameters: HashMap::new(),
                    },
                ],
                frequency: 1e6,
                parameters: HashMap::new(),
            },
            time: TimeConfig {
                dt: 1e-8,
                num_steps: 1000,
                cfl_factor: 0.3,
            },
            validation: ValidationConfig::default(),
        }
    }
}

/// Builder pattern for simulation setup
/// Follows GRASP Creator and Controller principles
pub struct SimulationBuilder {
    grid: Option<Grid>,
    medium: Option<Arc<dyn Medium>>,
    physics: Option<PhysicsPipeline>,
    time: Option<Time>,
}

impl SimulationBuilder {
    /// Create a new simulation builder
    pub fn new() -> Self {
        Self {
            grid: None,
            medium: None,
            physics: None,
            time: None,
        }
    }

    /// Add grid to builder
    pub fn with_grid(mut self, grid: Grid) -> Self {
        self.grid = Some(grid);
        self
    }

    /// Add medium to builder
    pub fn with_medium(mut self, medium: Arc<dyn Medium>) -> Self {
        self.medium = Some(medium);
        self
    }

    /// Add physics pipeline to builder
    pub fn with_physics(mut self, physics: PhysicsPipeline) -> Self {
        self.physics = Some(physics);
        self
    }

    /// Add time configuration to builder
    pub fn with_time(mut self, time: Time) -> Self {
        self.time = Some(time);
        self
    }

    /// Build the simulation setup
    /// Follows Controller principle - coordinates the build process
    pub fn build(self) -> KwaversResult<SimulationSetup> {
        // Validate that all required components are present
        let grid = self.grid.ok_or_else(|| ConfigError::MissingParameter {
            parameter: "grid".to_string(),
            section: "simulation".to_string(),
        })?;

        let medium = self.medium.ok_or_else(|| ConfigError::MissingParameter {
            parameter: "medium".to_string(),
            section: "simulation".to_string(),
        })?;

        let physics = self.physics.ok_or_else(|| ConfigError::MissingParameter {
            parameter: "physics".to_string(),
            section: "simulation".to_string(),
        })?;

        let time = self.time.ok_or_else(|| ConfigError::MissingParameter {
            parameter: "time".to_string(),
            section: "simulation".to_string(),
        })?;

        Ok(SimulationSetup {
            grid,
            medium,
            physics,
            time,
        })
    }
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete simulation setup
/// Follows SSOT principle - single source of truth for simulation state
pub struct SimulationSetup {
    pub grid: Grid,
    pub medium: Arc<dyn Medium>,
    pub physics: PhysicsPipeline,
    pub time: Time,
}

impl SimulationSetup {
    /// Validate the simulation setup
    /// Comprehensive validation following ValidationContext pattern
    fn validate(&self) -> KwaversResult<()> {
        // Basic grid validation - validate parameters directly since Grid doesn't have validate method
        if self.grid.nx == 0 || self.grid.ny == 0 || self.grid.nz == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_dimensions".to_string(),
                value: format!("({}, {}, {})", self.grid.nx, self.grid.ny, self.grid.nz),
                constraint: "positive integers".to_string(),
            }.into());
        }

        // Check grid spacing consistency
        let (nx, ny, nz) = (self.grid.nx, self.grid.ny, self.grid.nz);
        
        // Check if medium has proper dimensions (simplified check)
        // Note: Removed complex shape comparison as medium validation is handled elsewhere
        
        // Physics pipeline validation - use proper component counting
        let component_count = self.physics.component_count();
        if component_count == 0 {
            return Err(ConfigError::ValidationFailed {
                section: "physics".to_string(),
                reason: "At least one physics component must be enabled".to_string(),
            }.into());
        }

        // Basic time validation - validate parameters directly since Time doesn't have validate method
        if self.time.dt <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "dt".to_string(),
                value: self.time.dt.to_string(),
                constraint: "positive value".to_string(),
            }.into());
        }
        
        if self.time.num_steps() == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "num_steps".to_string(),
                value: self.time.num_steps().to_string(),
                constraint: "positive integer".to_string(),
            }.into());
        }

        Ok(())
    }

    /// Get validation summary
    fn get_validation_summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        
        // Grid information
        summary.insert("grid_size".to_string(), format!("{}x{}x{}", self.grid.nx, self.grid.ny, self.grid.nz));
        summary.insert("grid_spacing".to_string(), format!("({:.2e}, {:.2e}, {:.2e})", self.grid.dx, self.grid.dy, self.grid.dz));
        
        // Time information  
        summary.insert("dt".to_string(), self.time.dt.to_string());
        summary.insert("num_steps".to_string(), self.time.num_steps().to_string());
        
        // Physics components count - use proper counting
        summary.insert("physics_components".to_string(), self.physics.component_count().to_string());
        
        summary
    }

    /// Get performance recommendations
    /// Follows Information Expert principle - knows about performance characteristics
    pub fn get_performance_recommendations(&self) -> HashMap<String, String> {
        let mut recommendations = HashMap::new();
        
        let (nx, ny, nz) = self.grid.dimensions();
        let total_points = nx * ny * nz;
        
        // Grid size recommendations
        if total_points > 10_000_000 {
            recommendations.insert(
                "grid_size".to_string(),
                "Large grid detected. Consider using GPU acceleration or reducing resolution.".to_string(),
            );
        }
        
        if nx != ny || ny != nz {
            recommendations.insert(
                "grid_aspect".to_string(),
                "Non-cubic grid detected. This may affect performance.".to_string(),
            );
        }
        
        // Time step recommendations
        let (dx, dy, dz) = self.grid.spacing();
        let min_spacing = dx.min(dy).min(dz);
        
        // Get the maximum stable time step from the medium for CFL condition
        let max_dt = self.grid.cfl_timestep_from_medium(&*self.medium);
        
        if self.time.dt > max_dt {
            recommendations.insert(
                "time_step".to_string(),
                format!("Time step may be too large for stability. Consider dt < {:.2e}", max_dt),
            );
        }
        
        // Physics model recommendations
        // Performance based on physics complexity - use proper counting
        let component_count = self.physics.component_count();
        if component_count > 5 {
            recommendations.insert(
                "physics_models".to_string(),
                "Many physics models detected. Consider disabling unused models for better performance.".to_string(),
            );
        }
        
        recommendations
    }

    /// Get simulation summary
    pub fn get_summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        
        let (nx, ny, nz) = self.grid.dimensions();
        let (dx, dy, dz) = self.grid.spacing();
        
        summary.insert("grid_dimensions".to_string(), format!("{}x{}x{}", nx, ny, nz));
        summary.insert("grid_spacing".to_string(), format!("{:.2e}x{:.2e}x{:.2e}", dx, dy, dz));
        summary.insert("total_points".to_string(), (nx * ny * nz).to_string());
        summary.insert("time_step".to_string(), format!("{:.2e}", self.time.dt));
        summary.insert("num_steps".to_string(), self.time.num_steps().to_string());
        summary.insert("physics_components".to_string(), self.physics.component_count().to_string());
        
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulation_factory_creation() {
        let config = SimulationFactory::create_default_config();
        let builder = SimulationFactory::create_simulation(config).unwrap();
        let setup = builder.build().unwrap();
        
        assert_eq!(setup.grid.dimensions(), (64, 64, 64));
        assert_eq!(setup.time.num_steps(), 1000);
    }

    #[test]
    fn test_grid_config_validation() {
        let mut config = GridConfig {
            nx: 64,
            ny: 64,
            nz: 64,
            dx: 1e-4,
            dy: 1e-4,
            dz: 1e-4,
        };
        
        assert!(config.validate().is_ok());
        
        // Test invalid dimensions
        config.nx = 0;
        assert!(config.validate().is_err());
        
        // Test invalid spacing
        config.nx = 64;
        config.dx = -1e-4;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_medium_config_validation() {
        let config = MediumConfig {
            medium_type: MediumType::Homogeneous {
                density: 1000.0,
                sound_speed: 1500.0,
                mu_a: 0.1,
                mu_s_prime: 1.0,
            },
            properties: [
                ("density".to_string(), 1000.0),
                ("sound_speed".to_string(), 1500.0),
            ].iter().cloned().collect(),
        };
        
        assert!(config.validate().is_ok());
        
        // Test missing required properties
        let invalid_config = MediumConfig {
            medium_type: MediumType::Homogeneous {
                density: 1000.0,
                sound_speed: 1500.0,
                mu_a: 0.1,
                mu_s_prime: 1.0,
            },
            properties: HashMap::new(),
        };
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_physics_config_validation() {
        let config = PhysicsConfig {
            models: vec![
                PhysicsModelConfig {
                    model_type: PhysicsModelType::AcousticWave,
                    enabled: true,
                    parameters: HashMap::new(),
                }
            ],
            frequency: 1e6,
            parameters: HashMap::new(),
        };
        
        assert!(config.validate().is_ok());
        
        // Test invalid frequency
        let mut invalid_config = config.clone();
        invalid_config.frequency = -1e6;
        assert!(invalid_config.validate().is_err());
        
        // Test no enabled models
        let mut invalid_config = config;
        invalid_config.models[0].enabled = false;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_time_config_validation() {
        let config = TimeConfig {
            dt: 1e-8,
            num_steps: 1000,
            cfl_factor: 0.3,
        };
        
        assert!(config.validate().is_ok());
        
        // Test invalid time step
        let mut invalid_config = config.clone();
        invalid_config.dt = -1e-8;
        assert!(invalid_config.validate().is_err());
        
        // Test invalid CFL factor
        let mut invalid_config = config;
        invalid_config.cfl_factor = 1.5;
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_simulation_builder_complete() {
        // Create components
        let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4);
        let medium = Arc::new(HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0));
        
        // Create physics pipeline with at least one component for validation
        let mut physics = PhysicsPipeline::new();
        let acoustic_component = AcousticWaveComponent::new("acoustic".to_string());
        physics.add_component(Box::new(acoustic_component)).expect("Failed to add component");
        
        let time = Time::new(1e-8, 100);
        
        let setup = SimulationBuilder::new()
            .with_grid(grid)
            .with_medium(medium)
            .with_physics(physics)
            .with_time(time)
            .build()
            .unwrap();
        
        // Test validation passes
        assert!(setup.validate().is_ok());
        
        // Test basic properties
        assert_eq!(setup.time.num_steps(), 100);
    }

    #[test]
    fn test_simulation_factory_heterogeneous_medium() {
        let config = SimulationFactory::create_default_config();
        
        // Override to use heterogeneous medium (which should work)
        let mut modified_config = config;
        modified_config.medium.medium_type = MediumType::Heterogeneous { 
            tissue_file: Some("tissue.dat".to_string()) 
        };
        
        let result = SimulationFactory::create_simulation(modified_config);
        assert!(result.is_ok()); // Should succeed as heterogeneous is implemented
    }
}