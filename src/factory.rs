// src/factory.rs
//! Factory patterns for creating simulation components
//! 
//! This module follows GRASP principles:
//! - Information Expert: Objects that have the information needed to fulfill a responsibility
//! - Creator: Objects responsible for creating other objects they use
//! - Controller: Objects that coordinate and control system operations
//! - Low Coupling: Minimize dependencies between objects
//! - High Cohesion: Keep related functionality together

use crate::error::{KwaversResult, ConfigError, PhysicsError};
use crate::grid::Grid;
use crate::medium::{Medium, homogeneous::HomogeneousMedium, heterogeneous::HeterogeneousMedium};
use crate::physics::{PhysicsComponent, PhysicsPipeline, AcousticWaveComponent, ThermalDiffusionComponent};
use crate::time::Time;
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for creating simulation components
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

#[derive(Debug, Clone)]
pub struct MediumConfig {
    pub medium_type: MediumType,
    pub properties: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum MediumType {
    Homogeneous,
    Heterogeneous { tissue_file: Option<String> },
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

#[derive(Debug, Clone)]
pub struct TimeConfig {
    pub dt: f64,
    pub num_steps: usize,
    pub cfl_factor: f64,
}

/// Factory for creating simulation components following GRASP Creator principle
pub struct SimulationFactory;

impl SimulationFactory {
    /// Create a complete simulation setup from configuration
    /// Follows GRASP Controller principle - coordinates the creation process
    pub fn create_simulation(config: SimulationConfig) -> KwaversResult<SimulationBuilder> {
        let mut builder = SimulationBuilder::new();
        
        // Create grid (Information Expert - Grid knows how to validate itself)
        let grid = Self::create_grid(config.grid)?;
        builder = builder.with_grid(grid);
        
        // Create medium (Creator - Factory creates medium with required information)
        let medium = Self::create_medium(config.medium, &builder.grid.as_ref().unwrap())?;
        builder = builder.with_medium(medium);
        
        // Create physics pipeline (Low Coupling - separate creation concerns)
        let physics = Self::create_physics_pipeline(config.physics)?;
        builder = builder.with_physics(physics);
        
        // Create time configuration
        let time = Self::create_time(config.time)?;
        builder = builder.with_time(time);
        
        Ok(builder)
    }
    
    /// Create grid with validation (Information Expert principle)
    fn create_grid(config: GridConfig) -> KwaversResult<Grid> {
        if config.nx == 0 || config.ny == 0 || config.nz == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_dimensions".to_string(),
                value: format!("{}x{}x{}", config.nx, config.ny, config.nz),
                reason: "All dimensions must be > 0".to_string(),
            }.into());
        }
        
        if config.dx <= 0.0 || config.dy <= 0.0 || config.dz <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "grid_spacing".to_string(),
                value: format!("dx={}, dy={}, dz={}", config.dx, config.dy, config.dz),
                reason: "All spacing values must be > 0".to_string(),
            }.into());
        }
        
        Ok(Grid::new(config.nx, config.ny, config.nz, config.dx, config.dy, config.dz))
    }
    
    /// Create medium based on configuration (Creator principle)
    fn create_medium(config: MediumConfig, grid: &Grid) -> KwaversResult<Arc<dyn Medium>> {
        match config.medium_type {
            MediumType::Homogeneous => {
                // Use default values for now - in a real implementation, these would come from config
                let density = config.properties.get("density").copied().unwrap_or(1000.0);
                let sound_speed = config.properties.get("sound_speed").copied().unwrap_or(1500.0);
                let mu_a = config.properties.get("mu_a").copied().unwrap_or(0.1);
                let mu_s_prime = config.properties.get("mu_s_prime").copied().unwrap_or(1.0);
                
                let medium = HomogeneousMedium::new(density, sound_speed, grid, mu_a, mu_s_prime);
                Ok(Arc::new(medium))
            }
            MediumType::Heterogeneous { .. } => {
                let medium = HeterogeneousMedium::new_tissue(grid);
                Ok(Arc::new(medium))
            }
        }
    }
    
    /// Create physics pipeline (High Cohesion - related physics components together)
    fn create_physics_pipeline(config: PhysicsConfig) -> KwaversResult<PhysicsPipeline> {
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
                _ => {
                    return Err(ConfigError::UnsupportedFeature {
                        feature: format!("{:?}", model_config.model_type),
                    }.into());
                }
            };
            
            pipeline.add_component(component)?;
        }
        
        Ok(pipeline)
    }
    
    /// Create time configuration with validation
    fn create_time(config: TimeConfig) -> KwaversResult<Time> {
        if config.dt <= 0.0 {
            return Err(ConfigError::InvalidValue {
                parameter: "dt".to_string(),
                value: config.dt.to_string(),
                reason: "Time step must be > 0".to_string(),
            }.into());
        }
        
        if config.num_steps == 0 {
            return Err(ConfigError::InvalidValue {
                parameter: "num_steps".to_string(),
                value: config.num_steps.to_string(),
                reason: "Number of steps must be > 0".to_string(),
            }.into());
        }
        
        Ok(Time::new(config.dt, config.num_steps))
    }
}

/// Builder pattern for constructing simulations (GRASP Creator and Controller)
pub struct SimulationBuilder {
    grid: Option<Grid>,
    medium: Option<Arc<dyn Medium>>,
    physics: Option<PhysicsPipeline>,
    time: Option<Time>,
}

impl SimulationBuilder {
    pub fn new() -> Self {
        Self {
            grid: None,
            medium: None,
            physics: None,
            time: None,
        }
    }
    
    pub fn with_grid(mut self, grid: Grid) -> Self {
        self.grid = Some(grid);
        self
    }
    
    pub fn with_medium(mut self, medium: Arc<dyn Medium>) -> Self {
        self.medium = Some(medium);
        self
    }
    
    pub fn with_physics(mut self, physics: PhysicsPipeline) -> Self {
        self.physics = Some(physics);
        self
    }
    
    pub fn with_time(mut self, time: Time) -> Self {
        self.time = Some(time);
        self
    }
    
    /// Build the complete simulation setup
    /// Follows GRASP Controller principle - validates and assembles components
    pub fn build(self) -> KwaversResult<SimulationSetup> {
        let grid = self.grid.ok_or_else(|| ConfigError::MissingParameter {
            parameter: "grid".to_string(),
        })?;
        
        let medium = self.medium.ok_or_else(|| ConfigError::MissingParameter {
            parameter: "medium".to_string(),
        })?;
        
        let physics = self.physics.ok_or_else(|| ConfigError::MissingParameter {
            parameter: "physics".to_string(),
        })?;
        
        let time = self.time.ok_or_else(|| ConfigError::MissingParameter {
            parameter: "time".to_string(),
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

/// Complete simulation setup (High Cohesion - all related components together)
pub struct SimulationSetup {
    pub grid: Grid,
    pub medium: Arc<dyn Medium>,
    pub physics: PhysicsPipeline,
    pub time: Time,
}

impl SimulationSetup {
    /// Validate the complete setup (Information Expert - each component validates itself)
    pub fn validate(&self) -> KwaversResult<()> {
        // Validate CFL condition for time stepping
        let max_sound_speed = 1500.0; // Simplified - would get from medium
        let min_spacing = self.grid.dx.min(self.grid.dy).min(self.grid.dz);
        let max_dt = 0.5 * min_spacing / max_sound_speed; // CFL condition
        
        if self.time.dt > max_dt {
            return Err(PhysicsError::InvalidTimeStep {
                dt: self.time.dt,
                recommended_max: max_dt,
            }.into());
        }
        
        // Additional validation can be added here
        Ok(())
    }
    
    /// Get recommended performance settings
    pub fn get_performance_recommendations(&self) -> HashMap<String, String> {
        let mut recommendations = HashMap::new();
        
        let total_points = self.grid.nx * self.grid.ny * self.grid.nz;
        if total_points > 1_000_000 {
            recommendations.insert(
                "memory".to_string(),
                "Consider using smaller grid or distributed computing".to_string(),
            );
        }
        
        if self.time.dt < 1e-8 {
            recommendations.insert(
                "time_step".to_string(),
                "Very small time step may lead to long simulation times".to_string(),
            );
        }
        
        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simulation_factory_creation() {
        let config = SimulationConfig {
            grid: GridConfig {
                nx: 64, ny: 64, nz: 64,
                dx: 1e-4, dy: 1e-4, dz: 1e-4,
            },
            medium: MediumConfig {
                medium_type: MediumType::Homogeneous,
                properties: [
                    ("density".to_string(), 1000.0),
                    ("sound_speed".to_string(), 1500.0),
                ].iter().cloned().collect(),
            },
            physics: PhysicsConfig {
                models: vec![
                    PhysicsModelConfig {
                        model_type: PhysicsModelType::AcousticWave,
                        enabled: true,
                        parameters: HashMap::new(),
                    },
                ],
                frequency: 1e6,
                parameters: HashMap::new(),
            },
            time: TimeConfig {
                dt: 1e-8,  // Smaller time step to satisfy CFL condition
                num_steps: 1000,
                cfl_factor: 0.5,
            },
        };
        
        let builder = SimulationFactory::create_simulation(config).unwrap();
        let setup = builder.build().unwrap();
        
        assert_eq!(setup.grid.nx, 64);
        
        // Check validation result
        match setup.validate() {
            Ok(()) => {},
            Err(e) => panic!("Validation failed: {}", e),
        }
    }
    
    #[test]
    fn test_builder_pattern() {
        let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4);
        let medium = Arc::new(HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0));
        let physics = PhysicsPipeline::new();
        let time = Time::new(1e-7, 500);
        
        let setup = SimulationBuilder::new()
            .with_grid(grid)
            .with_medium(medium)
            .with_physics(physics)
            .with_time(time)
            .build()
            .unwrap();
        
        assert_eq!(setup.grid.nx, 32);
    }
}