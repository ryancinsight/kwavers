// examples/enhanced_simulation.rs
//! Enhanced Kwavers Simulation Example
//! 
//! This example demonstrates the enhanced kwavers system with:
//! - SOLID principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
//! - CUPID principles: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
//! - GRASP principles: Information Expert, Creator, Controller, Low Coupling, High Cohesion
//! - DRY: Don't Repeat Yourself
//! - YAGNI: You Aren't Gonna Need It
//! - ACID: Atomicity, Consistency, Isolation, Durability (in terms of simulation state)

use kwavers::{
    FactorySimulationConfig, SimulationFactory, KwaversResult,
    PhysicsComponent, PhysicsPipeline, AcousticWaveComponent, ThermalDiffusionComponent,
    Grid, HomogeneousMedium, Time,
    factory::{GridConfig, MediumConfig, MediumType, PhysicsConfig, PhysicsModelConfig, PhysicsModelType, TimeConfig}
};
use std::collections::HashMap;
use std::sync::Arc;

fn main() -> KwaversResult<()> {
    // Initialize logging
    env_logger::init();
    
    println!("=== Enhanced Kwavers Simulation Example ===");
    println!("Demonstrating SOLID, CUPID, GRASP, DRY, YAGNI, and ACID principles\n");
    
    // Example 1: Factory Pattern (GRASP Creator principle)
    println!("1. Creating simulation using Factory pattern...");
    let config = create_simulation_config();
    let builder = SimulationFactory::create_simulation(config)?;
    let setup = builder.build()?;
    
    // Validate the setup (SOLID Single Responsibility)
    setup.validate()?;
    println!("   ✓ Simulation setup validated successfully");
    
    // Get performance recommendations (GRASP Information Expert)
    let recommendations = setup.get_performance_recommendations();
    if !recommendations.is_empty() {
        println!("   Performance recommendations:");
        for (category, recommendation) in recommendations {
            println!("     - {}: {}", category, recommendation);
        }
    }
    
    // Example 2: Composable Physics Pipeline (CUPID Composable principle)
    println!("\n2. Demonstrating composable physics pipeline...");
    demonstrate_composable_physics()?;
    
    // Example 3: Error Handling (SOLID principles)
    println!("\n3. Demonstrating enhanced error handling...");
    demonstrate_error_handling()?;
    
    // Example 4: Builder Pattern (GRASP Creator and Controller)
    println!("\n4. Demonstrating builder pattern...");
    demonstrate_builder_pattern()?;
    
    println!("\n=== Simulation completed successfully! ===");
    Ok(())
}

/// Create a simulation configuration demonstrating the factory pattern
fn create_simulation_config() -> FactorySimulationConfig {
    FactorySimulationConfig {
        grid: GridConfig {
            nx: 64,
            ny: 64,
            nz: 64,
            dx: 1e-4,
            dy: 1e-4,
            dz: 1e-4,
        },
        medium: MediumConfig {
            medium_type: MediumType::Homogeneous,
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
            dt: 1e-8,  // Small time step for stability
            num_steps: 100,
            cfl_factor: 0.5,
        },
    }
}

/// Demonstrate the composable physics pipeline (CUPID principles)
fn demonstrate_composable_physics() -> KwaversResult<()> {
    // Create individual physics components (Unix-like: each does one thing well)
    let acoustic = Box::new(AcousticWaveComponent::new("acoustic_wave".to_string()));
    let thermal = Box::new(ThermalDiffusionComponent::new("thermal_diffusion".to_string()));
    
    // Create a composable pipeline
    let mut pipeline = PhysicsPipeline::new();
    
    // Add components (the pipeline automatically handles dependency ordering)
    pipeline.add_component(acoustic)?;
    pipeline.add_component(thermal)?;
    
    println!("   ✓ Created physics pipeline with automatic dependency resolution");
    
    // Get performance metrics (Interface Segregation principle)
    let metrics = pipeline.get_all_metrics();
    println!("   ✓ Available physics components: {}", metrics.len());
    for (component_id, component_metrics) in metrics {
        println!("     - {}: {} metrics available", component_id, component_metrics.len());
    }
    
    Ok(())
}

/// Demonstrate enhanced error handling (SOLID principles)
fn demonstrate_error_handling() -> KwaversResult<()> {
    use kwavers::error::{GridError, ConfigError};
    
    // Example of specific error types (Single Responsibility)
    let grid_error = GridError::InvalidDimensions { nx: 0, ny: 10, nz: 10 };
    let config_error = ConfigError::MissingParameter { parameter: "frequency".to_string() };
    
    println!("   ✓ Grid error: {}", grid_error);
    println!("   ✓ Config error: {}", config_error);
    
    // Automatic error conversion (Dependency Inversion)
    let _kwavers_error_from_grid: kwavers::KwaversError = grid_error.into();
    let _kwavers_error_from_config: kwavers::KwaversError = config_error.into();
    
    println!("   ✓ Automatic error conversion working");
    
    Ok(())
}

/// Demonstrate the builder pattern (GRASP principles)
fn demonstrate_builder_pattern() -> KwaversResult<()> {
    use kwavers::SimulationBuilder;
    
    // Create components individually (Information Expert principle)
    let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4);
    let medium = Arc::new(HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0));
    let physics = PhysicsPipeline::new();
    let time = Time::new(1e-8, 50);
    
    // Use builder pattern (Creator and Controller principles)
    let setup = SimulationBuilder::new()
        .with_grid(grid)
        .with_medium(medium)
        .with_physics(physics)
        .with_time(time)
        .build()?;
    
    println!("   ✓ Built simulation using builder pattern");
    
    // Validate (each component knows how to validate itself)
    setup.validate()?;
    println!("   ✓ Builder-created simulation validated successfully");
    
    Ok(())
}

/// Custom physics component demonstrating extensibility (Open/Closed principle)
#[derive(Debug)]
#[allow(dead_code)]
struct CustomAcousticComponent {
    id: String,
    metrics: HashMap<String, f64>,
    custom_parameter: f64,
}

impl CustomAcousticComponent {
    #[allow(dead_code)]
    pub fn new(id: String, custom_parameter: f64) -> Self {
        Self {
            id,
            metrics: HashMap::new(),
            custom_parameter,
        }
    }
}

impl PhysicsComponent for CustomAcousticComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<&str> {
        vec![] // No dependencies
    }
    
    fn output_fields(&self) -> Vec<&str> {
        vec!["pressure", "custom_field"]
    }
    
    fn apply(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        grid: &Grid,
        _medium: &dyn kwavers::Medium,
        dt: f64,
        _t: f64,
        _context: &kwavers::physics::PhysicsContext,
    ) -> KwaversResult<()> {
        use std::time::Instant;
        let start = Instant::now();
        
        // Custom acoustic wave implementation with the custom parameter
        let pressure_idx = 0;
        let mut pressure = fields.index_axis(ndarray::Axis(0), pressure_idx).to_owned();
        
        let (nx, ny, nz) = (grid.nx, grid.ny, grid.nz);
        let c_squared = (1500.0 * self.custom_parameter).powi(2); // Use custom parameter
        
        // Simple wave equation with custom behavior
        for i in 1..nx-1 {
            for j in 1..ny-1 {
                for k in 1..nz-1 {
                    let laplacian = (pressure[[i+1,j,k]] + pressure[[i-1,j,k]] - 2.0*pressure[[i,j,k]]) / (grid.dx * grid.dx)
                                  + (pressure[[i,j+1,k]] + pressure[[i,j-1,k]] - 2.0*pressure[[i,j,k]]) / (grid.dy * grid.dy)
                                  + (pressure[[i,j,k+1]] + pressure[[i,j,k-1]] - 2.0*pressure[[i,j,k]]) / (grid.dz * grid.dz);
                    
                    pressure[[i,j,k]] += dt * dt * c_squared * laplacian;
                }
            }
        }
        
        fields.index_axis_mut(ndarray::Axis(0), pressure_idx).assign(&pressure);
        
        // Record metrics
        let elapsed = start.elapsed().as_secs_f64();
        self.metrics.insert("execution_time".to_string(), elapsed);
        self.metrics.insert("custom_parameter".to_string(), self.custom_parameter);
        
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_enhanced_simulation_example() {
        // Test that the example runs without errors
        assert!(main().is_ok());
    }
    
    #[test]
    fn test_custom_physics_component() {
        let mut component = CustomAcousticComponent::new("custom".to_string(), 1.2);
        
        assert_eq!(component.component_id(), "custom");
        assert_eq!(component.dependencies(), Vec::<&str>::new());
        assert_eq!(component.output_fields(), vec!["pressure", "custom_field"]);
        
        let metrics = component.get_metrics();
        assert!(metrics.contains_key("custom_parameter"));
        assert_eq!(metrics["custom_parameter"], 1.2);
    }
}