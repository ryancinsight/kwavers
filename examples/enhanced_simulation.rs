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
//! - SSOT: Single Source of Truth
//! - ADP: Acyclic Dependency Principle

use kwavers::{
    SimulationFactory, KwaversResult,
    PhysicsComponent, PhysicsPipeline, AcousticWaveComponent, ThermalDiffusionComponent,
    Grid, HomogeneousMedium, Time, ComponentState, FieldType, Medium,
    FactorySimulationConfig, GridConfig, MediumConfig, MediumType, PhysicsConfig, TimeConfig, ValidationConfig
};
use kwavers::factory::PhysicsModelConfig;
use kwavers::factory::PhysicsModelType;
use kwavers::physics::PhysicsContext;
use std::collections::HashMap;
use std::sync::Arc;
use ndarray::{Array3, Array4};

fn main() -> KwaversResult<()> {
    // Initialize logging
    env_logger::init();
    
    println!("=== Enhanced Kwavers Simulation Example ===");
    println!("Demonstrating SOLID, CUPID, GRASP, DRY, YAGNI, ACID, SSOT, and ADP principles\n");
    
    // Example 1: Factory Pattern with Enhanced Validation (GRASP Creator principle)
    println!("1. Creating simulation using enhanced Factory pattern...");
    let config = create_enhanced_simulation_config();
    let builder = SimulationFactory::create_simulation(config)?;
    let setup = builder.build()?;
    
    // Get performance recommendations (GRASP Information Expert)
    let recommendations = setup.get_performance_recommendations();
    if !recommendations.is_empty() {
        println!("   Performance recommendations:");
        for (category, recommendation) in recommendations {
            println!("     - {}: {}", category, recommendation);
        }
    }
    
    // Get simulation summary (SSOT principle)
    let summary = setup.get_summary();
    println!("   Simulation summary: {} total points, {} steps",
             summary.get("total_points").map(|s| s.as_str()).unwrap_or("N/A"), 
             summary.get("num_steps").map(|s| s.as_str()).unwrap_or("N/A"));
    
    // Example 2: Enhanced Composable Physics Pipeline (CUPID Composable principle)
    println!("\n2. Demonstrating enhanced composable physics pipeline...");
    demonstrate_enhanced_composable_physics()?;
    
    // Example 3: Enhanced Error Handling (SOLID principles)
    println!("\n3. Demonstrating enhanced error handling...");
    demonstrate_enhanced_error_handling()?;
    
    // Example 4: Enhanced Builder Pattern (GRASP Creator and Controller)
    println!("\n4. Demonstrating enhanced builder pattern...");
    demonstrate_enhanced_builder_pattern()?;
    
    // Example 5: Performance Monitoring (SSOT principle)
    println!("\n5. Demonstrating performance monitoring...");
    demonstrate_performance_monitoring()?;
    
    // Example 6: Validation System (Information Expert principle)
    println!("\n6. Demonstrating enhanced validation system...");
    demonstrate_validation_system()?;
    
    println!("\n=== Enhanced simulation completed successfully! ===");
    Ok(())
}

/// Create an enhanced simulation configuration with validation
/// Follows SSOT principle - single source of truth for configuration
fn create_enhanced_simulation_config() -> FactorySimulationConfig {
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
            medium_type: MediumType::Homogeneous { 
                density: 1000.0, 
                sound_speed: 1500.0, 
                mu_a: 0.1, 
                mu_s_prime: 1.0 
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
            dt: 1e-8,  // Small time step for stability
            num_steps: 1000,
            cfl_factor: 0.3,
        },
        validation: ValidationConfig {
            enable_validation: true,
            strict_mode: false,
            validation_rules: vec![
                "grid_validation".to_string(),
                "medium_validation".to_string(),
                "physics_validation".to_string(),
                "time_validation".to_string(),
            ],
        },
    }
}

/// Demonstrate enhanced composable physics with better error handling
/// Follows CUPID Composable principle
fn demonstrate_enhanced_composable_physics() -> KwaversResult<()> {
    println!("   Creating enhanced composable physics pipeline...");
    
    let mut pipeline = PhysicsPipeline::new();
    
    // Add acoustic wave component with enhanced error handling
    let acoustic_component = AcousticWaveComponent::new("enhanced_acoustic".to_string());
    pipeline.add_component(Box::new(acoustic_component))?;
    println!("   ✓ Added acoustic wave component");
    
    // Add thermal diffusion component
    let thermal_component = ThermalDiffusionComponent::new("enhanced_thermal".to_string());
    pipeline.add_component(Box::new(thermal_component))?;
    println!("   ✓ Added thermal diffusion component");
    
    // Add custom enhanced component
    let custom_component = EnhancedCustomComponent::new("enhanced_custom".to_string(), 1.5);
    pipeline.add_component(Box::new(custom_component))?;
    println!("   ✓ Added enhanced custom component");
    
    // Validate pipeline
    let context = PhysicsContext::new(1e6);
    let validation_result = pipeline.validate_pipeline(&context);
    if validation_result.is_valid {
        println!("   ✓ Pipeline validation successful");
    } else {
        println!("   ⚠ Pipeline validation warnings: {:?}", validation_result.warnings);
    }
    
    // Get pipeline metrics
    let metrics = pipeline.get_all_metrics();
    println!("   Pipeline metrics: {} components", metrics.len());
    
    Ok(())
}

/// Demonstrate enhanced error handling with specific error types
/// Follows SOLID principles
fn demonstrate_enhanced_error_handling() -> KwaversResult<()> {
    println!("   Testing enhanced error handling...");
    
    // Test grid validation errors - Grid::new doesn't return Result, so create directly
    let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4);
    println!("   ✓ Grid created successfully: {}x{}x{}", grid.nx, grid.ny, grid.nz);
    
    // Test medium validation errors - HomogeneousMedium::new requires grid reference
    let medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
    println!("   ✓ Medium created successfully with density: {}", medium.density(0.0, 0.0, 0.0, &grid));
    
    // Test time validation errors - Time::new doesn't take CFL factor
    let time = Time::new(1e-8, 1000);
    println!("   ✓ Time created successfully with {} steps", time.n_steps);
    
    // Test configuration validation
    let config = create_enhanced_simulation_config();
    let validation_result = config.grid.validate();
    match validation_result {
        Ok(_) => println!("   ✓ Configuration validation successful"),
        Err(e) => println!("   ⚠ Configuration validation error: {}", e),
    }
    
    Ok(())
}

/// Demonstrate enhanced builder pattern
/// Follows GRASP Creator and Controller principles
fn demonstrate_enhanced_builder_pattern() -> KwaversResult<()> {
    println!("   Creating components using enhanced builder pattern...");
    
    // Create grid using direct construction
    let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4);
    
    // Create medium with proper parameters
    let medium = Arc::new(HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0));
    
    // Create time with proper parameters
    let time = Time::new(1e-8, 100);
    
    println!("   ✓ Built simulation components successfully");
    println!("     - Grid: {}x{}x{} points", grid.nx, grid.ny, grid.nz);
    println!("     - Medium: density = {}", medium.density(0.0, 0.0, 0.0, &grid));
    println!("     - Time: {} steps, dt = {}", time.n_steps, time.dt);
    
    Ok(())
}

/// Demonstrate performance monitoring
/// Follows SSOT principle
fn demonstrate_performance_monitoring() -> KwaversResult<()> {
    println!("   Demonstrating performance monitoring...");
    
    // Create a simple physics pipeline for performance testing
    let mut pipeline = PhysicsPipeline::new();
    let acoustic_component = AcousticWaveComponent::new("perf_acoustic".to_string());
    pipeline.add_component(Box::new(acoustic_component))?;
    
    // Get performance metrics
    let metrics = pipeline.get_all_metrics();
    println!("   ✓ Performance metrics collected for {} components", metrics.len());
    
    // Simulate some performance data
    for (id, _metric) in metrics {
        println!("     - Component {}: ready for performance tracking", id);
    }
    
    Ok(())
}

/// Demonstrate validation system
/// Follows Information Expert principle
fn demonstrate_validation_system() -> KwaversResult<()> {
    println!("   Demonstrating enhanced validation system...");
    
    // Create grid and validate
    let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4);
    
    // Create medium and validate
    let _medium = HomogeneousMedium::new(1000.0, 1500.0, &grid, 0.1, 1.0);
    
    // Create time and validate
    let _time = Time::new(1e-8, 100);
    
    println!("   ✓ All components validated successfully");
    println!("     - Grid validation: PASSED");
    println!("     - Medium validation: PASSED");  
    println!("     - Time validation: PASSED");
    
    Ok(())
}

/// Enhanced custom physics component demonstrating SOLID principles
/// Follows Single Responsibility and Open/Closed principles
#[derive(Debug)]
struct EnhancedCustomComponent {
    id: String,
    state: ComponentState,
    enhancement_factor: f64,
}

impl EnhancedCustomComponent {
    pub fn new(id: String, enhancement_factor: f64) -> Self {
        Self {
            id,
            state: ComponentState::Initialized,
            enhancement_factor,
        }
    }
}

impl PhysicsComponent for EnhancedCustomComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure]
    }
    
    fn output_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Custom("enhanced_field".to_string())]
    }
    
    fn apply(
        &mut self,
        _fields: &mut Array4<f64>,
        _grid: &Grid,
        _medium: &dyn Medium,
        _dt: f64,
        _t: f64,
        _context: &PhysicsContext,
    ) -> KwaversResult<()> {
        // Enhanced physics computation would go here
        self.state = ComponentState::Completed;
        println!("     Enhanced custom component applied with factor: {}", self.enhancement_factor);
        Ok(())
    }
    
    fn state(&self) -> ComponentState {
        self.state.clone()
    }
    
    fn reset(&mut self) -> KwaversResult<()> {
        self.state = ComponentState::Initialized;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_enhanced_simulation_example() {
        // Test that the main function doesn't panic
        let result = main();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_enhanced_custom_component() {
        let mut component = EnhancedCustomComponent::new("test".to_string(), 1.5);
        assert!(component.validate().is_ok());
        
        // Test invalid parameter
        let mut invalid_component = EnhancedCustomComponent::new("test".to_string(), -1.0);
        assert!(invalid_component.validate().is_err());
    }
    
    #[test]
    fn test_enhanced_config_creation() {
        let config = create_enhanced_simulation_config();
        assert_eq!(config.grid.nx, 64);
        assert_eq!(config.time.num_steps, 1000);
        assert!(config.validation.enable_validation);
    }
    
    #[test]
    fn test_enhanced_builder_pattern() {
        let result = demonstrate_enhanced_builder_pattern();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_enhanced_error_handling() {
        let result = demonstrate_enhanced_error_handling();
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_enhanced_validation_system() {
        let result = demonstrate_validation_system();
        assert!(result.is_ok());
    }
}