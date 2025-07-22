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
    SimulationConfig, SimulationFactory, KwaversResult,
    PhysicsComponent, PhysicsPipeline, AcousticWaveComponent, ThermalDiffusionComponent,
    Grid, HomogeneousMedium, Time,
    factory::{GridConfig, MediumConfig, MediumType, PhysicsConfig, PhysicsModelConfig, PhysicsModelType, TimeConfig, ValidationConfig}
};
use std::collections::HashMap;
use std::sync::Arc;

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
    
    // Get simulation summary (SSOT principle)
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
fn create_enhanced_simulation_config() -> SimulationConfig {
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
    let mut context = kwavers::physics::PhysicsContext::new(1e6);
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
    
    // Test grid validation errors
    let invalid_grid_result = Grid::new(0, 64, 64, 1e-4, 1e-4, 1e-4);
    match invalid_grid_result {
        Ok(_) => println!("   ⚠ Unexpected success for invalid grid"),
        Err(e) => println!("   ✓ Caught grid validation error: {}", e),
    }
    
    // Test medium validation errors
    let invalid_medium_result = HomogeneousMedium::new(-1000.0, 1500.0, 0.1, 1.0);
    match invalid_medium_result {
        Ok(_) => println!("   ⚠ Unexpected success for invalid medium"),
        Err(e) => println!("   ✓ Caught medium validation error: {}", e),
    }
    
    // Test time validation errors
    let invalid_time_result = Time::new(-1e-8, 1000, 0.3);
    match invalid_time_result {
        Ok(_) => println!("   ⚠ Unexpected success for invalid time"),
        Err(e) => println!("   ✓ Caught time validation error: {}", e),
    }
    
    // Test configuration validation
    let mut invalid_config = create_enhanced_simulation_config();
    invalid_config.grid.nx = 0; // Invalid dimension
    
    let invalid_builder_result = SimulationFactory::create_simulation(invalid_config);
    match invalid_builder_result {
        Ok(_) => println!("   ⚠ Unexpected success for invalid config"),
        Err(e) => println!("   ✓ Caught configuration validation error: {}", e),
    }
    
    Ok(())
}

/// Demonstrate enhanced builder pattern with validation
/// Follows GRASP Creator and Controller principles
fn demonstrate_enhanced_builder_pattern() -> KwaversResult<()> {
    println!("   Creating simulation with enhanced builder pattern...");
    
    // Create components with validation
    let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4)?;
    let medium = Arc::new(HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0)?);
    let mut physics = PhysicsPipeline::new();
    let time = Time::new(1e-8, 100, 0.3)?;
    
    // Add physics components
    let acoustic = AcousticWaveComponent::new("builder_acoustic".to_string());
    physics.add_component(Box::new(acoustic))?;
    
    let thermal = ThermalDiffusionComponent::new("builder_thermal".to_string());
    physics.add_component(Box::new(thermal))?;
    
    // Build simulation setup
    let setup = kwavers::factory::SimulationBuilder::new()
        .with_grid(grid)
        .with_medium(medium)
        .with_physics(physics)
        .with_time(time)
        .build()?;
    
    // Validate setup
    setup.validate()?;
    println!("   ✓ Enhanced builder pattern simulation created and validated");
    
    // Get summary
    let summary = setup.get_summary();
    println!("   Built simulation: {} points, {} steps", 
             summary.get("total_points").unwrap(), 
             summary.get("num_steps").unwrap());
    
    Ok(())
}

/// Demonstrate performance monitoring with SSOT metrics
/// Follows SSOT principle
fn demonstrate_performance_monitoring() -> KwaversResult<()> {
    println!("   Setting up performance monitoring...");
    
    // Create a simple simulation for performance testing
    let grid = Grid::new(16, 16, 16, 1e-4, 1e-4, 1e-4)?;
    let medium = Arc::new(HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0)?);
    let mut physics = PhysicsPipeline::new();
    let time = Time::new(1e-8, 10, 0.3)?;
    
    // Add components with performance tracking
    let acoustic = AcousticWaveComponent::new("perf_acoustic".to_string());
    physics.add_component(Box::new(acoustic))?;
    
    let thermal = ThermalDiffusionComponent::new("perf_thermal".to_string());
    physics.add_component(Box::new(thermal))?;
    
    let setup = kwavers::factory::SimulationBuilder::new()
        .with_grid(grid)
        .with_medium(medium)
        .with_physics(physics)
        .with_time(time)
        .build()?;
    
    println!("   ✓ Performance monitoring setup complete");
    println!("   Simulation ready for performance analysis");
    
    Ok(())
}

/// Demonstrate enhanced validation system
/// Follows Information Expert principle
fn demonstrate_validation_system() -> KwaversResult<()> {
    println!("   Testing enhanced validation system...");
    
    // Test grid validation
    let grid = Grid::new(32, 32, 32, 1e-4, 1e-4, 1e-4)?;
    grid.validate()?;
    println!("   ✓ Grid validation passed");
    
    // Test medium validation
    let medium = HomogeneousMedium::new(1000.0, 1500.0, 0.1, 1.0)?;
    medium.validate()?;
    println!("   ✓ Medium validation passed");
    
    // Test time validation
    let time = Time::new(1e-8, 100, 0.3)?;
    time.validate()?;
    println!("   ✓ Time validation passed");
    
    // Test physics pipeline validation
    let mut pipeline = PhysicsPipeline::new();
    let acoustic = AcousticWaveComponent::new("validation_acoustic".to_string());
    pipeline.add_component(Box::new(acoustic))?;
    
    let mut context = kwavers::physics::PhysicsContext::new(1e6);
    let validation_result = pipeline.validate_pipeline(&context);
    if validation_result.is_valid {
        println!("   ✓ Physics pipeline validation passed");
    } else {
        println!("   ⚠ Physics pipeline validation warnings: {:?}", validation_result.warnings);
    }
    
    Ok(())
}

/// Enhanced custom physics component demonstrating design principles
/// Follows Open/Closed principle - extends functionality without modification
#[derive(Debug)]
struct EnhancedCustomComponent {
    id: String,
    metrics: HashMap<String, f64>,
    custom_parameter: f64,
    state: kwavers::physics::ComponentState,
}

impl EnhancedCustomComponent {
    /// Create new enhanced custom component
    /// Follows GRASP Creator principle
    pub fn new(id: String, custom_parameter: f64) -> Self {
        Self {
            id,
            metrics: HashMap::new(),
            custom_parameter,
            state: kwavers::physics::ComponentState::Initialized,
        }
    }
    
    /// Validate component configuration
    /// Follows Information Expert principle
    pub fn validate(&self) -> KwaversResult<()> {
        if self.custom_parameter <= 0.0 {
            return Err(kwavers::error::PhysicsError::InvalidConfiguration {
                component: self.id.clone(),
                reason: "Custom parameter must be positive".to_string(),
            }.into());
        }
        Ok(())
    }
}

impl PhysicsComponent for EnhancedCustomComponent {
    fn component_id(&self) -> &str {
        &self.id
    }
    
    fn dependencies(&self) -> Vec<kwavers::physics::FieldType> {
        vec![kwavers::physics::FieldType::Pressure]
    }
    
    fn output_fields(&self) -> Vec<kwavers::physics::FieldType> {
        vec![kwavers::physics::FieldType::Custom("enhanced_field".to_string())]
    }
    
    fn apply(
        &mut self,
        fields: &mut ndarray::Array4<f64>,
        _grid: &Grid,
        _medium: &dyn kwavers::Medium,
        dt: f64,
        _t: f64,
        _context: &kwavers::physics::PhysicsContext,
    ) -> KwaversResult<()> {
        // Validate component state
        self.validate()?;
        
        // Update state
        self.state = kwavers::physics::ComponentState::Running;
        
        // Simple physics update (placeholder)
        // In a real implementation, this would contain actual physics calculations
        let pressure_field = fields.slice(ndarray::s![0, .., .., ..]);
        let mut enhanced_field = fields.slice_mut(ndarray::s![1, .., .., ..]);
        
        // Apply custom physics with parameter
        enhanced_field.assign(&(&pressure_field * self.custom_parameter * dt));
        
        // Update metrics
        self.metrics.insert("custom_parameter".to_string(), self.custom_parameter);
        self.metrics.insert("dt".to_string(), dt);
        
        self.state = kwavers::physics::ComponentState::Completed;
        Ok(())
    }
    
    fn get_metrics(&self) -> HashMap<String, f64> {
        self.metrics.clone()
    }
    
    fn state(&self) -> kwavers::physics::ComponentState {
        self.state.clone()
    }
    
    fn reset(&mut self) -> KwaversResult<()> {
        self.state = kwavers::physics::ComponentState::Initialized;
        self.metrics.clear();
        Ok(())
    }
    
    fn priority(&self) -> u32 {
        1 // High priority
    }
    
    fn is_optional(&self) -> bool {
        true // This component is optional
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_enhanced_simulation_example() {
        // Test that the main function doesn't panic
        assert!(result.is_ok());
    
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