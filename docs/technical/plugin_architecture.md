# Plugin Architecture Documentation

## Overview

The Kwavers plugin architecture provides a flexible, extensible system for adding new physics modules and numerical methods to the simulation framework. It follows key design principles including SOLID, CUPID, GRASP, and others to ensure maintainability and scalability.

## Design Principles

### SOLID Principles
- **Single Responsibility**: Each plugin handles one specific physics phenomenon
- **Open/Closed**: System is open for extension via plugins, closed for modification
- **Liskov Substitution**: All plugins implement the same interface
- **Interface Segregation**: Minimal, focused plugin interface
- **Dependency Inversion**: Plugins depend on abstractions, not concrete implementations

### Additional Principles
- **CUPID**: Composable, Unix-like, Predictable, Idiomatic, Domain-focused
- **GRASP**: Information expert pattern with proper encapsulation
- **DRY**: Shared utilities and common interfaces
- **KISS**: Simple, intuitive plugin API
- **YAGNI**: Only essential features implemented

## Core Components

### 1. PhysicsPlugin Trait

The main interface that all plugins must implement:

```rust
pub trait PhysicsPlugin: Debug + Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;
    
    /// Get required input fields
    fn required_fields(&self) -> Vec<FieldType>;
    
    /// Get provided output fields
    fn provided_fields(&self) -> Vec<FieldType>;
    
    /// Initialize the plugin
    fn initialize(
        &mut self,
        config: Option<Box<dyn PluginConfig>>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()>;
    
    /// Update physics for one time step
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()>;
    
    /// Validate plugin configuration
    fn validate(&self, grid: &Grid, medium: &dyn Medium) -> ValidationResult;
}
```

### 2. PluginMetadata

Describes plugin capabilities and information:

```rust
pub struct PluginMetadata {
    pub id: String,              // Unique identifier
    pub name: String,            // Human-readable name
    pub version: String,         // Semantic version
    pub author: String,          // Plugin author
    pub description: String,     // Brief description
    pub license: String,         // License type
}
```

### 3. PluginManager

Manages plugin lifecycle and execution:

```rust
pub struct PluginManager {
    // Register a new plugin
    pub fn register(&mut self, plugin: Box<dyn PhysicsPlugin>) -> KwaversResult<()>;
    
    // Compute execution order based on dependencies
    pub fn compute_execution_order(&mut self) -> KwaversResult<()>;
    
    // Execute all plugins for one time step
    pub fn update_all(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()>;
}
```

## Creating a Plugin

### Step 1: Define Your Plugin Structure

```rust
use kwavers::physics::{PhysicsPlugin, PluginMetadata, FieldType};

#[derive(Debug)]
struct MyCustomPlugin {
    metadata: PluginMetadata,
    // Your plugin-specific fields
    coefficient: f64,
    enabled: bool,
}
```

### Step 2: Implement the PhysicsPlugin Trait

```rust
impl PhysicsPlugin for MyCustomPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    
    fn required_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Pressure, FieldType::Velocity]
    }
    
    fn provided_fields(&self) -> Vec<FieldType> {
        vec![FieldType::Custom("MyOutput".to_string())]
    }
    
    fn initialize(
        &mut self,
        config: Option<Box<dyn PluginConfig>>,
        grid: &Grid,
        medium: &dyn Medium,
    ) -> KwaversResult<()> {
        // Perform initialization
        if let Some(cfg) = config {
            // Apply configuration
        }
        Ok(())
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &PluginContext,
    ) -> KwaversResult<()> {
        // Implement your physics calculations
        let pressure = &fields.slice(s![field_indices::PRESSURE, .., .., ..]);
        let velocity = &fields.slice(s![field_indices::VELOCITY_X, .., .., ..]);
        
        // Perform computations...
        
        Ok(())
    }
}
```

### Step 3: Register and Use Your Plugin

```rust
let mut manager = PluginManager::new();

// Create and register your plugin
let plugin = Box::new(MyCustomPlugin {
    metadata: PluginMetadata {
        id: "my_custom_plugin".to_string(),
        name: "My Custom Physics".to_string(),
        version: "1.0.0".to_string(),
        author: "Your Name".to_string(),
        description: "Custom physics implementation".to_string(),
        license: "MIT".to_string(),
    },
    coefficient: 1.5,
    enabled: true,
});

manager.register(plugin)?;
manager.compute_execution_order()?;

// Use in simulation loop
manager.update_all(&mut fields, &grid, &medium, dt, t, &context)?;
```

## Adapting Existing Components

The plugin system provides adapters to wrap existing PhysicsComponent implementations:

```rust
use kwavers::physics::plugin::adapters::ComponentPluginAdapter;

// Wrap an existing component
let acoustic_component = AcousticWaveComponent::new("acoustic".to_string());
let adapter = ComponentPluginAdapter::new(
    acoustic_component,
    PluginMetadata {
        id: "acoustic_wave".to_string(),
        name: "Acoustic Wave Propagation".to_string(),
        version: "1.0.0".to_string(),
        author: "Kwavers".to_string(),
        description: "Linear acoustic wave propagation".to_string(),
        license: "MIT".to_string(),
    },
);

manager.register(Box::new(adapter))?;
```

## Field Dependencies

Plugins declare their input and output fields, allowing the PluginManager to:
- Automatically resolve execution order
- Validate that all required fields are available
- Optimize execution by running independent plugins in parallel (future enhancement)

### Standard Field Types

```rust
pub enum FieldType {
    Pressure,
    Velocity,
    Temperature,
    Density,
    StressXX, StressYY, StressZZ,
    StressXY, StressXZ, StressYZ,
    Custom(String),
}
```

## Best Practices

### 1. Plugin Design
- Keep plugins focused on a single physics phenomenon
- Use descriptive IDs and metadata
- Properly declare all dependencies
- Validate configuration in the `initialize` method

### 2. Performance
- Minimize allocations in the `update` method
- Use SIMD operations where possible
- Cache frequently used calculations
- Profile your plugin's performance impact

### 3. Error Handling
- Return appropriate errors from `initialize` and `update`
- Validate inputs before processing
- Provide clear error messages

### 4. Testing
- Write unit tests for your plugin
- Test with various grid sizes and configurations
- Verify correct handling of edge cases
- Test interaction with other plugins

## Example Plugins

### 1. Frequency-Dependent Absorption Plugin

```rust
#[derive(Debug)]
struct FrequencyAbsorptionPlugin {
    metadata: PluginMetadata,
    absorption_coefficients: HashMap<f64, f64>,
}

impl FrequencyAbsorptionPlugin {
    fn calculate_absorption(&self, frequency: f64) -> f64 {
        // Interpolate absorption coefficient for given frequency
        // ...
    }
}
```

### 2. Nonlinear Enhancement Plugin

```rust
#[derive(Debug)]
struct NonlinearEnhancementPlugin {
    metadata: PluginMetadata,
    b_over_a: f64,
    threshold: f64,
}

impl NonlinearEnhancementPlugin {
    fn apply_nonlinear_effects(&self, pressure: &Array3<f64>) -> Array3<f64> {
        // Apply nonlinear acoustic effects
        // ...
    }
}
```

## Future Enhancements

1. **Dynamic Loading**: Support for loading plugins from shared libraries
2. **Plugin Repository**: Central repository for community plugins
3. **Parallel Execution**: Automatic parallel execution of independent plugins
4. **GPU Support**: CUDA/ROCm kernels for plugin computations
5. **Python Bindings**: Allow plugins written in Python
6. **Hot Reloading**: Update plugins without restarting simulation

## Troubleshooting

### Common Issues

1. **Circular Dependencies**: Ensure plugins don't create dependency cycles
2. **Missing Fields**: Verify all required fields are provided by other plugins
3. **Performance Impact**: Profile plugins to identify bottlenecks
4. **Memory Usage**: Monitor memory allocation in update methods

### Debug Tips

- Use the `RUST_LOG=debug` environment variable for detailed logging
- Check plugin execution order with `manager.execution_order`
- Validate plugins before running: `manager.validate_all(&grid, &medium)`
- Use performance metrics: `manager.get_all_metrics()`

## Contributing

To contribute a new plugin:

1. Follow the design principles outlined above
2. Ensure comprehensive test coverage
3. Document your plugin thoroughly
4. Submit a pull request with examples
5. Consider adding to the plugin gallery

For more information, see the [contribution guidelines](../CONTRIBUTING.md).