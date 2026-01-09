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

### 1. Plugin Trait

The main interface that all plugins must implement:

```rust
use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::Medium;
use kwavers::domain::source::Source;
use kwavers::domain::boundary::Boundary;
use kwavers::physics::field_mapping::UnifiedFieldType;
use kwavers::physics::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use ndarray::Array4;
use std::any::Any;
use std::fmt::Debug;

pub trait Plugin: Debug + Send + Sync {
    /// Get plugin metadata
    fn metadata(&self) -> &PluginMetadata;

    /// Get current plugin state
    fn state(&self) -> PluginState;

    /// Set plugin state
    fn set_state(&mut self, state: PluginState);

    /// Get required fields for this plugin
    fn required_fields(&self) -> Vec<UnifiedFieldType>;

    /// Get fields provided by this plugin
    fn provided_fields(&self) -> Vec<UnifiedFieldType>;

    /// Update physics for one time step
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()>;

    fn initialize(&mut self, _grid: &Grid, _medium: &dyn Medium) -> KwaversResult<()> {
        Ok(())
    }

    fn finalize(&mut self) -> KwaversResult<()> {
        Ok(())
    }

    fn reset(&mut self) -> KwaversResult<()> {
        Ok(())
    }

    fn diagnostics(&self) -> String {
        format!("Plugin: {:?}", self.metadata())
    }

    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
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
    // Create a new plugin manager
    pub fn new() -> Self;

    // Add a plugin to the manager (dependency order resolved internally)
    pub fn add_plugin(&mut self, plugin: Box<dyn Plugin>) -> KwaversResult<()>;

    // Initialize all plugins
    pub fn initialize(&mut self, grid: &Grid, medium: &dyn Medium) -> KwaversResult<()>;

    // Execute all plugins for one time step
    pub fn execute(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        sources: &[Box<dyn Source>],
        boundary: &mut dyn Boundary,
        dt: f64,
        t: f64,
    ) -> KwaversResult<()>;
}
```

## Creating a Plugin

### Step 1: Define Your Plugin Structure

```rust
use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::domain::medium::Medium;
use kwavers::physics::field_mapping::UnifiedFieldType;
use kwavers::physics::plugin::{Plugin, PluginContext, PluginMetadata, PluginState};
use ndarray::Array4;
use std::any::Any;
use std::fmt::Debug;

#[derive(Debug)]
struct MyCustomPlugin {
    metadata: PluginMetadata,
    state: PluginState,
    // Your plugin-specific fields
    coefficient: f64,
    enabled: bool,
}
```

### Step 2: Implement the Plugin Trait

```rust
impl Plugin for MyCustomPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }

    fn state(&self) -> PluginState {
        self.state
    }

    fn set_state(&mut self, state: PluginState) {
        self.state = state;
    }

    fn required_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }

    fn provided_fields(&self) -> Vec<UnifiedFieldType> {
        vec![UnifiedFieldType::Pressure]
    }
    
    fn update(
        &mut self,
        fields: &mut Array4<f64>,
        grid: &Grid,
        medium: &dyn Medium,
        dt: f64,
        t: f64,
        context: &mut PluginContext<'_>,
    ) -> KwaversResult<()> {
        let _ = (fields, grid, medium, dt, t, context);
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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
    state: PluginState::Created,
    coefficient: 1.5,
    enabled: true,
});

manager.add_plugin(plugin)?;

// Use in simulation loop
manager.initialize(&grid, &medium)?;
manager.execute(&mut fields, &grid, &medium, &sources, &mut boundary, dt, t)?;
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
