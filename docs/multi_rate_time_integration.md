# Multi-Rate Time Integration Documentation

## Overview

The Multi-Rate Time Integration module provides sophisticated time stepping methods that allow different physics components to evolve at different time scales. This is essential for efficiently simulating multi-physics problems where different phenomena have vastly different characteristic time scales.

## Key Features

- **Multiple Time Steppers**: Support for Runge-Kutta, Adams-Bashforth, and other methods
- **Adaptive Time Stepping**: Automatic time step adjustment based on error estimation
- **Multi-Rate Control**: Intelligent management of different time scales
- **Stability Analysis**: CFL condition monitoring and enforcement
- **Flexible Coupling**: Multiple strategies for coupling physics components

## Architecture

The module follows SOLID design principles with clear separation of concerns:

```
time_integration/
├── mod.rs              # Main module and MultiRateTimeIntegrator
├── traits.rs           # Common interfaces (ISP)
├── time_stepper.rs     # Time stepping methods (SRP)
├── adaptive_stepping.rs # Adaptive control (SRP)
├── multi_rate_controller.rs # Multi-rate management
├── stability.rs        # Stability analysis
├── coupling.rs         # Coupling strategies
└── tests.rs           # Comprehensive tests
```

## Usage Examples

### Basic Time Stepping

```rust
use kwavers::solver::time_integration::*;

// Create a 4th-order Runge-Kutta stepper
let config = RK4Config::default();
let mut stepper = RungeKutta4::new(config);

// Define your RHS function
let rhs_fn = |field: &Array3<f64>| -> KwaversResult<Array3<f64>> {
    // Compute time derivatives
    physics.evaluate(field, &grid)
};

// Take a time step
let new_field = stepper.step(&field, rhs_fn, dt, &grid)?;
```

### Adaptive Time Stepping

```rust
// Create adaptive time stepper
let base_stepper = RungeKutta4::new(RK4Config::default());
let low_order = RungeKutta4::new(RK4Config::default());
let error_estimator = Box::new(RichardsonErrorEstimator::new(4));

let mut adaptive = AdaptiveTimeStepper::new(
    base_stepper,
    low_order,
    error_estimator,
    0.01,    // initial dt
    1e-6,    // min dt
    1.0,     // max dt
    1e-4,    // tolerance
);

// Take adaptive step
let (new_field, actual_dt) = adaptive.adaptive_step(&field, rhs_fn, &grid)?;
```

### Multi-Rate Integration

```rust
// Configure multi-rate integration
let mut config = MultiRateConfig::default();
config.max_subcycles = 10;
config.stability_factor = 0.9;

// Create integrator
let mut integrator = MultiRateTimeIntegrator::new(config);

// Set up physics components
let mut fields = HashMap::new();
fields.insert("acoustic".to_string(), acoustic_field);
fields.insert("thermal".to_string(), thermal_field);

let mut physics = HashMap::new();
physics.insert("acoustic".to_string(), Box::new(acoustic_physics));
physics.insert("thermal".to_string(), Box::new(thermal_physics));

// Advance the coupled system
let final_time = integrator.advance(
    &mut fields,
    &physics,
    0.0,      // start time
    1.0,      // target time
    &grid,
)?;
```

## Time Stepping Methods

### Runge-Kutta 4 (RK4)
- **Order**: 4th
- **Stages**: 4
- **Stability**: Good for moderate CFL numbers
- **Use Case**: General purpose, high accuracy

### Adams-Bashforth
- **Order**: 2nd or 3rd
- **Stages**: 1 (multi-step)
- **Stability**: More restrictive than RK4
- **Use Case**: Efficient for smooth solutions

### Forward Euler
- **Order**: 1st
- **Stages**: 1
- **Stability**: Most restrictive
- **Use Case**: Simple problems, debugging

## Coupling Strategies

### Subcycling Strategy
Each component advances with its own time step, with synchronization at global time steps:
- Fast components take multiple small steps
- Slow components take fewer large steps
- Synchronization ensures consistency

### Averaging Strategy
Time-averaged coupling between components:
- Components advance independently
- Coupling terms are time-averaged
- Higher-order interpolation available

### Predictor-Corrector Strategy
Iterative coupling for strong interactions:
- Predictor step with extrapolated values
- Corrector iterations for consistency
- Configurable iteration count

## Stability Analysis

The module provides comprehensive stability analysis:

```rust
let analyzer = StabilityAnalyzer::new(0.9); // 90% of CFL limit

// Compute stable time step
let max_dt = analyzer.compute_stable_dt(&physics, &field, &grid)?;

// Check stability
let cfl = CFLCondition::new(dt, max_wave_speed, &grid, 0.9);
println!("{}", cfl.report());
```

## Performance Considerations

1. **Efficiency Ratio**: Monitor the efficiency compared to single-rate integration
2. **Subcycle Limits**: Balance accuracy vs computational cost
3. **Memory Usage**: Time steppers cache intermediate results
4. **Parallel Potential**: Different components can be advanced in parallel

## Best Practices

1. **Choose Appropriate Methods**:
   - RK4 for general problems
   - Adams-Bashforth for smooth solutions
   - Adaptive stepping for varying time scales

2. **Set Reasonable Tolerances**:
   - Too tight: Excessive computation
   - Too loose: Inaccurate results

3. **Monitor Stability**:
   - Check CFL conditions regularly
   - Use safety factors (typically 0.8-0.9)

4. **Profile Multi-Rate Performance**:
   - Use statistics to verify efficiency gains
   - Adjust subcycle limits based on results

## Error Estimation

Two error estimators are provided:

### Richardson Extrapolation
- Compares solutions at different orders
- Provides reliable error estimates
- Computational overhead: ~2x

### Embedded Methods
- Uses embedded error estimates
- Lower computational cost
- Less reliable for stiff problems

## API Reference

### Core Types

- `MultiRateTimeIntegrator`: Main integration controller
- `TimeStepper`: Trait for time stepping methods
- `AdaptiveTimeStepper`: Wrapper for adaptive control
- `MultiRateConfig`: Configuration structure

### Key Methods

- `advance()`: Advance the multi-physics system
- `compute_stable_dt()`: Calculate CFL-limited time step
- `get_statistics()`: Retrieve performance metrics

## Future Enhancements

- Implicit time stepping methods
- IMEX (Implicit-Explicit) schemes
- Higher-order coupling strategies
- GPU acceleration for subcycling