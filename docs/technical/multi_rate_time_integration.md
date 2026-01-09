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
solver/integration/time_integration/
├── mod.rs                   # MultiRateTimeIntegrator + re-exports
├── traits.rs                # Interfaces and configuration types
├── time_stepper.rs          # Runge-Kutta, Adams-Bashforth implementations
├── adaptive_stepping.rs     # AdaptiveTimeStepper + error estimators
├── multi_rate_controller.rs # Subcycling orchestration
├── time_scale_separation.rs # TimeScaleSeparator
├── stability.rs             # StabilityAnalyzer + CFLCondition
├── coupling.rs              # Coupling strategies
└── tests.rs                 # Module tests
```

## Usage Examples

### Basic Time Stepping

```rust
use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::solver::time_integration::{RungeKutta4, TimeStepper};
use kwavers::solver::time_integration::time_stepper::RK4Config;
use ndarray::Array3;

fn rk4_step_zero_rhs() -> KwaversResult<()> {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).expect("grid creation failed");
    let mut field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

    let rhs_fn = |y: &Array3<f64>| -> KwaversResult<Array3<f64>> {
        Ok(Array3::zeros(y.dim()))
    };

    let mut stepper = RungeKutta4::new(RK4Config::default());
    stepper.step(&mut field, rhs_fn, 1e-7, &grid)?;

    Ok(())
}
```

### Adaptive Time Stepping

```rust
use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::solver::time_integration::{AdaptiveTimeStepper, ErrorEstimator, RungeKutta4, TimeStepper};
use kwavers::solver::time_integration::time_stepper::RK4Config;
use ndarray::Array3;

fn adaptive_step_zero_rhs() -> KwaversResult<()> {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).expect("grid creation failed");
    let field = Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz));

    let rhs_fn = |y: &Array3<f64>| -> KwaversResult<Array3<f64>> {
        Ok(Array3::zeros(y.dim()))
    };

    let base_stepper = RungeKutta4::new(RK4Config::default());
    let low_order_stepper = RungeKutta4::new(RK4Config::default());

    let mut adaptive = AdaptiveTimeStepper::new(
        base_stepper,
        low_order_stepper,
        Box::new(ErrorEstimator::new(4)),
        1e-7,
        1e-10,
        1e-4,
        1e-8,
    );

    let (_new_field, _actual_dt) = adaptive.adaptive_step(&field, rhs_fn, &grid)?;
    Ok(())
}
```

### Multi-Rate Integration

```rust
use kwavers::core::error::KwaversResult;
use kwavers::domain::grid::Grid;
use kwavers::physics::plugin::AcousticWavePlugin;
use kwavers::solver::time_integration::{MultiRateConfig, MultiRateTimeIntegrator};
use ndarray::Array3;
use std::collections::HashMap;

fn multirate_advance_example() -> KwaversResult<()> {
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3).expect("grid creation failed");

    let mut config = MultiRateConfig::default();
    config.max_subcycles = 10;
    config.stability_factor = 0.9;

    let mut integrator = MultiRateTimeIntegrator::new(config, &grid);

    let mut fields = HashMap::new();
    fields.insert("acoustic".to_string(), Array3::<f64>::zeros((grid.nx, grid.ny, grid.nz)));

    let mut physics_components: HashMap<String, Box<dyn kwavers::physics::plugin::Plugin>> =
        HashMap::new();
    physics_components.insert("acoustic".to_string(), Box::new(AcousticWavePlugin::new(0.3)));

    let _final_time = integrator.advance(&mut fields, &physics_components, 0.0, 1e-6, &grid)?;
    Ok(())
}
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
