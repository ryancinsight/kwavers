# Physics Module Architecture

## Overview

The kwavers physics modules have been reorganized to follow a clear separation of concerns, where each module focuses on its specific domain of physics while sharing common data through well-defined interfaces.

## Core Architecture

### 1. Bubble Dynamics Module (`physics/bubble_dynamics/`)
**Purpose**: Core bubble physics calculations shared by all other modules

**Components**:
- `bubble_state.rs`: Complete bubble state (radius, temperature, pressure, etc.)
- `rayleigh_plesset.rs`: Bubble dynamics equations (Rayleigh-Plesset, Keller-Miksis)
- `bubble_field.rs`: Management of single and multiple bubbles
- `interactions.rs`: Bubble-bubble interactions (Bjerknes forces)

**Key Features**:
- Compressible bubble dynamics (Keller-Miksis equation)
- Thermal effects and heat transfer
- Mass transfer (evaporation/condensation)
- Support for different gas species
- Bubble cloud dynamics

### 2. Mechanics Module (`physics/mechanics/`)
**Purpose**: Mechanical effects and damage from cavitation

**Cavitation Submodule** (`mechanics/cavitation/`):
- `damage.rs`: Cavitation erosion and material fatigue
  - Impact pressure calculations
  - Erosion rate modeling
  - Fatigue damage accumulation
  - Material property definitions

**Key Features**:
- Water hammer and jet impact pressure
- Material-specific damage models
- Erosion depth predictions
- Mean time to failure calculations

### 3. Optics Module (`physics/optics/`)
**Purpose**: Light emission from sonoluminescence

**Sonoluminescence Submodule** (`optics/sonoluminescence/`):
- `blackbody.rs`: Thermal radiation (Planck's law)
- `bremsstrahlung.rs`: Free-free emission from plasma
- `spectral.rs`: Spectral analysis tools
- `emission.rs`: Integrated emission calculations

**Key Features**:
- Temperature-dependent emission
- Multiple radiation mechanisms
- Spectral analysis (peak wavelength, FWHM)
- Time-resolved emission tracking

### 4. Chemistry Module (`physics/chemistry/`)
**Purpose**: Chemical reactions and ROS generation

**ROS-Plasma Submodule** (`chemistry/ros_plasma/`):
- `ros_species.rs`: Reactive oxygen species definitions
- `plasma_reactions.rs`: High-temperature plasma chemistry
- `radical_kinetics.rs`: Aqueous phase radical reactions
- `sonochemistry.rs`: Integrated sonochemical effects

**Key Features**:
- Comprehensive ROS tracking (•OH, H₂O₂, O₂•⁻, etc.)
- Temperature-dependent reaction rates
- pH-dependent kinetics
- Diffusion and decay modeling

## Data Flow

```
┌─────────────────────┐
│  Acoustic Field     │
│  (Pressure, dp/dt)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Bubble Dynamics    │◄─── Core Physics Engine
│  - Radius           │
│  - Temperature      │
│  - Pressure         │
│  - Velocity         │
│  - Collapse state   │
└──────────┬──────────┘
           │
           ├─────────────┬─────────────┬─────────────┐
           ▼             ▼             ▼             ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Mechanics   │ │   Optics     │ │  Chemistry   │ │Thermodynamics│
│              │ │              │ │              │ │              │
│ - Damage     │ │ - Light      │ │ - ROS        │ │ - Heat       │
│ - Erosion    │ │ - Spectrum   │ │ - Reactions  │ │ - Shock      │
│ - Fatigue    │ │ - Photons    │ │ - pH         │ │ - Transfer   │
└──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘
```

## Usage Example

```rust
// 1. Initialize bubble dynamics
let bubble_params = BubbleParameters {
    r0: 5e-6,  // 5 μm bubble
    gas_species: GasSpecies::Argon,
    // ... other parameters
};
let mut bubble_field = BubbleField::new(grid_shape, bubble_params);

// 2. Initialize physics modules
let mut damage = CavitationDamage::new(grid_shape, material, params);
let mut light = SonoluminescenceEmission::new(grid_shape, emission_params);
let mut chemistry = SonochemistryModel::new(nx, ny, nz, initial_ph);

// 3. Main simulation loop
for step in 0..n_steps {
    // Update bubble dynamics
    bubble_field.update(&pressure, &dp_dt, dt, t);
    
    // Get bubble states
    let states = bubble_field.get_state_fields();
    
    // Update each physics domain
    damage.update_damage(&states, liquid_props, dt);
    light.calculate_emission(&states.temperature, &states.pressure, &states.radius, t);
    // chemistry.update_ros_generation(...);
}
```

## Benefits of This Architecture

1. **Separation of Concerns**: Each module focuses on its specific physics domain
2. **Reusability**: Bubble dynamics can be used by any module that needs it
3. **Modularity**: Easy to add new physics modules or enhance existing ones
4. **Physical Accuracy**: Each domain uses appropriate models from literature
5. **Extensibility**: New phenomena can be added without disrupting existing code

## Key Improvements from Previous Version

1. **Removed Legacy Cavitation Model**: The old model mixed bubble dynamics with effects
2. **Proper Module Boundaries**: Clear separation between dynamics, damage, light, and chemistry
3. **Shared State Management**: BubbleStateFields provides consistent interface
4. **Literature-Based Models**: Each module implements established scientific models
5. **Comprehensive Testing**: Unit tests for each physics component

## Future Enhancements

1. **GPU Acceleration**: Parallel computation for bubble clouds
2. **Adaptive Time Stepping**: Different time scales for different physics
3. **Multi-Scale Modeling**: Coupling molecular dynamics at interfaces
4. **Machine Learning**: Parameter optimization from experimental data
5. **Real-Time Visualization**: 3D rendering of bubble dynamics and effects