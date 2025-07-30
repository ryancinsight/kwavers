# Sonoluminescence Physics Module Improvements

## Overview

This document summarizes the comprehensive improvements made to the kwavers physics module to support detailed sonoluminescence simulations based on scientific literature.

## Enhanced Physics Modules

### 1. Mechanics/Cavitation Enhancements

#### Improved Bubble Dynamics (dynamics.rs)
- **Keller-Miksis Equation**: Implemented compressible Rayleigh-Plesset equation
  - Accounts for liquid compressibility effects
  - Mach number corrections for high-speed bubble wall motion
  - Pressure gradient terms for acoustic radiation

- **Enhanced Thermal Model**:
  - Peclet number-based effective polytropic index
  - Shock heating during violent collapse (Rankine-Hugoniot relations)
  - Radiation damping for small bubbles
  - Van der Waals hard-core corrections for extreme compression

- **Additional State Variables**:
  - Equilibrium radius tracking (r0)
  - Gas and vapor molecule counts (n_gas, n_vapor)
  - Internal pressure calculation
  - Maximum temperature and compression tracking

### 2. Optics/Sonoluminescence Module

#### Blackbody Radiation (blackbody.rs)
- **Planck's Law Implementation**:
  - Full spectral radiance calculation
  - Wien and Rayleigh-Jeans approximations for efficiency
  - Emissivity and optical depth corrections
  - Color temperature estimation

- **Key Features**:
  - Temperature range: 1,000 - 50,000 K
  - Wavelength range: 200-800 nm (UV to near-IR)
  - Stefan-Boltzmann total power calculation

#### Bremsstrahlung Emission (bremsstrahlung.rs)
- **Free-Free Emission Model**:
  - Quantum-corrected emission coefficients
  - Gaunt factor implementation
  - Temperature-dependent ionization (Saha equation)

- **Plasma Parameters**:
  - Ion charge states
  - Electron density calculations
  - Spectral emission for weakly ionized plasma

#### Spectral Analysis (spectral.rs)
- **Analysis Tools**:
  - Wavelength-to-RGB conversion
  - Peak wavelength detection
  - FWHM calculation
  - Time-resolved spectral tracking
  - Spectral integration

### 3. Chemistry/ROS-Plasma Module

#### Reactive Oxygen Species (ros_species.rs)
- **Comprehensive ROS Tracking**:
  - Hydroxyl radical (•OH)
  - Hydrogen peroxide (H₂O₂)
  - Superoxide (O₂•⁻)
  - Singlet oxygen (¹O₂)
  - Ozone (O₃)
  - Atomic species (H, O)

- **Physical Properties**:
  - Diffusion coefficients
  - Lifetimes in water
  - Reduction potentials
  - Oxidative stress calculation

#### Plasma Chemistry (plasma_reactions.rs)
- **High-Temperature Reactions**:
  - Water dissociation: H₂O → H + OH
  - Oxygen dissociation: O₂ → 2O
  - Nitrogen reactions (Zeldovich NO mechanism)
  - Ionization at T > 10,000 K

- **Reaction Kinetics**:
  - Modified Arrhenius equations
  - Temperature-dependent rate constants
  - Equilibrium composition solver

#### Radical Kinetics (radical_kinetics.rs)
- **Aqueous Phase Reactions**:
  - OH radical recombination
  - Superoxide dismutation
  - Fenton-like reactions
  - pH-dependent kinetics

- **Diffusion and Transport**:
  - 3D diffusion solver
  - Radical lifetime calculations
  - Scavenger reaction support

## Example Implementations

### Single-Bubble Sonoluminescence (SBSL)
Based on Gaitan et al. (1992) and Brenner et al. (2002):
- Frequency: 26.5 kHz
- Pressure: 1.2-1.5 atm
- Bubble radius: 4-5 μm
- Noble gas (Argon)
- Degassed water

**Key Physics**:
- Maximum compression ratio: 10-15x
- Peak temperature: 10,000-50,000 K
- Light pulse duration: 50-300 ps
- Photon yield: 10⁴-10⁶ per flash

### Multi-Bubble Sonoluminescence (MBSL)
Based on Yasui et al. (2008) and Lauterborn & Kurz (2010):
- Frequency: 100-500 kHz
- Pressure: 2+ atm
- Bubble density: 10⁹-10¹⁰ m⁻³
- Size distribution: Lognormal
- Aerated water

**Key Physics**:
- Bubble-bubble interactions (Bjerknes forces)
- Cloud dynamics
- Enhanced sonochemical yields
- Spatial ROS distribution

## Scientific Validation

### Literature Comparisons
1. **Compression Ratios**: Match experimental values (10-15 for SBSL)
2. **Temperature Peaks**: Consistent with spectral measurements
3. **Pulse Duration**: Picosecond timescales verified
4. **Chemical Yields**: OH radical production rates match dosimetry

### Numerical Methods
- **Keller-Miksis**: 2nd-order accurate compressible bubble dynamics
- **Spectral Integration**: Trapezoidal rule with adaptive wavelength spacing
- **Chemical Kinetics**: Explicit Euler with stability checks
- **Diffusion**: Central differences with CFL condition

## Usage Guidelines

### For SBSL Simulations
```rust
// Key parameters for stable SBSL
let params = SBSLParameters {
    frequency: 26.5e3,
    pressure_amplitude: 1.35 * 101325.0,
    equilibrium_radius: 4.5e-6,
    // ...
};
```

### For MBSL Simulations
```rust
// Parameters for cavitation clouds
let params = MBSLParameters {
    frequency: 200e3,
    bubble_density: 1e9,
    size_distribution: "lognormal",
    // ...
};
```

## Future Enhancements

1. **Multi-species diffusion** in bubble interior
2. **Non-equilibrium plasma** models
3. **Molecular dynamics** at bubble interface
4. **GPU acceleration** for large-scale MBSL
5. **Adaptive mesh refinement** near bubbles

## References

1. Brenner, M. P., Hilgenfeldt, S., & Lohse, D. (2002). Single-bubble sonoluminescence. Reviews of Modern Physics, 74(2), 425.
2. Gaitan, D. F., et al. (1992). Sonoluminescence and bubble dynamics for a single, stable, cavitation bubble. JASA, 91(6), 3166-3183.
3. Yasui, K. (1997). Alternative model of single-bubble sonoluminescence. Physical Review E, 56(6), 6750.
4. Suslick, K. S., & Flannigan, D. J. (2008). Inside a collapsing bubble. Annual Review of Physical Chemistry, 59, 659-683.