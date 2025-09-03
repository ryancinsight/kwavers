# KWave Parity Module Refactoring Plan

## Current Issues (520 lines - SLAP violation)
- Monolithic structure mixing configuration, solver logic, and utilities
- Multiple responsibilities violating SOC
- Complex nested implementations

## Proposed Structure

### 1. `config.rs` (~50 lines)
- `KWaveConfig` struct
- `AbsorptionMode` enum
- Default implementations

### 2. `solver.rs` (~200 lines)
- `KWaveSolver` struct
- Core step() method
- Field management

### 3. `operators.rs` (~100 lines)
- K-space operator computation
- FFT utilities
- Spectral derivatives

### 4. `velocity.rs` (~100 lines)
- Velocity field updates
- PML application
- Source injection

### 5. `pressure.rs` (~100 lines)
- Pressure field updates
- Nonlinear terms
- Source smoothing

This refactoring enforces SLAP by separating concerns and keeping each module focused on a single abstraction level.