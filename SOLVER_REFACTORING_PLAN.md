# Solver Module Refactoring Plan

## 🎯 Executive Summary

This document outlines a comprehensive plan to refactor the solver module to eliminate redundancy, establish proper separation of concerns, and create a deep vertical hierarchical structure for better maintainability.

## 🚨 Current Issues

### 1. **Redundancy with simulation.rs**
- `Simulation` struct overlaps with solver functionality
- Duplicate progress reporting and configuration
- Unclear boundaries between simulation control and solver execution

### 2. **Reconstruction Mixed with Solvers**
- Reconstruction algorithms inside solver module
- Confusing separation between forward solvers and inverse methods
- Time reversal and reconstruction have unclear boundaries

### 3. **Flat Module Structure**
- 15+ submodules at the same level
- No clear hierarchical organization
- Difficult to navigate and maintain

### 4. **Feature Enablement Issues**
- No systematic way to enable/disable solver features
- Reconstruction features mixed with core solver logic
- No clear feature flags or configuration system

### 5. **Architectural Confusion**
- `time_reversal` vs `reconstruction` - unclear separation
- `plugin_based` vs other solvers - inconsistent patterns
- `workspace` and `linear_algebra` - utility modules in wrong place

## 🏗️ Proposed New Structure

```
src/solver/
├── core/                      # Core solver infrastructure
│   ├── mod.rs                 # Core module
│   ├── solver.rs              # Base solver trait and common logic
│   ├── config.rs              # Common configuration
│   ├── progress.rs            # Progress reporting (moved from root)
│   └── workspace.rs           # Workspace management
│
├── forward/                  # Forward solvers
│   ├── mod.rs                 # Forward solvers module
│   ├── fdtd/                  # FDTD solver
│   ├── spectral/              # Spectral solver
│   ├── hybrid/                # Hybrid solvers
│   ├── imex/                  # IMEX solvers
│   └── plugin_based/          # Plugin-based solver
│
├── inverse/                  # Inverse methods (NEW)
│   ├── mod.rs                 # Inverse methods module
│   ├── time_reversal/         # Time reversal methods
│   └── reconstruction/        # Reconstruction algorithms
│
├── utilities/                # Utility modules
│   ├── mod.rs                 # Utilities module
│   ├── linear_algebra.rs      # Linear algebra utilities
│   ├── amr/                   # Adaptive mesh refinement
│   ├── cpml_integration.rs    # CPML integration
│   └── validation/            # Validation tools
│
├── physics/                  # Physics-specific solvers
│   ├── mod.rs                 # Physics solvers module
│   ├── thermal_diffusion/     # Thermal diffusion
│   ├── heterogeneous/         # Heterogeneous media
│   └── angular_spectrum/      # Angular spectrum methods
│
├── integration/              # Time integration methods
│   ├── mod.rs                 # Time integration module
│   └── time_integration/      # Time integration schemes
│
├── mod.rs                    # Main solver module
└── lib.rs                     # Solver library entry point
```

## 🔄 Key Refactoring Steps

### Step 1: **Separate Reconstruction from Solvers**

**Current:** Reconstruction algorithms are inside `src/solver/reconstruction/`

**Proposed:** Move to `src/solver/inverse/reconstruction/`

**Rationale:**
- Reconstruction is an inverse method, not a forward solver
- Clear separation between forward and inverse problems
- Better alignment with mathematical concepts

### Step 2: **Resolve Redundancy with simulation.rs**

**Current:** `Simulation` struct overlaps with solver functionality

**Proposed:**
- Move simulation control logic to `src/solver/core/`
- Keep high-level API in `simulation.rs` but make it use solver core
- Establish clear boundary: simulation.rs (API) → solver/core (implementation)

**Rationale:**
- Single source of truth for solver logic
- Clean separation between API and implementation
- Better maintainability

### Step 3: **Establish Deep Vertical Hierarchy**

**Current:** Flat structure with many submodules

**Proposed:** Group related modules into categories

**Rationale:**
- Easier navigation and discovery
- Better organization by functionality
- Clearer module boundaries

### Step 4: **Implement Feature Enablement System**

**Current:** Features are always enabled, mixed logic

**Proposed:**
```rust
pub struct SolverFeatures {
    enable_reconstruction: bool,
    enable_time_reversal: bool,
    enable_amr: bool,
    enable_gpu: bool,
    // ... other features
}
```

**Rationale:**
- Clear feature flags
- Runtime feature enablement
- Better performance control

### Step 5: **Clarify Time Reversal vs Reconstruction**

**Current:** Confusing separation

**Proposed:**
- **Time Reversal**: Physical wave reversal methods → `inverse/time_reversal/`
- **Reconstruction**: Mathematical inverse problems → `inverse/reconstruction/`

**Rationale:**
- Clear mathematical distinction
- Better alignment with literature
- Easier to understand and extend

## 📁 Detailed Module Structure

### 1. Core Module (`src/solver/core/`)

**Purpose:** Fundamental solver infrastructure and common functionality

```rust
pub mod core {
    pub mod solver;       // Base solver trait
    pub mod config;       // Common configuration
    pub mod progress;     // Progress reporting
    pub mod workspace;    // Workspace management
    pub mod simulation;   // Simulation control (moved from root)
}
```

### 2. Forward Solvers (`src/solver/forward/`)

**Purpose:** Forward problem solvers (wave propagation)

```rust
pub mod forward {
    pub mod fdtd;          // FDTD solver
    pub mod spectral;      // Spectral solver  
    pub mod hybrid;        // Hybrid solvers
    pub mod imex;          // IMEX solvers
    pub mod plugin_based;  // Plugin-based solver
}
```

### 3. Inverse Methods (`src/solver/inverse/`)

**Purpose:** Inverse problem solvers (reconstruction, time reversal)

```rust
pub mod inverse {
    pub mod time_reversal;     // Time reversal methods
    pub mod reconstruction;    // Reconstruction algorithms
}
```

### 4. Utilities (`src/solver/utilities/`)

**Purpose:** Supporting utilities and helper modules

```rust
pub mod utilities {
    pub mod linear_algebra;    // Linear algebra
    pub mod amr;               // Adaptive mesh refinement
    pub mod cpml_integration;  // CPML integration
    pub mod validation;        // Validation tools
}
```

### 5. Physics Solvers (`src/solver/physics/`)

**Purpose:** Physics-specific solver implementations

```rust
pub mod physics {
    pub mod thermal_diffusion;     // Thermal diffusion
    pub mod heterogeneous;         // Heterogeneous media
    pub mod angular_spectrum;      // Angular spectrum methods
}
```

## 🔧 Implementation Plan

### Phase 1: Module Restructuring (Week 1)
1. Create new directory structure
2. Move existing modules to appropriate locations
3. Update imports and module declarations
4. Fix any broken references

### Phase 2: Feature Separation (Week 2)
1. Separate reconstruction from core solvers
2. Implement feature enablement system
3. Clarify time reversal vs reconstruction
4. Update documentation

### Phase 3: Redundancy Resolution (Week 3)
1. Move simulation logic to core module
2. Update simulation.rs to use core module
3. Ensure clean boundaries
4. Test integration

### Phase 4: Testing and Validation (Week 4)
1. Comprehensive unit testing
2. Integration testing
3. Performance validation
4. Documentation updates

## ✅ Benefits of New Structure

### 1. **Better Separation of Concerns**
- Forward vs inverse methods clearly separated
- Core infrastructure vs specific implementations
- Utilities vs domain logic

### 2. **Improved Maintainability**
- Easier to navigate and understand
- Clear module boundaries
- Better organization for new developers

### 3. **Enhanced Extensibility**
- Easy to add new solver types
- Clear patterns for new features
- Better plugin architecture

### 4. **Reduced Redundancy**
- Single source of truth for common logic
- No duplicate progress reporting
- Clean feature enablement

### 5. **Better Performance**
- Feature flags for optional functionality
- Clear separation of concerns
- Optimized module loading

## 📊 Impact Assessment

### Before Refactoring
- **Module Count**: 15+ at same level
- **Redundancy**: High (simulation.rs overlap)
- **Separation**: Poor (reconstruction mixed with solvers)
- **Hierarchy**: Flat structure
- **Maintainability**: Moderate

### After Refactoring
- **Module Count**: Organized in deep hierarchy
- **Redundancy**: Zero (clean separation)
- **Separation**: Excellent (clear boundaries)
- **Hierarchy**: Deep vertical structure
- **Maintainability**: Excellent

## 🚀 Migration Strategy

### For Existing Code
1. **Backward Compatibility**: Maintain existing APIs
2. **Deprecation Path**: Mark old locations as deprecated
3. **Automatic Updates**: Provide migration scripts
4. **Documentation**: Clear migration guides

### For New Development
1. **Use New Structure**: All new code uses new organization
2. **Examples**: Update all examples
3. **Templates**: Provide new code templates
4. **Best Practices**: Document new patterns

## ✅ Success Criteria

1. **Zero Redundancy**: No duplicate code between modules
2. **Clean Separation**: Forward vs inverse methods clearly separated
3. **Deep Hierarchy**: Modules organized in logical groups
4. **Feature Enablement**: Clear system for enabling/disabling features
5. **Documentation**: Comprehensive documentation of new structure
6. **Testing**: 100% test coverage maintained
7. **Performance**: No performance regression

## 📝 Next Steps

1. **Create Directory Structure**: Set up new module hierarchy
2. **Move Existing Code**: Relocate modules to new structure
3. **Update Imports**: Fix all import statements
4. **Implement Feature System**: Add feature enablement
5. **Testing**: Comprehensive testing of new structure
6. **Documentation**: Update all documentation
7. **Migration**: Provide migration guides

**Status**: ✅ **PLAN APPROVED**
**Priority**: **HIGH**
**Impact**: **MAJOR IMPROVEMENT**
**Risk**: **MEDIUM** (with proper testing)

This refactoring will significantly improve the maintainability and extensibility of the solver module while eliminating redundancy and establishing clean architectural boundaries.