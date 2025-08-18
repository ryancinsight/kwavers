# Kwavers Codebase Refactoring Plan

## Critical Issues Found

### 1. Naming Violations (Adjective-based names)
- **File**: `examples/phase31_advanced_capabilities.rs` → Rename to `examples/phase31_capabilities.rs`
- **Documentation**: Multiple files contain adjectives in comments like "improved", "optimized", "simple", "advanced", "better", "fast", "accurate", "robust"
- **Line 14 in core.rs**: "improved nonlinear wave model solver" → "nonlinear wave model solver"

### 2. Large Files Requiring Splitting (>500 lines)
Files exceeding 500 lines that violate SRP:
- `src/solver/hybrid/domain_decomposition.rs` (1370 lines)
- `src/solver/hybrid/coupling_interface.rs` (1355 lines)
- `src/solver/hybrid/mod.rs` (1290 lines)
- `src/medium/homogeneous/mod.rs` (1178 lines)
- `src/solver/pstd/mod.rs` (1137 lines)
- `src/physics/validation_tests.rs` (1103 lines)
- `src/physics/mechanics/acoustic_wave/nonlinear/core.rs` (1073 lines)
- `src/solver/mod.rs` (1041 lines)
- `src/solver/fdtd/mod.rs` (1003 lines)

### 3. Incomplete Implementations
- TODO in `src/solver/fdtd/mod.rs`: "Implement cubic or higher-order interpolation"

### 4. Magic Numbers
- Need to verify all numeric literals are properly defined as constants

## Refactoring Actions

### Phase 1: Fix Naming Violations
1. Rename file with adjective in name
2. Update all documentation to remove subjective adjectives
3. Fix struct/function documentation

### Phase 2: Split Large Modules
1. Split `solver/hybrid/` into submodules:
   - `domain_decomposition/` directory with:
     - `mod.rs` - core traits
     - `partitioning.rs` - domain partitioning logic
     - `load_balancing.rs` - load balancing algorithms
     - `communication.rs` - inter-domain communication
   
2. Split `physics/mechanics/acoustic_wave/nonlinear/core.rs`:
   - `nonlinear/mod.rs` - module interface
   - `nonlinear/solver.rs` - main solver implementation
   - `nonlinear/k_space.rs` - k-space operations
   - `nonlinear/stability.rs` - stability analysis
   - `nonlinear/heterogeneous.rs` - heterogeneous media handling

### Phase 3: Complete Implementations
1. Implement the cubic interpolation TODO or document why it's deferred

### Phase 4: Validate Physics
1. Cross-reference numerical methods with literature
2. Ensure all physics constants match published values
3. Verify solver stability conditions