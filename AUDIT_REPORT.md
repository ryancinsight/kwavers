# Kwavers Codebase Comprehensive Audit Report
**Date**: 2026-01-29  
**Scope**: Complete kwavers v3.0.0 codebase audit  
**Status**: CRITICAL - Multiple compilation errors blocking build

---

## Executive Summary

The kwavers codebase has **6 compilation errors** and **31+ warnings** that prevent successful compilation. The primary issues are:
1. **Architectural breakage** from recent refactoring (removal of localization module)
2. **API mismatch** between distributed beamforming and PINN interface
3. **Missing imports** and unresolved types across multiple test files
4. **Dead code** and unused implementations in test/benchmark infrastructure

---

## 1. CRITICAL COMPILATION ERRORS

### 1.1 Error: Broken Sensor Array Module (BLOCKER)

**File**: `D:\kwavers\tests\sensor_delay_test.rs`  
**Lines**: 3-4  
**Severity**: CRITICAL - Build Failure

```rust
use kwavers::domain::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
use kwavers::domain::sensor::localization::Position;
```

**Issue**: Module `localization` was removed from `domain::sensor`. According to git history (commit 5c25ae1e), the `domain.sensor.localization` module was completely removed in a BREAKING refactor.

**Root Cause**: The `localization` submodule has been fully refactored/removed. The types now live directly in:
- `D:\kwavers\src\domain\sensor\array.rs` - Contains `Position`, `Sensor`, `SensorArray`, `ArrayGeometry`

**Fix Required**: Update imports in `tests/sensor_delay_test.rs`:
```rust
use kwavers::domain::sensor::array::{ArrayGeometry, Sensor, SensorArray, Position};
// OR
use kwavers::domain::sensor::{ArrayGeometry, Position, Sensor, SensorArray};
```

**Related Files**:
- `D:\kwavers\src\domain\sensor\mod.rs` - Shows re-exports: `pub use array::{ArrayGeometry, Position, Sensor, SensorArray};`
- `D:\kwavers\src\domain\sensor\array.rs` - Contains all the types

---

### 1.2 Error: NIFTI API Mismatch (BLOCKER)

**File**: `D:\kwavers\tests\ct_nifti_integration_test.rs`  
**Lines**: 46, 99, 223, 301, 348  
**Severity**: CRITICAL - Build Failure

```rust
let nifti = InMemNiftiObject::from_header_and_data(header, volume);
```

**Issue**: The `nifti` crate v0.17.0 doesn't expose `from_header_and_data()` method.

**Root Cause**: API mismatch with nifti crate. The method `from_header_and_data()` is not available in the public API.

**Available Methods** (from nifti v0.17.0):
- `GenericNiftiObject::from_file()` 
- `GenericNiftiObject::from_file_pair()`
- `GenericNiftiObject::from_reader()`

**Fix Required**: Refactor to use available APIs or create wrapper helper:
```rust
// Option 1: Write to temp file and read back
// Option 2: Use from_reader() with in-memory buffer
// Option 3: Manually construct using lower-level APIs
```

**Related Files**:
- `D:\kwavers\tests\ct_nifti_integration_test.rs` - Lines 46, 99, 223, 301, 348 (5 occurrences)
- All instances need coordinated fix

**Associated Error**: Line 47, 100, 224, 302, 349
```rust
WriterOptions::new(path).write_nifti(&nifti)?;
```
Returns `nifti::NiftiError` but function signature expects `std::io::Error` - need error conversion wrapper.

---

### 1.3 Error: Missing Simulation Factory (BLOCKER)

**File**: `D:\kwavers\examples\phase2_factory.rs`  
**Lines**: 8-9, 112, 122  
**Severity**: CRITICAL - Build Failure

```rust
use kwavers::simulation::factory::{
    AccuracyLevel, CFLCalculator, GridSpacingCalculator, SimulationFactory, SimulationPreset,
};
```

**Issue**: These types don't exist in `kwavers::simulation::factory` module.

**Root Cause**: The factory pattern types are missing or not exported from `simulation::factory` module.

**Fix Required**: Either:
1. Implement these types in `D:\kwavers\src\simulation\factory.rs`
2. Update imports to use actual types that exist in the codebase
3. Remove example if factory pattern is not yet implemented

---

### 1.4 Error: Distributed Beamforming API Mismatch (BLOCKER)

**File**: `D:\kwavers\src\analysis\signal_processing\beamforming\neural\distributed\core.rs`  
**Lines**: 269-297  
**Severity**: CRITICAL - Build Failure

```rust
let result = DistributedNeuralBeamformingProcessor::new(
    config,
    2,                                          // ERROR: Extra argument
    DecompositionStrategy::Spatial {
        dimensions: 3,                          // ERROR: Field doesn't exist
        overlap: 0.0,                          // ERROR: Field doesn't exist
    },
    LoadBalancingAlgorithm::Static,            // ERROR: Type undefined
)
.await;                                         // ERROR: Result is not Future
```

**Issues**:
- Line 276: `LoadBalancingAlgorithm` type not declared in this module
- Lines 269-277: Constructor signature mismatch - takes 2 args, 4 provided
- Line 273-274: `DecompositionStrategy::Spatial` doesn't have `dimensions`/`overlap` fields
- Line 278: Result is not awaitable

**Root Cause**: API refactoring incomplete. The `new()` method signature was changed but test code not updated.

**Fix Required**: 
1. Check `DistributedNeuralBeamformingProcessor::new()` signature (line 151)
2. Check `DecompositionStrategy` enum definition
3. Import or define `LoadBalancingAlgorithm`
4. Remove `.await` if not async

**Related Test Code Issues**:
- Line 297: Call to non-existent `initialize_communication_channels()` method

---

### 1.5 Warnings: Large Enum Variant (Performance Issue)

**File**: `D:\kwavers\src\analysis\signal_processing\beamforming\slsc\mod.rs`  
**Lines**: 143-152  
**Severity**: WARNING (clippy::large_enum_variant)

```rust
pub enum LagWeighting {
    Uniform,
    Triangular,
    Hamming,
    Custom { weights: [f64; 64], len: usize },  // 528+ bytes
}
```

**Issue**: `Custom` variant is 520+ bytes while others are 0-8 bytes. This causes enum to be at least 528 bytes.

**Recommendation**: Box the custom variant:
```rust
Custom { weights: Box<Vec<f64>>, len: usize },
```

**Impact**: Memory bloat for all enum instances, not just Custom variant.

---

### 1.6 Warning: Missing Debug Implementation

**File**: `D:\kwavers\src\solver\inverse\pinn\ml\beamforming_provider.rs`  
**Lines**: 34-45  
**Severity**: WARNING (missing_debug_implementations)

```rust
pub struct BurnPinnBeamformingAdapter<B: burn::tensor::backend::Backend> {
    /// Underlying Burn PINN model
    model: Arc<Mutex<Option<BurnPINN1DWave<B>>>>,
    // ... more fields
}
```

**Issue**: No `Debug` implementation. The generic type parameter `B` may not implement Debug.

**Fix**: Add conditional derive or manual implementation:
```rust
#[cfg_attr(feature = "pinn", derive(Debug))]
pub struct BurnPinnBeamformingAdapter<B> { ... }
```

---

## 2. WARNING ANALYSIS

### 2.1 Benchmark Issues (pinn_elastic_2d_training.rs)

**File**: `D:\kwavers\benches\pinn_elastic_2d_training.rs`

| Line | Issue | Type |
|------|-------|------|
| 121 | Unused variable `model` | warning::unused_variables |
| 211 | Unused variable `loss_computer` | warning::unused_variables |
| 58, 119, 208, 265, 404, 452 | Field reassignment after `Default::default()` | clippy::field_reassign_with_default |

**Fix**: Use struct literal initialization instead:
```rust
// Before (8 warnings)
let mut config = Config::default();
config.hidden_layers = vec![64, 64, 64];

// After (1 line, 0 warnings)
let config = Config { hidden_layers: vec![64, 64, 64], ..Default::default() };
```

**Affected Lines**: 58, 119, 208, 265, 404, 452 (6 warnings)

---

### 2.2 Test Framework Dead Code (elastic_wave_validation_framework.rs)

**File**: `D:\kwavers\tests\elastic_wave_validation_framework.rs`

| Lines | Item | Type | Status |
|-------|------|------|--------|
| 36-41 | `ValidationResult` fields: `error_l2`, `error_linf`, `tolerance` | dead_code | Never read |
| 109 | `validate_material_properties()` | dead_code | Never called |
| 331 | `validate_wave_speeds()` | dead_code | Never called |
| 461 | `PlaneWaveSolution::amplitude` field | dead_code | Never read |
| 528-555 | 4 methods: `displacement`, `velocity`, `acceleration`, `displacement_gradient` | dead_code | Never used |
| 584 | `validate_plane_wave_pde()` | dead_code | Never called |
| 681 | `validate_energy_conservation()` | dead_code | Never called |
| 704 | `run_full_validation_suite()` | dead_code | Never called |

**Total Dead Code**: 8 functions/methods + 4 unused fields in test framework

**Recommendation**: Either use these or remove them. If framework provides utilities, document which are for future use.

---

### 2.3 Unused Imports

**File**: `D:\kwavers\benches\hilbert_benchmark.rs` - Line 2
```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};
//              ^^^^^^^^ UNUSED
```

**File**: `D:\kwavers\tests\pinn_elastic_validation.rs` - Line 33-39
```rust
use ndarray::{Array2, ArrayD};  // UNUSED
//           ^^^^^^  ^^^^^^

use ... WaveType,                // Line 39 UNUSED
//           ^^^^^^^^
```

**Fix**: Remove unused imports

---

### 2.4 Unused Variables

| File | Line | Variable | Context |
|------|------|----------|---------|
| `benches/pinn_elastic_2d_training.rs` | 121 | `model` | Benchmark setup |
| `benches/pinn_elastic_2d_training.rs` | 211 | `loss_computer` | Benchmark criterion |
| `examples/monte_carlo_validation.rs` | 295 | `dims` | Unused parameter |
| `tests/pinn_elastic_validation.rs` | 311, 334 | `wave_vector` | 2 occurrences |
| `tests/pinn_elastic_validation.rs` | 312, 335 | `amplitude` | 2 occurrences |
| `src/analysis/signal_processing/beamforming/neural/pinn_interface.rs` | 402 | `registry` (mut) | Doesn't need mutability |

**Fix Pattern**: Prefix with underscore if intentional:
```rust
let _model = ...;
let mut _registry = ...;
```

---

## 3. ARCHITECTURAL ISSUES

### 3.1 Module Removal Breaking Change

**Issue**: Removal of `domain::sensor::localization` module (commit 5c25ae1e)

**Impact**:
- `tests/sensor_delay_test.rs` - Cannot import from removed module
- Test file still references old module path
- Indicates incomplete refactoring

**Files Affected**:
1. `D:\kwavers\tests\sensor_delay_test.rs` (Test file using old imports)
2. `D:\kwavers\src\domain\sensor\mod.rs` (Shows types moved to `array.rs`)
3. `D:\kwavers\src\domain\sensor\array.rs` (New home for types)

**Resolution**: Update all references to use new module structure

---

### 3.2 Solver-Agnostic Interface Incomplete

**Issue**: Distributed neural beamforming refactoring incomplete

**Evidence**:
- `D:\kwavers\src\analysis\signal_processing\beamforming\neural\distributed\core.rs` - Contains multiple TODOs
- Line 108: Comment states "TODO: Implement communication channel initialization"
- API signatures don't match test expectations (lines 269-297)
- Methods referenced in tests don't exist (initialize_communication_channels)

**Status**: Placeholder implementation in progress

**Files Affected**:
- `D:\kwavers\src\analysis\signal_processing\beamforming\neural\distributed\core.rs`
- `D:\kwavers\src\analysis\signal_processing\beamforming\neural\pinn_interface.rs`
- Tests expecting full implementation

---

### 3.3 NIFTI Integration Incomplete

**Issue**: External crate API incompatibility

**Problem**: Tests assume `from_header_and_data()` method exists in nifti crate, but it doesn't in v0.17.0.

**Solution Options**:
1. Create wrapper function in `infra::io::nifti` module
2. Use available nifti APIs (from_reader, from_file)
3. Update nifti crate version
4. Remove NIFTI tests if feature not ready

---

### 3.4 Simulation Factory Module (Missing Implementation)

**Issue**: Example code references non-existent factory types

**Missing Types**:
- `AccuracyLevel`
- `CFLCalculator`
- `GridSpacingCalculator`
- `SimulationFactory`
- `SimulationPreset`
- `PhysicsValidator`

**File**: `D:\kwavers\examples\phase2_factory.rs` (Lines 8-9, 112)

**Status**: Either:
1. Types not yet implemented
2. Types exported from wrong module
3. Example outdated

---

## 4. DEAD CODE & UNUSED IMPLEMENTATIONS

### 4.1 Test Framework Dead Code

**Location**: `D:\kwavers\tests\elastic_wave_validation_framework.rs`

**Dead Items** (verified by clippy):

```rust
// Unused fields (4)
pub struct ValidationResult {
    pub error_l2: f64,           // LINE 39 - Never read
    pub error_linf: f64,         // LINE 40 - Never read
    pub tolerance: f64,          // LINE 41 - Never read
}

pub struct PlaneWaveSolution {
    pub amplitude: f64,          // LINE 461 - Never read
}

// Unused functions (4 + 4 methods)
pub fn validate_material_properties<T>(...) -> ValidationResult  // LINE 109
pub fn validate_wave_speeds<T>(...) -> ...                       // LINE 331
pub fn validate_plane_wave_pde<T>(...) -> ...                    // LINE 584
pub fn validate_energy_conservation<T>(...) -> ...               // LINE 681
pub fn run_full_validation_suite<T>(...) -> ...                  // LINE 704

// Unused methods (4)
impl PlaneWaveSolution {
    pub fn displacement(&self, ...) -> [f64; 2]                  // LINE 528
    pub fn velocity(&self, ...) -> [f64; 2]                      // LINE 537
    pub fn acceleration(&self, ...) -> [f64; 2]                  // LINE 546
    pub fn displacement_gradient(&self, ...) -> [[f64; 2]; 2]   // LINE 555
}
```

**Decision Required**: Keep for future use or delete?

### 4.2 Performance/SIMD Dead Code

**Location**: `D:\kwavers\src\analysis\performance/`

Multiple modules have `#[allow(dead_code)]` attributes:
- `cache.rs` - Line unreported
- `memory.rs` - Line unreported
- `mod.rs` - Line unreported
- `simd.rs` - Multiple lines

These may be intentional (for future feature enabling) but should be documented.

---

## 5. LARGE FILES ANALYSIS

### 5.1 Top 30 Largest Files

| Rank | File | Size | Lines |
|------|------|------|-------|
| 1 | `src/domain/boundary/coupling.rs` | Large | 1827 |
| 2 | `src/infra/api/clinical_handlers.rs` | Large | 1116 |
| 3 | `src/solver/forward/hybrid/bem_fem_coupling.rs` | Large | 1015 |
| 4 | `src/clinical/therapy/swe_3d_workflows.rs` | Large | 985 |
| 5 | `src/solver/inverse/pinn/ml/electromagnetic_gpu.rs` | Large | 966 |
| 6 | `src/physics/optics/sonoluminescence/emission.rs` | Large | 957 |
| 7 | `src/solver/forward/bem/solver.rs` | Large | 947 |
| 8 | `src/solver/inverse/pinn/ml/burn_wave_equation_3d/solver.rs` | Large | 922 |
| 9 | `src/solver/inverse/pinn/ml/universal_solver.rs` | Large | 913 |
| 10 | `src/clinical/safety.rs` | Large | 880 |

**Recommendations for Large Files**:
1. `src/domain/boundary/coupling.rs` (1827 lines) - Split boundary and coupling concerns
2. `src/solver/inverse/pinn/ml/burn_wave_equation_3d/solver.rs` (922 lines) - Extract specific solvers
3. `src/solver/inverse/pinn/ml/universal_solver.rs` (913 lines) - Extract backend adapters

**Observation**: Large files often indicate mixed concerns or could benefit from extraction.

---

## 6. FEATURE FLAG USAGE

### 6.1 Conditional Code with #[allow(...)]

Code marked with `#[allow(dead_code)]` that may be feature-gated:

**Files**:
- `src/analysis/signal_processing/beamforming/three_dimensional/apodization.rs` - 2 items
- `src/analysis/signal_processing/beamforming/three_dimensional/delay_sum.rs` - 3 items
- `src/analysis/signal_processing/beamforming/three_dimensional/metrics.rs` - 1 item

**Recommendation**: Use `#[cfg(feature = "...")]` instead of allow() for clarity

---

## 7. DUPLICATE IMPLEMENTATIONS

### 7.1 ArrayGeometry Re-export

**Location**: `D:\kwavers\src\domain\sensor\mod.rs`

```rust
pub use array::{ArrayGeometry, Position, Sensor, SensorArray};
pub use passive_acoustic_mapping::{
    ArrayElement, ArrayGeometry as PAMArrayGeometry, DirectivityPattern,
};
```

**Issue**: Two types with similar names:
- `array::ArrayGeometry` - Sensor array geometry
- `passive_acoustic_mapping::ArrayGeometry` (aliased as PAMArrayGeometry)

**Risk**: Potential confusion or accidental import of wrong type. Consider more distinctive naming.

---

## 8. CRATE DEPENDENCY ISSUES

### 8.1 NIFTI Crate Compatibility

**Dependency**: `nifti = "0.17.0"` in Cargo.toml

**Problem**: Expected method `from_header_and_data()` doesn't exist

**Verification Needed**:
1. Check if upgrade to newer nifti version available
2. Check API documentation for v0.17.0
3. Determine proper way to create NiftiObject in-memory

---

## 9. BUILD CONFIGURATION ISSUES

### 9.1 PINN Feature Gate Conditional Code

**File**: `D:\kwavers\src\analysis\signal_processing\beamforming\neural\distributed\core.rs`

Multiple items have `#[cfg(feature = "pinn")]` guards:
- Lines 77-164: Struct definitions
- Lines 88-97: Impl Debug
- Lines 128-158: FaultToleranceState impl
- Lines 262-310: processor implementation

**Status**: Properly gated, but test code at lines 262-310 has bugs that prevent compilation when feature is enabled

---

## 10. CODE QUALITY METRICS

### 10.1 Summary Statistics

| Metric | Value | Status |
|--------|-------|--------|
| Compilation Errors | 6 | CRITICAL |
| Warnings | 31+ | WARNING |
| Dead Code Items | 12 | REVIEW |
| Large Files (>900 lines) | 10 | REFACTOR |
| Unused Imports | 3 | FIX |
| Unused Variables | 6 | FIX |
| Field Reassignments | 6 | FIX |
| Tests with Dead Code | 1 | REVIEW |
| allow(dead_code) Attributes | 20+ | AUDIT |

### 10.2 Code Health Indicators

**Architecture**:
- ✅ Good layering (domain, analysis, solver, infra)
- ❌ Incomplete refactoring (sensor localization removal)
- ⚠️ Placeholder implementations (distributed beamforming)

**Testing**:
- ✅ Good test coverage
- ❌ Dead code in test framework
- ⚠️ Tests referencing removed/missing APIs

**Documentation**:
- ✅ Module documentation present
- ✅ Architecture diagrams included
- ❌ Missing implementation status for placeholder code

---

## 11. PRIORITY REMEDIATION PLAN

### Phase 1: FIX BLOCKERS (Required for Compilation)

**Must Complete Before Any PR**:

1. **Fix sensor_delay_test.rs imports** (5 mins)
   - Update import paths for relocated types
   - File: `D:\kwavers\tests\sensor_delay_test.rs` lines 3-4

2. **Fix NIFTI API calls** (30 mins)
   - Refactor ct_nifti_integration_test.rs to use available APIs
   - File: `D:\kwavers\tests\ct_nifti_integration_test.rs` lines 46, 99, 223, 301, 348
   - Create error conversion wrapper for NiftiError → io::Error

3. **Fix factory example/remove** (15 mins)
   - Either implement missing types or remove example
   - File: `D:\kwavers\examples\phase2_factory.rs` lines 8-9, 112

4. **Fix distributed beamforming API** (1 hour)
   - Update test code to match actual API
   - Verify API signature matches usage
   - File: `D:\kwavers\src/analysis/signal_processing/beamforming/neural/distributed/core.rs` lines 269-297

### Phase 2: FIX WARNINGS (Code Quality)

**Priority High - Prevents clippy warnings**:

1. **Fix LagWeighting enum** (10 mins)
   - Box the Custom variant
   - File: `D:\kwavers\src/analysis/signal_processing/beamforming/slsc/mod.rs` line 151

2. **Add Debug implementation** (10 mins)
   - Conditional derive for BurnPinnBeamformingAdapter
   - File: `D:\kwavers\src/solver/inverse/pinn/ml/beamforming_provider.rs` line 34

3. **Fix benchmark field reassignments** (10 mins)
   - Use struct literal initialization
   - File: `D:\kwavers\benches/pinn_elastic_2d_training.rs` lines 58, 119, 208, 265, 404, 452

4. **Fix unused imports** (5 mins)
   - Remove black_box, Array2, ArrayD, WaveType
   - Files: hilbert_benchmark.rs, pinn_elastic_validation.rs

5. **Fix unused variables** (5 mins)
   - Prefix with underscore or use in code
   - Multiple files

### Phase 3: CLEAN UP DEAD CODE (Optional)

1. **Review test framework dead code** (1 hour)
   - Decide what to keep/remove
   - File: `D:\kwavers\tests/elastic_wave_validation_framework.rs`

2. **Document feature-gated allow()** (30 mins)
   - Add comments explaining why dead_code is allowed
   - Files: beamforming/three_dimensional/*.rs

---

## 12. SPECIFIC FILE FIX RECOMMENDATIONS

### Fix #1: sensor_delay_test.rs

**Current** (lines 3-4):
```rust
use kwavers::domain::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
use kwavers::domain::sensor::localization::Position;
```

**Corrected**:
```rust
use kwavers::domain::sensor::array::{ArrayGeometry, Sensor, SensorArray, Position};
// OR (more concise)
use kwavers::domain::sensor::{ArrayGeometry, Position, Sensor, SensorArray};
```

---

### Fix #2: ct_nifti_integration_test.rs

**Current** (line 46):
```rust
let nifti = InMemNiftiObject::from_header_and_data(header, volume);
WriterOptions::new(path).write_nifti(&nifti)?;
```

**Corrected** (Option A - use in-memory approach):
```rust
// Create in-memory representation
let nifti = InMemNiftiObject::from_reader(vec![])?; // Placeholder approach
// Better: Implement wrapper function in infra::io::nifti module
```

---

### Fix #3: pinn_elastic_2d_training.rs

**Current** (lines 57-58):
```rust
let mut config = Config::default();
config.hidden_layers = vec![64, 64, 64];
```

**Corrected**:
```rust
let config = Config { 
    hidden_layers: vec![64, 64, 64], 
    ..Default::default() 
};
```

---

### Fix #4: slsc/mod.rs

**Current** (lines 143-152):
```rust
pub enum LagWeighting {
    // ...
    Custom { weights: [f64; 64], len: usize },
}
```

**Corrected** (if only Custom variant is large):
```rust
pub enum LagWeighting {
    Uniform,
    Triangular,
    Hamming,
    Custom(Box<CustomWeights>),
}

pub struct CustomWeights {
    weights: Vec<f64>,
}
```

---

## 13. VERIFICATION CHECKLIST

After fixes are applied:

```
□ cargo clippy --all-features --all-targets passes with 0 errors
□ cargo build --release succeeds
□ cargo test --all passes
□ No dead_code warnings in production code (allow! in tests only)
□ All TODO comments documented with tickets or removal plan
□ Distributed beamforming refactoring complete or clearly marked as WIP
□ NIFTI integration either working or disabled with clear reason
□ Factory example either implemented or removed
□ All unused variables prefixed with _ or used
□ Large files reviewed for refactoring opportunities
```

---

## 14. SUMMARY TABLE

| Category | Count | Severity | Status |
|----------|-------|----------|--------|
| **Compilation Errors** | 6 | CRITICAL | Blocking |
| **Warnings** | 31+ | High | Needs Fix |
| **Dead Code Items** | 12 | Medium | Review |
| **Architecture Issues** | 4 | Medium | Plan |
| **Large Files** | 10 | Low | Consider |
| **Feature Gaps** | 3 | Medium | Plan |

**Total Issues Found**: 66+

**Estimated Fix Time**: 3-4 hours for all fixes

---

## 15. RECOMMENDATIONS

1. **Establish Build Gate**: Make `cargo clippy --all-features --all-targets` pass as CI requirement
2. **Document Placeholder Code**: Mark all WIP/placeholder implementations clearly
3. **Regular Audits**: Schedule quarterly code audits to catch issues early
4. **Deprecation Plan**: Document how deprecated modules should be migrated (like localization)
5. **Dead Code Policy**: Decide on dead code tolerance for feature-gated code
6. **Large File Limits**: Set soft limits (900-1000 lines) and review larger files

---

**Report Generated**: 2026-01-29  
**Total Findings**: 66+ issues across 15 categories  
**Critical Path**: 6 compilation errors must be fixed
