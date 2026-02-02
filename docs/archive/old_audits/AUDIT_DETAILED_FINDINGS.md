# Detailed Audit Findings - Code Snippets and Line Numbers

## Table of Contents
1. [Compilation Errors - Detailed Analysis](#compilation-errors)
2. [Warning Details](#warnings)
3. [Dead Code Inventory](#dead-code)
4. [allow(dead_code) Attributes](#allow-attributes)
5. [File Size Analysis](#file-sizes)

---

## COMPILATION ERRORS

### ERROR 1: sensor_delay_test.rs - Removed Module Reference

**File**: `D:\kwavers\tests\sensor_delay_test.rs`

```
ERROR MESSAGES:
error[E0432]: unresolved import `kwavers::domain::sensor::localization`
error[E0433]: failed to resolve: could not find `localization` in `sensor`
```

**Affected Lines**:
- Line 3: `use kwavers::domain::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};`
- Line 4: `use kwavers::domain::sensor::localization::Position;`

**Root Cause Location**: Git commit 5c25ae1e - refactor(BREAKING): Remove deprecated domain.sensor.localization module completely

**Current Module Structure** (D:\kwavers\src\domain\sensor\mod.rs):
```rust
pub mod array;                     // Sensor array geometry (domain concept)
pub mod beamforming;
pub mod grid_sampling;
pub mod passive_acoustic_mapping;
pub mod recorder;
pub mod sonoluminescence;
pub mod ultrafast;

// Re-exported at module root:
pub use array::{ArrayGeometry, Position, Sensor, SensorArray};
```

**Types Now Located In**: `D:\kwavers\src\domain\sensor\array.rs`
- struct `Position` (lines 6-50)
- struct `Sensor` (lines 52-83)
- enum `ArrayGeometry` (lines 85-95)
- struct `SensorArray` (lines 97-140)

**Fix**:
```rust
// WRONG (removed module)
use kwavers::domain::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
use kwavers::domain::sensor::localization::Position;

// CORRECT (updated paths)
use kwavers::domain::sensor::array::{ArrayGeometry, Position, Sensor, SensorArray};
// OR shorter:
use kwavers::domain::sensor::{ArrayGeometry, Position, Sensor, SensorArray};
```

---

### ERROR 2: ct_nifti_integration_test.rs - API Mismatch (5 instances)

**File**: `D:\kwavers\tests\ct_nifti_integration_test.rs`

**Error Type**: `error[E0433]` - failed to resolve: use of undeclared type `NiftiObject`

**Affected Line Groups**:

**Instance 1 - Lines 46-47**:
```rust
46:     let nifti = InMemNiftiObject::from_header_and_data(header, volume);
47:     WriterOptions::new(path).write_nifti(&nifti)?;
```
Error: `E0599` - no function or associated item named `from_header_and_data` found

**Instance 2 - Lines 99-100**:
```rust
99:     let nifti = NiftiObject::from_header_and_data(header, volume);
100:    WriterOptions::new(path).write_nifti(&nifti)?;
```
Error: `E0433` - failed to resolve: use of undeclared type `NiftiObject`
Error: `E0599` - no function or associated item named `from_header_and_data`

**Instance 3 - Lines 223-224**:
```rust
223:    let nifti = NiftiObject::from_header_and_data(header, volume);
224:    WriterOptions::new(path).write_nifti(&nifti)?;
```

**Instance 4 - Lines 301-302**:
```rust
301:    let nifti = NiftiObject::from_header_and_data(header, volume);
302:    WriterOptions::new(path).write_nifti(&nifti)?;
```

**Instance 5 - Lines 348-349**:
```rust
348:    let nifti = NiftiObject::from_header_and_data(header, original_ct.clone());
349:    WriterOptions::new(path).write_nifti(&nifti)?;
```

**Root Cause**: 
1. `nifti::NiftiObject` is not imported (line 99 context shows no import)
2. Method `from_header_and_data()` doesn't exist in nifti v0.17.0 API
3. Available methods per nifti v0.17.0 docs:
   - `GenericNiftiObject::from_file(path: P) -> Result<Self>`
   - `GenericNiftiObject::from_file_pair(hdr: P, vol: Q) -> Result<Self>`
   - `GenericNiftiObject::from_reader(source: R) -> Result<Self>`

**Error Chain**:
```
E0599 at line 46: from_header_and_data() doesn't exist
     help: try from_reader() instead
E0277 at line 47: Result<_, NiftiError> cannot convert to io::Error
     help: need From<NiftiError> for io::Error
```

**Additional Error at Line 47, 100, 224, 302, 349**:
```rust
WriterOptions::new(path).write_nifti(&nifti)?;
```
`error[E0277]`: `?` couldn't convert the error to `std::io::Error`

This is because `write_nifti()` returns `Result<_, NiftiError>` but function signature is:
```rust
fn create_synthetic_nifti(...) -> std::io::Result<()>
```

**Fix Approach**:
Option 1 - Use in-memory approach with from_reader:
```rust
let mut cursor = std::io::Cursor::new(Vec::new());
// Write header to cursor...
// Use from_reader on cursor
```

Option 2 - Create wrapper function:
```rust
fn write_nifti_to_file(nifti: &InMemNiftiObject, path: &str) -> KwaversResult<()> {
    WriterOptions::new(path).write_nifti(&nifti)
        .map_err(|e| KwaversError::IO(format!("NIFTI error: {}", e)))
}
```

Option 3 - Change return type:
```rust
fn create_synthetic_nifti(...) -> nifti::Result<()> {
    // ...
    WriterOptions::new(path).write_nifti(&nifti)?;
    Ok(())
}
```

---

### ERROR 3: phase2_factory.rs - Missing Factory Types

**File**: `D:\kwavers\examples\phase2_factory.rs`

**Error**: `error[E0432]` - unresolved imports

**Affected Lines - Import Block (lines 8-9)**:
```rust
use kwavers::simulation::factory::{
    AccuracyLevel, CFLCalculator, GridSpacingCalculator, SimulationFactory, SimulationPreset,
};
                                   ^ ^ ^ ^ 
                                   | | | |
                                   All MISSING from module
```

**Missing Types**:
- `AccuracyLevel` - Not found in `simulation::factory`
- `CFLCalculator` - Not found in `simulation::factory`
- `GridSpacingCalculator` - Not found in `simulation::factory`
- `SimulationFactory` - Not found in `simulation::factory`
- `SimulationPreset` - Not found in `simulation::factory`

**Additional Error at Line 112**:
```rust
error[E0432]: unresolved import `kwavers::simulation::factory::PhysicsValidator`
   --> examples\phase2_factory.rs:112:9
112 |     use kwavers::simulation::factory::PhysicsValidator;
```

**Additional Error at Line 122**:
```rust
error[E0282]: type annotations needed
   --> examples\phase2_factory.rs:122:13
122 |             report.print();
       ^^^^^^ cannot infer type for variable `report`
```
This cascades from the missing types above.

**Current simulation::factory status**: 
Check `D:\kwavers\src\simulation\factory.rs` to see what actually exists.

**Fix Options**:
1. Implement all missing types in `simulation/factory.rs`
2. Change imports to use actual types in the codebase
3. Remove example if factory pattern not yet implemented

---

### ERROR 4: distributed/core.rs - API Mismatch (5 errors at lines 269-278)

**File**: `D:\kwavers\src\analysis\signal_processing\beamforming\neural\distributed\core.rs`

**Context**: Test function starting at line 262

**Error 1 - Line 269: Wrong constructor arguments**
```rust
269: let result = DistributedNeuralBeamformingProcessor::new(
270:     config,
271:     2,                                              // ERROR: Extra argument
272:     DecompositionStrategy::Spatial {
273:         dimensions: 3,                              // ERROR: Field doesn't exist
274:         overlap: 0.0,                               // ERROR: Field doesn't exist
275:     },
276:     LoadBalancingAlgorithm::Static,                // ERROR: Type undefined
277: )
278: .await;                                             // ERROR: Not a future
```

**Actual Constructor Signature** (lines 151-153):
```rust
pub fn new(
    beamforming_config: PINNBeamformingConfig,
    distributed_config: DistributedConfig,
) -> KwaversResult<Self> {
```

**Takes 2 args**, not 4. Constructor expects:
- `PINNBeamformingConfig` - NOT provided
- `DistributedConfig` - NOT provided
- All other args (2, DecompositionStrategy, LoadBalancingAlgorithm) are unexpected

**Error Details**:
```
error[E0061]: this function takes 2 arguments but 4 arguments were supplied
error[E0559]: variant `DecompositionStrategy::Spatial` has no field named `dimensions`
error[E0559]: variant `DecompositionStrategy::Spatial` has no field named `overlap`
error[E0433]: failed to resolve: use of undeclared type `LoadBalancingAlgorithm`
error[E0277]: `Result<...>` is not a future (cannot .await)
```

**Error 2 - Line 297: Non-existent method**
```rust
297: DistributedNeuralBeamformingProcessor::initialize_communication_channels(4).unwrap();
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 
     This method doesn't exist
```

**Actual Available Methods** (lines 167-183 show):
- `config(&self) -> &DistributedConfig`
- `decomposition_strategy(&self) -> &DecompositionStrategy`
- `load_balancing_strategy(&self) -> &LoadBalancingStrategy`
- `gpu_devices(&self) -> &[usize]`
- `num_processors(&self) -> usize`
- `metrics(&self) -> &DistributedNeuralBeamformingMetrics`
- `fault_tolerance_config(&self) -> &FaultToleranceState`
- `process_volume_distributed(...) -> KwaversResult<...>` (not implemented)

**Status**: Test code (lines 262-310) is outdated and doesn't match current API.

**Fix**: Update test to use actual API or mark as WIP/skip.

---

## WARNINGS

### WARNING 1: Large Enum Variant - slsc/mod.rs

**File**: `D:\kwavers\src\analysis\signal_processing\beamforming\slsc\mod.rs`

**Warning**: `large_enum_variant` (clippy::large_enum_variant)

**Location - Lines 143-152**:
```rust
143 | pub enum LagWeighting {
144 |     /// Uniform weighting (all lags equal)
145 |     Uniform,
146 |     /// Triangular weighting (linear decrease with lag)
147 |     Triangular,
148 |     /// Hamming window weighting
149 |     Hamming,
150 |     /// Custom weighting with user-defined weights
151 |     Custom { weights: [f64; 64], len: usize },
    |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

**Size Analysis**:
- `Uniform`: 0 bytes
- `Triangular`: 0 bytes
- `Hamming`: 0 bytes
- `Custom`: 8 bytes (usize) + 512 bytes ([f64; 64]) = 520 bytes
- **Enum total**: 528 bytes (all instances)

**Why It Matters**: Every instance of `LagWeighting` is 528 bytes, even if it's just `Uniform`.

**Suggested Fix**:
```rust
pub enum LagWeighting {
    Uniform,
    Triangular,
    Hamming,
    Custom(Box<CustomWeights>),  // Indirect reference only
}

pub struct CustomWeights {
    weights: Vec<f64>,  // Only allocated if needed
    len: usize,
}
```

Or use Box<[f64]>:
```rust
Custom { weights: Box<[f64]>, },
```

---

### WARNING 2: Missing Debug Implementation - beamforming_provider.rs

**File**: `D:\kwavers\src\solver\inverse\pinn\ml\beamforming_provider.rs`

**Warning**: `missing_debug_implementations`

**Location - Lines 34-45**:
```rust
34 | pub struct BurnPinnBeamformingAdapter<B: burn::tensor::backend::Backend> {
35 |     /// Underlying Burn PINN model
36 |     model: Arc<Mutex<Option<BurnPINN1DWave<B>>>>,
37 |     /// Model configuration
38 |     config: BurnPINNConfig,
39 |     /// Backend device
40 |     device: B::Device,
41 |     /// Is model trained?
42 |     is_trained: bool,
43 |     /// Model metadata
44 |     metadata: ModelInfo,
45 | }
```

**Issue**: The generic type `B` may not implement `Debug`, so derive won't work unconditionally.

**Fix**:
```rust
#[cfg_attr(feature = "pinn", derive(Debug))]
pub struct BurnPinnBeamformingAdapter<B: burn::tensor::backend::Backend> {
    // ...
}

// Or manual impl if needed:
impl<B: burn::tensor::backend::Backend> std::fmt::Debug 
    for BurnPinnBeamformingAdapter<B> 
where
    B::Device: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BurnPinnBeamformingAdapter")
            .field("is_trained", &self.is_trained)
            .finish()
    }
}
```

---

### WARNING 3: Unused Variables - pinn_elastic_2d_training.rs

**File**: `D:\kwavers\benches\pinn_elastic_2d_training.rs`

**Warning**: `unused_variables`

**Instance 1 - Line 121**:
```rust
121 | let model = ElasticPINN2D::<Backend>::new(&config, &device)
    |     ^^^^^ help: if this is intentional, prefix with an underscore: `_model`
    |
    = note: `#[warn(unused)]` is reported here
```

**Instance 2 - Line 211**:
```rust
211 | let loss_computer = LossComputer::new(config.loss_weights);
    |     ^^^^^^^^^^^^^ help: if this is intentional, prefix with an underscore: `_loss_computer`
```

**Fix**: Either use the variable or prefix with underscore:
```rust
let _model = ElasticPINN2D::<Backend>::new(&config, &device)?;
let _loss_computer = LossComputer::new(config.loss_weights);
```

---

### WARNING 4: Field Reassignment - pinn_elastic_2d_training.rs (6 instances)

**File**: `D:\kwavers\benches\pinn_elastic_2d_training.rs`

**Warning**: `field_reassign_with_default` (clippy::field_reassign_with_default)

**All Instances**:

**Line 57-58**:
```rust
57: let mut config = Config::default();
58: config.hidden_layers = vec![64, 64, 64];
```
Should be: `let config = Config { hidden_layers: vec![64, 64, 64], ..Default::default() };`

**Line 118-119**:
```rust
118: let mut config = Config::default();
119: config.hidden_layers = vec![64, 64, 64];
```

**Line 207-208**:
```rust
207: let mut config = Config::default();
208: config.hidden_layers = vec![64, 64, 64];
```

**Line 264-265**:
```rust
264: let mut config = Config::default();
265: config.hidden_layers = vec![64, 64, 64];
```
Note: This one also has additional fields:
```rust
265: config.hidden_layers = vec![64, 64, 64];
266: config.n_collocation_interior = 1000;
267: config.n_collocation_boundary = 100;
268: config.n_collocation_initial = 100;
```

**Line 403-404**:
```rust
403: let mut config = Config::default();
404: config.hidden_layers = layers.clone();
```

**Line 451-452**:
```rust
451: let mut config = Config::default();
452: config.hidden_layers = vec![64, 64, 64];
```

**Pattern**: All 6 instances follow the same antipattern of creating default then reassigning fields.

---

### WARNING 5: Unused Imports (3 instances)

**Instance 1 - benches/hilbert_benchmark.rs, Line 2**:
```rust
2 | use criterion::{black_box, criterion_group, criterion_main, Criterion};
  |                 ^^^^^^^^ 
```
Remove: `black_box` is imported but never used.

**Instance 2 - tests/pinn_elastic_validation.rs, Lines 33, 39**:
```rust
33 | use ndarray::{Array2, ArrayD};
   |              ^^^^^^  ^^^^^^
39 | use ... WaveType,
   |          ^^^^^^^^
```
Three unused imports: `Array2`, `ArrayD`, `WaveType`

---

### WARNING 6: Unused Variables - Additional (4 instances)

**Instance 1 - tests/pinn_elastic_validation.rs, Line 311**:
```rust
311 | let wave_vector = [2.0 * std::f64::consts::PI, 0.0];
    |     ^^^^^^^^^^^ help: if this is intentional, prefix with an underscore: `_wave_vector`
```

**Instance 2 - tests/pinn_elastic_validation.rs, Line 312**:
```rust
312 | let amplitude = 1e-6;
    |     ^^^^^^^^^ help: if this is intentional, prefix with an underscore: `_amplitude`
```

**Instance 3 - tests/pinn_elastic_validation.rs, Line 334**:
```rust
334 | let wave_vector = [k / std::f64::consts::SQRT_2, k / std::f64::consts::SQRT_2];
    |     ^^^^^^^^^^^ help: if this is intentional, prefix with an underscore: `_wave_vector`
```

**Instance 4 - tests/pinn_elastic_validation.rs, Line 335**:
```rust
335 | let amplitude = 1e-6;
    |     ^^^^^^^^^ help: if this is intentional, prefix with an underscore: `_amplitude`
```

**Instance 5 - examples/monte_carlo_validation.rs, Line 295**:
```rust
295 | dims: GridDimensions,
    |     ^^^^ help: if this is intentional, prefix with an underscore: `_dims`
```

**Instance 6 - pinn_interface.rs, Line 402**:
```rust
402 | let mut registry = PinnProviderRegistry::new();
    |     ----^^^^^^^^
```
Variable is mutable but doesn't need to be.

---

### WARNING 7: Mutable Variable Not Needed - pinn_interface.rs

**File**: `D:\kwavers\src\analysis\signal_processing\beamforming\neural\pinn_interface.rs`

**Warning**: `variable_not_need_mut`

**Location - Line 402**:
```rust
402 | let mut registry = PinnProviderRegistry::new();
    |     ---- help: remove this `mut`
```

The variable `registry` is never mutated. Should be:
```rust
let registry = PinnProviderRegistry::new();
```

---

## DEAD CODE

### DEAD CODE INVENTORY - elastic_wave_validation_framework.rs

**File**: `D:\kwavers\tests\elastic_wave_validation_framework.rs`

**Total Dead Code Items**: 8 functions/methods + 4 fields

### Dead Fields:

**1. ValidationResult::error_l2 (Line 39)**
```rust
36 | pub struct ValidationResult {
37 |     pub pass: bool,
38 |     pub message: String,
39 |     pub error_l2: f64,  // DEAD - Never read
   |         ^^^^^^^^
40 |     pub error_linf: f64,  // DEAD - Never read
   |         ^^^^^^^^^^
41 |     pub tolerance: f64,  // DEAD - Never read
   |         ^^^^^^^^^
```

**2. PlaneWaveSolution::amplitude (Line 461)**
```rust
458 | pub struct PlaneWaveSolution {
459 |     pub wavenumber: f64,
460 |     pub frequency: f64,
461 |     pub amplitude: f64,  // DEAD - Never read
   |         ^^^^^^^^^
462 |     pub wave_speed: f64,
```

### Dead Functions:

**3. validate_material_properties (Line 109)**
```rust
109 | pub fn validate_material_properties<T: ElasticWaveEquation>(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |        Function is never called
```

**4. validate_wave_speeds (Line 331)**
```rust
331 | pub fn validate_wave_speeds<T: ElasticWaveEquation>(
    |        ^^^^^^^^^^^^^^^^^^^^
    |        Function is never called
```

**5. validate_plane_wave_pde (Line 584)**
```rust
584 | pub fn validate_plane_wave_pde<T: ElasticWaveEquation>(
    |        ^^^^^^^^^^^^^^^^^^^^^^^
    |        Function is never called
```

**6. validate_energy_conservation (Line 681)**
```rust
681 | pub fn validate_energy_conservation<T: ElasticWaveEquation>(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |        Function is never called
```

**7. run_full_validation_suite (Line 704)**
```rust
704 | pub fn run_full_validation_suite<T: ElasticWaveEquation>(
    |        ^^^^^^^^^^^^^^^^^^^^^^^^^
    |        Function is never called
```

### Dead Methods (4 in PlaneWaveSolution impl):

**Lines 473-600+ contain PlaneWaveSolution implementation:**

**8. PlaneWaveSolution::displacement (Line 528)**
```rust
528 | pub fn displacement(&self, x: f64, y: f64, t: f64) -> [f64; 2] {
    |        ^^^^^^^^^^^^
    |        Method is never called
```

**9. PlaneWaveSolution::velocity (Line 537)**
```rust
537 | pub fn velocity(&self, x: f64, y: f64, t: f64) -> [f64; 2] {
    |        ^^^^^^^^
    |        Method is never called
```

**10. PlaneWaveSolution::acceleration (Line 546)**
```rust
546 | pub fn acceleration(&self, x: f64, y: f64, t: f64) -> [f64; 2] {
    |        ^^^^^^^^^^^^
    |        Method is never called
```

**11. PlaneWaveSolution::displacement_gradient (Line 555)**
```rust
555 | pub fn displacement_gradient(&self, x: f64, y: f64, t: f64) -> [[f64; 2]; 2] {
    |        ^^^^^^^^^^^^^^^^^^^^^
    |        Method is never called
```

**Status**: These items are part of test framework but may be:
- Infrastructure for future tests
- Deprecated/superseded by other validation approach
- Intentionally kept for reference

---

## allow(dead_code) ATTRIBUTES

**Files with allow(dead_code)**:

### analysis/ml/inference.rs
- 2 items with `#[allow(dead_code)]`

### analysis/ml/uncertainty/bayesian_networks.rs
- 1 item with `#[allow(dead_code)]`

### analysis/ml/uncertainty/ensemble_methods.rs
- 1 item with `#[allow(dead_code)]`

### analysis/performance/optimization/cache.rs
- 2 items with `#[allow(dead_code)]` - "Cache configuration for advanced optimization"

### analysis/performance/optimization/memory.rs
- 1 item with `#[allow(dead_code)]` - "Memory optimization configuration for advanced systems"

### analysis/performance/optimization/mod.rs
- 1 item with `#[allow(dead_code)]` - "Used for parallel optimization strategies"

### analysis/performance/simd.rs
- 6 items with `#[allow(dead_code)]`

### analysis/signal_processing/beamforming/three_dimensional/apodization.rs
- 2 items with `#[allow(dead_code)]`

### analysis/signal_processing/beamforming/three_dimensional/delay_sum.rs
- 3 items with `#[allow(dead_code)]` (one mentions "Methods used only with GPU feature enabled")

### analysis/signal_processing/beamforming/three_dimensional/metrics.rs
- 1 item with `#[allow(dead_code)]`

**Total**: 20+ items across 10 files

**Recommendation**: These are likely feature-gated or for future use. Add documentation comments explaining why they're needed.

---

## FILE SIZES

### Large Files Requiring Review

| Rank | File | Lines | Concerns |
|------|------|-------|----------|
| 1 | `src/domain/boundary/coupling.rs` | 1827 | Very large - consider splitting |
| 2 | `src/infra/api/clinical_handlers.rs` | 1116 | Large - multiple handler concerns? |
| 3 | `src/solver/forward/hybrid/bem_fem_coupling.rs` | 1015 | Large - BEM+FEM integration complex |
| 4 | `src/clinical/therapy/swe_3d_workflows.rs` | 985 | Large - multiple workflows |
| 5 | `src/solver/inverse/pinn/ml/electromagnetic_gpu.rs` | 966 | Large - GPU-specific code |
| 6 | `src/physics/optics/sonoluminescence/emission.rs` | 957 | Large - complex physics |
| 7 | `src/solver/forward/bem/solver.rs` | 947 | Large - full BEM solver |
| 8 | `src/solver/inverse/pinn/ml/burn_wave_equation_3d/solver.rs` | 922 | Large - 3D solver implementation |
| 9 | `src/solver/inverse/pinn/ml/universal_solver.rs` | 913 | Large - multiple solver backends |
| 10 | `src/clinical/safety.rs` | 880 | Large - safety logic |

**Refactoring Candidates**:
1. `boundary/coupling.rs` (1827 lines) - Split coupling logic from boundary conditions
2. `burn_wave_equation_3d/solver.rs` (922 lines) - Extract specific numerical schemes
3. `universal_solver.rs` (913 lines) - Extract backend adapters

---

**End of Detailed Findings**
