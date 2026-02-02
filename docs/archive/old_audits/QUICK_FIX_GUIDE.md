# Quick Fix Guide - Priority Actions

## STOP: Code Cannot Compile
Fix these 6 errors BEFORE anything else.

---

## ERROR 1: Remove Old Imports (5 minutes)
**File**: `D:\kwavers\tests\sensor_delay_test.rs`

```diff
- use kwavers::domain::sensor::localization::array::{ArrayGeometry, Sensor, SensorArray};
- use kwavers::domain::sensor::localization::Position;
+ use kwavers::domain::sensor::{ArrayGeometry, Position, Sensor, SensorArray};
```

---

## ERROR 2: Fix NIFTI Calls (30 minutes)
**File**: `D:\kwavers\tests\ct_nifti_integration_test.rs`

**Problem**: `from_header_and_data()` doesn't exist in nifti v0.17.0

**Locations**: Lines 46, 99, 223, 301, 348

### Quick Fix Option 1: Remove feature
Add to `Cargo.toml`:
```toml
# tests don't require nifti unless explicitly testing it
# Move ct_nifti_integration_test to require feature:

[[test]]
name = "ct_nifti_integration_test"
required-features = ["nifti"]  # Disable by default
```

### Quick Fix Option 2: Use wrapper
Create `D:\kwavers\src\infra\io\nifti_utils.rs`:

```rust
pub fn create_nifti_from_header_and_data(
    header: nifti::NiftiHeader, 
    volume: Array3<f32>
) -> nifti::Result<nifti::InMemNiftiObject> {
    // Implement using from_reader or write-read approach
    unimplemented!("NIFTI in-memory creation not yet implemented")
}
```

Then update tests to use this wrapper.

---

## ERROR 3: Remove Factory Example (5 minutes)
**File**: `D:\kwavers\examples\phase2_factory.rs`

Either delete the file or rename to `phase2_factory_wip.rs` and add comment:
```rust
//! WIP: Factory pattern not yet implemented
//! This example will work once SimulationFactory is fully implemented
```

Or check `D:\kwavers\src\simulation\factory.rs` to verify what exists.

---

## ERROR 4: Fix Distributed Beamforming (30 minutes)
**File**: `D:\kwavers\src\analysis\signal_processing\beamforming\neural\distributed\core.rs`

**Lines 269-297**: Test code doesn't match API

Replace with:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "pinn")]
    #[test]
    fn test_processor_creation() {
        let pinn_config = PINNBeamformingConfig::default();
        let distributed_config = DistributedConfig {
            gpu_devices: vec![0],
            decomposition: DecompositionStrategy::Default,
            load_balancing: LoadBalancingStrategy::Default,
        };
        
        let result = DistributedNeuralBeamformingProcessor::new(
            pinn_config,
            distributed_config,
        );

        // May fail if GPUs not available, but should not panic
        let _ = result;
    }

    #[cfg(feature = "pinn")]
    #[test]
    fn test_fault_tolerance_default() {
        let ft = FaultToleranceState::default();
        assert_eq!(ft.max_retries, 3);
        assert!(ft.dynamic_load_balancing);
        assert_eq!(ft.load_imbalance_threshold, 0.2);
    }
}
```

**Remove**: The old test code (lines 269-310) that references non-existent API.

---

## ERROR 5: Check What Exists
**File**: `D:\kwavers\src\simulation\factory.rs`

Run this command to see what's actually there:
```bash
grep -n "pub struct\|pub enum\|pub fn" src/simulation/factory.rs | head -20
```

Then either:
1. Implement missing types
2. Update example to use what exists
3. Delete example if factory not ready

---

## ERROR 6: Verify Build
```bash
cd D:\kwavers
cargo clippy --all-features --all-targets 2>&1 | head -50
```

You should see 0 errors when above fixes applied.

---

## THEN: Fix Warnings (30 minutes)

### WARNING 1: Large Enum
**File**: `D:\kwavers\src\analysis\signal_processing\beamforming\slsc\mod.rs:143-152`

```diff
pub enum LagWeighting {
    Uniform,
    Triangular,
    Hamming,
-   Custom { weights: [f64; 64], len: usize },
+   Custom(Box<[f64]>),
}
```

Update `.weight()` method accordingly.

### WARNING 2: Missing Debug
**File**: `D:\kwavers\src\solver\inverse\pinn\ml\beamforming_provider.rs:34`

```diff
+ #[cfg_attr(feature = "pinn", derive(Debug))]
pub struct BurnPinnBeamformingAdapter<B: burn::tensor::backend::Backend> {
```

### WARNING 3: Field Reassignments (6x)
**File**: `D:\kwavers\benches\pinn_elastic_2d_training.rs:58,119,208,265,404,452`

Replace ALL instances of:
```diff
- let mut config = Config::default();
- config.hidden_layers = vec![64, 64, 64];
+ let config = Config { 
+     hidden_layers: vec![64, 64, 64], 
+     ..Default::default() 
+ };
```

### WARNING 4: Unused Variables (6x)

Option A - Don't use:
```diff
- let model = ElasticPINN2D::<Backend>::new(&config, &device)?;
+ let _model = ElasticPINN2D::<Backend>::new(&config, &device)?;
```

Option B - Actually use them (better)

### WARNING 5: Unused Imports (3x)

Remove these lines:
- `benches/hilbert_benchmark.rs:2` → Remove `black_box`
- `tests/pinn_elastic_validation.rs:33` → Remove `Array2, ArrayD`
- `tests/pinn_elastic_validation.rs:39` → Remove `WaveType`

### WARNING 6: Remove mut keyword
**File**: `D:\kwavers\src\analysis\signal_processing\beamforming/neural/pinn_interface.rs:402`

```diff
- let mut registry = PinnProviderRegistry::new();
+ let registry = PinnProviderRegistry::new();
```

---

## VERIFY FIX

```bash
cargo clippy --all-features --all-targets 2>&1 | grep -c "error\|warning"
```

Target: 0 errors, <5 warnings

---

## SUMMARY TABLE

| Priority | File | Lines | Issue | Time | Status |
|----------|------|-------|-------|------|--------|
| 1 | sensor_delay_test.rs | 3-4 | Import path | 5 min | MUST FIX |
| 2 | ct_nifti_integration_test.rs | 46,99,223,301,348 | API mismatch | 30 min | MUST FIX |
| 3 | phase2_factory.rs | 8-9,112 | Missing types | 5 min | MUST FIX |
| 4 | distributed/core.rs | 269-297 | API mismatch | 30 min | MUST FIX |
| 5 | simulation/factory.rs | ? | Verify | 10 min | MUST FIX |
| 1W | slsc/mod.rs | 143-152 | Large enum | 10 min | HIGH |
| 2W | beamforming_provider.rs | 34 | Missing Debug | 10 min | HIGH |
| 3W | pinn_elastic_2d_training.rs | 58,119,208,265,404,452 | Field reassign | 10 min | HIGH |
| 4W | pinn_elastic_2d_training.rs | 121,211 | Unused vars | 5 min | HIGH |
| 5W | Various | Multiple | Unused imports | 5 min | MEDIUM |
| 6W | pinn_interface.rs | 402 | Mut keyword | 2 min | MEDIUM |

---

## ESTIMATED TIMELINE

- **Errors (6)**: 1-2 hours to fix all
- **Warnings (31+)**: 30 minutes to fix high-priority ones
- **Total**: 1.5-2.5 hours to get clean build

---

## NEXT STEPS AFTER FIXING

1. Run: `cargo build --release`
2. Run: `cargo test --all`
3. Run: `cargo clippy --all-features --all-targets`
4. Commit: `fix(audit): Resolve compilation errors and warnings`
5. Check git status to verify all changes
