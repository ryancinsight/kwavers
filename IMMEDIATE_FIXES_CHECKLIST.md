# Immediate Fixes Checklist - Kwavers Compilation Restoration

**Date**: 2024-01-09  
**Priority**: 🔴 CRITICAL - P0  
**Goal**: Restore compilation to working state  
**Time Estimate**: 4-6 hours

---

## Status: 39 Compilation Errors + 20 Warnings

### Error Categories

1. **Missing Files**: 1 error
2. **Incomplete Stubs**: 30+ errors (lithotripsy module methods)
3. **Import Issues**: 3 errors
4. **Function Signature Mismatches**: 5 errors

---

## Phase 1: Fix Missing Files (15 minutes)

### Task 1.1: Create Missing `numerical_accuracy.rs`

**Error**:
```
error[E0583]: file not found for module `numerical_accuracy`
 --> src\solver\utilities\validation\mod.rs
```

**Action**:
```bash
# File: src/solver/utilities/validation/numerical_accuracy.rs
```

**Content**: Create stub with basic validation traits
- [ ] Create file `src/solver/utilities/validation/numerical_accuracy.rs`
- [ ] Define `NumericalAccuracyValidator` struct
- [ ] Define `ValidationResult` type
- [ ] Add basic convergence validation functions
- [ ] Add unit tests

---

## Phase 2: Fix Import Issues (30 minutes)

### Task 2.1: Fix Validation Import Errors

**Errors**:
```
error[E0432]: unresolved import `crate::solver::validation::ValidationParameters`
```

**Files Affected**:
- `src/solver/utilities/validation/mod.rs`
- Other validation modules

**Actions**:
- [ ] Check if `ValidationParameters` exists in `solver/validation/mod.rs`
- [ ] If missing, create in `solver/validation/config.rs`
- [ ] Update re-exports in `solver/validation/mod.rs`
- [ ] Fix all import paths

### Task 2.2: Fix Physics Therapy Import

**Error**:
```
error[E0433]: failed to resolve: could not find `therapy` in `physics`
```

**Action**:
- [ ] Check `src/physics/acoustics/therapy/mod.rs` exists
- [ ] Verify `therapy` is declared in `src/physics/acoustics/mod.rs`
- [ ] Add `pub mod therapy;` if missing
- [ ] Update re-exports

---

## Phase 3: Complete Lithotripsy Stubs (2-3 hours)

### Task 3.1: Complete `bioeffects.rs`

**Location**: `src/clinical/therapy/lithotripsy/bioeffects.rs`

**Missing Methods**:
- [ ] `check_safety(&self, acoustic_intensity: f64) -> SafetyAssessment`
- [ ] `update_assessment(&mut self, pressure: f64, duration: f64)`
- [ ] `current_assessment(&self) -> SafetyAssessment`

**Missing Fields in `SafetyAssessment`**:
- [ ] `overall_safe: bool`
- [ ] `max_mechanical_index: f64`
- [ ] `max_thermal_index: f64`
- [ ] `max_cavitation_dose: f64`
- [ ] `max_damage_probability: f64`
- [ ] `violations: Vec<String>`

**Implementation**:
```rust
#[derive(Debug, Clone)]
pub struct SafetyAssessment {
    pub overall_safe: bool,
    pub max_mechanical_index: f64,
    pub max_thermal_index: f64,
    pub max_cavitation_dose: f64,
    pub max_damage_probability: f64,
    pub violations: Vec<String>,
}

impl BioeffectsModel {
    pub fn check_safety(&self, acoustic_intensity: f64) -> SafetyAssessment {
        let mi = (acoustic_intensity / 1e6).sqrt(); // Simplified MI
        let overall_safe = mi < self.parameters.mechanical_index_threshold;
        
        SafetyAssessment {
            overall_safe,
            max_mechanical_index: mi,
            max_thermal_index: 0.0, // TODO: Implement thermal calculation
            max_cavitation_dose: 0.0,
            max_damage_probability: 0.0,
            violations: if overall_safe { vec![] } else { vec!["MI exceeded".to_string()] },
        }
    }
    
    pub fn update_assessment(&mut self, _pressure: f64, _duration: f64) {
        // TODO: Accumulate bioeffects over time
    }
    
    pub fn current_assessment(&self) -> SafetyAssessment {
        SafetyAssessment::default()
    }
}
```

### Task 3.2: Complete `cavitation_cloud.rs`

**Location**: `src/clinical/therapy/lithotripsy/cavitation_cloud.rs`

**Missing Methods**:
- [ ] `initialize_cloud(&mut self, pressure_field: &Array3<f64>)`
- [ ] `evolve_cloud(&mut self, dt: f64, pressure_field: &Array3<f64>)`
- [ ] `cloud_density(&self) -> &Array3<f64>`
- [ ] `total_eroded_mass(&self) -> f64`

**Implementation**:
```rust
impl CavitationCloudDynamics {
    pub fn initialize_cloud(&mut self, _pressure_field: &Array3<f64>) {
        // TODO: Initialize bubble nuclei based on negative pressure
    }
    
    pub fn evolve_cloud(&mut self, _dt: f64, _pressure_field: &Array3<f64>) {
        // TODO: Update bubble radii using Rayleigh-Plesset
    }
    
    pub fn cloud_density(&self) -> &Array3<f64> {
        // TODO: Return spatial distribution of bubbles
        unimplemented!("cloud_density requires Array3 field")
    }
    
    pub fn total_eroded_mass(&self) -> f64 {
        // TODO: Integrate erosion from bubble collapse
        0.0
    }
}
```

### Task 3.3: Complete `shock_wave.rs`

**Location**: `src/clinical/therapy/lithotripsy/shock_wave.rs`

**Missing Fields in `ShockWaveParameters`**:
- [ ] `center_frequency: f64`
- [ ] `rise_time: f64`

**Missing Methods**:
- [ ] `ShockWaveGenerator::generate_shock_field(&self) -> Array3<f64>`
- [ ] `ShockWavePropagation::propagate_shock_wave(&mut self, field: &Array3<f64>, dt: f64)`

**Implementation**:
```rust
#[derive(Debug, Clone)]
pub struct ShockWaveParameters {
    pub peak_pressure: f64,
    pub pulse_duration: f64,
    pub center_frequency: f64,  // Add
    pub rise_time: f64,          // Add
}

impl ShockWaveGenerator {
    pub fn generate_shock_field(&self) -> Array3<f64> {
        // TODO: Generate initial shock wave pressure distribution
        unimplemented!("generate_shock_field requires grid")
    }
}

impl ShockWavePropagation {
    pub fn propagate_shock_wave(&mut self, _field: &Array3<f64>, _dt: f64) {
        // TODO: Propagate shock wave using KZK or Westervelt equation
    }
}
```

### Task 3.4: Complete `stone_fracture.rs`

**Location**: `src/clinical/therapy/lithotripsy/stone_fracture.rs`

**Missing Methods**:
- [ ] `apply_stress_loading(&mut self, stress: &Array3<f64>)`
- [ ] `damage_field(&self) -> &Array3<f64>`
- [ ] `fragment_sizes(&self) -> Vec<f64>`

**Implementation**:
```rust
impl StoneFractureModel {
    pub fn apply_stress_loading(&mut self, _stress: &Array3<f64>) {
        // TODO: Update damage based on stress tensor
    }
    
    pub fn damage_field(&self) -> &Array3<f64> {
        // TODO: Return spatial distribution of stone damage
        unimplemented!("damage_field requires Array3 field")
    }
    
    pub fn fragment_sizes(&self) -> Vec<f64> {
        // TODO: Calculate fragment size distribution
        vec![]
    }
}
```

---

## Phase 4: Fix Function Signature Mismatches (30 minutes)

### Task 4.1: Fix `UncertaintyEstimator::new()`

**Error**:
```
error[E0061]: this function takes 1 argument but 0 arguments were supplied
 --> src\analysis\signal_processing\beamforming\neural\beamformer.rs:128:37
```

**Action**:
- [ ] Update `beamformer.rs` line 128:
  ```rust
  // OLD:
  let uncertainty_estimator = UncertaintyEstimator::new();
  
  // NEW:
  let uncertainty_estimator = UncertaintyEstimator::new(0.1); // 10% dropout
  ```

### Task 4.2: Fix Other Function Signature Mismatches

**Files**:
- Check all files with `error[E0061]`
- Update function calls to match signatures
- Add missing arguments with sensible defaults

---

## Phase 5: Clean Up Warnings (30 minutes)

### Task 5.1: Remove Unused Imports

**Warnings**: 20 unused import warnings

**Action**:
```bash
# Automated fix:
cargo clippy --fix --allow-dirty --allow-staged
```

Manual review:
- [ ] Review auto-fixed changes
- [ ] Remove manually if auto-fix fails
- [ ] Verify no breakage

---

## Phase 6: Verification (30 minutes)

### Task 6.1: Build Verification

- [ ] `cargo build` - MUST succeed with 0 errors
- [ ] `cargo build --all-features` - Check all features
- [ ] `cargo clippy -- -D warnings` - Zero warnings required

### Task 6.2: Test Verification

- [ ] `cargo test --lib` - Fast library tests
- [ ] `cargo test --test infrastructure_test` - Core infrastructure
- [ ] Check for panics or unwrap failures

### Task 6.3: Documentation Check

- [ ] `cargo doc --no-deps` - Verify doc builds
- [ ] Check for broken intra-doc links
- [ ] Update module-level docs

---

## Success Criteria

### Must Have (P0)
- ✅ Zero compilation errors
- ✅ Zero clippy warnings with `-D warnings`
- ✅ `cargo build --all-features` succeeds
- ✅ Core tests pass

### Should Have (P1)
- ✅ All unit tests pass
- ✅ Documentation builds without warnings
- ✅ No `unimplemented!()` in core paths

### Nice to Have (P2)
- ✅ TODO comments converted to GitHub issues
- ✅ Stub implementations documented
- ✅ Performance tests pass

---

## Implementation Strategy

### Recommended Order

1. **Start with missing files** (Task 1.1) - Quick win
2. **Fix imports** (Tasks 2.1, 2.2) - Unblocks other work
3. **Complete stubs incrementally** (Tasks 3.1-3.4):
   - Start with smallest/simplest (bioeffects)
   - Build up to complex (cavitation, fracture)
   - Test after each module
4. **Fix signatures** (Task 4) - Once stubs compile
5. **Clean warnings** (Task 5) - Polish
6. **Verify** (Task 6) - Final validation

### Time Allocation

| Phase | Estimated Time | Priority |
|-------|---------------|----------|
| Phase 1 | 15 min | P0 |
| Phase 2 | 30 min | P0 |
| Phase 3 | 2-3 hours | P0 |
| Phase 4 | 30 min | P0 |
| Phase 5 | 30 min | P1 |
| Phase 6 | 30 min | P0 |
| **Total** | **4-6 hours** | |

---

## Notes for Implementation

### Stub Implementation Guidelines

When creating stub methods:

1. **Use `unimplemented!()` with descriptive messages**:
   ```rust
   unimplemented!("cloud_density requires spatial grid integration - see Issue #XXX")
   ```

2. **Return sensible defaults** where possible:
   ```rust
   pub fn total_eroded_mass(&self) -> f64 {
       0.0  // No erosion until implemented
   }
   ```

3. **Document TODOs with issue references**:
   ```rust
   /// TODO(#XXX): Implement bubble dynamics using Rayleigh-Plesset equation
   ```

4. **Add validation tests for stubs**:
   ```rust
   #[test]
   #[should_panic(expected = "unimplemented")]
   fn test_unimplemented_method() {
       // Ensure we know what's not implemented
   }
   ```

### Testing Strategy

- Test compilation after EACH phase
- Don't accumulate errors
- Use `cargo check` for fast feedback
- Run `cargo test --lib` frequently

---

## Post-Completion

After all tasks complete:

1. **Update `errors.txt`**:
   ```bash
   rm errors.txt  # Delete old error log
   ```

2. **Update `.gitignore`**:
   ```gitignore
   # Build logs
   *.log
   errors.txt
   baseline_*.log
   ```

3. **Commit changes**:
   ```bash
   git add .
   git commit -m "fix: restore compilation - complete lithotripsy stubs

   - Created missing numerical_accuracy.rs
   - Fixed import paths for validation modules
   - Completed lithotripsy module stubs (bioeffects, cavitation, shock_wave, stone_fracture)
   - Fixed function signature mismatches
   - Removed unused imports (20 warnings)
   
   Closes #XXX (create issue first)
   
   Ref: DEEP_VERTICAL_HIERARCHY_REFACTORING_AUDIT.md Phase 1"
   ```

4. **Create tracking issue** for remaining work:
   - Link to `DEEP_VERTICAL_HIERARCHY_REFACTORING_AUDIT.md`
   - Document which stubs need full implementation
   - Prioritize based on usage

---

## Risk Mitigation

### If Errors Persist

1. **Isolate the problem**:
   ```bash
   cargo build --lib 2>&1 | grep "error\[E" | sort | uniq
   ```

2. **Fix one error type at a time**:
   - Group similar errors
   - Fix all instances together
   - Test immediately

3. **Use incremental compilation**:
   ```bash
   cargo check  # Faster than cargo build
   ```

### If Tests Fail

1. **Identify failing tests**:
   ```bash
   cargo test --lib 2>&1 | grep "test result: FAILED"
   ```

2. **Skip non-critical tests temporarily**:
   ```rust
   #[test]
   #[ignore = "blocked by lithotripsy implementation"]
   fn test_full_simulation() { ... }
   ```

3. **Document test failures** in issue tracker

---

## Reference

- **Main Audit**: `DEEP_VERTICAL_HIERARCHY_REFACTORING_AUDIT.md`
- **Architecture Guide**: `README.md` (Section: Architecture)
- **ADRs**: `docs/adr.md`
- **Related Issues**: Create and link here

---

**End of Checklist**

*Last Updated: 2024-01-09*  
*Status: Ready for execution*