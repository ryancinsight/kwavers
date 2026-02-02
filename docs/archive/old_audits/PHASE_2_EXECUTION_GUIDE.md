# Phase 2 Execution Guide: High-Priority Architecture Fixes

**Start Date:** 2026-01-28  
**Target Completion:** 2026-02-02  
**Duration:** 3-5 days

---

## Overview

Phase 2 addresses critical architectural violations preventing proper layering:

1. **Solver→Analysis reverse dependency** (2.1) - 3-4 hours
2. **Domain localization misplacement** (2.2) - 4-5 hours
3. **Duplicate imports cleanup** (1.3) - 1 hour

---

## Task 2.1: Fix Solver→Analysis Reverse Dependency

### Current State
```
VIOLATION: Solver Layer (4) → Analysis Layer (6)
Location: src/solver/inverse/pinn/ml/beamforming_provider.rs:1-20
Problem: Imports from analysis::signal_processing::beamforming::neural::pinn_interface
```

### What Should Happen
```
CORRECT: Analysis Layer (6) → Solver Layer (4)
Analysis imports solver interfaces, not vice versa
```

### Root Cause
PINN configuration types are defined in Analysis layer, but Solver needs them.

### Solution Architecture

**Step 1: Move interface types from Analysis→Solver**
```
FROM: src/analysis/signal_processing/beamforming/neural/pinn_interface.rs
TO:   src/solver/inverse/pinn/interface.rs

Types to move:
- PinnBeamformingConfig
- PinnModelConfig  
- ModelArchitecture
- ActivationFunction
- InferenceConfig
- UncertaintyConfig
- ModelInfo
- PinnBeamformingProvider trait
- TrainingMetrics
- PinnBeamformingResult
```

**Step 2: Create backward-compatible re-exports in Analysis**
```rust
// src/analysis/signal_processing/beamforming/neural/pinn_interface.rs
pub use crate::solver::inverse::pinn::interface::{
    PinnBeamformingConfig, PinnModelConfig, ModelArchitecture,
    ActivationFunction, InferenceConfig, UncertaintyConfig,
    ModelInfo, PinnBeamformingProvider, TrainingMetrics,
    PinnBeamformingResult,
};
```

This maintains backward compatibility while fixing the dependency direction.

**Step 3: Update imports in Solver**
```rust
// src/solver/inverse/pinn/ml/beamforming_provider.rs
// BEFORE
use crate::analysis::signal_processing::beamforming::neural::pinn_interface::{...};

// AFTER
use crate::solver::inverse::pinn::interface::{...};
```

**Step 4: Update all Analysis consumers**
```rust
// Imports continue to work via re-exports
use crate::analysis::signal_processing::beamforming::neural::pinn_interface::{...};
// ✅ Still works, but now goes through Solver layer
```

### Implementation Checklist

- [ ] Create `src/solver/inverse/pinn/interface.rs` with all types
- [ ] Update `src/solver/inverse/pinn/ml/beamforming_provider.rs` imports
- [ ] Create re-exports in `src/analysis/signal_processing/beamforming/neural/pinn_interface.rs`
- [ ] Update imports in all Analysis files
- [ ] Run `cargo check` - verify clean compile
- [ ] Run `cargo test --lib` - verify tests pass
- [ ] Verify no reverse dependencies remain with grep:
  ```bash
  grep -r "analysis::" src/solver/ --include="*.rs" | grep -v "^src/solver/backend" # Should be empty
  ```

### Files to Modify

**Create (NEW):**
- `src/solver/inverse/pinn/interface.rs` (move types here)

**Modify:**
- `src/solver/inverse/pinn/ml/beamforming_provider.rs` (update imports)
- `src/solver/inverse/pinn/mod.rs` (export interface module)
- `src/analysis/signal_processing/beamforming/neural/pinn_interface.rs` (re-export types)

**Check/Update (dependent files):**
- `src/analysis/signal_processing/beamforming/neural/beamformer.rs`
- `src/analysis/signal_processing/beamforming/neural/*.rs` (all files in neural/)
- `src/clinical/imaging/workflows/neural/*.rs` (any clinical code)

### Verification

```bash
# 1. Build check
cargo check 2>&1 | grep -i "error" # Should be empty

# 2. Build library
cargo build --lib 2>&1 | grep -i "error" # Should be empty

# 3. Run tests
cargo test --lib --no-run 2>&1 | grep -i "error" # Should be empty

# 4. Verify no reverse dependencies
grep -r "use crate::analysis" src/solver --include="*.rs" | wc -l # Should be 0

# 5. Verify re-exports work
grep -r "pinn_interface" src/analysis --include="*.rs" | head -3 # Should find re-exports
```

---

## Task 2.2: Move Localization from Domain to Analysis

### Current State
```
VIOLATION: Localization algorithms in Domain Layer (3)
Locations:
- src/domain/sensor/localization/
- src/domain/sensor/localization/mod.rs
- src/domain/sensor/localization/multilateration.rs
- src/domain/sensor/localization/music.rs
- src/domain/sensor/localization/array/mod.rs
```

### What Should Happen
```
CORRECT: Localization algorithms in Analysis Layer (6)
Domain: Only defines data types (Position, SensorArray)
Analysis: Algorithms that compute positions from sensor data
```

### Why This Matters
- **Domain** = What (data structures)
- **Analysis** = How (algorithms)
- Localization is an algorithm/analysis, not a domain model

### Solution Architecture

**Step 1: Identify what stays in Domain vs moves to Analysis**

**STAY in Domain (data types only):**
```rust
// src/domain/sensor/array/mod.rs
pub struct SensorArray {
    pub positions: Vec<Position>,
    // ...
}

pub struct Position {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}
```

**MOVE to Analysis (algorithms):**
```rust
// src/analysis/signal_processing/localization/mod.rs
pub mod multilateration;
pub mod music;

pub trait LocalizationAlgorithm {
    fn estimate_position(&self, delays: &[f64]) -> KwaversResult<Position>;
}
```

**Step 2: Create target structure in Analysis layer**

```
src/analysis/signal_processing/localization/
├── mod.rs                      (main module, re-exports)
├── multilateration.rs          (multilateration algorithm)
├── music.rs                    (MUSIC algorithm)
├── types.rs                    (minimal types if needed)
└── tests/
    ├── multilateration_tests.rs
    └── music_tests.rs
```

**Step 3: Move implementations**

Copy files:
- `src/domain/sensor/localization/multilateration.rs` → `src/analysis/signal_processing/localization/multilateration.rs`
- `src/domain/sensor/localization/music.rs` → `src/analysis/signal_processing/localization/music.rs`

Update imports in moved files:
```rust
// BEFORE
use crate::domain::sensor::localization::Position;

// AFTER  
use crate::domain::sensor::Position; // or define minimal type
use crate::core::error::KwaversResult;
```

**Step 4: Create deprecation shim in Domain**

```rust
// src/domain/sensor/localization/mod.rs (NEW - deprecation wrapper)
#![deprecated(
    since = "3.2.0",
    note = "Localization algorithms moved to analysis::signal_processing::localization for proper architectural separation"
)]

pub use crate::analysis::signal_processing::localization::*;
```

**Step 5: Update all imports**

Find and update all files that import from `domain::sensor::localization`:

```bash
grep -r "domain::sensor::localization\|domain::sensor::" src --include="*.rs" | grep -v "Binary" | head -30
```

Files to update:
- `src/analysis/signal_processing/localization/beamforming_search.rs`
- `src/domain/sensor/beamforming/sensor_beamformer.rs`
- `src/clinical/imaging/workflows/neural/ai_beamforming_processor.rs`
- Any test files importing localization

### Implementation Checklist

- [ ] Create `src/analysis/signal_processing/localization/` directory
- [ ] Move `multilateration.rs` and `music.rs`
- [ ] Create `src/analysis/signal_processing/localization/mod.rs`
- [ ] Create `src/analysis/signal_processing/localization/types.rs` if needed
- [ ] Update imports in moved files
- [ ] Create deprecation wrapper in `src/domain/sensor/localization/mod.rs`
- [ ] Find all imports:
  ```bash
  grep -r "localization" src --include="*.rs" | grep "use\|mod" | head -50
  ```
- [ ] Update all dependent imports
- [ ] Run `cargo check`
- [ ] Run `cargo test --lib`
- [ ] Verify deprecation warnings:
  ```bash
  cargo build --lib 2>&1 | grep "localization" | head -10
  ```

### Files Affected

**Create (NEW):**
- `src/analysis/signal_processing/localization/mod.rs`
- `src/analysis/signal_processing/localization/multilateration.rs` (moved)
- `src/analysis/signal_processing/localization/music.rs` (moved)
- `src/analysis/signal_processing/localization/types.rs` (if needed)

**Modify:**
- `src/domain/sensor/localization/mod.rs` (deprecation wrapper)
- `src/analysis/signal_processing/mod.rs` (add localization module)
- Update all `use` statements in dependent files

**Check Imports In:**
- `src/analysis/signal_processing/localization/beamforming_search.rs`
- `src/domain/sensor/beamforming/sensor_beamformer.rs`  
- `src/clinical/imaging/workflows/neural/ai_beamforming_processor.rs`
- Test files

### Verification

```bash
# 1. Verify moved files exist
ls src/analysis/signal_processing/localization/multilateration.rs
ls src/analysis/signal_processing/localization/music.rs

# 2. Build and check for errors
cargo check 2>&1 | grep -i "error"

# 3. Run tests
cargo test --lib localization 2>&1 | tail -20

# 4. Verify deprecation works
cargo build --lib 2>&1 | grep "deprecated" | grep -i "localization"

# 5. Clean build
cargo build --lib 2>&1 | tail -3
```

---

## Task 1.3: Fix Duplicate Imports

### Current State

**File:** `src/clinical/imaging/workflows/mod.rs`

```rust
// Line 5 - FIRST IMPORT
use crate::core::error::KwaversResult;

// Line 248 - DUPLICATE IMPORT
use crate::core::error::KwaversResult;
```

### Solution

Simple: Remove the duplicate import on line 248.

### Implementation

```bash
# 1. View the file around line 248
head -260 src/clinical/imaging/workflows/mod.rs | tail -30

# 2. Check for other duplicates
grep -n "use crate::core::error::KwaversResult" src/clinical/imaging/workflows/mod.rs

# 3. Remove duplicate line 248
# (Use editor to remove or sed command below)

# Or use sed to remove duplicate imports (keep first occurrence only)
sed -i '248d' src/clinical/imaging/workflows/mod.rs
```

### Verification

```bash
# Verify no duplicates remain
cargo clippy --lib 2>&1 | grep "unused import" 

# Build to confirm
cargo check 2>&1 | grep -i "error"
```

---

## Execution Sequence

### Day 1-2: Task 2.1 (Solver→Analysis Dependency)

```bash
# Morning: Understand current structure
cd src/solver/inverse/pinn
grep -r "use crate::analysis" .

# Mid-morning: Create new interface file
cp src/analysis/signal_processing/beamforming/neural/pinn_interface.rs \
   src/solver/inverse/pinn/interface.rs

# Afternoon: Update imports
# - Edit src/solver/inverse/pinn/ml/beamforming_provider.rs
# - Update imports to use solver::inverse::pinn::interface

# Late afternoon: Create re-exports
# - Edit src/analysis/signal_processing/beamforming/neural/pinn_interface.rs
# - Add re-exports for backward compatibility

# Evening: Testing
cargo check
cargo test --lib
```

### Day 3-4: Task 2.2 (Move Localization)

```bash
# Morning: Create target structure
mkdir -p src/analysis/signal_processing/localization

# Mid-morning: Copy files
cp src/domain/sensor/localization/multilateration.rs \
   src/analysis/signal_processing/localization/

cp src/domain/sensor/localization/music.rs \
   src/analysis/signal_processing/localization/

# Afternoon: Update imports in moved files
# - Fix all imports in multilateration.rs
# - Fix all imports in music.rs

# Late afternoon: Create deprecation wrapper
# - Edit src/domain/sensor/localization/mod.rs
# - Add deprecation notice and re-exports

# Evening: Fix dependent imports
# - Find all files importing from domain::localization
# - Update to use analysis::localization
```

### Day 5: Task 1.3 + Verification

```bash
# Morning: Fix duplicate imports
sed -i '248d' src/clinical/imaging/workflows/mod.rs

# Afternoon: Full verification
cargo check
cargo build --lib
cargo clippy --lib
cargo test --lib

# Evening: Final validation
cargo build --release
```

---

## Risk Mitigation

### Risk: Breaking Analysis code by moving PINN types
**Mitigation:** 
- Create re-exports in Analysis layer first
- All existing code continues to work
- Provide migration guide with deprecation notice

### Risk: Test failures after moving localization
**Mitigation:**
- Run full test suite before and after
- Compare test results
- Document any failures

### Risk: Circular dependencies created
**Mitigation:**
- Use `grep` to verify dependencies after changes
- Run `cargo tree` to visualize dependency graph

---

## Success Criteria

### Phase 2 Complete When:

✅ `cargo check` produces zero errors  
✅ `cargo test --lib` passes 100%  
✅ No reverse dependencies from Solver→Analysis  
✅ Localization moved to Analysis layer  
✅ Duplicate imports removed  
✅ All deprecation notices in place  
✅ Backward compatibility maintained  

### Build Quality:
```bash
cargo build --lib 2>&1 | grep "error" # Empty output
cargo check 2>&1 | grep "error" # Empty output  
cargo clippy --lib 2>&1 | grep "error" # Empty output
```

### Dependency Verification:
```bash
# No solver importing from analysis
grep -r "use.*analysis::" src/solver | grep -v "backend" | wc -l # Should be 0

# All localization in analysis
ls -la src/analysis/signal_processing/localization/ | wc -l # Should be > 0
ls -la src/domain/sensor/localization/*.rs | wc -l # Should be minimal

# No circular imports
cargo tree --duplicates | grep -i "circular" # Should be empty
```

---

## Communication Plan

After completing Phase 2, document:
1. **What changed** - Summary of modifications
2. **Why it changed** - Architecture rationale
3. **Migration guide** - For users of affected APIs
4. **Deprecation timeline** - When old imports will be removed

---

## Next Steps After Phase 2

Once Phase 2 is complete:
1. Proceed to Phase 3.1 (Dead Code Cleanup)
2. Address high-priority TODOs (API implementation, PINN training)
3. Begin Phase 4 deduplication tasks

---

**Ready to Begin:** Execute according to sequence above, one task per day minimum.

**Expected Outcome:** Clean 8-layer architecture with proper dependency flow (always toward lower layers).
