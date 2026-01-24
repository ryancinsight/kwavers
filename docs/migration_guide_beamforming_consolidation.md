# Migration Guide: Beamforming Consolidation

**Sprint:** 213-215  
**Goal:** Eliminate beamforming duplication between `domain/sensor/` and `analysis/signal_processing/`  
**Pattern:** k-Wave-Python's `reconstruction/beamform` model

---

## Overview

This guide provides step-by-step instructions for migrating beamforming code from the domain layer to the analysis layer, following industry best practices.

**Current State:**
- `domain/sensor/beamforming/` - 120+ files (❌ remove)
- `analysis/signal_processing/beamforming/` - Parallel implementation (✅ keep)

**Target State:**
- `domain/sensor/` - Geometry exports only
- `analysis/signal_processing/beamforming/` - Single source of truth for all algorithms

---

## Phase 1: Deprecation Warnings (Sprint 213)

### Step 1.1: Add Deprecation Attributes

```rust
// src/domain/sensor/beamforming/mod.rs

#![deprecated(
    since = "0.8.0",
    note = "This module is deprecated. Use analysis::signal_processing::beamforming instead. \
            Migration guide: docs/migration_guide_beamforming_consolidation.md. \
            This module will be removed in v0.9.0."
)]

// Existing modules with deprecation
#[deprecated(since = "0.8.0", note = "Use analysis::signal_processing::beamforming::adaptive")]
pub mod adaptive;

#[deprecated(since = "0.8.0", note = "Use analysis::signal_processing::beamforming::neural")]
pub mod neural;

#[deprecated(since = "0.8.0", note = "Use analysis::signal_processing::beamforming::three_dimensional")]
pub mod beamforming_3d;

// Keep only geometry exports (NOT deprecated)
pub mod geometry;  // ✅ Keep - exports sensor positions for beamforming
```

### Step 1.2: Add Transitional Delegation

For backwards compatibility during migration period:

```rust
// src/domain/sensor/beamforming/mod.rs

/// Transitional wrapper - delegates to analysis layer
/// 
/// # Deprecated
/// This function is deprecated. Use `analysis::signal_processing::beamforming::beamform` directly.
#[deprecated(
    since = "0.8.0",
    note = "Use analysis::signal_processing::beamforming::beamform directly"
)]
pub fn beamform_sensor_data(
    data: &Array3<f32>,
    config: &BeamformingConfig,
) -> crate::core::error::KwaversResult<Array3<f32>> {
    // Delegate to analysis layer
    crate::analysis::signal_processing::beamforming::beamform(data, config)
}
```

### Step 1.3: Update Cargo.toml

```toml
[package]
version = "0.8.0"  # Bump minor version for deprecation
```

### Step 1.4: Add Migration Guide to Documentation

```rust
// src/domain/sensor/beamforming/mod.rs

//! # Migration Guide
//!
//! This module is deprecated. Please migrate to `analysis::signal_processing::beamforming`.
//!
//! ## Before (v0.7.x)
//! ```rust
//! use kwavers::domain::sensor::beamforming::{BeamformingAlgorithm3D, Beamformer3D};
//!
//! let beamformer = Beamformer3D::new(config)?;
//! let image = beamformer.process(rf_data)?;
//! ```
//!
//! ## After (v0.8.0+)
//! ```rust
//! use kwavers::analysis::signal_processing::beamforming::{BeamformingAlgorithm, beamform};
//!
//! let image = beamform(rf_data, &geometry, &config)?;
//! ```
//!
//! ## Key Changes
//! - **Import path:** `domain::sensor::beamforming` → `analysis::signal_processing::beamforming`
//! - **API style:** Object-oriented (`Beamformer3D::new().process()`) → Functional (`beamform()`)
//! - **Geometry:** Embedded in struct → Explicit parameter (`&geometry`)
//!
//! See `docs/migration_guide_beamforming_consolidation.md` for detailed examples.
```

---

## Phase 2: Code Migration (Sprint 214)

### Step 2.1: Update Clinical Workflows

**Before:**
```rust
// clinical/imaging/workflows/neural/workflow.rs (OLD)

use crate::domain::sensor::beamforming::neural::{NeuralBeamformer, NeuralConfig};
use crate::domain::sensor::beamforming::BeamformingAlgorithm3D;

impl NeuralWorkflow {
    pub fn process_acquisition(&self, rf_data: &Array3<f32>) -> KwaversResult<DiagnosisResult> {
        // ❌ Using domain layer beamforming
        let beamformer = NeuralBeamformer::new(self.beamforming_config.clone())?;
        let beamformed = beamformer.process(rf_data)?;
        
        // ... rest of workflow
    }
}
```

**After:**
```rust
// clinical/imaging/workflows/neural/workflow.rs (NEW)

use crate::analysis::signal_processing::beamforming::{BeamformingAlgorithm, beamform};
use crate::domain::sensor::geometry::BeamformingGeometry;

impl NeuralWorkflow {
    pub fn process_acquisition(&self, acquisition: &RawAcquisition) -> KwaversResult<DiagnosisResult> {
        // ✅ Using analysis layer beamforming
        let geometry = acquisition.sensor_geometry();
        
        let beamformed = beamform(
            &acquisition.rf_data,
            &geometry,
            &BeamformingAlgorithm::Neural {
                model_path: self.model_path.clone(),
                uncertainty: true,
            },
        )?;
        
        // ... rest of workflow
    }
}
```

### Step 2.2: Update Examples

**Before:**
```rust
// examples/adaptive_beamforming_refactored.rs (OLD)

use kwavers::domain::sensor::beamforming::adaptive::{AdaptiveBeamformer, MVDRConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ❌ Using domain layer
    let beamformer = AdaptiveBeamformer::new(MVDRConfig {
        diagonal_loading: 0.01,
        subarray_size: [32, 32, 16],
    })?;
    
    let image = beamformer.process(&rf_data)?;
    Ok(())
}
```

**After:**
```rust
// examples/adaptive_beamforming_refactored.rs (NEW)

use kwavers::analysis::signal_processing::beamforming::{BeamformingAlgorithm, beamform};
use kwavers::domain::sensor::geometry::BeamformingGeometry;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ✅ Using analysis layer
    let geometry = BeamformingGeometry {
        positions: sensor_positions,
        orientations: sensor_normals,
    };
    
    let image = beamform(
        &rf_data,
        &geometry,
        &BeamformingAlgorithm::MVDR {
            diagonal_loading: 0.01,
            subarray_size: [32, 32, 16],
        },
    )?;
    
    Ok(())
}
```

### Step 2.3: Update Tests

**Before:**
```rust
// tests/beamforming_tests.rs (OLD)

use kwavers::domain::sensor::beamforming::time_domain::{DASBeamformer, DASConfig};

#[test]
fn test_das_beamforming() {
    let config = DASConfig::default();
    let beamformer = DASBeamformer::new(config).unwrap();
    let result = beamformer.process(&test_data).unwrap();
    assert_eq!(result.shape(), &[100, 100, 50]);
}
```

**After:**
```rust
// tests/beamforming_tests.rs (NEW)

use kwavers::analysis::signal_processing::beamforming::{BeamformingAlgorithm, beamform};
use kwavers::domain::sensor::geometry::BeamformingGeometry;

#[test]
fn test_das_beamforming() {
    let geometry = create_test_geometry();
    let result = beamform(
        &test_data,
        &geometry,
        &BeamformingAlgorithm::DAS,
    ).unwrap();
    assert_eq!(result.shape(), &[100, 100, 50]);
}
```

### Step 2.4: Update API Handlers

**Before:**
```rust
// infra/api/clinical_handlers.rs (OLD)

use crate::domain::sensor::beamforming::BeamformingAlgorithm3D;

async fn beamform_endpoint(
    Json(request): Json<BeamformRequest>,
) -> Result<Json<BeamformResponse>, ApiError> {
    let beamformer = match request.algorithm {
        "DAS" => Beamformer3D::new(BeamformingAlgorithm3D::DAS)?,
        "MVDR" => Beamformer3D::new(BeamformingAlgorithm3D::MVDR { ... })?,
        _ => return Err(ApiError::InvalidAlgorithm),
    };
    
    let image = beamformer.process(&request.rf_data)?;
    Ok(Json(BeamformResponse { image }))
}
```

**After:**
```rust
// infra/api/clinical_handlers.rs (NEW)

use crate::analysis::signal_processing::beamforming::{BeamformingAlgorithm, beamform};

async fn beamform_endpoint(
    Json(request): Json<BeamformRequest>,
) -> Result<Json<BeamformResponse>, ApiError> {
    let algorithm = match request.algorithm.as_str() {
        "DAS" => BeamformingAlgorithm::DAS,
        "MVDR" => BeamformingAlgorithm::MVDR {
            diagonal_loading: request.diagonal_loading.unwrap_or(0.01),
            subarray_size: request.subarray_size,
        },
        _ => return Err(ApiError::InvalidAlgorithm),
    };
    
    let image = beamform(&request.rf_data, &request.geometry, &algorithm)?;
    Ok(Json(BeamformResponse { image }))
}
```

---

## Phase 3: Cleanup (Sprint 215)

### Step 3.1: Remove Deprecated Code

```bash
# Remove deprecated beamforming modules
rm -rf src/domain/sensor/beamforming/adaptive/
rm -rf src/domain/sensor/beamforming/neural/
rm -rf src/domain/sensor/beamforming/beamforming_3d/
rm -rf src/domain/sensor/beamforming/time_domain/

# Keep only geometry exports
# src/domain/sensor/beamforming/
#   ├── mod.rs (stub with re-exports)
#   └── geometry.rs (sensor geometry for beamforming)
```

### Step 3.2: Final Module Structure

```rust
// src/domain/sensor/beamforming/mod.rs (FINAL)

//! Sensor geometry exports for beamforming
//!
//! This module provides sensor geometry data structures used by beamforming algorithms
//! in `analysis::signal_processing::beamforming`.
//!
//! **Note:** All beamforming algorithms have been moved to the analysis layer.
//! See `analysis::signal_processing::beamforming` for beamforming implementations.

pub mod geometry;
pub use geometry::{BeamformingGeometry, SensorGeometry};

// Re-export from analysis layer for convenience (optional)
// Users should prefer direct imports from analysis layer
pub use crate::analysis::signal_processing::beamforming as algorithms;
```

```rust
// src/domain/sensor/beamforming/geometry.rs

//! Sensor geometry data structures for beamforming
//!
//! These types describe sensor array geometry (positions, orientations)
//! without implementing any beamforming algorithms.

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Sensor geometry for beamforming calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamformingGeometry {
    /// Sensor positions [N_sensors, 3] (x, y, z coordinates)
    pub positions: Array2<f64>,
    
    /// Sensor orientations [N_sensors, 3] (normal vectors)
    pub orientations: Array2<f64>,
    
    /// Sensor element sizes (optional)
    pub element_sizes: Option<Array1<f64>>,
    
    /// Sampling frequency (Hz)
    pub sampling_frequency: f64,
    
    /// Speed of sound (m/s) - for time-of-flight calculations
    pub speed_of_sound: f64,
}

impl BeamformingGeometry {
    /// Create geometry from sensor array
    pub fn from_sensor_array(sensors: &crate::domain::sensor::GridSensorSet) -> Self {
        BeamformingGeometry {
            positions: sensors.positions(),
            orientations: sensors.orientations(),
            element_sizes: Some(sensors.element_sizes()),
            sampling_frequency: sensors.sampling_frequency(),
            speed_of_sound: 1500.0,  // Default, override if needed
        }
    }
    
    /// Number of sensors
    pub fn num_sensors(&self) -> usize {
        self.positions.nrows()
    }
}
```

### Step 3.3: Update Cargo.toml

```toml
[package]
version = "0.9.0"  # Major version bump for removal of deprecated code
```

### Step 3.4: Update lib.rs Exports

```rust
// src/lib.rs

// Remove deprecated re-exports
// ❌ DELETE:
// pub use domain::sensor::beamforming::{Beamformer3D, BeamformingAlgorithm3D};

// Add new re-exports
pub mod beamforming {
    //! Beamforming algorithms and utilities
    //! 
    //! All beamforming algorithms are in the analysis layer.
    //! This re-export provides convenient access.
    
    pub use crate::analysis::signal_processing::beamforming::*;
}

pub mod sensor_geometry {
    //! Sensor geometry for beamforming
    pub use crate::domain::sensor::beamforming::geometry::*;
}
```

---

## Verification Checklist

### Code Quality Checks

- [ ] **Compile without warnings:** `cargo build --all-features 2>&1 | grep -i "warning.*deprecated"`
- [ ] **All tests pass:** `cargo test --all-features`
- [ ] **Benchmarks run:** `cargo bench --bench beamforming_benchmark`
- [ ] **Examples compile:** `cargo build --examples`
- [ ] **Documentation builds:** `cargo doc --no-deps --all-features`

### Deprecation Checks (Sprint 213-214)

- [ ] Deprecation warnings appear when using old API
- [ ] Migration guide linked in deprecation messages
- [ ] Transitional delegation functions work correctly
- [ ] No performance regression from delegation

### Migration Checks (Sprint 214)

- [ ] All clinical workflows use `analysis::signal_processing::beamforming`
- [ ] All examples updated to new API
- [ ] All tests updated to new API
- [ ] API handlers updated (if applicable)
- [ ] No imports from `domain::sensor::beamforming::*` (except `geometry`)

### Cleanup Checks (Sprint 215)

- [ ] Deprecated code removed
- [ ] Only `geometry.rs` remains in `domain/sensor/beamforming/`
- [ ] Version bumped to 0.9.0
- [ ] Changelog updated with breaking changes
- [ ] Migration guide published in documentation

### Functional Checks

- [ ] DAS beamforming produces identical results (regression test)
- [ ] MVDR beamforming produces identical results
- [ ] Neural beamforming works with new API
- [ ] 3D beamforming performance unchanged
- [ ] GPU beamforming (if implemented) works

---

## Automated Migration Script

For large codebases, use this script to automate some migrations:

```bash
#!/bin/bash
# migrate_beamforming.sh

# Replace imports
find src examples tests benches -name "*.rs" -exec sed -i \
  's/use crate::domain::sensor::beamforming::/use crate::analysis::signal_processing::beamforming::/g' {} \;

# Replace old beamformer instantiation patterns
find src examples tests benches -name "*.rs" -exec sed -i \
  's/Beamformer3D::new(BeamformingAlgorithm3D::DAS)/beamform(\&rf_data, \&geometry, \&BeamformingAlgorithm::DAS)/g' {} \;

# Note: This is a rough approximation. Manual review required!
echo "Automated migration complete. REVIEW ALL CHANGES before committing!"
echo "Run: git diff"
```

**Warning:** This script is a starting point. Manual review is essential.

---

## Common Migration Patterns

### Pattern 1: Object-Oriented to Functional

**Before:**
```rust
let beamformer = DASBeamformer::new(config)?;
let image = beamformer.process(rf_data)?;
```

**After:**
```rust
let image = beamform(rf_data, &geometry, &BeamformingAlgorithm::DAS)?;
```

### Pattern 2: Embedded Geometry to Explicit Geometry

**Before:**
```rust
let beamformer = Beamformer3D {
    sensor_positions: positions,
    sensor_orientations: orientations,
    algorithm: BeamformingAlgorithm3D::MVDR { ... },
};
let image = beamformer.process(rf_data)?;
```

**After:**
```rust
let geometry = BeamformingGeometry {
    positions,
    orientations,
    sampling_frequency: 10e6,
    speed_of_sound: 1500.0,
};
let image = beamform(rf_data, &geometry, &BeamformingAlgorithm::MVDR { ... })?;
```

### Pattern 3: Builder Pattern to Direct Configuration

**Before:**
```rust
let config = AdaptiveBeamformingConfig::builder()
    .diagonal_loading(0.01)
    .subarray_size([32, 32, 16])
    .build()?;
let beamformer = AdaptiveBeamformer::new(config)?;
```

**After:**
```rust
let algorithm = BeamformingAlgorithm::MVDR {
    diagonal_loading: 0.01,
    subarray_size: [32, 32, 16],
};
// Use directly in beamform()
```

### Pattern 4: Stateful to Stateless

**Before:**
```rust
let mut beamformer = NeuralBeamformer::new(config)?;
beamformer.load_model(model_path)?;
for frame in frames {
    let image = beamformer.process(frame)?;
}
```

**After:**
```rust
let algorithm = BeamformingAlgorithm::Neural {
    model_path: model_path.clone(),
    uncertainty: true,
};
for frame in frames {
    let image = beamform(frame, &geometry, &algorithm)?;
}
// Model loaded once per beamform() call (cached internally)
```

---

## Performance Considerations

### Optimization: Geometry Caching

If calling `beamform()` repeatedly with same geometry:

```rust
// Cache geometry calculation
let geometry = BeamformingGeometry::from_sensor_array(&sensors);

// Reuse for all frames (avoid recalculation)
for frame in acquisition.frames() {
    let image = beamform(&frame.rf_data, &geometry, &algorithm)?;
}
```

### Optimization: Pre-compiled Neural Models

For neural beamforming, pre-compile models:

```rust
// Before loop
let model = NeuralBeamformer::load_model(&model_path)?;
let algorithm = BeamformingAlgorithm::NeuralPrecompiled { model };

// In loop (no model loading overhead)
for frame in frames {
    let image = beamform(&frame, &geometry, &algorithm)?;
}
```

---

## Breaking Changes Summary (v0.9.0)

1. **Removed modules:**
   - `domain::sensor::beamforming::adaptive`
   - `domain::sensor::beamforming::neural`
   - `domain::sensor::beamforming::beamforming_3d`
   - `domain::sensor::beamforming::time_domain`

2. **Removed types:**
   - `Beamformer3D`
   - `BeamformingAlgorithm3D`
   - `AdaptiveBeamformer`
   - `NeuralBeamformer`
   - `DASBeamformer`

3. **New API:**
   - `analysis::signal_processing::beamforming::beamform()`
   - `analysis::signal_processing::beamforming::BeamformingAlgorithm`
   - `domain::sensor::beamforming::geometry::BeamformingGeometry`

4. **Migration path:**
   - v0.7.x: Old API (no warnings)
   - v0.8.x: Both APIs (deprecation warnings)
   - v0.9.0+: New API only (old API removed)

---

## Support and Questions

- **Documentation:** `cargo doc --open --all-features`
- **Migration issues:** GitHub Issues with tag `migration-beamforming`
- **Examples:** See `examples/adaptive_beamforming_refactored.rs` (updated)
- **Tests:** See `tests/beamforming_tests.rs` (updated)

---

## Appendix: Side-by-Side Comparison

### Complete Example: Adaptive Beamforming Workflow

#### Before (v0.7.x - Domain Layer)

```rust
use kwavers::domain::sensor::beamforming::adaptive::{AdaptiveBeamformer, MVDRConfig};
use kwavers::domain::sensor::GridSensorSet;
use kwavers::simulation::configuration::Configuration;

fn run_adaptive_beamforming() -> Result<(), Box<dyn std::error::Error>> {
    // Configure sensors
    let config = Configuration::default();
    let sensors = GridSensorSet::new(/* ... */)?;
    
    // Create beamformer (domain layer)
    let beamformer = AdaptiveBeamformer::new(MVDRConfig {
        diagonal_loading: 0.01,
        subarray_size: [32, 32, 16],
    })?;
    
    // Process RF data
    let rf_data = acquire_data(&sensors)?;
    let image = beamformer.process(&rf_data)?;
    
    // Visualize
    save_image("adaptive_beamformed.png", &image)?;
    Ok(())
}
```

#### After (v0.9.0+ - Analysis Layer)

```rust
use kwavers::analysis::signal_processing::beamforming::{BeamformingAlgorithm, beamform};
use kwavers::domain::sensor::beamforming::geometry::BeamformingGeometry;
use kwavers::domain::sensor::GridSensorSet;
use kwavers::simulation::configuration::Configuration;

fn run_adaptive_beamforming() -> Result<(), Box<dyn std::error::Error>> {
    // Configure sensors
    let config = Configuration::default();
    let sensors = GridSensorSet::new(/* ... */)?;
    
    // Extract geometry for beamforming
    let geometry = BeamformingGeometry::from_sensor_array(&sensors);
    
    // Process RF data (analysis layer)
    let rf_data = acquire_data(&sensors)?;
    let image = beamform(
        &rf_data,
        &geometry,
        &BeamformingAlgorithm::MVDR {
            diagonal_loading: 0.01,
            subarray_size: [32, 32, 16],
        },
    )?;
    
    // Visualize
    save_image("adaptive_beamformed.png", &image)?;
    Ok(())
}
```

**Changes:**
1. Import from `analysis::signal_processing::beamforming` instead of `domain::sensor::beamforming::adaptive`
2. Extract geometry explicitly: `BeamformingGeometry::from_sensor_array(&sensors)`
3. Use functional `beamform()` instead of object-oriented `AdaptiveBeamformer::new().process()`
4. Algorithm specified as enum variant: `BeamformingAlgorithm::MVDR { ... }`

---

**Migration Guide Version:** 1.0  
**Last Updated:** 2026-01-23  
**Applies To:** kwavers v0.8.0 → v0.9.0  
**Status:** Ready for Sprint 213 Implementation
