# Sprint 188 - Phase 3: Domain Layer Cleanup
## Audit Document

**Date**: 2024-01-XX  
**Phase**: 3 of 5  
**Focus**: Domain Layer Purity - Remove Application Logic  
**Status**: 🔍 AUDIT COMPLETE

---

## Executive Summary

Phase 3 focuses on enforcing **domain layer purity** by ensuring `src/domain/` contains only **pure domain entities** (primitives, specifications, core business concepts) and moving all **application logic** to appropriate upper layers (`analysis/`, `clinical/`).

### Key Findings

- ✅ **Grid, Medium, Geometry, Source, Sensor**: Already pure entities
- ⚠️ **Signal Processing**: Contains application-level filter implementations
- ⚠️ **Imaging**: Contains application-level photoacoustic workflow types
- ⚠️ **Therapy**: Contains application-level therapy modality enums
- ✅ **Physics**: Already moved to `physics/foundations/` in Phase 1

### Violation Summary

| Module | Issue | Severity | Action |
|--------|-------|----------|--------|
| `domain/signal/filter.rs` | Filter implementations (FFT-based) | Medium | Move to `analysis/signal_processing/filtering/` |
| `domain/imaging/photoacoustic.rs` | Workflow parameters/results | High | Move to `clinical/imaging/photoacoustic/` |
| `domain/therapy/modalities.rs` | Application-level enums | Medium | Move to `clinical/therapy/modalities.rs` |
| `domain/therapy/metrics.rs` | Application metrics | Medium | Move to `clinical/therapy/metrics.rs` |
| `domain/therapy/parameters.rs` | Application parameters | Medium | Move to `clinical/therapy/parameters.rs` |

---

## Architecture Principles

### What Belongs in Domain Layer

**✅ ALLOWED: Pure Domain Entities**
- Primitives: Grid, Geometry, Medium properties
- Interfaces: Trait definitions (`Signal`, `CoreMedium`, `Source`)
- Value Objects: `PointLocation`, `Dimension`, `TensorShape`
- Specifications: Mathematical contracts (moved to `physics/`)
- Domain Services: Pure domain logic with no I/O

**❌ FORBIDDEN: Application Logic**
- Workflow orchestration (belongs in `clinical/`, `simulation/`)
- Algorithm implementations (belongs in `analysis/`, `solver/`)
- Configuration management (belongs in application layer)
- Complex business rules combining multiple domains

### Clean Architecture Layer Rules

```text
┌──────────────────────────────────────────────────────┐
│  Application Layer (clinical/, api/)                 │
│  - Workflows, use cases, orchestration               │
│  - Can import: ALL lower layers                      │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│  Analysis Layer (analysis/)                          │
│  - Signal processing, beamforming, validation        │
│  - Can import: domain, physics, solver, math, core   │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│  Solver Layer (solver/)                              │
│  - Numerical methods (FDTD, PSTD, DG, PINN)          │
│  - Can import: physics, domain, math, core           │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│  Physics Layer (physics/)                            │
│  - Physics specifications, wave equations            │
│  - Can import: domain, math, core                    │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│  Domain Layer (domain/)  ← PHASE 3 FOCUS             │
│  - Pure entities, primitives, interfaces             │
│  - Can import: math, core ONLY                       │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│  Math Layer (math/)                                  │
│  - Mathematical primitives, FFT, linear algebra      │
│  - Can import: core ONLY                             │
└───────────────────┬──────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────┐
│  Core Layer (core/)                                  │
│  - Error types, common utilities                     │
│  - No dependencies (foundation)                      │
└──────────────────────────────────────────────────────┘
```

---

## Detailed Audit Results

### 1. Signal Module (`domain/signal/`)

#### Current Structure
```
domain/signal/
├── mod.rs              ✅ Signal trait, sampling utilities
├── traits.rs           ✅ Core Signal trait
├── filter.rs           ⚠️ Filter implementations (FFT-based)
├── analytic.rs         ✅ Analytic signal (Hilbert transform helper)
├── window.rs           ✅ Window functions (primitives)
├── waveform/           ✅ Basic waveforms (sine, square, triangle)
├── pulse/              ✅ Pulse signals (Gaussian, Ricker, tone burst)
├── frequency_sweep/    ✅ Chirp signals
├── modulation/         ✅ AM/FM/PM modulation
├── amplitude/          ✅ Amplitude envelopes
├── phase/              ✅ Phase signals
├── frequency/          ✅ Frequency signals
└── special/            ✅ Null signal, time-varying
```

#### Analysis

**Keep in Domain (Pure Primitives):**
- ✅ `Signal` trait - Core interface
- ✅ Waveform generators (sine, square, triangle) - Pure signal primitives
- ✅ Pulse shapes (Gaussian, Ricker, tone burst) - Domain concepts
- ✅ Window functions - Mathematical primitives
- ✅ Modulation types - Signal generation primitives
- ✅ Helper functions: `sample_signal()`, `next_pow2()`, `pad_zeros()`

**Move to Analysis Layer:**
- ⚠️ `filter.rs::FrequencyFilter` - FFT-based filtering is signal processing
- ⚠️ `filter.rs::Filter` trait - Can stay as interface, implementation moves

**Rationale:**
- Signal *generation* (waveforms, pulses) is a domain primitive
- Signal *processing* (filtering, transformation) is analysis
- The `Filter` trait defines an interface (domain concept)
- `FrequencyFilter` implements algorithms (analysis concern)

#### Action Items
1. Create `analysis/signal_processing/filtering/frequency_filter.rs`
2. Move `FrequencyFilter` implementation
3. Keep `Filter` trait in `domain/signal/filter.rs` as interface
4. Update imports in tests/examples

---

### 2. Imaging Module (`domain/imaging/`)

#### Current Structure
```
domain/imaging/
├── mod.rs                  ✅ Module exports
├── ultrasound.rs           ⚠️ Application-level types
└── photoacoustic.rs        ⚠️ Workflow parameters, results
```

#### Analysis: `photoacoustic.rs`

**Current Contents:**
```rust
pub struct PhotoacousticResult {
    pub pressure_fields: Vec<Array3<f64>>,
    pub time: Vec<f64>,
    pub reconstructed_image: Array3<f64>,
    pub snr: f64,
}

pub struct PhotoacousticParameters {
    pub wavelengths: Vec<f64>,
    pub absorption_coefficients: Vec<f64>,
    pub scattering_coefficients: Vec<f64>,
    // ... 10+ fields combining optics + acoustics
}

pub struct OpticalProperties {
    pub absorption: f64,
    pub scattering: f64,
    pub anisotropy: f64,
    pub refractive_index: f64,
}
```

**Problem:**
- `PhotoacousticResult` is a workflow output DTO
- `PhotoacousticParameters` combines optics + acoustics in application-specific way
- These are **use case** types, not domain primitives
- Violates Single Responsibility Principle at domain layer

**Solution:**
Move to `clinical/imaging/photoacoustic/` where workflow logic belongs

**Keep in Domain:**
- Pure optical property definitions could stay if needed as primitives
- But current `OpticalProperties::blood()`, `::soft_tissue()` are application presets

#### Action Items
1. Move entire `domain/imaging/photoacoustic.rs` to `clinical/imaging/photoacoustic/types.rs`
2. Update `domain/imaging/mod.rs` to remove photoacoustic exports
3. Create `clinical/imaging/photoacoustic/mod.rs` with proper re-exports
4. Update all imports (`domain::imaging::photoacoustic` → `clinical::imaging::photoacoustic`)

---

### 3. Therapy Module (`domain/therapy/`)

#### Current Structure
```
domain/therapy/
├── mod.rs            ✅ Module definition
├── modalities.rs     ⚠️ Application-level enums
├── metrics.rs        ⚠️ Therapy outcome metrics
└── parameters.rs     ⚠️ Treatment parameters
```

#### Analysis: `modalities.rs`

**Current Contents:**
```rust
pub enum TherapyMechanism {
    Thermal,
    Mechanical,
    Combined,
}

pub enum TherapyModality {
    HIFU,
    LIFU,
    Histotripsy,
    BBBOpening,
    Sonodynamic,
    Sonoporation,
}

impl TherapyModality {
    pub fn has_thermal_effects(&self) -> bool { ... }
    pub fn has_cavitation(&self) -> bool { ... }
    pub fn primary_mechanism(&self) -> TherapyMechanism { ... }
}
```

**Problem:**
- `TherapyModality` is a **clinical concept**, not a domain primitive
- Business rules like `has_thermal_effects()` belong in application layer
- These are treatment protocols, not fundamental physics concepts

**Solution:**
Move entire `domain/therapy/` module to `clinical/therapy/`

**Alternative Consideration:**
Could keep minimal physics-level enums in domain, but current implementation is too high-level

#### Action Items
1. Move `domain/therapy/modalities.rs` → `clinical/therapy/modalities.rs`
2. Move `domain/therapy/metrics.rs` → `clinical/therapy/metrics.rs`
3. Move `domain/therapy/parameters.rs` → `clinical/therapy/parameters.rs`
4. Update `domain/mod.rs` to remove `therapy` module
5. Update `clinical/therapy/mod.rs` to include moved files
6. Update all imports across codebase

---

### 4. What Stays in Domain (Verified Clean)

#### ✅ Grid Module (`domain/grid/`)
- **Purpose**: Computational grid primitives
- **Contents**: Grid dimensions, spacing, indexing
- **Status**: ✅ Pure domain entity - NO CHANGES NEEDED

#### ✅ Geometry Module (`domain/geometry/`)
- **Purpose**: Spatial domain definitions
- **Contents**: `RectangularDomain`, `SphericalDomain`, `PointLocation`
- **Status**: ✅ Pure domain primitives - NO CHANGES NEEDED

#### ✅ Medium Module (`domain/medium/`)
- **Purpose**: Material property specifications
- **Contents**: `CoreMedium` trait, `HomogeneousMedium`, property access
- **Status**: ✅ Pure domain interface - NO CHANGES NEEDED

#### ✅ Source Module (`domain/source/`)
- **Purpose**: Acoustic source primitives
- **Contents**: `Source` trait, source geometry, parameters
- **Status**: ✅ Pure domain primitives - NO CHANGES NEEDED

#### ✅ Sensor Module (`domain/sensor/`)
- **Purpose**: Sensor geometry and data collection
- **Contents**: Sensor positioning, sampling rates, data buffers
- **Status**: ✅ Pure domain primitives (beamforming already moved in Sprint 4)

#### ✅ Boundary Module (`domain/boundary/`)
- **Purpose**: Boundary condition specifications
- **Contents**: PML, Dirichlet, Neumann conditions
- **Status**: ✅ Pure domain specifications - NO CHANGES NEEDED

#### ✅ Field Module (`domain/field/`)
- **Purpose**: Physical field representations
- **Contents**: Pressure, velocity, stress tensor fields
- **Status**: ✅ Pure domain entities - NO CHANGES NEEDED

#### ✅ Tensor Module (`domain/tensor/`)
- **Purpose**: Computational array abstractions
- **Contents**: `TensorView`, `TensorMut`, backend abstraction
- **Status**: ✅ Pure domain primitives - NO CHANGES NEEDED

---

## Migration Strategy

### Phase 3.1: Signal Processing Filter (Est. 30 min)

**Step 1: Create Analysis Structure**
```bash
mkdir -p src/analysis/signal_processing/filtering
touch src/analysis/signal_processing/filtering/mod.rs
touch src/analysis/signal_processing/filtering/frequency_filter.rs
```

**Step 2: Move Implementation**
- Copy `FrequencyFilter` from `domain/signal/filter.rs`
- Keep `Filter` trait in domain as interface
- Add proper module docs and imports

**Step 3: Update Exports**
- Update `analysis/signal_processing/mod.rs`
- Add re-exports for backward compatibility

**Step 4: Update Callers**
- Search for `use.*domain::signal::filter::FrequencyFilter`
- Replace with `analysis::signal_processing::filtering::FrequencyFilter`
- Estimated: ~5-10 files

---

### Phase 3.2: Imaging Types (Est. 45 min)

**Step 1: Create Clinical Structure**
```bash
mkdir -p src/clinical/imaging/photoacoustic
touch src/clinical/imaging/photoacoustic/mod.rs
touch src/clinical/imaging/photoacoustic/types.rs
```

**Step 2: Move Types**
- Move entire `domain/imaging/photoacoustic.rs` content
- Add proper module documentation
- Update trait bounds if needed

**Step 3: Clean Domain**
- Remove `domain/imaging/photoacoustic.rs`
- Update `domain/imaging/mod.rs`

**Step 4: Update Callers**
- Search for `use.*domain::imaging::photoacoustic`
- Replace with `clinical::imaging::photoacoustic`
- Estimated: ~10-15 files (tests + examples)

---

### Phase 3.3: Therapy Types (Est. 45 min)

**Step 1: Move Files**
```bash
mv src/domain/therapy/modalities.rs src/clinical/therapy/
mv src/domain/therapy/metrics.rs src/clinical/therapy/
mv src/domain/therapy/parameters.rs src/clinical/therapy/
```

**Step 2: Update Modules**
- Update `clinical/therapy/mod.rs` to include moved files
- Remove `domain/therapy/` directory
- Update `domain/mod.rs`

**Step 3: Update Callers**
- Search for `use.*domain::therapy`
- Replace with `clinical::therapy`
- Estimated: ~8-12 files

---

### Phase 3.4: Documentation & Validation (Est. 30 min)

**Step 1: Update Domain Documentation**
- Update `src/domain/mod.rs` header comments
- Remove references to moved modules
- Emphasize "pure entities only" principle

**Step 2: Verify Build**
```bash
cargo check --workspace --all-features
cargo test --workspace --all-features
cargo clippy --workspace --all-features
```

**Step 3: Update Architecture Docs**
- Update `README.md` architecture section
- Update `docs/adr.md` with Phase 3 decisions
- Create `docs/sprint_188_phase3_complete.md`

---

## Success Criteria

### Functional Requirements
- ✅ All moved types accessible from new locations
- ✅ Zero compilation errors
- ✅ All existing tests pass (1052/1052 or better)
- ✅ No new test failures introduced

### Architectural Requirements
- ✅ Domain layer contains only pure entities
- ✅ No application logic in domain
- ✅ Clear layer separation maintained
- ✅ Dependency flow: Application → Analysis → Solver → Physics → Domain → Math → Core

### Documentation Requirements
- ✅ Module documentation updated
- ✅ Migration guide in completion doc
- ✅ README reflects new architecture
- ✅ ADR updated with decisions

---

## Risk Assessment

### Low Risk
- ✅ Signal filter move - isolated, well-defined interface
- ✅ Limited number of files affected (~20-30 total)

### Medium Risk
- ⚠️ Photoacoustic types - used in multiple test files
- ⚠️ Need to verify feature flag compatibility

### Mitigation Strategies
1. **Incremental Migration**: Move one module at a time
2. **Continuous Testing**: Run tests after each module move
3. **Backward Compatibility**: Consider deprecation warnings before removal
4. **Feature Flags**: Ensure GPU/PINN features still work correctly

---

## Estimated Effort

| Phase | Task | Time | Complexity |
|-------|------|------|------------|
| 3.1 | Signal filter migration | 30 min | Low |
| 3.2 | Imaging types migration | 45 min | Medium |
| 3.3 | Therapy types migration | 45 min | Medium |
| 3.4 | Documentation & validation | 30 min | Low |
| **Total** | **Phase 3 Complete** | **2.5 hours** | **Medium** |

---

## Next Steps

1. **Immediate**: Begin Phase 3.1 (Signal filter migration)
2. **After 3.1**: Proceed with 3.2 (Imaging types)
3. **After 3.2**: Complete 3.3 (Therapy types)
4. **Final**: Documentation and validation (3.4)

---

## References

### Architecture Decisions
- ADR-022: Physics Layer Consolidation (Sprint 188 Phase 1)
- ADR-023: Solver Interface Standardization (Planned Phase 4)
- ADR-024: Domain Layer Purity (This Phase)

### Design Principles
- Clean Architecture (Robert C. Martin)
- Domain-Driven Design (Eric Evans)
- SOLID Principles (especially SRP, DIP)
- Hexagonal Architecture (Ports & Adapters)

### Related Documents
- `docs/sprint_188_phase1_complete.md` - Physics consolidation
- `docs/sprint_188_phase2_complete.md` - Circular dependency removal
- `docs/prd.md` - Product requirements
- `docs/srs.md` - Software requirements

---

**End of Phase 3 Audit**

Ready to proceed with implementation upon approval.