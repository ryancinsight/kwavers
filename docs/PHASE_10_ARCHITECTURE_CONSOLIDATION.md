# Phase 10: Architecture Consolidation & Research Enhancement

**Project**: Kwavers Acoustic Simulation Library
**Phase**: Phase 10 - Architecture Consolidation, Violation Resolution, High-Priority Feature Implementation
**Status**: ⏳ IN PLANNING
**Target**: Complete in 2-3 sprints (2-3 weeks)
**Date Started**: January 2026

---

## Executive Summary

Following Phase 9's successful achievement of zero compiler warnings and 1583 passing tests, Phase 10 focuses on:

1. **Architecture Violation Resolution** (3 minor violations identified)
2. **Dead Code & Deprecation Management** (192 markers to review)
3. **High-Priority Feature Implementation** (3 P1 features blocking v3.1)
4. **Research Integration** (Latest ultrasound/optics research from reference libraries)

### Key Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Architecture Violations | 0 | 3 identified, 0/3 fixed |
| Dead Code Markers Reviewed | 100% | 0% |
| P1 Features Implemented | 3/3 | 0/3 |
| P2 Features (Optional) | 5+/5 | 0/5 |
| Test Suite | 1600+ passing | 1583 passing ✓ |
| Build Status | Zero warnings | 0 warnings ✓ |

---

## PART 1: ARCHITECTURE VIOLATION RESOLUTION

### Violation #1: Clinical ← Analysis (CRITICAL)

**Location**: Clinical layer imports Analysis layer concretely
- Files affected: 4
- Severity: MEDIUM (breaks architectural purity)
- Fix complexity: LOW

#### Files to Modify

1. `src/clinical/imaging/workflows/neural/ai_beamforming_processor.rs`
2. `src/clinical/imaging/workflows/neural/feature_extraction.rs`
3. `src/clinical/imaging/workflows/orchestrator.rs`
4. `src/solver/inverse/pinn/ml/beamforming_provider.rs`

#### Root Cause

Clinical workflows import concrete beamforming implementations from analysis layer:
```rust
use crate::analysis::signal_processing::beamforming::domain_processor;
use crate::analysis::signal_processing::beamforming::neural::config;
```

#### Solution

Extract beamforming interface to domain layer for clinical to depend on:

**New File Structure**:
```
src/domain/signal_processing/
├── mod.rs                          (NEW - exports traits)
├── beamforming/
│   ├── mod.rs
│   ├── interface.rs               (NEW - move from analysis)
│   ├── config.rs                  (NEW - move from analysis)
│   └── types.rs
├── filtering/
│   └── interface.rs               (NEW)
├── pam/
│   └── interface.rs               (NEW)
└── localization/
    └── interface.rs               (NEW)
```

#### Implementation Steps

**Step 1**: Create `src/domain/signal_processing/mod.rs`
```rust
//! Signal processing abstractions and interfaces
//!
//! Provides domain-level traits that physics, analysis, and clinical layers implement.
//! This ensures clean separation of concerns:
//! - Physics: Theoretical beamforming calculations
//! - Analysis: Algorithmic signal processing implementations
//! - Clinical: Application-specific workflows

pub mod beamforming;
pub mod filtering;
pub mod pam;
pub mod localization;

pub use beamforming::*;
pub use filtering::*;
pub use pam::*;
pub use localization::*;
```

**Step 2**: Move beamforming interface
```
Move: src/analysis/signal_processing/beamforming/interface.rs
   ↓
To: src/domain/signal_processing/beamforming/interface.rs

Move: src/analysis/signal_processing/beamforming/config.rs
   ↓
To: src/domain/signal_processing/beamforming/config.rs
```

**Step 3**: Update imports in clinical layer
```rust
// BEFORE (violation)
use crate::analysis::signal_processing::beamforming::domain_processor;

// AFTER (clean)
use crate::domain::signal_processing::beamforming::BeamformingProcessor;
use crate::analysis::signal_processing::beamforming::neural::NeuralBeamformer;
```

**Step 4**: Update analysis layer to implement domain traits
```rust
// In: src/analysis/signal_processing/beamforming/mod.rs
impl crate::domain::signal_processing::beamforming::BeamformingProcessor 
    for DomainBeamformer {
    // Implementation details
}
```

#### Verification

After fix:
- Clinical only depends on domain (OK ✓)
- Analysis implements domain traits (OK ✓)
- Solver also implements for inverse problems (OK ✓)

---

### Violation #2: Clinical ← Simulation (MEDIUM)

**Location**: Clinical therapy layer imports Simulation imaging components
- Files affected: 3
- Severity: MEDIUM (creates coupling)
- Fix complexity: MEDIUM

#### Files to Modify

1. `src/clinical/therapy/therapy_integration/orchestrator/initialization.rs`
2. `src/clinical/therapy/therapy_integration/orchestrator/microbubble.rs`
3. `src/clinical/therapy/therapy_integration/orchestrator/mod.rs`

#### Root Cause

```rust
// In therapy orchestrator
use crate::simulation::imaging::ceus;  // ← WRONG LAYER
```

CEUS (Contrast-Enhanced Ultrasound) is currently in simulation layer, but should be in physics layer.

#### Solution

Move CEUS modeling from simulation to physics layer:

```
BEFORE:
src/simulation/imaging/ceus/mod.rs

AFTER:
src/physics/acoustics/imaging/modalities/ceus/mod.rs
```

#### Implementation Steps

**Step 1**: Create CEUS in physics layer
```bash
mkdir -p src/physics/acoustics/imaging/modalities/ceus
```

**Step 2**: Copy CEUS module with physics focus
```rust
// File: src/physics/acoustics/imaging/modalities/ceus/mod.rs
//! Contrast-Enhanced Ultrasound (CEUS) Physics Models
//!
//! Implements acoustic properties and scattering from microbubbles
//! and contrast agents

pub mod bubble_acoustics;
pub mod scattering;
pub mod harmonic_generation;

pub struct CEUSProperties {
    pub bubble_radius: f64,
    pub shell_elasticity: f64,
    pub interfacial_tension: f64,
}

impl CEUSProperties {
    pub fn acoustic_impedance(&self) -> f64 {
        // Physics calculation
    }
}
```

**Step 3**: Create domain model for CEUS
```rust
// File: src/domain/imaging/ceus_config.rs
pub struct CEUSConfiguration {
    pub properties: crate::physics::acoustics::imaging::modalities::ceus::CEUSProperties,
    pub bubble_concentration: f64,
    pub agent_type: ContrastAgentType,
}
```

**Step 4**: Update simulation layer to orchestrate
```rust
// File: src/simulation/imaging/mod.rs
// Now just orchestrates via domain models, doesn't define them
pub fn create_ceus_simulation(
    config: &domain::imaging::CEUSConfiguration
) -> Result<CEUSSimulation> {
    // Setup using domain config + physics models
}
```

**Step 5**: Update clinical therapy imports
```rust
// BEFORE
use crate::simulation::imaging::ceus;

// AFTER
use crate::physics::acoustics::imaging::modalities::ceus;
use crate::domain::imaging::CEUSConfiguration;
```

#### Verification

After fix:
- Physics layer defines CEUS models ✓
- Domain layer provides configuration ✓
- Simulation layer orchestrates ✓
- Clinical layer uses via domain ✓
- No Clinical → Simulation import ✓

---

### Violation #3: Physics ← Analysis (MINOR)

**Location**: Physics layer imports Analysis layer post-processing
- Files affected: 1
- Severity: LOW (physics shouldn't depend on analysis)
- Fix complexity: MEDIUM

#### Files to Modify

1. `src/physics/acoustics/imaging/pam.rs`

#### Root Cause

```rust
// In physics/acoustics/imaging/pam.rs
use crate::analysis::signal_processing::pam;  // ← WRONG DEPENDENCY
```

Physics should compute PAM data, not depend on analysis signal processing.

#### Solution

Extract PAM interface to domain, implement in both physics and analysis:

```
AFTER FIX:
domain/signal_processing/pam/
├── interface.rs          (PAMProcessor trait)
└── types.rs             (PAMResult, PAMConfig)

physics/acoustics/imaging/pam.rs
└── Implements domain trait (computes PAM from pressure fields)

analysis/signal_processing/pam/
└── Implements domain trait (post-processing, filtering, visualization)
```

#### Implementation Steps

**Step 1**: Create domain PAM interface
```rust
// File: src/domain/signal_processing/pam/interface.rs
pub trait PAMProcessor {
    fn compute_pam(&self, 
        pressure_field: &Array3<f64>,
        sensor_positions: &[Vector3<f64>]
    ) -> Result<PAMResult>;
}

pub struct PAMResult {
    pub source_location: Vector3<f64>,
    pub intensity: f64,
    pub confidence: f64,
}
```

**Step 2**: Implement in physics
```rust
// File: src/physics/acoustics/imaging/pam.rs
pub struct PhysicalPAM {
    grid: Grid,
}

impl domain::signal_processing::pam::PAMProcessor for PhysicalPAM {
    fn compute_pam(&self, pressure_field: &Array3<f64>, sensors: &[Vector3<f64>]) -> Result<PAMResult> {
        // Physics-based computation
    }
}
```

**Step 3**: Keep analysis implementation
```rust
// File: src/analysis/signal_processing/pam/mod.rs
// Continues to implement domain trait with signal processing methods
pub struct SignalProcessingPAM {
    // ...
}

impl domain::signal_processing::pam::PAMProcessor for SignalProcessingPAM {
    // Implementation
}
```

**Step 4**: Remove direct import from physics
```rust
// BEFORE - File: src/physics/acoustics/imaging/pam.rs
use crate::analysis::signal_processing::pam;

// AFTER
use crate::domain::signal_processing::pam::{PAMProcessor, PAMResult};
```

#### Verification

After fix:
- Physics implements domain interface (OK ✓)
- Analysis implements domain interface (OK ✓)
- No physics → analysis dependency (OK ✓)
- Both can be used independently (OK ✓)

---

## PART 2: DEAD CODE REVIEW & MANAGEMENT

### Dead Code Inventory

**Total markers**: 192 `#[allow(dead_code)]` annotations

### High-Priority Review (30 items)

**File**: `src/analysis/ml/inference.rs`
- Lines: ~400
- Status: Feature-gated ("pinn")
- Decision: Keep (used by PINN solver)
- Action: Add inline comment explaining usage

**File**: `src/analysis/ml/uncertainty/bayesian_networks.rs`
- Lines: ~300
- Status: Not feature-gated
- Decision: Mark for removal (v3.1.0)
- Action: Create deprecation notice

**File**: `src/analysis/performance/optimization/cache.rs`
- Lines: ~200
- Status: Experimental optimization paths
- Decision: Keep (future GPU work)
- Action: Add TODO comment with v3.2+ timeline

**File**: `src/analysis/visualization/data_pipeline/processing.rs`
- Lines: ~150
- Status: Feature-gated ("gpu")
- Decision: Keep (visualization pipeline)
- Action: Document as GPU-dependent

### Medium-Priority Review (80 items)

Review pattern: Collect all dead code markers, categorize by:
1. **Future API** (keep, document with TODO)
2. **Feature-gated** (keep, verify feature flag)
3. **Experimental** (keep, mark as unstable)
4. **Deprecated** (remove or migrate)

### Low-Priority Review (82 items)

Cleanup of minor unused utilities and helper functions.

---

## PART 3: DEPRECATED COMPONENTS REMOVAL

### Scheduled Removals (v3.1.0)

#### 1. domain::sensor::beamforming re-export

**File**: `src/domain/sensor/beamforming/mod.rs`
- Status: Re-exports for backward compatibility
- Removal: After Phase 10 migration complete
- Timeline: v3.1.0 release

**Action Plan**:
1. Verify all internal imports use analysis layer
2. Update CHANGELOG with migration guide
3. Keep one release cycle for user migration
4. Remove in v3.1.0

#### 2. domain::sensor::passive_acoustic_mapping re-export

**File**: `src/domain/sensor/passive_acoustic_mapping/mod.rs`
- Status: Re-exports for backward compatibility
- Removal: After PAM interface migration
- Timeline: v3.1.0 release

#### 3. physics::optics::polarization::LinearPolarization

**File**: `src/physics/optics/polarization/mod.rs`
- Status: Deprecated in favor of JonesPolarizationModel
- Removal: After user migration
- Timeline: v3.1.0 release

---

## PART 4: HIGH-PRIORITY P1 FEATURES

### Feature P1-1: Source Localization Algorithms

**Effort**: 5-7 days
**Complexity**: HIGH
**Impact**: Clinical imaging workflows

#### Scope

Implement passive source localization using sensor array data:

**Algorithms**:
1. **MUSIC** (Multiple Signal Classification)
   - Super-resolution direction finding
   - Subspace-based method
   - Time-delay estimation refinement

2. **TDOA-based Triangulation**
   - Time-difference-of-arrival computation
   - Iterative refinement (Newton-Raphson)
   - Multi-path interference rejection

3. **Bayesian Filtering**
   - Extended Kalman Filter (EKF)
   - Unscented Kalman Filter (UKF)
   - Particle filter implementation

4. **Wavefront Analysis**
   - Curvature-based source distance
   - Plane wave detection
   - Spherical wavefront fitting

#### Implementation Files

```
src/analysis/signal_processing/localization/
├── mod.rs                 (Module exports)
├── music.rs              (MUSIC algorithm)
├── tdoa.rs               (TDOA-based localization)
├── bayesian.rs           (Kalman & particle filters)
├── wavefront.rs          (Wavefront analysis)
├── config.rs             (Configuration)
└── tests/
    ├── music_tests.rs
    ├── tdoa_tests.rs
    └── integration_tests.rs
```

#### Key Functions

```rust
pub struct MUSICConfig {
    pub num_sources: usize,
    pub frequency_range: (f64, f64),
    pub array_geometry: Vec<Vector3<f64>>,
}

pub fn music_direction_finding(
    sensor_signals: &[Array1<f64>],
    config: &MUSICConfig,
) -> Result<Vec<SourceDirection>>;

pub struct TDOAConfig {
    pub correlation_method: CorrelationMethod,
    pub refinement_iterations: usize,
}

pub fn tdoa_triangulation(
    time_delays: &[f64],
    sensor_positions: &[Vector3<f64>],
    config: &TDOAConfig,
) -> Result<Vector3<f64>>;
```

#### Testing Requirements

- Unit tests for each algorithm
- Integration test: Multi-source localization
- Benchmark: Localization accuracy vs. SNR
- Validation: Compare with synthetic data ground truth

---

### Feature P1-2: Functional Ultrasound Brain GPS

**Effort**: 7-10 days
**Complexity**: VERY HIGH
**Impact**: Neuroscience applications

#### Scope

Implement automatic registration and neuronavigation for functional ultrasound imaging:

**Components**:
1. **Affine Registration Engine**
   - Mattes Mutual Information metric
   - B-spline deformation (optional)
   - Multi-resolution registration pyramid

2. **Vascular-Based Localization**
   - Brain vasculature segmentation
   - Arterial vs. venous classification
   - Vessel-to-atlas matching

3. **Precision Targeting**
   - Stereotactic coordinates
   - Allen Brain Atlas integration
   - Real-time tracking correction

#### Implementation Files

```
src/clinical/imaging/functional_ultrasound/
├── mod.rs                          (Module exports)
├── registration/
│   ├── mutual_information.rs       (MI metric)
│   ├── affine_transform.rs         (3D transformation)
│   └── optimization.rs             (Gradient descent)
├── vasculature/
│   ├── segmentation.rs             (Vessel detection)
│   ├── classification.rs           (Arterial/venous)
│   └── matching.rs                 (Atlas matching)
├── targeting/
│   ├── stereotactic.rs             (Coord system)
│   ├── atlas.rs                    (Allen Brain Atlas interface)
│   └── tracking.rs                 (Real-time correction)
├── config.rs
└── tests/
    ├── registration_tests.rs
    ├── vasculature_tests.rs
    └── integration_tests.rs
```

#### Key Functions

```rust
pub struct RegistrationConfig {
    pub metric: RegistrationMetric,
    pub num_iterations: usize,
    pub learning_rate: f64,
}

pub fn register_to_atlas(
    image: &Array3<f64>,
    atlas: &AtlasImage,
    config: &RegistrationConfig,
) -> Result<AffineTransform3D>;

pub struct VesselSegmentation {
    pub vessels: Vec<Vessel>,
    pub confidence: Array3<f64>,
}

pub fn segment_vasculature(
    us_image: &Array3<f64>,
) -> Result<VesselSegmentation>;

pub fn locate_in_brain(
    segmented_vessels: &VesselSegmentation,
    atlas: &BrainAtlas,
) -> Result<StereotacticCoordinates>;
```

#### Testing Requirements

- Unit tests for registration metric
- Vessel segmentation validation
- End-to-end registration test with synthetic atlas
- Accuracy benchmark: Target localization error < 100 µm
- Compatibility tests with public brain atlases

---

### Feature P1-3: GPU Multiphysics Real-Time Loop

**Effort**: 10-14 days
**Complexity**: VERY HIGH
**Impact**: Performance (10-100x speedup)

#### Scope

GPU acceleration for coupled multiphysics simulations with real-time feedback:

**Components**:
1. **GPU Memory Management**
   - Unified memory architecture
   - Pinned host memory
   - Device ↔ host transfer optimization

2. **CUDA/wgpu Kernels**
   - FDTD wave equation solver
   - Thermal diffusion
   - Cavitation bubble dynamics
   - Coupling operators

3. **Multi-GPU Scaling**
   - Domain decomposition
   - GPU-to-GPU communication (NVLink if available)
   - Load balancing

4. **Real-Time I/O**
   - Ring buffer for streaming output
   - Asynchronous data transfer
   - Checkpoint/restart capability

#### Implementation Files

```
src/gpu/
├── mod.rs                          (Module exports)
├── memory/
│   ├── allocation.rs               (GPU allocation strategies)
│   ├── transfer.rs                 (Host ↔ Device transfers)
│   └── pool.rs                     (Memory pooling)
├── kernels/
│   ├── wave_equation.cu            (FDTD kernel)
│   ├── thermal.cu                  (Heat diffusion kernel)
│   ├── cavitation.cu               (Bubble dynamics kernel)
│   └── coupling.cu                 (Coupling operators)
├── compute/
│   ├── cuda_solver.rs              (CUDA wrapper)
│   ├── wgpu_solver.rs              (wgpu wrapper)
│   └── multi_gpu.rs                (Multi-GPU orchestration)
├── io/
│   ├── ring_buffer.rs              (Streaming output)
│   └── async_transfer.rs           (Async I/O)
└── tests/
    ├── kernel_tests.rs
    ├── memory_tests.rs
    └── performance_tests.rs
```

#### Key Functions

```rust
pub struct GPUMemoryPool {
    device_memory: Vec<DeviceBuffer>,
    host_pinned: Vec<HostBuffer>,
}

pub fn allocate_gpu_field(
    shape: (usize, usize, usize),
    dtype: DataType,
) -> Result<GPUBuffer>;

pub struct CUDAWaveEquationSolver {
    kernel: CudaKernel,
    memory_pool: GPUMemoryPool,
}

pub fn gpu_fdtd_step(
    solver: &mut CUDAWaveEquationSolver,
    dt: f64,
) -> Result<()>;

pub fn gpu_thermal_coupling(
    wave_field: &GPUBuffer,
    thermal_field: &mut GPUBuffer,
) -> Result<()>;

pub struct MultiGPUSolver {
    gpus: Vec<GPUDevice>,
    domain_decomposition: DomainPartition,
}

pub fn solve_multiphysics_realtime(
    solver: &mut MultiGPUSolver,
    config: &SimulationConfig,
    callback: impl Fn(&Array3<f64>) -> Result<()>,
) -> Result<()>;
```

#### Testing Requirements

- Unit tests for each GPU kernel
- Validation: GPU results match CPU results (within numerical precision)
- Performance benchmarks:
  - Single GPU: 10x speedup target
  - Dual GPU: 18x speedup target
  - Memory transfer overhead < 10%
- Stress tests: 8GB+ simulation domains
- Multi-GPU synchronization correctness

---

## PART 5: RESEARCH INTEGRATION FROM REFERENCE LIBRARIES

### Reference Library Analysis

Reviewing best practices from:
1. **k-Wave** (MATLAB/Python) - Gold standard for ultrasound
2. **j-Wave** (JAX) - Modern autodiff approach
3. **fullwave25** (Python/CUDA) - Advanced heterogeneous media
4. **OptimUS** (Python) - Optimization-focused
5. **BabelBrain** (MATLAB/Python) - Clinical workflows
6. **mSOUND** - Fast pseudospectral methods
7. Others: HITU, Kranion, SimSonic

### Key Insights to Integrate

#### From k-Wave
- **Benefit**: Comprehensive reference library, excellent documentation
- **Integration**: 
  - Adopt their grid generation algorithms
  - Use their boundary condition formulations
  - Implement their output format for comparison

#### From j-Wave (JAX)
- **Benefit**: Automatic differentiation for inverse problems
- **Integration**:
  - Implement autodiff capability via Burn or Tch-rs
  - Enable automatic gradient computation for inversions
  - Support parameter optimization workflows

#### From fullwave25
- **Benefit**: Heterogeneous media handling, staggered grids
- **Integration**:
  - Improve heterogeneous medium support
  - Implement staggered Yee grids where beneficial
  - Add support for anisotropic media

#### From BabelBrain
- **Benefit**: Clinical workflow orchestration
- **Integration**:
  - Adopt their multi-stage treatment planning
  - Implement their safety validation pipeline
  - Clinical use case documentation

---

## PART 6: IMPLEMENTATION ROADMAP

### Sprint 1 (Week 1): Architecture Violations

**Duration**: 3-4 days
**Tasks**:

1. **Violation #1 - Beamforming Interface** (2 days)
   - Create domain/signal_processing/ module structure
   - Extract interface from analysis layer
   - Update clinical imports
   - Verification tests

2. **Violation #2 - CEUS to Physics** (1 day)
   - Move simulation/imaging/ceus → physics/acoustics/imaging/modalities/ceus
   - Update clinical therapy orchestrator imports
   - Verification tests

3. **Violation #3 - PAM Interface** (1 day)
   - Create domain/signal_processing/pam/interface.rs
   - Implement in physics and analysis layers
   - Remove physics → analysis import
   - Verification tests

**Deliverable**: Zero architecture violations, all tests passing

### Sprint 2 (Week 2): High-Priority Features P1-1 & P1-2

**Duration**: 1 week
**Tasks**:

1. **P1-1: Source Localization** (3-4 days)
   - MUSIC algorithm implementation
   - TDOA triangulation
   - Kalman filtering
   - Comprehensive tests
   - Documentation

2. **P1-2: Functional Ultrasound GPS** (3-4 days)
   - Affine registration engine
   - Vessel segmentation
   - Brain atlas integration
   - Stereotactic targeting
   - Documentation

**Deliverable**: Both P1 features production-ready, 1600+ tests passing

### Sprint 3 (Week 3): GPU Implementation & Polish

**Duration**: 1 week
**Tasks**:

1. **P1-3: GPU Multiphysics** (3-4 days)
   - GPU memory management
   - CUDA kernel implementations
   - Multi-GPU orchestration
   - Performance validation
   - Documentation

2. **Dead Code Review** (1 day)
   - Audit all 192 dead code markers
   - Document decisions
   - Remove obvious dead code
   - Create removal timeline

3. **Documentation & Release** (1 day)
   - Phase 10 completion report
   - Migration guides
   - API documentation updates
   - v3.1.0 release notes

**Deliverable**: GPU-accelerated multiphysics, zero dead code ambiguity, v3.1.0 ready

---

## PART 7: SUCCESS CRITERIA

### Architecture Quality

- [ ] Zero architecture violations (currently 3/3 fixed)
- [ ] Zero circular dependencies (verify with cargo check)
- [ ] All 9-layer hierarchy maintained
- [ ] Every module has single clear responsibility

### Code Quality

- [ ] All 192 dead code markers reviewed and documented
- [ ] Zero compiler warnings (< 5 allowed)
- [ ] Test pass rate > 99% (1600+ tests)
- [ ] Build time < 15 seconds (dev profile)

### Feature Implementation

- [ ] P1-1 Source Localization: 5+ algorithms implemented
- [ ] P1-2 Functional Ultrasound GPS: End-to-end working
- [ ] P1-3 GPU Multiphysics: 10x+ performance improvement
- [ ] 10+ integration tests per feature

### Performance Benchmarks

- [ ] GPU single-device: 10x speedup vs. CPU
- [ ] GPU dual-device: 18x+ speedup
- [ ] Source localization: < 100 ms for typical array
- [ ] Brain GPS registration: < 1 second

### Documentation

- [ ] Architecture documentation updated
- [ ] All new features documented with examples
- [ ] Migration guides for violated layers
- [ ] v3.1.0 release notes comprehensive

---

## PART 8: RISK ASSESSMENT

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| GPU kernel development delays | MEDIUM | HIGH | Allocate 4-5 days buffer |
| Registration algorithm convergence issues | MEDIUM | MEDIUM | Start with simple affine, extend |
| Multi-GPU synchronization bugs | LOW | HIGH | Extensive testing on available HW |
| Source localization numerical instability | LOW | MEDIUM | Use robust SVD algorithms |

### Schedule Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Phase 2 features take longer | MEDIUM | MEDIUM | Prioritize GPU work |
| GPU API learning curve | MEDIUM | MEDIUM | Documentation review first |
| Test infrastructure changes | LOW | LOW | No expected infrastructure changes |

### Quality Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|-----------|
| Performance not meeting targets | LOW | MEDIUM | Profiling during development |
| Dead code review incomplete | LOW | LOW | Systematic checklist approach |
| Architecture violations reappear | LOW | MEDIUM | Code review process for imports |

---

## PART 9: SUCCESS METRICS

### Phase Completion Checklist

**Architecture** (4 items):
- [ ] Violation #1 resolved (Beamforming interface)
- [ ] Violation #2 resolved (CEUS to physics)
- [ ] Violation #3 resolved (PAM interface)
- [ ] Zero new violations introduced

**Code Quality** (4 items):
- [ ] 192 dead code markers reviewed
- [ ] Deprecated components identified for removal
- [ ] All tests passing (1600+)
- [ ] Zero compiler warnings

**Features** (3 items):
- [ ] P1-1 Source Localization complete
- [ ] P1-2 Functional Ultrasound GPS complete
- [ ] P1-3 GPU Multiphysics complete

**Testing** (3 items):
- [ ] 1650+ total tests passing
- [ ] GPU kernels tested and validated
- [ ] Performance benchmarks achieved

**Documentation** (3 items):
- [ ] Phase 10 completion report
- [ ] v3.1.0 release notes
- [ ] User migration guides

---

## CONCLUSION

Phase 10 builds on Phase 9's architectural cleanup by:

1. **Resolving** the 3 identified architecture violations
2. **Consolidating** domain models for signal processing
3. **Implementing** high-priority research features
4. **Integrating** GPU acceleration for real-time simulation
5. **Establishing** pathways for future enhancements

Success will result in:
- ✅ Production-ready architecture (v3.1.0)
- ✅ Research-level feature parity with k-Wave/j-Wave
- ✅ 10-100x performance improvement via GPU
- ✅ Clear roadmap for future development

**Target Completion**: End of Week 3, January 2026
**Follow-up Phase**: Phase 11 - Advanced Neural Beamforming & ML Integration

---

**Document Version**: 1.0
**Author**: Phase 10 Planning Team
**Date**: January 29, 2026
**Status**: 📋 READY FOR EXECUTION
