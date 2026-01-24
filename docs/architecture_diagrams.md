# Kwavers Architecture Diagrams

**Date:** 2026-01-23  
**Purpose:** Visual representations of current vs. recommended architecture  
**Based On:** Analysis of 11 ultrasound simulation libraries

---

## Current Architecture (Issues Highlighted)

```
┌─────────────────────────────────────────────────────────────┐
│                     CLINICAL LAYER                           │
│                                                              │
│  ┌────────────────────────────────────────┐                 │
│  │ clinical/imaging/workflows/            │                 │
│  │  ├─ advanced_imaging.rs                │                 │
│  │  ├─ neural/ai_beamforming_processor.rs │ ❌ Contains    │
│  │  └─ simulation.rs                      │    algorithms  │
│  └───────────┬────────────────────────────┘                 │
│              │                                               │
│              │ ❌ Direct import: solver::forward::fdtd      │
│              ↓                                               │
└──────────────┼───────────────────────────────────────────────┘
               │
┌──────────────┼───────────────────────────────────────────────┐
│              ↓         SOLVER LAYER                          │
│  ┌─────────────────────────────┐                             │
│  │ solver/forward/fdtd/        │                             │
│  │ solver/forward/pstd/        │                             │
│  │ solver/plugin/executor.rs   │ ✅ Good abstraction         │
│  └─────────────────────────────┘    (underutilized)         │
└──────────────────────────────────────────────────────────────┘
               
┌──────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER                             │
│                                                              │
│  ┌────────────────────────────┐   ┌─────────────────────┐   │
│  │ domain/sensor/             │   │ domain/medium/      │   │
│  │  ├─ mod.rs                 │   │  ├─ heterogeneous/  │   │
│  │  └─ beamforming/  ❌       │   │  └─ homogeneous/    │   │
│  │      ├─ adaptive/          │   └─────────────────────┘   │
│  │      ├─ neural/            │                             │
│  │      └─ beamforming_3d/    │                             │
│  │         (120+ files!)      │                             │
│  └────────────────────────────┘                             │
│         ↑                                                    │
│         │ ❌ DUPLICATION                                     │
│         │                                                    │
└─────────┼──────────────────────────────────────────────────┘
          │
┌─────────┼──────────────────────────────────────────────────┐
│         ↓        ANALYSIS LAYER                             │
│  ┌────────────────────────────────────────┐                 │
│  │ analysis/signal_processing/            │                 │
│  │  └─ beamforming/  ✅                   │                 │
│  │      ├─ adaptive/                      │                 │
│  │      ├─ neural/                        │                 │
│  │      ├─ narrowband/                    │                 │
│  │      └─ three_dimensional/             │                 │
│  │         (Same algorithms as domain!)   │                 │
│  └────────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────┘
```

**Issues:**
1. ❌ Beamforming code duplicated in domain and analysis layers (120+ files)
2. ❌ Clinical workflows directly import specific solvers (tight coupling)
3. ❌ Domain layer contains signal processing algorithms (layering violation)

---

## Recommended Architecture (Industry Best Practices)

```
┌───────────────────────────────────────────────────────────────────┐
│                     CLINICAL LAYER                                 │
│                                                                    │
│  ┌──────────────────────────────────────────────────────┐         │
│  │ clinical/imaging/workflows/orchestrator.rs           │         │
│  │                                                      │         │
│  │  fn run_workflow() {                                │         │
│  │    // Step 1: Domain preparation (clinical)         │         │
│  │    let domain = self.prepare_domain()?;             │         │
│  │                                                      │         │
│  │    // Step 2: Acoustic calculation (abstraction)    │         │
│  │    let executor = PluginExecutor::new(config)?; ────┼─────┐   │
│  │    let pressure = executor.execute(&domain)?;       │     │   │
│  │                                                      │     │   │
│  │    // Step 3: Beamforming (analysis layer) ─────────┼───┐ │   │
│  │    let image = beamforming::beamform(               │   │ │   │
│  │        &pressure, &config                           │   │ │   │
│  │    )?;                                               │   │ │   │
│  │  }                                                   │   │ │   │
│  └──────────────────────────────────────────────────────┘   │ │   │
│         ✅ Orchestrates, doesn't implement                  │ │   │
└─────────────────────────────────────────────────────────────┼─┼───┘
                                                              │ │
┌─────────────────────────────────────────────────────────────┼─┼───┐
│                     SOLVER LAYER                            │ │   │
│                                                              ↓ │   │
│  ┌────────────────────────────────────────────────┐          │   │
│  │ solver/plugin/executor.rs (ABSTRACTION)        │          │   │
│  │                                                │          │   │
│  │  impl PluginExecutor {                        │          │   │
│  │    fn execute(&self, domain: &Domain) {       │          │   │
│  │      match self.config.solver_type {          │          │   │
│  │        SolverType::FDTD => fdtd::solve(...),  │          │   │
│  │        SolverType::PSTD => pstd::solve(...),  │          │   │
│  │        SolverType::BEM => bem::solve(...),    │          │   │
│  │      }                                         │          │   │
│  │    }                                           │          │   │
│  │  }                                             │          │   │
│  └────────────────────────────────────────────────┘          │   │
│         ↓                                                    │   │
│  ┌────────────────────────────────────────────────┐          │   │
│  │ solver/forward/                                │          │   │
│  │  ├─ fdtd/ (FDTD implementation)               │          │   │
│  │  ├─ pstd/ (PSTD implementation)               │          │   │
│  │  └─ bem/  (BEM implementation)                │          │   │
│  └────────────────────────────────────────────────┘          │   │
│         ✅ Solver selection via configuration               │   │
└─────────────────────────────────────────────────────────────┼───┘
                                                              │
┌─────────────────────────────────────────────────────────────┼───┐
│                     DOMAIN LAYER                            │   │
│                                                              │   │
│  ┌────────────────────────────┐   ┌─────────────────────┐   │   │
│  │ domain/sensor/             │   │ domain/medium/      │   │   │
│  │  ├─ mod.rs                 │   │  ├─ heterogeneous/  │   │   │
│  │  └─ geometry.rs ✅         │   │  ├─ tissue_db.rs    │   │   │
│  │     (Exports only)         │   │  └─ builder.rs      │   │   │
│  └────────────────────────────┘   └─────────────────────┘   │   │
│                                                              │   │
│  pub struct SensorArray {                                   │   │
│    pub fn geometry() -> BeamformingGeometry { ... }         │   │
│  }                                                           │   │
│  ✅ Geometry only, NO algorithms                            │   │
└─────────────────────────────────────────────────────────────┼───┘
                                                              │
┌─────────────────────────────────────────────────────────────┼───┐
│                     ANALYSIS LAYER                          ↓   │
│                                                                  │
│  ┌─────────────────────────────────────────────────────┐        │
│  │ analysis/signal_processing/beamforming/             │        │
│  │  ├─ mod.rs (ALL beamforming algorithms)            │        │
│  │  ├─ adaptive/                                       │        │
│  │  │   ├─ mvdr.rs                                     │        │
│  │  │   ├─ music.rs                                    │        │
│  │  │   └─ capon.rs                                    │        │
│  │  ├─ neural/                                         │        │
│  │  │   ├─ beamformer.rs                               │        │
│  │  │   └─ pinn/                                       │        │
│  │  ├─ narrowband/                                     │        │
│  │  └─ three_dimensional/                              │        │
│  │      ├─ das.rs                                      │        │
│  │      ├─ saft.rs                                     │        │
│  │      └─ coherence.rs                                │        │
│  │                                                      │        │
│  │  pub fn beamform(                                   │        │
│  │    rf_data: &Array3<f32>,                           │        │
│  │    geometry: &BeamformingGeometry,                  │        │
│  │    algorithm: BeamformingAlgorithm,                 │        │
│  │  ) -> Result<Array3<f32>>                           │        │
│  └─────────────────────────────────────────────────────┘        │
│  ✅ SINGLE SOURCE OF TRUTH for all beamforming                  │
└──────────────────────────────────────────────────────────────────┘
```

**Improvements:**
1. ✅ Beamforming consolidated in analysis layer (SSOT)
2. ✅ Clinical workflows use `PluginExecutor` abstraction (loose coupling)
3. ✅ Domain layer exports geometry only (proper layering)
4. ✅ Step-based workflow pattern (BabelBrain model)

---

## Dependency Graph: Current vs. Recommended

### Current (With Violations)

```
    Clinical ──────┐
       │           │
       │ Direct    │ ❌ Should use
       │ import    │    abstraction
       ↓           ↓
    Solver ←── Analysis
       ↓           ↑
    Domain ────────┘
       │           ↑
       │ ❌ Beamforming
       │    duplication
       ↓
    Physics
       ↓
     Math
```

### Recommended (Clean Architecture)

```
    Clinical ──────────────┐
       │                   │
       │ Via               │ Via
       │ PluginExecutor    │ beamforming::
       ↓                   ↓
    Solver            Analysis
       ↓                   ↑
    Domain ────────────────┘
       │                   │
       │ Geometry          │ Data only
       │ export            │
       ↓                   │
    Physics ←──────────────┘
       ↓
     Math

✅ No cycles, clean downward dependencies
```

---

## Data Flow: Beamforming (Recommended)

### Current (Problematic)

```
Sensor Acquisition
       │
       ├─→ domain/sensor/beamforming/  ❌ Coupled
       │      │
       │      └─→ Beamformed Image
       │
       └─→ analysis/signal_processing/beamforming/  ✅ Correct location
              │
              └─→ Beamformed Image

Problem: Same data, two paths, duplicated code
```

### Recommended (Single Path)

```
┌─────────────────┐
│ Sensor          │
│ Acquisition     │
│                 │
│ - RF data       │
│ - Geometry      │
│ - Timestamps    │
└────────┬────────┘
         │ Exports RawAcquisition
         ↓
┌─────────────────────────────────────┐
│ analysis/signal_processing/         │
│   beamforming/                      │
│                                     │
│ pub fn beamform(                    │
│   acquisition: &RawAcquisition,     │
│   algorithm: BeamformingAlgorithm,  │
│ ) -> Result<BeamformedImage>        │
│                                     │
│ match algorithm {                   │
│   DAS => delay_and_sum(...),        │
│   MVDR => mvdr_beamformer(...),     │
│   Neural => neural_network(...),    │
│ }                                   │
└────────┬────────────────────────────┘
         │
         ↓
┌─────────────────┐
│ Beamformed      │
│ Image           │
│                 │
│ - Intensity map │
│ - Metadata      │
└─────────────────┘

✅ Single source of truth, all algorithms in one place
```

---

## Clinical Workflow: BabelBrain Pattern

### Industry Best Practice (BabelBrain)

```
┌────────────────────────────────────────────────────┐
│ Step 1: Domain Preparation (Clinical Layer)       │
│                                                    │
│  - Load MRI/CT imaging data                       │
│  - Segment tissues (brain, skull, skin)           │
│  - Register modalities (Elastix)                  │
│  - Generate computational mesh (trimesh)          │
│  - Export domain → Nifti files                    │
└───────────────────┬────────────────────────────────┘
                    │ Data handoff
                    ↓
┌────────────────────────────────────────────────────┐
│ Step 2: Acoustic Calculation (Physics Backend)    │
│                                                    │
│  - Load domain from Step 1                        │
│  - Configure transducer (source)                  │
│  - Run FDTD solver (BabelViscoFDTD)              │
│  - Export pressure fields → HDF5                  │
└───────────────────┬────────────────────────────────┘
                    │ Data handoff
                    ↓
┌────────────────────────────────────────────────────┐
│ Step 3: Thermal Analysis (Thermal Backend)        │
│                                                    │
│  - Load pressure from Step 2                      │
│  - Compute acoustic intensity                     │
│  - Solve bioheat transfer equation (BHTE)         │
│  - Calculate thermal dose                         │
│  - Export results → Nifti (for Brainsight)        │
└────────────────────────────────────────────────────┘
```

### Kwavers Adaptation

```
┌────────────────────────────────────────────────────┐
│ clinical/imaging/workflows/comprehensive.rs        │
│                                                    │
│ impl ComprehensiveWorkflow {                      │
│   fn execute(&self) -> Result<ClinicalResult> {   │
│                                                    │
│     // Step 1: Domain Preparation                 │
│     let domain = self.prepare_domain()?;          │
│     // - Load phantom or patient data             │
│     // - Configure grid and medium                │
│     // - Set up sources and sensors               │
│                                                    │
│     // Step 2: Acoustic Simulation                │
│     let executor = PluginExecutor::new(            │
│       self.solver_config                          │
│     )?;                                            │
│     let acquisition = executor.execute_with_sensors(
│       &domain,                                     │
│       &self.sensors                                │
│     )?;                                            │
│                                                    │
│     // Step 3: Image Reconstruction                │
│     let image = analysis::signal_processing::      │
│       beamforming::beamform(                       │
│         &acquisition.rf_data,                      │
│         &acquisition.geometry,                     │
│         self.beamforming_algorithm,                │
│       )?;                                          │
│                                                    │
│     // Step 4: Diagnosis (if neural workflow)      │
│     let diagnosis = if self.enable_ai {            │
│       analysis::ml::diagnosis::diagnose(&image)?   │
│     } else { None };                               │
│                                                    │
│     Ok(ClinicalResult {                           │
│       image,                                       │
│       diagnosis,                                   │
│       metadata: self.collect_metadata(),          │
│     })                                             │
│   }                                                │
│ }                                                  │
└────────────────────────────────────────────────────┘

✅ Clean separation: orchestration vs. implementation
✅ Each step delegates to appropriate layer
✅ Testable: can mock executor and beamforming
```

---

## Module Dependency Rules (Enforced by Lints)

### Allowed Dependencies (Green)

```
                Clinical
                   │
         ┌─────────┼─────────┐
         ↓         ↓         ↓
      Solver   Analysis   Domain
         │         │         │
         └────┬────┴────┬────┘
              ↓         ↓
           Physics   Domain
              │         │
              └────┬────┘
                   ↓
                 Math
                 
Infra ──→ All (cross-cutting: logging, I/O, API)

✅ Downward dependencies only (no cycles)
```

### Forbidden Dependencies (Red)

```
❌ Domain ──X→ Solver
❌ Domain ──X→ Analysis
❌ Domain ──X→ Clinical

❌ Physics ─X→ Domain
❌ Physics ─X→ Solver

❌ Math ────X→ Physics
❌ Math ────X→ Domain

❌ Solver ──X→ Clinical
❌ Solver ──X→ Analysis

❌ Analysis ─X→ Clinical
```

### Implementation (CI Check)

```rust
// src/architecture.rs
const DEPENDENCY_RULES: &[(&str, &[&str])] = &[
    ("math", &[]),  // Math depends on nothing
    ("physics", &["math"]),
    ("domain", &["physics", "math"]),
    ("solver", &["physics", "domain", "math"]),
    ("analysis", &["domain", "physics", "math"]),
    ("clinical", &["solver", "analysis", "domain", "physics", "math"]),
    ("infra", &["*"]),  // Infra can use anything (cross-cutting)
];

#[test]
fn enforce_dependency_rules() {
    for (module, allowed_deps) in DEPENDENCY_RULES {
        let violations = check_imports(module, allowed_deps);
        assert!(violations.is_empty(), 
            "Module {} has forbidden dependencies: {:?}", 
            module, violations
        );
    }
}
```

---

## GPU Backend Architecture (Already Good)

### Current Implementation (wgpu)

```
┌─────────────────────────────────────┐
│ Application Layer                   │
│  - FdtdSolver::new(config)          │
│  - solver.step()                    │
└────────────┬────────────────────────┘
             │
             ↓
┌─────────────────────────────────────┐
│ GPU Executor (src/gpu/executor.rs)  │
│                                     │
│  impl GpuExecutor {                 │
│    fn dispatch_kernel(              │
│      kernel: ComputeKernel,         │
│      data: &GpuBuffer                │
│    )                                 │
│  }                                   │
└────────────┬────────────────────────┘
             │ wgpu API
             ↓
┌─────────────────────────────────────┐
│ wgpu Multi-Backend Abstraction      │
│                                     │
│  ┌─────────┐  ┌────────┐  ┌──────┐ │
│  │ Vulkan  │  │ Metal  │  │ DX12 │ │
│  └─────────┘  └────────┘  └──────┘ │
│                                     │
│  Runtime selection based on:        │
│  - Platform (Linux/macOS/Windows)   │
│  - Hardware (NVIDIA/AMD/Apple)      │
│  - User preference (config)         │
└─────────────────────────────────────┘

✅ Already matches BabelBrain's multi-backend pattern
✅ No changes needed, just expose configuration
```

### Recommended Enhancement

```rust
// src/gpu/config.rs
pub enum PreferredBackend {
    Auto,       // wgpu selects best (default)
    Vulkan,     // Force Vulkan
    Metal,      // Force Metal (macOS)
    Dx12,       // Force DirectX 12 (Windows)
    OpenGL,     // Fallback (compatibility)
}

// src/simulation/configuration.rs
pub struct Configuration {
    // ... existing fields ...
    
    #[serde(default)]
    pub gpu_backend: PreferredBackend,
}

// Usage in examples
let config = Configuration {
    gpu_backend: PreferredBackend::Vulkan,  // For benchmarking
    // ...
};
```

---

## Beamforming Algorithm Organization

### Recommended Structure

```
analysis/signal_processing/beamforming/
├── mod.rs                  # Public API: beamform()
│
├── traits.rs               # Beamformer trait
│   pub trait Beamformer {
│     fn beamform(&self, data: &RawData) -> Result<Image>;
│   }
│
├── time_domain/            # Time-domain beamformers
│   ├── mod.rs
│   ├── das.rs              # Delay-and-sum
│   ├── dmas.rs             # Delay-multiply-and-sum
│   └── saft.rs             # Synthetic aperture focusing
│
├── adaptive/               # Adaptive beamformers
│   ├── mod.rs
│   ├── mvdr.rs             # Minimum variance distortionless response
│   ├── music.rs            # Multiple signal classification
│   ├── capon.rs            # Capon beamformer
│   └── eig_space.rs        # Eigenspace-based methods
│
├── neural/                 # Neural beamformers
│   ├── mod.rs
│   ├── beamformer.rs       # Neural network beamformer
│   ├── pinn/               # Physics-informed neural networks
│   │   ├── inference.rs    # PINN inference
│   │   └── processor.rs    # PINN processing
│   └── uncertainty.rs      # Uncertainty quantification
│
├── narrowband/             # Narrowband beamformers
│   ├── mod.rs
│   ├── steering.rs         # Steering vectors
│   └── snapshots/          # Snapshot processing
│
├── coherence/              # Coherence-based methods
│   ├── mod.rs
│   ├── slsc.rs             # Short-lag spatial coherence
│   └── gcc_phat.rs         # Generalized cross-correlation
│
└── utils/                  # Shared utilities
    ├── apodization.rs      # Window functions
    ├── delay_calc.rs       # Time-of-flight calculations
    └── interpolation.rs    # Sub-sample interpolation

✅ All beamforming code in ONE module tree
✅ Clear categorization by algorithm type
✅ Shared utilities prevent duplication
```

### Public API Design

```rust
// analysis/signal_processing/beamforming/mod.rs
pub enum BeamformingAlgorithm {
    // Time-domain
    DAS,
    DMAS,
    SAFT,
    
    // Adaptive
    MVDR { diagonal_loading: f64 },
    MUSIC { num_sources: usize },
    Capon,
    
    // Neural
    NeuralNetwork { model_path: PathBuf },
    PINN { physics_weight: f64 },
    
    // Coherence
    SLSC { lag_range: (usize, usize) },
}

pub struct BeamformingConfig {
    pub algorithm: BeamformingAlgorithm,
    pub apodization: WindowType,
    pub interpolation: InterpolationMethod,
    pub speed_of_sound: f64,
}

pub fn beamform(
    rf_data: &Array3<f32>,          // [channels, samples, frames]
    geometry: &BeamformingGeometry,  // Sensor positions
    config: &BeamformingConfig,
) -> KwaversResult<Array3<f32>> {   // [x, y, z] image
    match config.algorithm {
        BeamformingAlgorithm::DAS => {
            time_domain::das::beamform(rf_data, geometry, config)
        }
        BeamformingAlgorithm::MVDR { diagonal_loading } => {
            adaptive::mvdr::beamform(rf_data, geometry, diagonal_loading)
        }
        // ... dispatch to appropriate implementation
    }
}
```

---

## Tissue Database Schema

### Recommended Data Model

```rust
// domain/medium/tissue_database.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TissueProperties {
    // Identification
    pub id: String,              // "brain_grey_matter"
    pub name: String,            // "Brain (Grey Matter)"
    pub category: TissueCategory, // Organ::Brain
    
    // Acoustic properties
    pub sound_speed: f64,        // m/s (1540.0)
    pub density: f64,            // kg/m³ (1045.0)
    pub attenuation: f64,        // dB/cm/MHz (0.6)
    pub attenuation_exponent: f64, // Power law exponent (1.1)
    pub nonlinearity: f64,       // B/A ratio (6.5)
    
    // Thermal properties
    pub thermal_conductivity: f64,  // W/(m·K)
    pub specific_heat: f64,         // J/(kg·K)
    pub perfusion: f64,             // kg/(m³·s)
    
    // Metadata
    pub source: DataSource,      // ITISFoundation, Literature
    pub reference: Option<String>, // DOI or citation
    pub temperature: f64,        // °C (measurement condition)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TissueCategory {
    Organ(OrganType),
    Tissue(TissueType),
    Fluid(FluidType),
    Bone,
}

pub struct TissueDatabase {
    tissues: HashMap<String, TissueProperties>,
}

impl TissueDatabase {
    pub fn load_itis_foundation() -> KwaversResult<Self> {
        // Load from embedded JSON (IT'IS Foundation data)
        let json = include_str!("../../data/itis_tissues.json");
        serde_json::from_str(json)
    }
    
    pub fn get(&self, id: &str) -> Option<&TissueProperties> {
        self.tissues.get(id)
    }
    
    pub fn search(&self, category: TissueCategory) -> Vec<&TissueProperties> {
        self.tissues.values()
            .filter(|t| t.category == category)
            .collect()
    }
}

// Usage
let db = TissueDatabase::load_itis_foundation()?;
let brain = db.get("brain_grey_matter").unwrap();
let medium = Medium::from_tissue(brain, &grid)?;
```

### Data File Format

```json
{
  "version": "4.1",
  "source": "IT'IS Foundation",
  "url": "https://itis.swiss/virtual-population/tissue-properties/",
  "tissues": [
    {
      "id": "brain_grey_matter",
      "name": "Brain (Grey Matter)",
      "category": "organ_brain",
      "sound_speed": 1540.0,
      "density": 1045.0,
      "attenuation": 0.6,
      "attenuation_exponent": 1.1,
      "nonlinearity": 6.5,
      "thermal_conductivity": 0.565,
      "specific_heat": 3630.0,
      "perfusion": 8.0,
      "source": "itis_foundation",
      "reference": "https://itis.swiss/virtual-population/tissue-properties/database/acoustic-properties/",
      "temperature": 37.0
    }
  ]
}
```

---

## Summary

This architecture follows industry best practices from:
- **BabelBrain:** Step-based workflows, backend delegation
- **j-Wave / k-Wave:** Component composition, clean separation
- **Fullwave:** Python API + optimized backend
- **DBUA:** Physics-guided learning

Key improvements:
1. ✅ Beamforming consolidated (120+ files → analysis layer only)
2. ✅ Clinical workflows orchestrate via abstractions
3. ✅ Module boundaries enforced (CI lints)
4. ✅ Tissue database (standardized properties)
5. ✅ GPU backend already optimal (wgpu)

**Next:** Implement in Sprints 213-217 per action plan.
