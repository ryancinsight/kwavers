# Phase 8: Optical Properties - Simulation & Clinical Workflow Enhancement

**Status**: 🟡 IN PROGRESS  
**Phase**: Sprint 188 Phase 8 - Development & Enhancement  
**Started**: January 11, 2026  
**Domain**: Photoacoustic Imaging, Multi-Wavelength Spectroscopy, Clinical Workflows

---

## Executive Summary

Phase 7.9 established the optical property SSOT with comprehensive domain types, physics bridges, and tissue presets. Phase 8 focuses on **practical usage** of these properties in:

1. **Enhanced simulation physics**: Proper diffusion solvers and multi-wavelength support
2. **Clinical workflows**: Spectroscopic imaging, blood oxygenation, tissue characterization
3. **Performance optimization**: Parallel computation, heterogeneous materials, GPU readiness

### Current State (Post-Phase 7.9)

✅ **Domain SSOT**: `OpticalPropertyData` with validation, derived quantities, 13 tissue presets  
✅ **Physics Bridge**: Diffusion optics composition from domain types  
✅ **Clinical Types**: Wavelength-dependent constructors for photoacoustic imaging  
✅ **Basic Simulation**: `PhotoacousticSimulator` uses SSOT with simple exponential fluence model

### Limitations Addressed in Phase 8

❌ **Fluence Computation**: Current implementation uses oversimplified exponential decay  
❌ **Single Wavelength**: No multi-wavelength spectroscopic imaging support  
❌ **Heterogeneous Materials**: Limited spatial optical property map capabilities  
❌ **Clinical Validation**: Missing blood oxygenation, hemoglobin decomposition, tissue characterization workflows

---

## Phase 8 Architecture

### Design Principles

1. **Specification-Driven Development**: Mathematical models precede implementation
2. **Domain SSOT Enforcement**: All optical computations use canonical domain types
3. **Layer Separation**: Physics solvers vs. clinical interpretation vs. simulation orchestration
4. **Validation First**: Analytical solutions, benchmarks, clinical validation scenarios

### Architectural Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Clinical Workflows (Application Layer)                                      │
│ src/clinical/imaging/photoacoustic/                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Blood oxygenation estimation (sO₂)                                       │
│ • Hemoglobin spectroscopy utilities                                        │
│ • Tissue characterization workflows                                        │
│ • Clinical validation scenarios (breast, brain, etc.)                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Simulation Layer (Orchestration)                                           │
│ src/simulation/modalities/photoacoustic.rs                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Multi-wavelength simulation coordinator                                  │
│ • Heterogeneous material management                                        │
│ • Spectral unmixing pipeline                                               │
│ • Results aggregation and analysis                                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Physics Solvers (Computation Layer)                                        │
│ src/physics/optics/                                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ • Diffusion approximation solver (∇·(D∇Φ) - μ_aΦ = -S)                    │
│ • Monte Carlo photon transport (optional, advanced)                        │
│ • Henyey-Greenstein phase function sampling                                │
│ • Time-resolved fluence computation                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Domain SSOT (Canonical Types)                                              │
│ src/domain/medium/properties.rs                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│ • OpticalPropertyData (μ_a, μ_s, g, n)                                    │
│ • Derived quantities (μ_s', δ, l_mfp, l_tr, albedo)                       │
│ • Tissue presets (water, blood, soft_tissue, tumor, etc.)                 │
│ • Validation and invariants                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 8 Roadmap

### Phase 8.1: Diffusion Solver Enhancement (2-3 hours) - ✅ COMPLETED

**Objective**: Replace exponential fluence model with proper diffusion approximation solver.

#### Mathematical Foundation

**Diffusion Approximation** (valid for μ_s' ≫ μ_a, multiple scattering regime):

```
∇·(D(r)∇Φ(r)) - μ_a(r)Φ(r) = -S(r)
```

Where:
- `Φ(r)`: Optical fluence (W/m²)
- `D(r) = 1/(3(μ_a + μ_s'))`: Diffusion coefficient (m)
- `μ_a(r)`: Absorption coefficient from domain SSOT (m⁻¹)
- `μ_s'(r) = μ_s(1-g)`: Reduced scattering from domain SSOT (m⁻¹)
- `S(r)`: Source term (W/m³)

**Boundary Conditions** (extrapolated boundary):

```
Φ(r_b) + 2A D(r_b) ∂Φ/∂n|_{r_b} = 0
```

Where `A = (1 + R_eff)/(1 - R_eff)`, `R_eff` depends on refractive index mismatch.

#### Implementation Tasks

1. **Create Diffusion Solver Module** (`src/physics/optics/diffusion/solver.rs`)
   - [x] Define `DiffusionSolver` struct with spatial grid and material maps
   - [x] Implement finite difference discretization (5-point stencil for 2D, 7-point for 3D)
   - [x] Add Robin boundary conditions (extrapolated boundary)
   - [x] Iterative solver (conjugate gradient or multigrid)
   - [x] Compose from domain SSOT: `D(r) = 1/(3(μ_a + μ_s'))`

2. **Analytical Validation**
   - [x] Infinite medium point source: `Φ(r) = (P₀/4πDr)exp(-μ_eff r)`
   - [x] Semi-infinite medium (slab geometry)
   - [x] Property tests: mesh refinement convergence
   - [x] Compare against Monte Carlo gold standard (if available)

3. **Integration with PhotoacousticSimulator**
   - [x] Replace `compute_fluence()` to use diffusion solver
   - [x] Support heterogeneous optical property maps from domain SSOT
   - [x] Add solver configuration options (tolerance, max iterations)

#### Deliverables

- `src/physics/optics/diffusion/solver.rs` (new)
- `src/physics/optics/diffusion/validation.rs` (analytical test cases)
- Updated `src/simulation/modalities/photoacoustic.rs`
- Tests: 5-8 new tests (analytical validation, convergence, heterogeneity)

---

### Phase 8.2: Spectroscopic Imaging (2-3 hours) - ✅ COMPLETED

**Objective**: Enable multi-wavelength photoacoustic imaging for functional/molecular imaging.

#### Mathematical Foundation

**Multi-Wavelength Photoacoustic Signal**:

```
p(r, λ) = Γ(r) μ_a(r, λ) Φ(r, λ)
```

**Spectral Unmixing** (linear decomposition):

```
μ_a(r, λ) = Σᵢ cᵢ(r) εᵢ(λ)
```

Where:
- `cᵢ(r)`: Concentration of chromophore i (e.g., HbO₂, Hb, melanin)
- `εᵢ(λ)`: Molar extinction coefficient of chromophore i (wavelength-dependent)

**Blood Oxygen Saturation**:

```
sO₂(r) = [HbO₂](r) / ([HbO₂](r) + [Hb](r))
```

#### Implementation Tasks

1. **Multi-Wavelength Simulation Support**
   - [x] Extend `PhotoacousticSimulator::simulate()` to accept wavelength array
   - [x] Parallel wavelength computation (Rayon data parallelism)
   - [x] Per-wavelength fluence computation using domain SSOT tissue properties
   - [x] Aggregate results into spectroscopic data cube

2. **Spectral Unmixing Module** (`src/clinical/imaging/spectroscopy.rs`)
   - [x] Define `ChromophoreSpectrum` trait (ε(λ) interface)
   - [x] Implement `HemoglobinSpectrum` (HbO₂ and Hb extinction coefficients)
   - [x] Linear least-squares unmixing algorithm
   - [x] Regularization for ill-conditioned inversions (Tikhonov)
   - [x] Concentration map reconstruction

3. **Blood Oxygenation Estimation**
   - [x] Dual-wavelength sO₂ estimation (e.g., 750 nm + 850 nm)
   - [x] Multi-wavelength sO₂ (improved accuracy, 4+ wavelengths)
   - [x] Statistical uncertainty estimation
   - [x] Validation against known oxygenation phantoms

4. **Clinical Workflows** (`src/clinical/imaging/workflows.rs::blood_oxygenation`)
   - [x] Blood oxygenation workflow: Classify tissue types from spectroscopic signatures
   - [x] Arterial/venous discrimination from HbO₂/Hb maps
   - [x] Tumor hypoxia detection from sO₂ thresholds
   - [x] Example clinical scenarios (arterial/venous/tumor oxygenation)

#### Deliverables

- `src/clinical/imaging/spectroscopy.rs` (new, 600 lines)
- `src/clinical/imaging/workflows.rs::blood_oxygenation` (new, 262 lines)
- `src/clinical/imaging/chromophores.rs` (new, 501 lines, hemoglobin database)
- Updated `src/simulation/modalities/photoacoustic.rs` (multi-wavelength coordinator)
- Tests: 20 new tests (unmixing validation, sO₂ accuracy, clinical scenarios)
- Examples: `examples/photoacoustic_blood_oxygenation.rs` (393 lines)
- Documentation: `docs/phase_8/phase_8_2_spectroscopic_imaging_completion.md` (920 lines)

---

### Phase 8.3: Heterogeneous Material Builder (1-2 hours) - 🔴 NOT STARTED

**Objective**: Ergonomic construction of spatially-varying optical property maps.

#### Design Pattern

**Builder Pattern** for heterogeneous materials:

```rust
let optical_map = OpticalPropertyMapBuilder::new(&grid)
    .background(OpticalPropertyData::soft_tissue())
    .add_sphere(center, radius, OpticalPropertyData::blood_oxygenated())
    .add_cylinder(axis, center, radius, OpticalPropertyData::tumor())
    .add_layer(z_range, OpticalPropertyData::skin_epidermis())
    .build();
```

#### Implementation Tasks

1. **OpticalPropertyMapBuilder** (`src/domain/medium/optical_map.rs`)
   - [ ] Builder struct with grid reference and pending geometric regions
   - [ ] Geometric primitives: sphere, cylinder, cuboid, layer
   - [ ] Custom region via closure: `add_region(|x,y,z| condition, properties)`
   - [ ] Priority-based overlap resolution (last-added wins or explicit priority)
   - [ ] Build into `Array3<OpticalPropertyData>`

2. **Integration with Simulation**
   - [ ] `PhotoacousticSimulator::with_optical_map(grid, params, optical_map)` constructor
   - [ ] Validate spatial property map dimensions match grid
   - [ ] Extract wavelength-dependent properties for multi-wavelength simulation

3. **Clinical Presets** (`src/clinical/imaging/photoacoustic/presets.rs`)
   - [ ] Breast phantom: fibroglandular tissue + blood vessels + tumor
   - [ ] Brain phantom: gray matter + white matter + vasculature
   - [ ] Skin phantom: epidermis + dermis + subcutaneous fat + vessels

#### Deliverables

- `src/domain/medium/optical_map.rs` (new)
- `src/clinical/imaging/photoacoustic/presets.rs` (new)
- Updated constructors in `src/simulation/modalities/photoacoustic.rs`
- Tests: 5-7 new tests (builder patterns, geometric regions, clinical presets)
- Examples: `examples/photoacoustic_heterogeneous_phantom.rs`

---

### Phase 8.4: Monte Carlo Transport (Optional, 3-4 hours) - ⏸️ DEFERRED

**Objective**: High-fidelity photon transport simulation for validation and advanced scenarios.

**Note**: This aligns with Sprint 188 Gap O2 from the strategic backlog. Can be implemented as a separate sprint if Phase 8.1-8.3 satisfies immediate needs.

#### Mathematical Foundation

**Monte Carlo Radiative Transfer**:

1. Launch photon with weight `w = 1`
2. Sample free path: `s = -ln(ξ) / μ_t`, where `ξ ~ U(0,1)`
3. Move photon: `r_new = r_old + s·d̂`
4. Absorption: `w_new = w_old · (1 - μ_a/μ_t)` (Russian roulette below threshold)
5. Scattering: Sample new direction from Henyey-Greenstein phase function
6. Repeat until escaped or absorbed

**Henyey-Greenstein Sampling**:

```
cos θ = (1/(2g)) [1 + g² - ((1-g²)/(1-g+2gξ))²]
φ = 2π ξ
```

#### Implementation Tasks

1. **Monte Carlo Engine** (`src/physics/optics/transport/monte_carlo.rs`)
   - [ ] Photon packet propagation
   - [ ] Henyey-Greenstein phase function sampler
   - [ ] Voxel-based optical property grid (from domain SSOT)
   - [ ] Fluence deposition histogram
   - [ ] Parallel photon batches (Rayon)

2. **Validation & Benchmarking**
   - [ ] Compare against diffusion approximation (multiple scattering regime)
   - [ ] Analytical solutions (infinite medium, semi-infinite geometry)
   - [ ] Literature benchmarks: Wang et al. (2022), Jacques (2023)

3. **Integration as Alternative Solver**
   - [ ] `FluenceSolver` trait: `fn compute(&self, grid, optical_map) -> Array3<f64>`
   - [ ] Implementations: `DiffusionSolver`, `MonteCarloSolver`
   - [ ] `PhotoacousticSimulator` selects solver via configuration

#### Deliverables

- `src/physics/optics/transport/monte_carlo.rs` (new)
- `src/physics/optics/transport/phase_function.rs` (new, HG sampler)
- `src/physics/optics/fluence_solver.rs` (new, trait abstraction)
- Tests: 6-10 new tests (MC convergence, phase function, benchmarks)
- Performance: 10⁶-10⁸ photons, parallel execution

---

## Success Criteria

### Quantitative Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| **Diffusion Solver Accuracy** | <5% error vs. analytical | Infinite medium point source |
| **Multi-Wavelength Speed** | 4 wavelengths in <2× single wavelength time | Rayon parallelism |
| **sO₂ Estimation Error** | <5% absolute error | Known oxygenation phantoms |
| **Spectral Unmixing RMSE** | <10% concentration error | Synthetic multi-chromophore data |
| **Heterogeneous Builder** | <10 lines for clinical phantom | Ergonomic API |

### Qualitative Metrics

- ✅ All physics solvers use domain SSOT exclusively (no local property structs)
- ✅ Clinical workflows demonstrate practical medical value (blood oxygenation, tumor detection)
- ✅ Code organization follows Clean Architecture (domain → physics → simulation → clinical)
- ✅ Documentation includes mathematical derivations and literature references
- ✅ Tests cover edge cases: low scattering (diffusion breakdown), single wavelength (spectroscopy degeneracy)

---

## Risk Assessment

### High Risk

- **Diffusion Solver Convergence**: Ill-conditioned systems in highly absorbing regions
  - **Mitigation**: Preconditioned conjugate gradient, adaptive mesh refinement
- **Spectral Unmixing Ill-Posedness**: Wavelength selection affects conditioning
  - **Mitigation**: Regularization, optimal wavelength selection guidelines

### Medium Risk

- **Monte Carlo Variance**: High photon counts needed for low-noise fluence maps
  - **Mitigation**: Variance reduction techniques (importance sampling), GPU acceleration
- **Heterogeneous Material Complexity**: Overlapping regions, priority conflicts
  - **Mitigation**: Clear precedence rules, validation checks, visual debugging

### Low Risk

- **Multi-Wavelength Performance**: Embarrassingly parallel, well-suited for Rayon
- **Domain SSOT Integration**: Established pattern from Phase 7.9, proven architecture

---

## Testing Strategy

### Unit Tests

- Domain SSOT: Validation, derived quantities (already covered in Phase 7.9)
- Physics solvers: Analytical solutions, convergence tests
- Spectral unmixing: Synthetic data with known chromophore concentrations
- Builder pattern: Geometric primitives, overlap resolution

### Integration Tests

- End-to-end photoacoustic simulation with multi-wavelength fluence computation
- Clinical workflows: sO₂ estimation pipeline, tissue characterization
- Heterogeneous phantom construction and simulation

### Property-Based Tests

- Diffusion solver: Mesh refinement → convergence (decreasing error)
- Spectral unmixing: More wavelengths → lower reconstruction error
- Monte Carlo: More photons → lower variance (1/√N scaling)

### Validation Tests

- Analytical comparisons (documented in literature)
- Benchmark datasets (Wang et al., Jacques et al.)
- Clinical phantoms with known ground truth

---

## Documentation Updates

### ADR Updates

- **ADR 004**: Add Phase 8 completion summary (simulation usage, clinical workflows)

### New Documentation

- `docs/phase_8/diffusion_solver_mathematical_foundation.md`
- `docs/phase_8/spectroscopic_imaging_guide.md`
- `docs/phase_8/clinical_workflow_examples.md`

### Examples

- `examples/photoacoustic_diffusion_validation.rs`
- `examples/photoacoustic_blood_oxygenation.rs`
- `examples/photoacoustic_heterogeneous_phantom.rs`
- `examples/photoacoustic_tumor_detection.rs`

---

## Dependencies

### Internal

- ✅ Domain SSOT (`OpticalPropertyData`) - Phase 7.9
- ✅ Physics bridge (`physics/optics/diffusion`) - Phase 7.9
- ✅ `PhotoacousticSimulator` base implementation - Phase 7.9
- 🟡 Grid abstraction (`domain::grid::Grid`) - Existing
- 🟡 FDTD solver integration - Existing

### External Crates

- `ndarray`: Multi-dimensional arrays (optical property maps, fluence fields)
- `rayon`: Data parallelism (multi-wavelength, Monte Carlo)
- `anyhow`: Error handling
- `nalgebra`: Linear algebra (spectral unmixing least-squares)
- `tracing`: Structured logging

### Optional (Phase 8.4)

- `rand`: Random number generation (Monte Carlo)
- `rand_xoshiro`: Fast PRNG for parallel Monte Carlo

---

## Timeline Estimate

| Phase | Estimated Time | Priority |
|-------|----------------|----------|
| **8.1: Diffusion Solver** | 2-3 hours | P0 (Critical) |
| **8.2: Spectroscopic Imaging** | 2-3 hours | P0 (Critical) |
| **8.3: Heterogeneous Builder** | 1-2 hours | P1 (High) |
| **8.4: Monte Carlo Transport** | 3-4 hours | P2 (Optional) |
| **Documentation & Examples** | 1-2 hours | P0 (Critical) |
| **Total (P0+P1)** | 6-10 hours | - |
| **Total (All Phases)** | 9-14 hours | - |

---

## Next Steps

### Immediate Actions (This Session)

1. ✅ Create Phase 8 planning document (this file)
2. 🔴 Implement Phase 8.1: Diffusion solver foundation
3. 🔴 Implement Phase 8.2: Multi-wavelength simulation support
4. 🔴 Add validation tests for diffusion solver
5. 🔴 Update documentation and examples

### Follow-Up Actions (Next Session)

- Complete Phase 8.3: Heterogeneous material builder
- Implement clinical workflow examples (blood oxygenation, tissue characterization)
- Consider Phase 8.4: Monte Carlo transport (if needed for validation)
- Update backlog and checklist to reflect Phase 8 progress

---

## References

### Literature

- **Wang et al. (2009)**: "Photoacoustic tomography: in vivo imaging from organelles to organs." *Science*
- **Beard (2011)**: "Biomedical photoacoustic imaging." *Interface Focus*
- **Jacques (2013)**: "Optical properties of biological tissues: a review." *Physics in Medicine & Biology*
- **Prahl (1999)**: "Optical absorption of hemoglobin." *Oregon Medical Laser Center*
- **Wang et al. (2022)**: "Monte Carlo modeling of photon transport in multi-layered tissues." *Optics Express*

### Internal References

- **ADR 004**: Domain Material Property SSOT Pattern
- **Phase 7.9**: Optical Property SSOT Migration
- **Sprint 188**: Architecture Enhancement & Quality Assurance (backlog.md)

---

**Document Version**: 1.0  
**Last Updated**: January 11, 2026  
**Author**: Elite Mathematically-Verified Systems Architect