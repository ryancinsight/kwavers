# TODO Audit Report - Kwavers Codebase
**Generated**: 2025-01-14  
**Sprint**: 208 Phase 3 - Code Quality Audit  
**Auditor**: Automated Comprehensive Audit

---

## Executive Summary

This report documents all incomplete, simplified, or placeholder components found in the Kwavers codebase. Per the architectural principles mandating **zero tolerance for placeholders, stubs, and simplified implementations**, all identified issues have been tagged with TODO markers and documented with implementation requirements.

### Audit Scope
- ✅ Source code (`src/**/*.rs`)
- ✅ Test code (`tests/**/*.rs`)
- ✅ Benchmark code (`benches/**/*.rs`)
- ✅ Examples (`examples/**/*.rs`)
- ✅ Documentation (`README.md`, `docs/**/*.md`)
- ✅ Build configuration (`build.rs`, `Cargo.toml`)

### Critical Findings

**Priority P0 (Production Source Code Issues)**: 5 files, 12 incomplete methods  
**Priority P1 (Benchmark Simplifications)**: 5 files, 35+ simplified implementations  
**Priority P2 (Documentation Examples)**: 1 file, 1 tutorial simplification

---

## P0: Critical Production Code Gaps

These issues exist in **production source code** and violate the "zero placeholder" mandate. They MUST be fully implemented before production use.

### 1. Sensor Beamforming Module - Incomplete Implementation
**File**: `src/domain/sensor/beamforming/sensor_beamformer.rs`  
**Status**: 🔴 CRITICAL - Three methods return placeholder values  
**Estimated Effort**: 6-8 hours total

#### Issue 1.1: `calculate_delays()` - Zero-Filled Placeholder
**Lines**: 83-93  
**Problem**: Returns `Array2::zeros()` instead of computing geometric delays  
**Impact**: Beamforming algorithms receive invalid delay values → incorrect image reconstruction

**Required Implementation**:
```rust
// Geometric distance calculation: d_ij = ||sensor_i - point_j||
// Time-of-flight: τ_ij = d_ij / c
// Delay correction: Δτ_ij = τ_ij - τ_ref
```

**Mathematical Specification**:
- Distance: `d(s_i, p_j) = sqrt((x_i - x_j)² + (y_i - y_j)² + (z_i - z_j)²)`
- Delay: `τ(s_i, p_j) = d(s_i, p_j) / c`
- Reference: Choose geometric focus or first sensor as time reference
- Output: `N_sensors × N_points` matrix of delays in seconds

**Validation Requirements**:
- Causality: All delays ≥ 0
- Physical bounds: τ_max ≤ array_aperture / c
- Symmetry: Equal geometry → equal delays
- Unit test against analytical delays for linear array

**References**:
- Jensen, J.A. (1996) "Field: A Program for Simulating Ultrasound Systems", Med. Bio. Eng. Comp.
- Van Trees, H.L. (2002) "Optimum Array Processing", Wiley-Interscience, Chapter 2

---

#### Issue 1.2: `apply_windowing()` - Pass-Through Stub
**Lines**: 99-117  
**Problem**: Returns unmodified input regardless of `window_type` parameter  
**Impact**: No apodization applied → degraded image quality (high side lobes)

**Required Implementation**:
```rust
match window_type {
    WindowType::Hanning => {
        // w[n] = 0.5 * (1 - cos(2π*n/(N-1)))
    }
    WindowType::Hamming => {
        // w[n] = 0.54 - 0.46 * cos(2π*n/(N-1))
    }
    WindowType::Blackman => {
        // w[n] = 0.42 - 0.5*cos(2πn/(N-1)) + 0.08*cos(4πn/(N-1))
    }
    WindowType::Rectangular => {
        // w[n] = 1.0 (no windowing)
    }
}
```

**Mathematical Specification**:
- Apply window coefficients element-wise to delay matrix or signal
- Normalize to preserve signal energy (optional)
- Handle edge cases (N=1, N=2)

**Validation Requirements**:
- Rectangular: output == input (identity operation)
- Hanning: symmetric, smooth taper, main lobe width ≈ 2 bins
- Side lobe suppression: measure using FFT of window
- Unit tests for each window type

**References**:
- Harris, F.J. (1978) "On the Use of Windows for Harmonic Analysis with the DFT", Proc. IEEE
- Oppenheim, A.V. & Schafer, R.W. (1989) "Discrete-Time Signal Processing", Chapter 7

---

#### Issue 1.3: `calculate_steering()` - Identity Matrix Placeholder
**Lines**: 121-141  
**Problem**: Returns `Array2::eye()` instead of computing steering vectors  
**Impact**: Adaptive beamforming fails → no null steering, no interference rejection

**Required Implementation**:
```rust
// Array manifold: a(θ,φ,f) = [e^(-j*2π*f*τ₁), ..., e^(-j*2π*f*τₙ)]
// where τᵢ = (sensor_i · direction) / c
```

**Mathematical Specification**:
- For each angle (θ, φ):
  - Calculate unit direction vector: d = [sin(θ)cos(φ), sin(θ)sin(φ), cos(θ)]
  - For each sensor i: delay τᵢ = (sensor_position_i · d) / c
  - Steering vector element: a_i = exp(-j * 2π * f * τᵢ)
- Output: `N_sensors × N_angles` complex matrix

**Physical Constraints**:
- Unitary: ||a(θ,φ,f)|| = √N for uniform array
- Hermitian symmetry: a(-θ, φ, f) = a*(θ, φ, f)
- Frequency dependence: visible region changes with f

**Validation Requirements**:
- Broadside steering (θ=0): all phases equal → a = [1, 1, ..., 1]ᵀ
- Endfire steering: maximum phase progression
- Orthogonality: a(θ₁)ᴴa(θ₂) ≈ 0 for widely separated angles
- Property-based tests using Proptest

**References**:
- Van Trees, H.L. (2002) "Optimum Array Processing", Chapter 2 (Array Manifold)
- Krim, H. & Viberg, M. (1996) "Two Decades of Array Signal Processing Research", IEEE SP Magazine

---

### 2. Source Factory - Missing Source Models
**File**: `src/domain/source/factory.rs`  
**Status**: 🔴 CRITICAL - Four source models not implemented  
**Estimated Effort**: 28-36 hours total

#### Issue 2.1: LinearArray Source - Not Implemented
**Lines**: 132-156  
**Problem**: Returns error for `SourceModel::LinearArray`  
**Impact**: Cannot simulate linear transducer arrays (most common clinical configuration)

**Required Implementation**:
- Element positions: 1D array with uniform or non-uniform spacing
- Element geometry: Width, height, kerf (gap between elements)
- Element directivity: Sinc function or measured pattern
- Array factor: Σᵢ wᵢ * exp(j*k*xᵢ*sin(θ))
- Steering: Electronic focusing via time delays or phase shifts

**Estimated Effort**: 8-10 hours

**References**:
- Szabo, T.L. (2004) "Diagnostic Ultrasound Imaging", Chapter 6 (Array Transducers)
- IEC 62359:2017 Annex C (Array transducer specifications)

---

#### Issue 2.2: MatrixArray Source - Not Implemented
**Lines**: 132-156  
**Problem**: Returns error for `SourceModel::MatrixArray`  
**Impact**: Cannot simulate 2D matrix arrays (required for 3D imaging, beam steering)

**Required Implementation**:
- Element positions: 2D grid (rectangular, hexagonal, spiral)
- 2D steering: Azimuth and elevation control
- Element count: Typically 256-9216 elements
- Aperture selection: Active aperture optimization
- Grating lobes: Spatial aliasing mitigation

**Estimated Effort**: 10-12 hours

**References**:
- Turnbull, D.H. & Foster, F.S. (1991) "Beam Steering with Pulsed 2D Arrays", IEEE Trans. UFFC
- Jensen, J.A. et al. (2016) "Synthetic Aperture Ultrasound Imaging", Ultrasonics

---

#### Issue 2.3: Focused Source - Not Implemented
**Lines**: 132-156  
**Problem**: Returns error for `SourceModel::Focused`  
**Impact**: Cannot simulate focused transducers (single-element, mechanically focused)

**Required Implementation**:
- Focal point: (x_f, y_f, z_f)
- Aperture: Diameter or extent
- F-number: F = focal_length / aperture
- Focusing gain: Pressure amplitude increase at focus
- Rayleigh-Sommerfeld diffraction: Accurate near-field and focal zone

**Estimated Effort**: 6-8 hours

**References**:
- O'Neil, H.T. (1949) "Theory of Focusing Radiators", J. Acoust. Soc. Am.
- Kino, G.S. (1987) "Acoustic Waves: Devices, Imaging, and Analog Signal Processing", Chapter 3

---

#### Issue 2.4: Custom Source - Not Implemented
**Lines**: 132-156  
**Problem**: Returns error for `SourceModel::Custom`  
**Impact**: Users cannot define custom source patterns (research use cases)

**Required Implementation**:
- Trait-based extension: `trait CustomSource: Source`
- User-provided field calculation: `fn calculate_field(&self, point: [f64; 3]) -> Complex64`
- Validation hooks: Boundary conditions, power conservation
- Examples: Vortex beams, arbitrary phase/amplitude distributions

**Estimated Effort**: 4-6 hours

**Architecture**: This is the proper extension point for research/custom sources

---

## P0 Part 2: Cloud Infrastructure Incomplete Implementations

### 3. AWS Provider - Hardcoded Infrastructure IDs
**File**: `src/infra/cloud/providers/aws.rs`  
**Severity**: 🔴 CRITICAL - Hardcoded values prevent production deployment  
**Impact**: Cannot deploy to real AWS infrastructure

#### Issue 3.1: Hardcoded Subnet and Security Group IDs
**Lines**: 169-198  
**Problem**: Load balancer creation uses placeholder IDs
- `subnet-12345678` - Invalid subnet ID
- `subnet-87654321` - Invalid subnet ID  
- `sg-12345678` - Invalid security group ID

**Required Implementation**:
```rust
// Load from config
let subnet_ids: Vec<String> = config["subnet_ids"]
    .split(',')
    .map(|s| s.trim().to_string())
    .collect();
let security_groups: Vec<String> = config["security_group_ids"]
    .split(',')
    .map(|s| s.trim().to_string())
    .collect();

// Apply to load balancer
for subnet in &subnet_ids {
    load_balancer.subnets(subnet);
}
for sg in &security_groups {
    load_balancer.security_groups(sg);
}
```

**Configuration Requirements**:
- `config["vpc_id"]` - VPC for deployment
- `config["subnet_ids"]` - Comma-separated subnet IDs (multi-AZ)
- `config["security_group_ids"]` - Security groups for ALB
- `config["certificate_arn"]` - SSL/TLS certificate ARN

**Estimated Effort**: 4-6 hours  
**Priority**: P0 - Blocks AWS production deployment

---

### 4. Azure Provider - Missing Deployment Implementation
**File**: `src/infra/cloud/providers/azure.rs`  
**Severity**: 🔴 CRITICAL - Placeholder deployment without Azure ML calls  
**Impact**: Deployment creates fake endpoint URL without actual Azure resources

#### Issue 4.1: deploy_to_azure() - No Azure ML API Calls
**Lines**: 87-109  
**Problem**: Function generates URL but doesn't create Azure ML resources

**Missing Azure ML REST API Calls**:
1. `PUT /models/{modelName}` - Register model in workspace
2. `PUT /onlineEndpoints/{endpointName}` - Create endpoint
3. `PUT /onlineEndpoints/{endpointName}/deployments/{deploymentName}` - Deploy model
4. `PATCH /onlineEndpoints/{endpointName}` - Update traffic allocation

**Estimated Effort**: 10-12 hours  
**Priority**: P0 - Blocks Azure production deployment

#### Issue 4.2: scale_azure_deployment() - Not Implemented
**Lines**: 121-247  
**Problem**: Returns `FeatureNotAvailable` error instead of scaling

**Required Implementation**:
- Azure ML REST API integration for deployment updates
- Update `sku.capacity` property in deployment configuration
- Poll provisioning state until scaling complete
- Handle authentication token management

**Estimated Effort**: 6-8 hours  
**Priority**: P1 - Required for production auto-scaling

---

### 5. GCP Provider - Missing Deployment Implementation
**File**: `src/infra/cloud/providers/gcp.rs`  
**Severity**: 🔴 CRITICAL - Placeholder deployment without Vertex AI calls  
**Impact**: Deployment creates fake endpoint URL without actual GCP resources

#### Issue 5.1: deploy_to_gcp() - No Vertex AI API Calls
**Lines**: 92-115  
**Problem**: Function generates URL but doesn't create Vertex AI resources

**Missing Vertex AI REST API Calls**:
1. `POST /v1/projects/{project}/locations/{location}/models` - Upload model
2. `POST /v1/projects/{project}/locations/{location}/endpoints` - Create endpoint
3. `POST /v1/projects/{project}/locations/{location}/endpoints/{endpoint}:deployModel` - Deploy
4. `PATCH /v1/projects/{project}/locations/{location}/endpoints/{endpoint}` - Update config

**Estimated Effort**: 10-12 hours  
**Priority**: P0 - Blocks GCP production deployment

#### Issue 5.2: scale_gcp_deployment() - Not Implemented
**Lines**: 129-261  
**Problem**: Returns `FeatureNotAvailable` error instead of scaling

**Required Implementation**:
- Vertex AI REST API integration for replica count updates
- Update `dedicatedResources.minReplicaCount` and `maxReplicaCount`
- Poll long-running operation until complete
- Configure auto-scaling policies (CPU utilization, scale-down delay)

**Estimated Effort**: 8-10 hours  
**Priority**: P1 - Required for production auto-scaling

---

## P1: Benchmark Simplifications

These issues exist in **benchmark code** (`benches/**/*.rs`). While not production code, they violate the principle of "no simplified implementations." Benchmarks should measure **real physics**, not stubs.

### 3. Performance Benchmark - Extensive Stub Functions
**File**: `benches/performance_benchmark.rs`  
**Status**: 🟡 HIGH PRIORITY - 25+ helper methods are empty stubs  
**Estimated Effort**: 40-60 hours for full implementation

**Problem**: Benchmark suite times empty functions, not actual physics solvers. Measurements are meaningless.

#### Stub Methods (Partial List):
1. `update_velocity_fdtd()` - Line 913: Empty body, no velocity update
2. `update_pressure_fdtd()` - Line 932: Empty body, no pressure update
3. `update_pressure_nonlinear()` - Line 948: Empty body, no Westervelt terms
4. `simulate_fft_operations()` - Line 973: Empty body, no FFT
5. `simulate_angular_spectrum_propagation()` - Line 978: Empty body
6. `simulate_elastic_wave_step()` - Line 982: Empty body
7. `simulate_stiffness_estimation()` - Line 1001: Returns clone, not inverse problem
8. `compute_uncertainty_statistics()` - Line 1009: Returns zeros, not variance
9. `compute_ensemble_mean()` - Line 1015: Returns zeros, not mean
10. `compute_conformity_score()` - Line 1030: Returns 0.0, not score

**Impact**: 
- Benchmark results are invalid (timing infrastructure overhead, not physics)
- Cannot use for performance optimization decisions
- Misleading for capacity planning

**Recommendation**: 
- **Option A**: Implement real physics (40-60 hours)
- **Option B**: Remove benchmarks until implementations exist
- **Option C**: Clearly label as "infrastructure timing only" and isolate from physics benchmarks

---

### 4. Comparative Solver Benchmark - Simplified Energy Calculation
**File**: `benches/comparative_solver_benchmark.rs`  
**Status**: 🟡 MEDIUM PRIORITY  
**Estimated Effort**: 2-3 hours

**Problem**: `calculate_energy()` at line 163 uses L2 norm as energy proxy. This is dimensionally incorrect and not a physical energy.

**Current Implementation**:
```rust
fn calculate_energy(field: ArrayView3<f64>) -> f64 {
    field.iter().map(|&x| x * x).sum::<f64>().sqrt()
}
```

**Required Implementation**:
```rust
fn calculate_acoustic_energy(
    pressure: ArrayView3<f64>,
    velocity: ArrayView3<f64>,
    density: f64,
    sound_speed: f64,
    dx: f64, dy: f64, dz: f64
) -> f64 {
    let mut total_energy = 0.0;
    
    // Kinetic energy: E_k = ∫ (1/2) * ρ * |v|² dV
    // Potential energy: E_p = ∫ (1/2) * p² / (ρc²) dV
    
    for i in 0..pressure.dim().0 {
        for j in 0..pressure.dim().1 {
            for k in 0..pressure.dim().2 {
                let p = pressure[[i, j, k]];
                let v = velocity[[i, j, k]];
                
                let e_potential = 0.5 * p * p / (density * sound_speed * sound_speed);
                let e_kinetic = 0.5 * density * v * v;
                
                total_energy += (e_potential + e_kinetic) * dx * dy * dz;
            }
        }
    }
    
    total_energy
}
```

**References**:
- Blackstock, D.T. (2000) "Fundamentals of Physical Acoustics", Chapter 1 (Energy density)

---

### 5. FNM Performance Benchmark - Simplified Rayleigh-Sommerfeld
**File**: `benches/fnm_performance_benchmark.rs`  
**Status**: 🟢 LOW PRIORITY - Acceptable for benchmark comparison  
**Estimated Effort**: 15-20 hours if full implementation desired

**Problem**: `RayleighSommerfeldSolver` at line 16 is simplified for O(n²) timing comparison.

**Current Status**: This is **acceptable** as a reference implementation for complexity demonstration. The benchmark compares FNM (O(n)) vs. naive integration (O(n²)). The simplified version correctly demonstrates the scaling difference.

**Recommendation**: Keep as-is, but add clear documentation that this is a reference implementation, not production code.

---

### 6. SIMD FDTD Benchmark - Scalar Fallback Placeholder
**File**: `benches/simd_fdtd_benchmarks.rs`  
**Status**: 🟡 MEDIUM PRIORITY  
**Estimated Effort**: 10-15 hours

**Problem**: Line 100 uses `scalar_pressure_update()` as placeholder for SIMD implementation.

**Current Implementation**:
```rust
// TODO: SIMD implementation not integrated
let result = scalar_pressure_update(p.clone(), div, rho, c, dt);
```

**Required Implementation**:
- Use SIMD intrinsics (AVX2 or AVX-512) for vectorized operations
- Aligned memory layout: `#[repr(align(32))]` or `#[repr(align(64))]`
- Handle remainder elements (array size not multiple of SIMD width)
- Benchmark realistic: SIMD vs scalar vs auto-vectorization

**Architecture Note**: This is actually a gap in the production SIMD implementation, not just benchmark code. SIMD solver exists but may not be integrated.

---

### 7. Ultrasound Benchmarks - Simplified Clinical Analysis
**File**: `benches/ultrasound_benchmarks.rs`  
**Status**: 🟢 LOW PRIORITY  
**Estimated Effort**: 6-8 hours

**Problem**: Line 209 "clinical analysis" is just mean/std calculation, not real clinical workflow.

**Current Status**: This is **acceptable** for benchmark purposes. Clinical workflows involve human interaction (ROI selection), regulatory compliance, and reporting - inappropriate for automated benchmarks.

**Recommendation**: Rename to "basic_statistics_analysis" to avoid confusion. Keep simplified version.

---

## P2: Documentation Examples

### 8. README Example - Simplified Tutorial Code
**File**: `README.md`  
**Status**: ✅ ACCEPTABLE - Tutorial simplification justified  
**No Action Required**

**Location**: Lines 155-157  
**Issue**: Example 2 shows property values without creating full `HomogeneousMedium` object.

**Current Status**: This is **acceptable** for tutorial/getting-started documentation. The note clearly states "simplified example showing property values" and refers users to the full API.

**Recommendation**: Keep as-is. Added TODO marker for documentation tracking, but no implementation change needed.

---

## Summary Statistics

| Category | Count | Estimated Effort |
|----------|-------|------------------|
| **P0 Critical** (Production Code) | 5 files, 12 methods | 72-98 hours |
| **P1 High** (Benchmarks) | 5 files, 35+ methods | 73-103 hours |
| **P1 High** (Cloud Scaling) | 2 files, 2 methods | 14-18 hours |
| **P2 Low** (Documentation) | 1 file, 1 example | 0 hours (acceptable) |
| **TOTAL** | 12 files | 159-219 hours |

---

## Architectural Impact Assessment

### Violation Severity

#### 🔴 Critical Violations (P0)
The sensor beamforming and source factory gaps are **critical architectural violations**:

1. **Single Source of Truth**: Methods exist but return invalid data (zeros, identity, error)
2. **Zero Placeholder Policy**: Direct violation of "no placeholder" mandate
3. **Mathematical Correctness**: Invalid outputs will corrupt downstream algorithms
4. **Production Readiness**: Cannot be used in production without these implementations

**Status**: These MUST be implemented before any production deployment. Currently, any beamforming or array simulation is producing incorrect results.

**Cloud Infrastructure Status**: Cloud deployment functions exist but create placeholder endpoints without actual cloud resources. AWS has hardcoded infrastructure IDs, Azure and GCP return fake URLs without creating resources. This blocks production cloud deployment entirely.

#### 🟡 Moderate Violations (P1)
The benchmark simplifications are **moderate violations**:

1. **Measurement Validity**: Benchmarks measure infrastructure, not physics
2. **Decision-Making**: Cannot use for optimization or capacity planning
3. **Technical Debt**: Creates false confidence in "tested" components

**Status**: Should be implemented or removed. Keeping simplified benchmarks creates technical debt and confusion.

#### ✅ Acceptable (P2)
Documentation examples are **acceptable simplifications**:

1. **Tutorial Purpose**: Explicitly educational, not production
2. **Clear Labeling**: Marked as simplified with references to full API
3. **Progressive Disclosure**: Appropriate for getting-started guide

**Status**: No action required. This is good documentation practice.

---

## Recommendations

### Immediate Actions (Sprint 208 Phase 4)

1. **Document All Gaps**: ✅ COMPLETE (this report - updated with cloud findings)
2. **Add TODO Tags**: ✅ COMPLETE (12 files tagged)
3. **Update Backlog**: Update `backlog.md` with cloud infrastructure gaps
4. **Prioritization Meeting**: Decide implementation vs. removal for benchmarks
5. **Cloud Deployment Warning**: Add runtime warnings for incomplete cloud providers

### Short-term (Sprint 209-210)

1. **Implement P0 Critical - Core**: Sensor beamforming (6-8 hours), Source factory (28-36 hours)
2. **Implement P0 Critical - Cloud**: AWS infrastructure (4-6 hours), Azure deployment (10-12 hours), GCP deployment (10-12 hours)
3. **Benchmark Decision**: Implement or remove P1 benchmarks
4. **Integration Tests**: Add tests that fail until P0 implementations complete
5. **API Documentation**: Mark incomplete cloud APIs with warnings

### Medium-term (Sprint 211-212)

1. **P1 Cloud Scaling**: Azure ML scaling (6-8 hours), GCP Vertex AI scaling (8-10 hours)
2. **Cloud Integration Tests**: Real deployment tests (requires cloud credentials)
3. **Cloud Documentation**: Deployment guides for each provider

### Long-term (Sprint 213+)

1. **Full Benchmark Suite**: Implement all physics for benchmarks
2. **Performance Baseline**: Establish validated performance metrics
3. **Research Integration**: Use full implementations for k-Wave/jwave comparison
4. **Multi-Cloud Orchestration**: Cross-cloud deployment management

---

## Compliance Statement

This audit fulfills the architectural mandate:

> "Absolute Prohibition: TODOs, stubs, dummy data, zero-filled placeholders, 'simplified' paths, error masking, unwrap() without proof."

**Current Status**:
- ❌ **Non-Compliant**: 5 production files contain placeholders (P0 critical)
  - 2 core domain files (beamforming, source factory)
  - 3 cloud infrastructure files (AWS, Azure, GCP)
- ❌ **Non-Compliant**: 5 benchmark files contain simplified stubs (P1 high)
- ❌ **Non-Compliant**: 2 cloud scaling functions not implemented (P1 high)
- ✅ **Compliant**: All gaps now documented and tagged
- ✅ **Compliant**: Clear path to full compliance defined

**Path to Compliance**:
1. Complete P0 core implementations (34-44 hours)
2. Complete P0 cloud implementations (24-30 hours)
3. Complete P1 cloud scaling (14-18 hours)
4. Implement or remove P1 benchmarks (73-103 hours OR removal)
5. Verify all TODO tags resolved
6. Final audit: grep for TODO, placeholder, simplified, stub

---

## Appendix A: Audit Methodology

### Search Patterns
```regex
(?i)(todo!|unimplemented!|fixme|hack|simplified|placeholder|stub|dummy|zero-filled|xxx|temporary)
```

### Files Audited
- Source: 247 files in `src/` (including cloud infrastructure)
- Tests: 68 files in `tests/`
- Benchmarks: 12 files in `benches/`
- Examples: 43 files in `examples/`
- Documentation: 25 files in `docs/`

### False Positives Excluded
- Historical references in sprint documentation
- Quotes from architectural principles (describing what to avoid)
- Git ignore comments
- Dependency names (`text_placeholder` crate)

---

## Appendix B: Implementation Priorities

### P0 - Week 1-2 (Sprint 209) - Core Domain
- [ ] Sensor beamforming: `calculate_delays()` - 2-3 hours
- [ ] Sensor beamforming: `apply_windowing()` - 2-3 hours
- [ ] Sensor beamforming: `calculate_steering()` - 2-3 hours
- [ ] Test suite: Beamforming validation - 2 hours
- **Subtotal**: 8-11 hours

### P0 - Week 3-5 (Sprint 209-210) - Source Factory
- [ ] Source factory: `LinearArray` - 8-10 hours
- [ ] Source factory: `MatrixArray` - 10-12 hours
- [ ] Source factory: `Focused` - 6-8 hours
- [ ] Source factory: `Custom` trait - 4-6 hours
- [ ] Test suite: Source validation - 4 hours
- **Subtotal**: 32-40 hours

### P0 - Week 6-8 (Sprint 210-211) - Cloud Infrastructure
- [ ] AWS: Configuration-based infrastructure IDs - 4-6 hours
- [ ] Azure: Implement ML deployment API calls - 10-12 hours
- [ ] GCP: Implement Vertex AI deployment API calls - 10-12 hours
- [ ] Test suite: Cloud deployment integration tests - 4 hours
- **Subtotal**: 28-34 hours

### P1 - Week 9-10 (Sprint 211) - Cloud Scaling
- [ ] Azure ML: Implement scaling API - 6-8 hours
- [ ] GCP Vertex AI: Implement scaling API - 8-10 hours
- [ ] Test suite: Cloud scaling tests - 2 hours
- **Subtotal**: 16-20 hours

### P1 - Week 11-15 (Sprint 212-213) - Benchmarks
- [ ] Decision: Implement or remove benchmark stubs
- [ ] If implement: 73-103 hours
- [ ] If remove: 2-3 hours (clean removal + documentation)

---

## Appendix C: References

### Architecture Documents
- `prompt.yaml` - Zero tolerance policy
- `docs/ADR.md` - Architectural decision records
- `checklist.md` - Sprint tracking
- `backlog.md` - Work items

### Physics References
- Hamilton & Blackstock - "Nonlinear Acoustics"
- Szabo - "Diagnostic Ultrasound Imaging"
- Van Trees - "Optimum Array Processing"
- Jensen - "Field II Simulation"
- IEC 62359:2017 - Ultrasound transducer standards

### Code Quality Standards
- Rust API Guidelines
- Clean Architecture (Robert C. Martin)
- Domain-Driven Design (Eric Evans)
- Mathematical Software (Accuracy, Stability, Reproducibility)

---

**Report End**  
**Next Action**: Review with team, prioritize implementations, update Sprint 209 plan