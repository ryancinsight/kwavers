# Phase 4 Implementation Plan - Advanced Features and Performance

**Date:** January 28, 2026  
**Status:** 🔄 IN PROGRESS  
**Branch:** main  
**Goal:** Complete GPU backend, integrate PSTD/Hybrid solvers, add SIMD optimization

---

## Executive Summary

Phase 4 focuses on performance optimization and advanced solver integration to bring kwavers to 100% feature completion. This phase will:

1. **Complete GPU Backend** - Production-ready WGPU compute shaders
2. **Integrate PSTD Solver** - Full Pseudo-Spectral Time-Domain execution
3. **Integrate Hybrid Solver** - PSTD/FDTD adaptive coupling
4. **SIMD Optimization** - Vectorized operations for CPU backend
5. **Advanced Examples** - Demonstrations of all Phase 4 features

---

## Current State Analysis

### What Exists (Phase 1-3)

✅ **Architecture** - Clean 8-layer structure, zero circular dependencies  
✅ **FDTD Solver** - Complete implementation with API integration  
✅ **Backend Traits** - Abstract interface for CPU/GPU  
✅ **CPU Backend** - Full implementation with rayon parallelization  
✅ **GPU Backend Stub** - Trait implementation, initialization logic  
✅ **PSTD Solver** - Core implementation exists (`src/solver/forward/pstd/`)  
✅ **Hybrid Solver** - Core implementation exists (`src/solver/forward/hybrid/solver.rs`)  
✅ **Factory Pattern** - Auto-configuration with CFL/grid spacing  
✅ **Tiered APIs** - Simple/Standard/Advanced  
✅ **Execution Engine** - API-to-solver bridge  
✅ **Domain Builders** - Anatomical models and transducers  

### What Needs Completion

❌ **GPU Backend** - Compute shaders (FFT, element-wise ops, k-space)  
❌ **PSTD Integration** - Connection to ExecutionEngine  
❌ **Hybrid Integration** - Connection to ExecutionEngine  
❌ **SIMD Optimization** - Vectorized CPU operations  
❌ **Performance Benchmarks** - Comparative analysis  
❌ **Advanced Examples** - GPU, PSTD, Hybrid demonstrations  

---

## Phase 4 Objectives

### 1. GPU Backend Completion (Priority: HIGH)

**Current State:**
- `src/solver/backend/gpu.rs` - Trait implementation with TODOs
- Feature-gated with `#[cfg(feature = "gpu")]`
- Basic initialization logic exists

**Implementation Tasks:**

#### 1.1 WGPU Initialization
```rust
// src/solver/backend/gpu/init.rs
- Instance creation
- Adapter selection (prefer high-performance GPU)
- Device and queue initialization
- Error handling and fallback
```

#### 1.2 Compute Shaders (WGSL)
```wgsl
// src/solver/backend/gpu/shaders/fft.wgsl
- 1D FFT (Cooley-Tukey, radix-2/4)
- 2D FFT (row-column decomposition)
- 3D FFT (composition of 1D/2D)

// src/solver/backend/gpu/shaders/operators.wgsl
- Element-wise multiply
- Element-wise add/subtract
- Scalar multiplication
- K-space operators (spatial derivatives)

// src/solver/backend/gpu/shaders/utils.wgsl
- Buffer copy operations
- Data type conversions (f32/f64)
- Reduction operations (sum, max, min)
```

#### 1.3 GPU Buffer Management
```rust
// src/solver/backend/gpu/buffers.rs
- Buffer allocation and deallocation
- CPU-GPU data transfer
- Staging buffer management
- Memory pool for reuse
```

#### 1.4 GPU Pipeline Management
```rust
// src/solver/backend/gpu/pipeline.rs
- Compute pipeline creation
- Bind group management
- Command encoder/buffer handling
- Synchronization primitives
```

**Files to Create:**
- `src/solver/backend/gpu/mod.rs` (refactor existing gpu.rs)
- `src/solver/backend/gpu/init.rs`
- `src/solver/backend/gpu/buffers.rs`
- `src/solver/backend/gpu/pipeline.rs`
- `src/solver/backend/gpu/shaders/fft.wgsl`
- `src/solver/backend/gpu/shaders/operators.wgsl`
- `src/solver/backend/gpu/shaders/utils.wgsl`

**Testing:**
- Unit tests for each shader
- Integration tests for full GPU backend
- Performance benchmarks vs CPU
- Validation against CPU results (ensure correctness)

**Expected Performance:**
- 10-30× speedup for FFT operations (256³ grids)
- 5-10× speedup for element-wise operations
- Overall simulation speedup: 8-20× depending on problem size

---

### 2. PSTD Solver Integration (Priority: HIGH)

**Current State:**
- `src/solver/forward/pstd/` - Full implementation exists
- `PSTDSolver` struct in `implementation/core/orchestrator.rs`
- Not connected to ExecutionEngine

**Implementation Tasks:**

#### 2.1 Execution Engine Integration
```rust
// src/api/execution.rs - Add execute_pstd() method

impl ExecutionEngine {
    fn execute_pstd(&self) -> KwaversResult<SimulationOutput> {
        // 1. Create PSTD configuration from self.config
        // 2. Initialize PSTD solver
        // 3. Run time-stepping loop
        // 4. Extract results
        // 5. Return SimulationOutput
    }
}
```

#### 2.2 Configuration Mapping
```rust
// src/solver/forward/pstd/config.rs - Ensure compatibility

// Map from simulation::factory::Configuration to PSTDConfig
fn from_simulation_config(config: &Configuration) -> KwaversResult<PSTDConfig> {
    // Extract relevant parameters
    // Set PSTD-specific defaults
    // Validate constraints
}
```

#### 2.3 PSTD-Specific Factory Presets
```rust
// src/simulation/factory/presets.rs - Add PSTD presets

pub fn pstd_ultrasound() -> Configuration {
    // Optimized for PSTD solver
    // Appropriate spatial resolution
    // Time step for spectral accuracy
}
```

**Files to Modify:**
- `src/api/execution.rs` (add `execute_pstd()`)
- `src/solver/forward/pstd/config.rs` (add conversion from factory config)
- `src/simulation/factory/presets.rs` (add PSTD-optimized presets)

**Testing:**
- Unit tests for config conversion
- Integration test: Simple API → PSTD solver
- Validation against analytical solutions
- Comparison with FDTD results

---

### 3. Hybrid Solver Integration (Priority: MEDIUM)

**Current State:**
- `src/solver/forward/hybrid/solver.rs` - Full implementation exists
- Combines PSTD and FDTD adaptively
- Not connected to ExecutionEngine

**Implementation Tasks:**

#### 3.1 Execution Engine Integration
```rust
// src/api/execution.rs - Add execute_hybrid() method

impl ExecutionEngine {
    fn execute_hybrid(&self) -> KwaversResult<SimulationOutput> {
        // 1. Create Hybrid configuration
        // 2. Initialize HybridSolver
        // 3. Run adaptive time-stepping
        // 4. Monitor domain decomposition
        // 5. Return SimulationOutput
    }
}
```

#### 3.2 Hybrid Configuration Builder
```rust
// src/solver/forward/hybrid/config.rs - Configuration builder

impl HybridConfig {
    pub fn from_simulation_config(config: &Configuration) -> KwaversResult<Self> {
        // Create both FDTD and PSTD configs
        // Set decomposition strategy
        // Configure coupling interface
    }
}
```

#### 3.3 Adaptive Selection Criteria
```rust
// src/simulation/factory/solver_selection.rs - Add hybrid criteria

pub fn should_use_hybrid(config: &Configuration) -> bool {
    // Use hybrid for heterogeneous media
    // Use hybrid for multi-scale problems
    // Use hybrid when both smoothness and discontinuities present
}
```

**Files to Modify:**
- `src/api/execution.rs` (add `execute_hybrid()`)
- `src/solver/forward/hybrid/config.rs` (add conversion)
- `src/simulation/factory/solver_selection.rs` (add hybrid selection logic)

**Testing:**
- Integration test: Standard API → Hybrid solver
- Validation: Smooth region (compare to PSTD)
- Validation: Discontinuous region (compare to FDTD)
- Performance: Verify speedup vs pure FDTD

---

### 4. SIMD Optimization (Priority: MEDIUM)

**Current State:**
- `src/math/simd/` - Module exists but limited use
- CPU backend uses rayon but no explicit SIMD

**Implementation Tasks:**

#### 4.1 SIMD Element-Wise Operations
```rust
// src/math/simd/elementwise.rs

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn multiply_f64_simd(a: &[f64], b: &[f64], out: &mut [f64]) {
    // AVX2: 4x f64 per instruction
    // AVX-512: 8x f64 per instruction
    // Fallback to scalar for non-aligned
}

pub fn add_f64_simd(a: &[f64], b: &[f64], out: &mut [f64]) {
    // Vectorized addition
}
```

#### 4.2 SIMD FFT Operations
```rust
// src/math/fft/simd_fft.rs

// Vectorized butterfly operations for FFT
// Radix-2/4/8 with SIMD intrinsics
```

#### 4.3 CPU Backend Integration
```rust
// src/solver/backend/cpu.rs - Use SIMD operations

impl Backend for CPUBackend {
    fn element_wise_multiply(...) -> KwaversResult<()> {
        #[cfg(target_feature = "avx2")]
        {
            simd::multiply_f64_simd(...)
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            // Fallback to rayon parallel
        }
    }
}
```

**Files to Create/Modify:**
- `src/math/simd/elementwise.rs` (create)
- `src/math/simd/fft.rs` (create)
- `src/solver/backend/cpu.rs` (integrate SIMD)

**Testing:**
- Correctness tests (SIMD vs scalar)
- Performance benchmarks
- Cross-platform testing (x86_64, aarch64)

**Expected Performance:**
- 2-4× speedup for element-wise ops
- 1.5-2× speedup for FFT
- Overall CPU backend: 30-50% faster

---

### 5. Advanced Examples and Documentation (Priority: HIGH)

**Examples to Create:**

#### 5.1 GPU Backend Example
```rust
// examples/phase4_gpu_backend.rs

// 1. GPU initialization and selection
// 2. Run simulation with GPU backend
// 3. Compare performance to CPU
// 4. Demonstrate automatic fallback
```

#### 5.2 PSTD Solver Example
```rust
// examples/phase4_pstd_solver.rs

// 1. Smooth homogeneous medium
// 2. PSTD-optimized configuration
// 3. Compare accuracy to FDTD
// 4. Demonstrate spectral accuracy
```

#### 5.3 Hybrid Solver Example
```rust
// examples/phase4_hybrid_solver.rs

// 1. Heterogeneous medium (skull + tissue)
// 2. Automatic domain decomposition
// 3. Adaptive method selection
// 4. Performance analysis
```

#### 5.4 Performance Comparison
```rust
// examples/phase4_performance_comparison.rs

// 1. Same problem with FDTD, PSTD, Hybrid
// 2. CPU vs GPU backend
// 3. SIMD vs non-SIMD
// 4. Generate performance report
```

**Documentation:**
- Phase 4 completion report
- GPU backend user guide
- PSTD solver user guide
- Hybrid solver user guide
- Performance optimization guide

---

## Implementation Strategy

### Week 1: GPU Backend Foundation

**Days 1-2:**
- ✅ Create Phase 4 plan (this document)
- Refactor `gpu.rs` into `gpu/` module
- Implement WGPU initialization
- Create buffer management system

**Days 3-4:**
- Write compute shaders (FFT, operators)
- Implement pipeline management
- Write unit tests for shaders

**Days 5-7:**
- Integration testing
- Performance benchmarking
- Bug fixes and optimization

### Week 2: Solver Integration

**Days 1-3:**
- PSTD solver integration
- Configuration mapping
- Testing and validation

**Days 4-5:**
- Hybrid solver integration
- Adaptive selection logic
- Testing and validation

**Days 6-7:**
- SIMD optimization
- CPU backend enhancement
- Performance testing

### Week 3: Examples and Documentation

**Days 1-3:**
- Write all Phase 4 examples
- Test examples thoroughly
- Create example documentation

**Days 4-5:**
- Write user guides
- Create performance guide
- Update main documentation

**Days 6-7:**
- Phase 4 completion report
- Final testing and validation
- Release preparation

---

## Architecture Compliance

### Layer Placement

All Phase 4 work maintains the 8-layer clean architecture:

**Layer 1 (Math):**
- `src/math/simd/elementwise.rs` ✅
- `src/math/simd/fft.rs` ✅

**Layer 4 (Solver):**
- `src/solver/backend/gpu/` ✅
- `src/solver/backend/cpu.rs` (SIMD integration) ✅

**Layer 5 (Simulation):**
- `src/simulation/factory/presets.rs` (PSTD/Hybrid presets) ✅

**Layer 8 (Infrastructure/API):**
- `src/api/execution.rs` (PSTD/Hybrid execution) ✅

### Dependency Rules

All dependencies flow downward:
- GPU backend depends on: Core, Math
- SIMD math depends on: Core only
- Execution engine depends on: All solver layers
- No circular dependencies introduced ✅

---

## Success Criteria

### Functional Requirements

✅ **GPU Backend**
- All backend trait methods implemented
- Correctness validated against CPU
- Automatic fallback when GPU unavailable
- Feature-gated properly

✅ **PSTD Integration**
- Simple API → PSTD works
- Standard API → PSTD works
- Advanced API → PSTD works
- Results match analytical solutions

✅ **Hybrid Integration**
- Automatic domain decomposition
- Smooth coupling between PSTD/FDTD
- Performance better than pure FDTD

✅ **SIMD Optimization**
- Cross-platform compatibility
- Correctness validated
- Performance improvement measured

### Performance Requirements

| Metric | Target | Baseline |
|--------|--------|----------|
| GPU FFT speedup | 10-30× | CPU rayon |
| GPU overall speedup | 8-20× | CPU rayon |
| CPU SIMD speedup | 1.3-1.5× | CPU no-SIMD |
| PSTD vs FDTD (smooth) | 2-5× faster | FDTD |
| Hybrid vs FDTD (mixed) | 1.5-3× faster | FDTD |

### Quality Requirements

- Zero build errors ✅
- Zero circular dependencies ✅
- >90% test coverage for new code ✅
- All examples compile and run ✅
- Documentation complete ✅

---

## Risk Assessment

### Technical Risks

**Risk 1: GPU Availability**
- **Impact:** High (blocks GPU backend)
- **Probability:** Medium (some systems lack GPU)
- **Mitigation:** Robust fallback to CPU, clear error messages

**Risk 2: WGPU Complexity**
- **Impact:** High (delays GPU implementation)
- **Probability:** Medium (complex API)
- **Mitigation:** Use well-tested patterns, extensive testing

**Risk 3: SIMD Portability**
- **Impact:** Medium (reduced performance on some platforms)
- **Probability:** Low (use `std::arch`)
- **Mitigation:** Runtime feature detection, scalar fallback

**Risk 4: Solver Integration Bugs**
- **Impact:** High (incorrect results)
- **Probability:** Medium (complex integration)
- **Mitigation:** Extensive validation against analytical solutions

### Schedule Risks

**Risk 5: Underestimated Complexity**
- **Impact:** Medium (delayed completion)
- **Probability:** Medium
- **Mitigation:** Prioritize features, accept partial completion

---

## Dependencies and Tools

### Rust Crates Required

```toml
[dependencies]
# Existing
ndarray = { version = "0.16", features = ["rayon"] }
rayon = "1.8"
rustfft = "6.1"

# New for Phase 4
wgpu = { version = "0.18", optional = true }        # GPU backend
bytemuck = { version = "1.14", optional = true }    # GPU buffer casting
pollster = { version = "0.3", optional = true }     # Async runtime for GPU

[features]
gpu = ["wgpu", "bytemuck", "pollster"]
simd = []  # SIMD uses std::arch (no extra deps)
full = ["gpu", "simd", "plotting", "parallel"]
```

### Testing Tools

- `cargo test` - Unit and integration tests
- `cargo bench` - Performance benchmarks
- `cargo clippy` - Linting
- `scripts/validate_architecture.sh` - Architecture validation

---

## Deliverables

### Code

1. **GPU Backend** (8 files, ~1,500 LOC)
   - `src/solver/backend/gpu/mod.rs`
   - `src/solver/backend/gpu/init.rs`
   - `src/solver/backend/gpu/buffers.rs`
   - `src/solver/backend/gpu/pipeline.rs`
   - `src/solver/backend/gpu/shaders/fft.wgsl`
   - `src/solver/backend/gpu/shaders/operators.wgsl`
   - `src/solver/backend/gpu/shaders/utils.wgsl`
   - Tests: 60+ tests

2. **SIMD Optimization** (3 files, ~800 LOC)
   - `src/math/simd/elementwise.rs`
   - `src/math/simd/fft.rs`
   - `src/solver/backend/cpu.rs` (modifications)
   - Tests: 40+ tests

3. **Solver Integration** (3 files, ~600 LOC)
   - `src/api/execution.rs` (add PSTD/Hybrid methods)
   - `src/solver/forward/pstd/config.rs` (add conversion)
   - `src/solver/forward/hybrid/config.rs` (add conversion)
   - Tests: 30+ tests

4. **Examples** (4 files, ~800 LOC)
   - `examples/phase4_gpu_backend.rs`
   - `examples/phase4_pstd_solver.rs`
   - `examples/phase4_hybrid_solver.rs`
   - `examples/phase4_performance.rs`

### Documentation

1. **Phase 4 Completion Report** (~80 pages)
   - Implementation details
   - Performance analysis
   - User guides
   - Migration guide

2. **User Guides** (~40 pages)
   - GPU backend usage
   - PSTD solver guide
   - Hybrid solver guide
   - Performance optimization

3. **Updated Comprehensive Summary** (~10 pages)
   - Include Phase 4 statistics
   - Final feature comparison
   - Overall project status

---

## Total Statistics Estimate

### Phase 4 Contribution

| Metric | Estimate |
|--------|----------|
| New files | 18 |
| Lines of code | 3,700 |
| Tests added | 130+ |
| Examples | 4 |
| Documentation pages | 130 |

### Overall Project (Phase 1-4)

| Metric | Current (Phase 3) | After Phase 4 |
|--------|-------------------|---------------|
| Total files | 30 | 48 |
| Total LOC | 7,550 | 11,250 |
| Total tests | 227 | 357+ |
| Examples | 7 | 11 |
| Documentation | 12 docs | 18 docs |
| Feature completion | 85% | 100% |

---

## Next Steps

1. ✅ Create this implementation plan
2. Begin GPU backend refactoring
3. Implement WGPU initialization
4. Write compute shaders
5. Integrate PSTD solver
6. Integrate Hybrid solver
7. Add SIMD optimization
8. Create examples
9. Write documentation
10. Final testing and release

---

**Plan Created:** January 28, 2026  
**Target Completion:** February 18, 2026 (3 weeks)  
**Status:** Ready to begin implementation

---

## Appendix: Reference Implementations

### GPU Computing References

- **WGPU Documentation:** https://wgpu.rs/
- **Compute Shader Guide:** https://www.w3.org/TR/WGSL/
- **GPU FFT Algorithms:** Cooley-Tukey, Stockham, Mixed-Radix
- **k-Wave GPU:** https://github.com/ucl-bug/k-wave (CUDA reference)

### PSTD Solver References

- **k-Wave PSTD:** Pseudo-spectral implementation
- **jWave:** JAX-based spectral methods
- **Theory:** Treeby & Cox (2010) - k-space propagation

### Hybrid Solver References

- **Domain Decomposition:** Smith, Bjorstad, Gropp (1996)
- **Adaptive Methods:** Berger & Colella (1989)
- **Coupling Strategies:** Conservative interpolation

### SIMD References

- **Rust SIMD:** `std::arch` portable intrinsics
- **AVX2:** 256-bit vector operations
- **AVX-512:** 512-bit vector operations
- **Portable SIMD:** `std::simd` (nightly, future)

---

**End of Phase 4 Implementation Plan**
