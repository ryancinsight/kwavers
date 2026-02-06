# Quick Start Checklist: Complete PSTD & Hybrid Solvers

**Goal:** Wire PSTD and Hybrid solvers into PyKwavers Python bindings  
**Time:** 6 hours  
**File:** `kwavers/pykwavers/src/lib.rs`

---

## ☑️ Pre-Flight Check

- [ ] Read `PSTD_HYBRID_IMPLEMENTATION_GUIDE.md` (5 min)
- [ ] Verify Rust toolchain installed: `rustc --version`
- [ ] Verify maturin installed: `pip install maturin`
- [ ] Current directory: `kwavers/pykwavers/`
- [ ] Existing FDTD backend working: `pytest test_phase5_features.py -v`

---

## Part 1: PSTD Solver (4 hours)

### Step 1.1: Add Imports (5 min)

Location: Top of `src/lib.rs`, after existing solver imports

```rust
use kwavers::solver::forward::pstd::{PSTDSolver, PSTDConfig};
use kwavers::solver::forward::pstd::config::{
    BoundaryConfig, CompatibilityMode, KSpaceMethod,
};
use kwavers::solver::forward::pstd::numerics::spectral_correction::{
    CorrectionMethod, SpectralCorrectionConfig,
};
use kwavers::domain::source::GridSource;
```

- [ ] Imports added
- [ ] Code compiles: `cargo check -p pykwavers`

### Step 1.2: Replace run_pstd() Stub (3 hours)

Location: `src/lib.rs` lines 1121-1130

**Delete:**
```rust
fn run_pstd<'py>(&self, _py: Python<'py>, _time_steps: usize, _dt: Option<f64>) 
    -> PyResult<SimulationResult> 
{
    Err(PyRuntimeError::new_err(
        "PSTD solver not yet fully implemented. Use SolverType.FDTD instead."
    ))
}
```

**Replace with:** (See `PSTD_HYBRID_IMPLEMENTATION_GUIDE.md` Step 2, ~120 lines)

Key sections to implement:
1. CFL calculation (CFL = 0.5 for PSTD)
2. PSTDConfig creation with CPML boundary
3. PSTDSolver::new() instantiation
4. Multi-source injection loop
5. Time-stepping loop with `solver.step_forward()`
6. Sensor recording from `solver.pressure_field()`
7. NumPy array conversion and return

- [ ] Code copied and adapted from guide
- [ ] Builds without errors: `cargo build -p pykwavers --release`
- [ ] No warnings related to PSTD code

### Step 1.3: Create PSTD Tests (30 min)

Create file: `test_pstd_solver.py`

Tests to implement:
1. `test_pstd_initialization()` - Basic instantiation
2. `test_pstd_dispersion_free()` - Timing error <1%
3. `test_pstd_multi_source()` - Multiple sources work

- [ ] Test file created with 3 tests
- [ ] All tests pass: `pytest test_pstd_solver.py -v`
- [ ] Timing error <1% verified

### Step 1.4: Build and Install (10 min)

```bash
maturin develop --release
```

- [ ] Build successful (no errors)
- [ ] Can import in Python: `python -c "import pykwavers as kw; print(kw.SolverType.PSTD)"`
- [ ] PSTD solver runs: `python -c "import pykwavers as kw; sim = kw.Simulation(..., solver=kw.SolverType.PSTD); sim.run(10)"`

---

## Part 2: Hybrid Solver (2 hours)

### Step 2.1: Add Imports (5 min)

Location: Top of `src/lib.rs`

```rust
use kwavers::solver::forward::hybrid::{HybridSolver, HybridConfig};
use kwavers::solver::forward::hybrid::config::{
    DecompositionStrategy, OptimizationConfig, ValidationConfig,
};
use kwavers::solver::forward::fdtd::{FdtdConfig, SpatialOrder};
```

- [ ] Imports added
- [ ] Code compiles: `cargo check -p pykwavers`

### Step 2.2: Replace run_hybrid() Stub (1.5 hours)

Location: `src/lib.rs` lines 1133-1142

**Delete stub, replace with:** (See guide, ~100 lines)

Key sections:
1. CFL calculation (use FDTD's conservative CFL = 0.3)
2. PSTDConfig + FdtdConfig creation
3. HybridConfig with DecompositionStrategy::Smoothness
4. HybridSolver::new() instantiation
5. Source injection (may need ArcSourceWrapper - see guide)
6. Time-stepping loop with `solver.step_forward()`
7. Sensor recording and NumPy conversion

**Note:** Handle Arc→Box source conversion (see Common Issues section)

- [ ] Code implemented
- [ ] Builds without errors: `cargo build -p pykwavers --release`
- [ ] No warnings

### Step 2.3: Create Hybrid Tests (30 min)

Create file: `test_hybrid_solver.py`

Tests:
1. `test_hybrid_initialization()` - Basic instantiation
2. `test_hybrid_performance()` - Runs faster than FDTD
3. `test_hybrid_accuracy()` - Similar to PSTD in smooth media

- [ ] Test file created with 3 tests
- [ ] All tests pass: `pytest test_hybrid_solver.py -v`
- [ ] Performance between FDTD and PSTD verified

### Step 2.4: Rebuild (5 min)

```bash
maturin develop --release
```

- [ ] Build successful
- [ ] Hybrid imports: `python -c "import pykwavers as kw; print(kw.SolverType.Hybrid)"`
- [ ] Hybrid runs: Quick smoke test

---

## Part 3: Integration & Validation (1 hour)

### Step 3.1: Run Full Test Suite (30 min)

```bash
# All Phase 5 tests (should still pass)
pytest test_phase5_features.py -v

# New PSTD tests
pytest test_pstd_solver.py -v

# New Hybrid tests
pytest test_hybrid_solver.py -v

# All tests together
pytest -v
```

**Expected Results:**
- Phase 5: 18/18 pass
- PSTD: 3/3 pass
- Hybrid: 3/3 pass
- Total: 24/24 pass (or more if you added extras)

- [ ] All Phase 5 tests pass (18/18)
- [ ] All PSTD tests pass (3/3)
- [ ] All Hybrid tests pass (3/3)
- [ ] No test failures

### Step 3.2: Create All-Solvers Comparison (20 min)

Create file: `test_all_solvers.py`

Single test comparing all three:
```python
def test_fdtd_vs_pstd_vs_hybrid():
    # Run same problem with all 3 solvers
    # Compare timing errors and performance
    # PSTD should have <1% timing error
    # Hybrid should be between FDTD and PSTD
```

- [ ] Comparison test created
- [ ] Test passes
- [ ] Dispersion behavior documented in output

### Step 3.3: Build Release Wheel (10 min)

```bash
maturin build --release
ls target/wheels/
```

- [ ] Wheel built successfully
- [ ] Wheel filename correct (e.g., `pykwavers-0.1.0-*.whl`)
- [ ] Installable: `pip install --force-reinstall target/wheels/pykwavers-*.whl`

---

## Part 4: Documentation (30 min)

### Step 4.1: Update README.md (20 min)

Add section after "Quick Start":

```markdown
## Solver Selection

PyKwavers supports three solvers:

### FDTD (Finite-Difference Time-Domain)
- Dispersion: ~15% at standard resolution
- Best for: Complex boundaries, sharp interfaces
- Speed: 1.0× (baseline)

### PSTD (Pseudospectral Time-Domain)
- Dispersion: <1% (nearly dispersion-free)
- Best for: k-Wave comparison, timing-critical
- Speed: 0.8-1.2×

### Hybrid (PSTD + FDTD)
- Dispersion: ~5% (balanced)
- Best for: Mixed problems, performance
- Speed: 1.5-3×

### Usage
```python
import pykwavers as kw
sim = kw.Simulation(..., solver=kw.SolverType.PSTD)
result = sim.run(time_steps=1000)
```
```

- [ ] Solver section added to README
- [ ] Usage examples included
- [ ] Guidance table added

### Step 4.2: Create CHANGELOG Entry (10 min)

Add to `CHANGELOG.md` (or create if missing):

```markdown
## [Unreleased]

### Added
- PSTD solver Python bindings (dispersion-free propagation)
- Hybrid solver Python bindings (adaptive PSTD/FDTD)
- Multi-solver comparison tests
- Solver selection documentation

### Fixed
- PSTD/Hybrid stub errors replaced with full implementations
```

- [ ] CHANGELOG updated
- [ ] Version bump planned (e.g., 0.1.0 → 0.2.0)

---

## Final Checklist ✅

### Functionality
- [ ] PSTD solver runs without errors
- [ ] Hybrid solver runs without errors
- [ ] Multi-source support works with all solvers
- [ ] Sensor recording works correctly
- [ ] NumPy arrays returned properly

### Performance
- [ ] PSTD timing error <1%
- [ ] Hybrid performance between FDTD and PSTD
- [ ] No memory leaks (run long simulation)

### Code Quality
- [ ] No compiler warnings
- [ ] No clippy warnings: `cargo clippy -p pykwavers`
- [ ] Code formatted: `cargo fmt -p pykwavers`
- [ ] Python code formatted: `black test_*.py`

### Testing
- [ ] All existing tests pass (Phase 5: 18/18)
- [ ] New PSTD tests pass (3/3)
- [ ] New Hybrid tests pass (3/3)
- [ ] Integration test passes
- [ ] Dispersion validation complete

### Documentation
- [ ] README.md updated with solver selection
- [ ] CHANGELOG.md updated
- [ ] Implementation guide available
- [ ] Known issues documented

### Build & Deploy
- [ ] Development build works: `maturin develop --release`
- [ ] Release wheel builds: `maturin build --release`
- [ ] Wheel is installable
- [ ] Fresh environment test passes

---

## Success Criteria

**You're done when:**

1. ✅ Run this command successfully:
   ```bash
   python -c "
   import pykwavers as kw
   import numpy as np
   
   grid = kw.Grid(32, 32, 32, 0.1e-3, 0.1e-3, 0.1e-3)
   medium = kw.Medium.homogeneous(1500.0, 1000.0)
   source = kw.Source.plane_wave(grid, 1e6, 1e5)
   sensor = kw.Sensor.point((0.01, 0.01, 0.01))
   
   for solver_name, solver_type in [('FDTD', kw.SolverType.FDTD), 
                                      ('PSTD', kw.SolverType.PSTD),
                                      ('Hybrid', kw.SolverType.Hybrid)]:
       sim = kw.Simulation(grid, medium, source, sensor, solver=solver_type)
       result = sim.run(100)
       print(f'{solver_name}: {result.time_steps} steps completed')
   "
   ```

2. ✅ All tests pass:
   ```bash
   pytest -v
   # Expected: 24+ tests passing, 0 failures
   ```

3. ✅ PSTD dispersion validated:
   ```bash
   pytest test_pstd_solver.py::test_pstd_dispersion_free -v
   # Expected: Timing error < 1%
   ```

---

## Troubleshooting Quick Reference

### Issue: ArcSourceWrapper not found
**Solution:** See `PSTD_HYBRID_IMPLEMENTATION_GUIDE.md` Common Issues #1

### Issue: PSTD initialization fails
**Solution:** Check grid dimensions (prefer powers of 2), verify medium properties

### Issue: Hybrid source injection error
**Solution:** See guide for Arc→Box conversion patterns

### Issue: Tests fail with timing errors
**Expected:** FDTD ~15-30%, PSTD <1%, Hybrid <5%

---

## Time Tracking

| Task | Estimated | Actual | Notes |
|------|-----------|--------|-------|
| PSTD imports | 5 min | ___ | |
| PSTD implementation | 3.5 h | ___ | |
| PSTD tests | 30 min | ___ | |
| Hybrid imports | 5 min | ___ | |
| Hybrid implementation | 1.5 h | ___ | |
| Hybrid tests | 30 min | ___ | |
| Integration tests | 30 min | ___ | |
| Documentation | 30 min | ___ | |
| **Total** | **~7 h** | ___ | |

---

## Next Steps After Completion

1. **k-Wave Validation Suite** (4 hours)
   - Create automated comparison tests
   - Generate validation report
   - Document error metrics

2. **Performance Benchmarks** (2 hours)
   - Grid size scaling
   - Solver comparison plots
   - Memory usage analysis

3. **PyCoeus Integration Examples** (2 hours)
   - Migration guide
   - Best practices
   - Use case recommendations

---

## Resources

- **Full Audit:** `docs/PYCOEUS_SOLVER_AUDIT.md` (1133 lines)
- **Implementation Guide:** `PSTD_HYBRID_IMPLEMENTATION_GUIDE.md` (927 lines)
- **Integration Summary:** `docs/PYCOEUS_INTEGRATION_SUMMARY.md` (422 lines)
- **Phase 5 Plan:** `PHASE5_PLAN.md` (mathematical specs)

---

## Contact

**Questions?** ryanclanton@outlook.com | @ryancinsight  
**Issues?** GitHub with `[pykwavers]` tag  
**Sprint:** 217 Session 10 | **Date:** 2026-02-04

---

*Start time: ___________*  
*End time: ___________*  
*Total: ___________*  
*Status: [ ] Complete*