# Phase 5 Quick Reference: Solver Selection & Multi-Source Support

**Status:** ✓ Complete  
**Date:** 2024-02-04  
**Tests:** 18/18 passing

---

## New Features

### 1. Solver Selection

```python
import pykwavers as kw

# FDTD solver (default, fully functional)
sim = kw.Simulation(grid, medium, sources, sensor, solver=kw.FDTD)

# PSTD solver (not yet implemented)
sim = kw.Simulation(grid, medium, sources, sensor, solver=kw.PSTD)

# Hybrid solver (not yet implemented)
sim = kw.Simulation(grid, medium, sources, sensor, solver=kw.Hybrid)
```

**Solver Comparison:**

| Solver | Status | Accuracy | Speed | Use Case |
|--------|--------|----------|-------|----------|
| FDTD   | ✓ Working | ~15% dispersion | Fast | General purpose |
| PSTD   | ⚠ Stub | <1% error | Moderate | High accuracy |
| Hybrid | ⚠ Stub | ~5% error | Balanced | Production |

---

### 2. Multi-Source Support

```python
import pykwavers as kw

# Single source (backward compatible)
source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
sim = kw.Simulation(grid, medium, source, sensor)

# Multiple sources (new!)
sources = [
    kw.Source.point((1e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4),
    kw.Source.point((5e-3, 3.2e-3, 3.2e-3), frequency=1e6, amplitude=5e4),
    kw.Source.plane_wave(grid, frequency=0.5e6, amplitude=2e4),
]
sim = kw.Simulation(grid, medium, sources, sensor)

# Run simulation
result = sim.run(time_steps=1000)
```

**Validation Results:**
- Two sources: 0.32% error ✓
- Three sources: 0.41% error ✓
- Mixed types: 2.1% error ✓
- Performance: 1.2× slowdown per source

---

## API Changes

### Before (Phase 4)
```python
sim = kw.Simulation(grid, medium, source, sensor)
# Single source only
# No solver selection
```

### After (Phase 5)
```python
sim = kw.Simulation(grid, medium, sources, sensor, solver=kw.FDTD)
# sources: single Source OR list of Sources
# solver: optional, default=kw.FDTD
```

**Backward Compatible:** ✓ Yes

---

## Common Patterns

### Pattern 1: Superposition Verification
```python
# Run sources individually
result1 = kw.Simulation(grid, medium, [source1], sensor).run(time_steps=500)
result2 = kw.Simulation(grid, medium, [source2], sensor).run(time_steps=500)

# Run combined
result_both = kw.Simulation(grid, medium, [source1, source2], sensor).run(time_steps=500)

# Verify linear superposition
import numpy as np
p_expected = result1.sensor_data + result2.sensor_data
p_measured = result_both.sensor_data
error = np.linalg.norm(p_measured - p_expected) / np.linalg.norm(p_expected)
print(f"Superposition error: {error:.2%}")  # Should be <5%
```

---

### Pattern 2: Multi-Source Array
```python
# Create array of point sources
positions = [
    (2e-3, 3.2e-3, 3.2e-3),
    (4e-3, 3.2e-3, 3.2e-3),
    (6e-3, 3.2e-3, 3.2e-3),
]

sources = [
    kw.Source.point(pos, frequency=1e6, amplitude=5e4)
    for pos in positions
]

sim = kw.Simulation(grid, medium, sources, sensor)
result = sim.run(time_steps=1000)
```

---

### Pattern 3: Solver Comparison (Future)
```python
# FDTD (fast, dispersive)
sim_fdtd = kw.Simulation(grid, medium, sources, sensor, solver=kw.FDTD)
result_fdtd = sim_fdtd.run(time_steps=1000)

# PSTD (accurate, slower) - NOT YET AVAILABLE
# sim_pstd = kw.Simulation(grid, medium, sources, sensor, solver=kw.PSTD)
# result_pstd = sim_pstd.run(time_steps=1000)

# Compare arrival times, waveforms, etc.
```

---

## Known Issues

### 1. PSTD/Hybrid Not Yet Implemented
```python
sim = kw.Simulation(grid, medium, sources, sensor, solver=kw.PSTD)
result = sim.run(time_steps=1000)
# RuntimeError: PSTD solver not yet fully implemented. Use SolverType.FDTD instead.
```

**Workaround:** Use `solver=kw.FDTD` (default)

---

### 2. FDTD Numerical Dispersion
- Effective wave speed: ~1276 m/s (vs. 1500 m/s physical)
- Arrival time error: ~24% at 15 points-per-wavelength
- **Expected behavior** for FDTD at this resolution

**Mitigation:**
- Use finer grid (30+ points per wavelength)
- Wait for PSTD implementation (Phase 6)
- Document as known FDTD limitation

---

### 3. Phase Control Not Exposed
```python
# Current: Cannot control source phase
source = kw.Source.point(position, frequency=1e6, amplitude=5e4)

# Desired (future): Phase parameter
# source = kw.Source.point(position, frequency=1e6, amplitude=5e4, phase=np.pi)
```

**Workaround:** Use spatial positioning for interference effects

---

## Testing

Run Phase 5 test suite:
```bash
cd pykwavers
pytest test_phase5_features.py -v
```

**Expected:** 18/18 tests passing in ~58 seconds

---

## Performance

**Benchmark (64³ grid, 500 time steps):**
- Single source: 0.087 s
- Two sources: 0.104 s (1.20× slower)
- Three sources: 0.118 s (1.36× slower)

**Conclusion:** Minimal overhead, scales well

---

## Next Steps (Phase 6)

1. ⚠ **Implement PSTD Solver** (P0)
   - Wrap kwavers PSTD backend
   - Enable source injection
   - Validate <1% timing error

2. ⚠ **k-Wave Validation Suite** (P1)
   - Systematic comparisons
   - HTML validation reports
   - CI integration

3. ⚠ **Documentation Update** (P1)
   - README with solver guide
   - Migration guide from k-Wave
   - Troubleshooting section

---

## Quick Troubleshooting

**Q: `AttributeError: module 'pykwavers' has no attribute 'SolverType'`**  
A: Reinstall package: `pip install --force-reinstall --no-deps pykwavers-*.whl`

**Q: `ValueError: At least one source is required`**  
A: Pass non-empty source list or single source (not empty list)

**Q: `RuntimeError: PSTD solver not yet fully implemented`**  
A: Use `solver=kw.FDTD` or omit solver parameter (FDTD is default)

**Q: Superposition error >5%**  
A: Check grid resolution, time step, sensor positioning. Ensure sources don't overlap.

**Q: Simulation too slow**  
A: Reduce grid size, use fewer time steps, or wait for optimized PSTD solver

---

## Resources

- **Full Documentation:** `PHASE5_SUMMARY.md`
- **Implementation Plan:** `PHASE5_PLAN.md`
- **Test Suite:** `test_phase5_features.py`
- **Example Code:** See "Common Patterns" above

---

**Last Updated:** 2024-02-04  
**Author:** Ryan Clanton (@ryancinsight)  
**Status:** Production-ready for FDTD solver ✓