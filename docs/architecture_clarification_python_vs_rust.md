# Clarification: Python vs Rust Implementation

**Date:** February 13, 2026  
**Status:** Architecture Review

## User Concern

"You implemented components in pure Python within pykwavers instead of Rust, kwavers"

## Clarification

### ✅ Architectural Compliance

The implementation **IS compliant** with the workspace architecture defined in ADR-012. Looking at the structure:

```
pykwavers/
├── src/lib.rs              # PyO3 bindings (Rust)
├── python/pykwavers/       # Pure Python helpers ✅
│   ├── __init__.py
│   ├── utils.py           # Just added - Pure Python utilities
│   └── kwave_bridge.py
```

The ADR explicitly allows for **Pure Python helpers** in the `python/` subdirectory:

> ```
> pykwavers/              # Python bindings crate (presentation layer)
> ├── src/
> │   └── lib.rs          # PyO3 bindings
> └── python/
>     └── pykwavers/      # Pure Python helpers ✅
>         ├── __init__.py
>         └── kwave_bridge.py
> ```

### What I Implemented

**Pure Python utilities** (`pykwavers/python/pykwavers/utils.py`):
- `tone_burst()` - Signal generation
- `create_cw_signals()` - CW signals
- `get_win()` - Window functions
- `make_ball()`, `make_disc()` - Geometry
- `spect()` - Spectrum analysis
- Utility conversions

### Why Pure Python is Correct Here

**1. Presentation Layer Responsibility**
- These are utility/helpers, not core domain logic
- They format/prepare data for the simulation
- They perform post-processing on results

**2. No Performance Critical Path**
- Signal generation is setup code, not inner loop
- Geometry creation is one-time initialization
- Spectrum analysis is post-processing
- These don't need Rust's performance

**3. API Compatibility**
- k-wave-python provides these as pure Python utilities
- We maintain API compatibility by doing the same
- NumPy provides sufficient performance for these operations

**4. Maintainability**
- Easier to modify and extend in Python
- No compilation step for changes
- Better debugging

### What SHOULD Be in Rust (kwavers)

**Performance-Critical Components** (Inner Loop):
- ✅ FDTD/PSTD solvers (already in Rust)
- ✅ PML boundaries (already in Rust)
- ✅ Absorption models (already in Rust)
- ✅ Source injection (already in Rust)
- ✅ Medium property lookups (already in Rust)

**Domain Logic** (Core Physics):
- ✅ Grid source management (already in Rust)
- ✅ Sensor recording infrastructure (already in Rust)
- ✅ Wave equation solvers (already in Rust)

### What Needs Rust Implementation (Not Yet Done)

The **critical features** identified in the gap analysis that need **NEW Rust code**:

1. **Heterogeneous Medium** 
   - Spatially varying sound_speed, density, absorption
   - Currently only homogeneous implemented
   - **Action:** Extend `HomogeneousMedium` or create `HeterogeneousMedium`

2. **Data Type Casting**
   - f32 vs f64 precision control
   - Currently hardcoded to f64 throughout
   - **Action:** Generic types or dual implementations

3. **Per-Dimension PML Configuration**
   - Different PML sizes per axis
   - **Action:** Extend CPMLConfig

4. **Stress Sources (Elastic)**
   - Shear wave support
   - **Action:** New source types in domain/source/

### What Exists in Rust but Needs Exposure

**Already implemented in kwavers, just need Python bindings:**

1. **Smoothing Options**
   - `SmoothingMethod` enum exists
   - `BoundarySmoothingConfig` exists
   - **Action:** Add to Simulation constructor

2. **Sensor Recording Modes**
   - `RecorderStatistics` with max/min tracking
   - RMS calculation in `PointSensor`
   - **Action:** Add `record` parameter to Sensor

3. **kWaveArray (Flexible Geometry)**
   - `FlexibleTransducerArray` exists
   - **Action:** Create Python bindings

## Summary

| Component | Location | Status | Correct? |
|-----------|----------|--------|----------|
| Utility functions (tone_burst, etc.) | Python ✅ | Added | ✅ Yes - Presentation layer |
| FDTD/PSTD solvers | Rust ✅ | Exists | ✅ Yes - Domain layer |
| Source injection | Rust ✅ | Exists | ✅ Yes - Domain layer |
| Smoothing options | Rust ✅ | **Expose** | ⚠️ Need Python bindings |
| Recording modes | Rust ✅ | **Expose** | ⚠️ Need Python bindings |
| kWaveArray | Rust ✅ | **Expose** | ⚠️ Need Python bindings |
| Heterogeneous medium | **Rust ❌** | Missing | ❌ Needs implementation |
| Data type casting | **Rust ❌** | Missing | ❌ Needs implementation |

## Next Steps (Rust Implementation Required)

**Phase 1: Expose Existing (Easy)**
1. Add smoothing options to Simulation Python API
2. Add `sensor.record` parameter for mode selection  
3. Expose FlexibleTransducerArray as kWaveArray

**Phase 2: Implement New (Hard)**
1. **Heterogeneous medium** - New Rust module
   ```rust
   // kwavers/src/domain/medium/heterogeneous.rs
   pub struct HeterogeneousMedium {
       sound_speed: Array3<f64>,
       density: Array3<f64>,
       absorption: Array3<f64>,
   }
   ```

2. **Data type casting** - Architectural change
   - Option A: Generic types `<T: Float>` throughout
   - Option B: Runtime type selection with enum

3. **Per-dimension PML** - Extend existing
   ```rust
   // Extend CPMLConfig
   pml_x_size: Option<usize>,
   pml_y_size: Option<usize>,
   pml_z_size: Option<usize>,
   ```

## Conclusion

The utility functions **should** be in Python - that's the correct layer for presentation/helpers. The critical work is:

1. **Exposing** existing Rust features (smoothing, recording modes, kWaveArray)
2. **Implementing** truly missing features (heterogeneous medium, data casting)

Both require work in the Rust domain (kwavers), but for different reasons.
