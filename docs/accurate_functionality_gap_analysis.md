# Accurate Functionality Gap Analysis: kwavers vs k-wave-python

**Date:** February 13, 2026  
**Purpose:** Distinguish between features implemented in kwavers (Rust) vs exposed in pykwavers (Python)

## Legend
- ✅ **Exposed** - Available in pykwavers Python API
- 🔧 **Implemented** - Exists in kwavers Rust but NOT exposed to Python
- ❌ **Missing** - Not implemented anywhere
- ⚠️ **Partial** - Partially implemented

---

## 1. Source Types

### 1.1 Initial Pressure (p0) - Photoacoustic Sources
**Status:** 🔧 **IMPLEMENTED but NOT EXPOSED**

**kwavers Implementation:**
- Location: `kwavers/src/domain/source/grid_source.rs`
- Field: `GridSource.p0: Option<Array3<f64>>`
- Also: `domain/imaging/photoacoustic.rs` - `InitialPressure` struct

**pykwavers Status:** ✅ **EXPOSED via `Source.from_initial_pressure()`**
```python
source = kw.Source.from_initial_pressure(p0_array)
```

**Verified:** Working in pykwavers/src/lib.rs lines 607-632

### 1.2 Velocity Sources (ux, uy, uz)
**Status:** 🔧 **IMPLEMENTED but NOT FULLY EXPOSED**

**kwavers Implementation:**
- Location: `kwavers/src/domain/source/grid_source.rs`
- Fields: `u_mask`, `u_signal` (stores ux, uy, uz)
- Modes: `u_mode` with SourceMode enum
- Components: `SourceField` enum (VelocityX, VelocityY, VelocityZ)

**pykwavers Status:** ✅ **EXPOSED via `Source.from_velocity_mask()`**
```python
source = kw.Source.from_velocity_mask(
    mask, ux=signal_x, uy=signal_y, uz=signal_z
)
```

**Verified:** Working in pykwavers/src/lib.rs lines 635-733

### 1.3 Stress Sources (sxx, syy, szz, sxy, sxz, syz)
**Status:** ❌ **MISSING**

**kwavers:** No implementation found for stress tensor sources

**Impact:** Cannot simulate shear waves in elastic media

### 1.4 Source Modes
**Status:** ⚠️ **PARTIAL**

**kwavers:** `SourceMode` enum with:
- `Additive` ✅
- `AdditiveNoCorrection` ❌ (not implemented)
- `Dirichlet` ✅

**pykwavers Status:** Only "additive" and "dirichlet" exposed

---

## 2. Medium Properties

### 2.1 Power Law Absorption (alpha_coeff, alpha_power)
**Status:** ✅ **EXPOSED**

**kwavers Implementation:**
- `AbsorptionMode::PowerLaw { alpha_coeff, alpha_power }`
- Location: `physics/acoustics/mechanics/absorption.rs`
- PSTD solver integration: `solver/forward/pstd/physics/absorption.rs`

**pykwavers Status:** ✅ **EXPOSED via Medium.homogeneous()**
```python
medium = kw.Medium.homogeneous(
    sound_speed=1540.0,
    density=1060.0,
    absorption=0.5,      # alpha_coeff
    alpha_power=1.5      # power law exponent
)
```

**Verified:** Working in pykwavers/src/lib.rs lines 345-378

### 2.2 Nonlinear Medium (BonA)
**Status:** ✅ **EXPOSED**

**kwavers Implementation:**
- `AcousticPropertyData.nonlinearity_parameter` (B/A)
- Westervelt equation support in PSTD solver

**pykwavers Status:** ✅ **EXPOSED via Medium.homogeneous()**
```python
medium = kw.Medium.homogeneous(
    sound_speed=1540.0,
    density=1060.0,
    nonlinearity=6.0    # B/A parameter
)
```

**Verified:** Working in pykwavers/src/lib.rs

### 2.3 Heterogeneous Medium
**Status:** ❌ **MISSING**

**kwavers:** Currently only homogeneous medium is implemented
- No spatially varying sound_speed, density, or absorption

**Impact:** Cannot model realistic tissue structures

---

## 3. Sensor Recording Modes

### 3.1 Recordable Parameters
**Status:** ⚠️ **PARTIAL**

**kwavers Implementation:**
- ✅ Pressure recording: `extract_recorded_sensor_data()`
- 🔧 Max/Min tracking: `RecorderStatistics` in `domain/sensor/recorder/statistics.rs`
- 🔧 RMS calculation: `PointSensor.rms_pressure()` in `domain/sensor/point.rs`
- 🔧 Final field: Available from solver final state

**pykwavers Status:** ❌ Only pressure "p" is returned in SimulationResult
- No `sensor.record` list to select parameters
- No p_max, p_min, p_rms, p_final, u, u_max, etc.

**GAP:** Recording mode selection not exposed

### 3.2 Sensor Directivity
**Status:** 🔧 **IMPLEMENTED but NOT EXPOSED**

**kwavers Implementation:**
- `kSensorDirectivity` struct in k-wave-python/kwave/ksensor.py equivalent
- Actually: No directivity found in kwavers domain/sensor/

**Correction:** ❌ **NOT IMPLEMENTED**

### 3.3 Frequency Response
**Status:** ❌ **MISSING**

**kwavers:** No sensor frequency response filtering

---

## 4. Simulation Options

### 4.1 Data Type Casting
**Status:** ❌ **MISSING**

**kwavers:** All arrays are hardcoded to `f64` (double precision)
- No `data_cast` option for single precision

**Impact:** Cannot optimize memory/speed with f32

### 4.2 PML Configuration
**Status:** ⚠️ **PARTIAL**

**kwavers:** CPML implementation exists
- Basic `pml_size` parameter exposed

**Missing in pykwavers:**
- Per-dimension PML sizes (pml_x_size, pml_y_size, pml_z_size)
- PML absorption alpha (pml_alpha, pml_x_alpha, etc.)
- PML auto-sizing (pml_auto)
- Multi-axial PML ratio

### 4.3 Smoothing Options
**Status:** 🔧 **IMPLEMENTED but NOT EXPOSED**

**kwavers Implementation:**
- Location: `domain/boundary/smoothing/mod.rs`
- Methods: `SmoothingMethod` enum with None, Subgrid, GhostCell, ImmersedInterface
- Config: `BoundarySmoothingConfig`
- PSTD: `smooth_sources` boolean in config

**pykwavers Status:** ❌ Not exposed in Simulation options

### 4.4 Cartesian Interpolation
**Status:** ❌ **MISSING**

**kwavers:** No Cartesian sensor point interpolation

### 4.5 Source Scaling
**Status:** Unknown

**kwavers:** May exist but needs verification

### 4.6 K-Space Correction Toggle
**Status:** Unknown

**kwavers:** May be configurable in solver configs

### 4.7 Stream to Disk
**Status:** ❌ **MISSING**

**kwavers:** No streaming capability for large simulations

---

## 5. Transducer Arrays

### 5.1 kWaveArray (Flexible Geometry)
**Status:** 🔧 **IMPLEMENTED but NOT EXPOSED**

**kwavers Implementation:**
- `FlexibleTransducerArray` in `domain/source/flexible/array.rs`
- Features:
  - Arc elements
  - Rectangular elements  
  - Disc elements
  - Position and rotation control
  - Calibration (self-calibration, external tracking)
  - Deformation modeling

**pykwavers Status:** ❌ Not exposed
- Only `TransducerArray2D` (linear arrays) is exposed

### 5.2 Linear Array (TransducerArray2D)
**Status:** ✅ **EXPOSED**

**pykwavers:** `kw.TransducerArray2D` class
- Electronic steering ✅
- Electronic focusing ✅
- Apodization ✅
- Active element masking ✅

---

## 6. Utility Functions

### 6.1 Signal Processing
**Status:** ❌ **MISSING**

**kwavers:** No tone_burst, create_cw_signals, window functions exposed

### 6.2 Map Generation
**Status:** ❌ **MISSING**

**kwavers:** No make_ball, make_disc, make_bowl, etc. exposed

### 6.3 Conversion Utilities
**Status:** ❌ **MISSING**

**kwavers:** No cart2grid, db2neper exposed

---

## 7. Reconstruction Algorithms
**Status:** ❌ **MISSING**

**kwavers:** No time reversal, k-space reconstruction implemented

---

## 8. Simulation Types

### 8.1 Axisymmetric
**Status:** ❌ **MISSING**

### 8.2 Elastic
**Status:** ❌ **MISSING**

### 8.3 1D
**Status:** ❌ **MISSING**

---

## 9. GPU Acceleration
**Status:** ❌ **MISSING**

**kwavers:** No GPU solver implementations

---

## Summary: What's Actually Missing

### Critical Gaps (Need Implementation):
1. **Heterogeneous Medium** - Spatially varying properties
2. **Sensor Recording Modes** - p_max, p_min, p_rms, p_final selection
3. **Data Type Casting** - f32/f64 precision control
4. **PML Configuration** - Per-dimension control, alpha values
5. **kWaveArray Exposure** - Flexible geometry to Python

### High Priority (Expose Existing):
1. **Smoothing Options** - Already in kwavers, expose to Python
2. **Velocity Sources** - Already exposed via from_velocity_mask
3. **p0 Sources** - Already exposed via from_initial_pressure
4. **Absorption** - Already exposed

### Medium Priority (New Features):
1. **Stress Sources** - For elastic waves
2. **Sensor Directivity** - Directional response
3. **Utility Functions** - tone_burst, map generators
4. **Stream to Disk** - Large simulations

### Low Priority:
1. **Reconstruction** - Time reversal, k-space
2. **GPU Acceleration**
3. **Axisymmetric/Elastic simulations**

---

## Verified Working in pykwavers

✅ Source.from_mask() - Time-varying pressure  
✅ Source.from_initial_pressure() - p0  
✅ Source.from_velocity_mask() - Velocity sources  
✅ Source.point() - Point source  
✅ Source.plane_wave() - Plane wave  
✅ Medium.homogeneous() - With absorption and nonlinearity  
✅ TransducerArray2D - Linear arrays with beamforming  
✅ Sensor.point() - Point sensor  
✅ Sensor.from_mask() - Multi-sensor  
✅ Simulation.run() - Basic execution  

---

## Next Steps for Implementation

### Phase 1: Expose Existing Features
1. Add `record` parameter to Sensor class
2. Add smoothing options to Simulation
3. Add per-dimension PML configuration

### Phase 2: Implement Critical Missing Features
1. Heterogeneous medium (spatially varying maps)
2. Data type casting (f32 support)
3. Complete recording modes infrastructure

### Phase 3: Enhanced Features
1. kWaveArray Python bindings
2. Utility functions (tone_burst, etc.)
3. Stress sources for elastic waves
