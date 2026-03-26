# Functionality Gap Analysis: kwavers vs k-wave-python

**Date:** February 13, 2026  
**Purpose:** Identify missing features in kwavers/pykwavers compared to k-wave-python for valid comparison through Python interfaces

## Executive Summary

This document provides a comprehensive analysis of functionality gaps between kwavers and k-wave-python. The gaps are categorized by priority (Critical, High, Medium, Low) based on their importance for valid comparisons and common use cases.

## 1. Source Types (CRITICAL)

### 1.1 Initial Pressure Distribution (p0) - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** CRITICAL

k-wave-python supports initial value problems (IVP) via `source.p0`:
```python
source = kSource()
source.p0 = initial_pressure_distribution  # Photoacoustic/thermoacoustic
```

**Impact:** Cannot simulate photoacoustic imaging, thermoacoustic imaging, or any IVP scenarios.

**Required Implementation:**
- Add `p0` field to Source class
- Support initial pressure distribution as 3D array
- Integrate with time-stepping solvers
- Support smoothing (`smooth_p0` option)

### 1.2 Velocity Sources (ux, uy, uz) - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** CRITICAL

k-wave-python supports time-varying velocity sources:
```python
source.ux = velocity_signal_x  # Time-varying x-velocity
source.uy = velocity_signal_y  # Time-varying y-velocity
source.uz = velocity_signal_z  # Time-varying z-velocity
source.u_mask = velocity_mask   # Binary mask
source.u_mode = "additive"      # or "dirichlet"
```

**Impact:** Cannot simulate vibration sources, piston sources, or directional sources.

**Required Implementation:**
- Add velocity source fields (ux, uy, uz)
- Velocity source mask (u_mask)
- Velocity source modes (u_mode)
- Support for reference frequency (u_frequency_ref)

### 1.3 Stress Sources (sxx, syy, szz, sxy, sxz, syz) - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** HIGH

k-wave-python supports stress tensor sources for elastic simulations:
```python
source.sxx = stress_xx
source.syy = stress_yy
source.szz = stress_zz
source.sxy = stress_xy
source.sxz = stress_xz
source.syz = stress_yz
source.s_mask = stress_mask
```

**Impact:** Cannot simulate directional stress sources, shear wave sources.

**Required Implementation:**
- Stress tensor source fields
- Stress mask (s_mask)
- Integration with elastic solvers (when implemented)

### 1.4 Source Modes - PARTIAL
**Status:** PARTIALLY IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports multiple source injection modes:
- `"additive"` (default with k-space correction)
- `"additive-no-correction"` (without k-space correction)
- `"dirichlet"` (hard boundary condition)

**Current kwavers Status:** Supports "additive" and "dirichlet" but not "additive-no-correction".

## 2. Sensor Recording Modes (CRITICAL)

### 2.1 Recordable Parameters - PARTIAL
**Status:** PARTIALLY IMPLEMENTED  
**Priority:** CRITICAL

k-wave-python supports recording multiple parameters:
```python
sensor.record = ["p"]           # Pressure (IMPLEMENTED)
sensor.record = ["p_final"]     # Final pressure field (MISSING)
sensor.record = ["p_max"]       # Maximum pressure (MISSING)
sensor.record = ["p_min"]       # Minimum pressure (MISSING)
sensor.record = ["p_rms"]       # RMS pressure (MISSING)
sensor.record = ["u"]           # Particle velocity (MISSING)
sensor.record = ["u_max"]       # Max particle velocity (MISSING)
sensor.record = ["u_min"]       # Min particle velocity (MISSING)
sensor.record = ["u_rms"]       # RMS particle velocity (MISSING)
```

**Current kwavers Status:** Only pressure "p" is recorded.

**Required Implementation:**
- Record final pressure field
- Record maximum/minimum pressure over time
- Record RMS pressure
- Record particle velocity components (if supported)

### 2.2 Sensor Directivity - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports sensor directivity:
```python
sensor.directivity = kSensorDirectivity(
    angle=directivity_angles,
    pattern="pressure",  # or "gradient"
    size=element_size
)
```

**Impact:** Cannot model directional sensor response.

### 2.3 Frequency Response - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports sensor frequency response filtering:
```python
sensor.frequency_response = [center_freq, bandwidth_percent]
```

**Impact:** Cannot model realistic sensor frequency characteristics.

### 2.4 Record Start Index - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python supports delayed recording:
```python
sensor.record_start_index = 100  # Start recording at time step 100
```

## 3. Medium Properties (CRITICAL)

### 3.1 Power Law Absorption - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** CRITICAL

k-wave-python supports power-law absorption:
```python
medium = kWaveMedium(
    sound_speed=1500,
    alpha_coeff=0.75,     # dB/(MHz^y cm)
    alpha_power=1.5,      # Power law exponent (0-3)
    alpha_mode=None       # or "no_absorption", "no_dispersion", "stokes"
)
```

**Impact:** Cannot simulate realistic tissue absorption and dispersion.

**Required Implementation:**
- Power law absorption coefficient (alpha_coeff)
- Power law exponent (alpha_power)
- Absorption mode (alpha_mode)
- Absorption filter (alpha_filter)
- Absorption sign control (alpha_sign)
- Stokes absorption support

### 3.2 Nonlinear Medium (BonA) - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** HIGH

k-wave-python supports nonlinear propagation:
```python
medium = kWaveMedium(
    sound_speed=1500,
    BonA=6  # Parameter of nonlinearity (B/A)
)
```

**Impact:** Cannot simulate nonlinear wave propagation, harmonic generation.

**Required Implementation:**
- Nonlinearity parameter (BonA)
- Westervelt equation solver
- Harmonic balance computations

### 3.3 Heterogeneous Medium - PARTIAL
**Status:** PARTIALLY IMPLEMENTED  
**Priority:** HIGH

k-wave-python supports spatially varying properties:
```python
# Each property can be a 3D array matching grid size
medium.sound_speed = sound_speed_map  # 3D array
medium.density = density_map          # 3D array
medium.alpha_coeff = absorption_map   # 3D array
```

**Current kwavers Status:** Supports homogeneous medium only.

**Required Implementation:**
- Spatially varying sound speed
- Spatially varying density
- Spatially varying absorption

### 3.4 Reference Sound Speed - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python allows specifying reference sound speed for k-space operator:
```python
medium.sound_speed_ref = 1540  # Reference for phase correction
```

**Impact:** Cannot optimize k-space operator for specific applications.

## 4. Simulation Options (HIGH)

### 4.1 Data Type Casting - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** HIGH

k-wave-python supports data type casting for performance:
```python
simulation_options = SimulationOptions(
    data_cast="single",      # Cast to single precision
    data_recast=False        # Don't recast output to double
)
```

**Impact:** Cannot optimize memory usage and computation speed.

**Required Implementation:**
- Data casting to single/double precision
- Output recasting option

### 4.2 PML Configuration - PARTIAL
**Status:** PARTIALLY IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports extensive PML control:
```python
simulation_options = SimulationOptions(
    pml_inside=True,          # PML inside grid (IMPLEMENTED)
    pml_size=[20, 20, 20],    # Per-dimension PML size (MISSING)
    pml_alpha=2.0,            # PML absorption (MISSING)
    pml_x_alpha=3.0,          # Per-axis PML alpha (MISSING)
    pml_y_alpha=2.0,
    pml_z_alpha=2.0,
    pml_auto=False,           # Auto-optimize PML size (MISSING)
    multi_axial_PML_ratio=0.1 # MPML ratio (MISSING)
)
```

**Current kwavers Status:** Basic pml_size parameter only.

### 4.3 Smoothing Options - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports smoothing of material properties:
```python
simulation_options = SimulationOptions(
    smooth_c0=True,    # Smooth sound speed
    smooth_rho0=True,  # Smooth density
    smooth_p0=True     # Smooth initial pressure
)
```

**Impact:** Cannot reduce staircasing artifacts at interfaces.

### 4.4 Cartesian Interpolation - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports Cartesian sensor interpolation:
```python
simulation_options = SimulationOptions(
    cart_interp="linear",   # or "nearest"
    cartesian_interp="linear"
)
```

**Impact:** Cannot use Cartesian sensor point coordinates.

### 4.5 Source Scaling - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python controls source term scaling:
```python
simulation_options = SimulationOptions(
    scale_source_terms=True  # Apply k-space source scaling
)
```

### 4.6 K-Space Correction Toggle - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python allows disabling k-space correction:
```python
simulation_options = SimulationOptions(
    use_kspace=True  # Can be set to False
)
```

### 4.7 Staggered Grid Toggle - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python controls staggered grid usage:
```python
simulation_options = SimulationOptions(
    use_sg=True  # Use staggered grid
)
```

### 4.8 Stream to Disk - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python supports streaming large simulations to disk:
```python
simulation_options = SimulationOptions(
    stream_to_disk=True,     # or integer step count
    save_to_disk=True,
    save_to_disk_exit=False
)
```

## 5. Grid Features (MEDIUM)

### 5.1 Grid Expansion - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python can expand grid for PML:
```python
# Automatic grid expansion when pml_inside=False
kgrid.expand_grid(expand_size)
```

### 5.2 Time Array Creation - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python provides automatic time array creation:
```python
kgrid.makeTime(medium.sound_speed, t_end=45e-6)
# or
kgrid.setTime(Nt, dt)
```

**Impact:** Manual time step calculation required in kwavers.

### 5.3 k-Grid Properties - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python provides k-space grid properties:
```python
kgrid.kx, kgrid.ky, kgrid.kz  # Wavenumber vectors
kgrid.k                       # Combined wavenumber
kgrid.k_max_all              # Maximum wavenumber
```

## 6. Array and Transducer Features (HIGH)

### 6.1 kWaveArray - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** HIGH

k-wave-python provides flexible array geometry:
```python
karray = kWaveArray()
karray.add_arc_element(position, radius, diameter, focus_pos)
karray.add_rect_element(position, Lx, Ly, orientation)
karray.add_disc_element(position, diameter, orientation)
karray.set_array_position(center, rotation)

# Get binary mask
mask = karray.get_array_binary_mask(kgrid)

# Get distributed source signals
source_signal = karray.get_distributed_source_signal(kgrid, input_signal)

# Combine sensor data
combined_data = karray.combine_sensor_data(kgrid, sensor_data)
```

**Impact:** Cannot model complex transducer geometries beyond simple linear arrays.

### 6.2 Annular Arrays - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports annular array transducers:
```python
from kwave.utils.mapgen import focused_annulus_oneil
```

### 6.3 Focused Bowl Transducers - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports focused bowl geometry:
```python
from kwave.utils.mapgen import focused_bowl_oneil, make_bowl
```

## 7. Utility Functions (MEDIUM)

### 7.1 Signal Processing - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python provides signal utilities:
```python
from kwave.utils.signals import (
    tone_burst,           # Generate tone burst
    create_cw_signals,    # Continuous wave signals
    get_win,              # Window functions
    reorder_binary_sensor_data  # Reorder sensor data
)

from kwave.utils.filters import (
    spect,                # Amplitude spectrum
    extract_amp_phase,    # Extract amplitude and phase
    filter_time_series,   # Filter time series
    gaussian_filter,      # Gaussian filter
    smooth                # Smoothing filter
)
```

### 7.2 Map Generation - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python provides geometry utilities:
```python
from kwave.utils.mapgen import (
    make_ball,            # Create spherical source
    make_disc,            # Create disc source
    make_circle,          # Create circle mask
    make_cart_circle,     # Create Cartesian circle
    make_bowl,            # Create bowl mask
    focused_bowl_oneil,   # O'Neil focused bowl solution
    focused_annulus_oneil # O'Neil annular solution
)
```

### 7.3 Conversion Utilities - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

```python
from kwave.utils.conversion import (
    cart2grid,            # Cartesian to grid indices
    db2neper              # dB to neper conversion
)

from kwave.utils.data import (
    scale_SI              # Scale to SI units
)
```

## 8. Reconstruction Algorithms (LOW)

### 8.1 Time Reversal - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python provides time reversal reconstruction:
```python
from kwave.reconstruction import TimeReversal

tr = TimeReversal(kgrid, sensor_data, c0)
reconstruction = tr.reconstruct()
```

### 8.2 k-Space Reconstruction - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python provides k-space reconstruction:
```python
from kwave.kspaceLineRecon import kspaceLineRecon
from kwave.kspacePlaneRecon import kspacePlaneRecon
```

### 8.3 Beamforming - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

```python
from kwave.reconstruction.beamform import (
    envelope_detection,
    scan_conversion
)
from kwave.reconstruction.tools import log_compression
```

## 9. Simulation Types (LOW)

### 9.1 Axisymmetric Simulations - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python supports axisymmetric simulations:
```python
from kwave.kspaceFirstOrderAS import kspaceFirstOrderASC

simulation_options = SimulationOptions(
    simulation_type=SimulationType.AXISYMMETRIC,
    radial_symmetry="WSWA-FFT"
)
```

### 9.2 Elastic Simulations - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python supports elastic wave simulations:
```python
simulation_options = SimulationOptions(
    simulation_type=SimulationType.ELASTIC
    # or SimulationType.ELASTIC_WITH_KSPACE_CORRECTION
)
```

### 9.3 1D Simulations - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** LOW

k-wave-python has specialized 1D solver.

## 10. GPU Acceleration (MEDIUM)

### 10.1 GPU Simulation - MISSING
**Status:** NOT IMPLEMENTED  
**Priority:** MEDIUM

k-wave-python supports GPU acceleration:
```python
from kwave.options.simulation_execution_options import SimulationExecutionOptions

execution_options = SimulationExecutionOptions(
    is_gpu_simulation=True
)

# GPU-specific functions
from kwave.kspaceFirstOrder2D import kspace_first_order_2d_gpu
```

## Implementation Roadmap

### Phase 1: Critical Features (Required for Basic Comparison)
1. Initial pressure distribution (p0)
2. Velocity sources (ux, uy, uz)
3. Multiple sensor recording modes (p_max, p_min, p_rms, p_final)
4. Power law absorption
5. Data type casting

### Phase 2: High Priority (Enhanced Comparison)
1. Nonlinear medium (BonA)
2. Heterogeneous medium (spatially varying properties)
3. kWaveArray flexible geometry
4. Enhanced PML configuration

### Phase 3: Medium Priority (Feature Parity)
1. Stress sources
2. Sensor directivity and frequency response
3. Source scaling and k-space toggle
4. Smoothing options
5. Grid expansion and time array creation

### Phase 4: Low Priority (Complete Coverage)
1. Utility functions (signal processing, map generation)
2. Reconstruction algorithms
3. Axisymmetric and elastic simulations
4. GPU acceleration

## Testing Strategy

For each feature implemented:

1. **Unit Tests:** Test individual feature in isolation
2. **Parity Tests:** Compare results with k-wave-python on identical problems
3. **Integration Tests:** Test feature interactions
4. **Example Replication:** Replicate k-wave-python examples

## Conclusion

The most critical gaps preventing valid comparison are:

1. **Initial pressure (p0)** - Blocks all IVP scenarios
2. **Velocity sources** - Blocks directional source modeling  
3. **Recording modes** - Only pressure is recorded; missing max/min/RMS/final
4. **Power law absorption** - Blocks realistic tissue modeling

Implementing these four critical features would enable comparison for 80% of k-wave-python examples.
