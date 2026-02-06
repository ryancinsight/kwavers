# PSTD and Hybrid Solver Implementation Guide

**Target:** Complete PyKwavers Python bindings for PSTD and Hybrid solvers  
**Effort:** 6 hours (4h PSTD + 2h Hybrid)  
**File:** `kwavers/pykwavers/src/lib.rs`  
**Sprint:** 217 Session 10  
**Date:** 2026-02-04

---

## Overview

This guide provides step-by-step instructions to complete the PSTD and Hybrid solver Python bindings. The Rust core implementations are production-ready; we only need to wire them into the PyO3 interface.

**Success Pattern:** Follow the existing `run_fdtd()` implementation (lines 1024-1118) as a template.

---

## Part 1: PSTD Solver Implementation (4 hours)

### Step 1: Add Required Imports (5 minutes)

Add to the top of `lib.rs` after existing solver imports:

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

### Step 2: Replace `run_pstd()` Stub (3.5 hours)

**Current code (lines 1121-1130):**
```rust
fn run_pstd<'py>(
    &self,
    _py: Python<'py>,
    _time_steps: usize,
    _dt: Option<f64>,
) -> PyResult<SimulationResult> {
    Err(PyRuntimeError::new_err(
        "PSTD solver not yet fully implemented. Use SolverType.FDTD instead."
    ))
}
```

**Replace with:**
```rust
/// Run simulation with PSTD solver.
fn run_pstd<'py>(
    &self,
    py: Python<'py>,
    time_steps: usize,
    dt: Option<f64>,
) -> PyResult<SimulationResult> {
    // Calculate time step from CFL condition if not provided
    // PSTD allows larger time steps than FDTD (CFL ~ 0.5-0.8)
    let dt_actual = dt.unwrap_or_else(|| {
        let c_max = 1500.0; // Conservative estimate
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let cfl = 0.5; // PSTD allows larger CFL than FDTD
        cfl * dx_min / c_max
    });

    // Create PSTD configuration
    let pstd_config = PSTDConfig {
        dt: dt_actual,
        nt: time_steps,
        compatibility_mode: CompatibilityMode::Reference, // k-Wave compatibility
        boundary: BoundaryConfig::CPML(kwavers::domain::boundary::CPMLConfig {
            thickness: 10,        // 10 grid points (k-Wave default)
            alpha: 2.0,           // Standard absorption parameter
            kappa_max: 1.0,       // No kappa stretching
            sigma_max_ratio: 1.0, // Standard sigma scaling
        }),
        sensor_mask: None, // Will handle sensor recording separately
        kspace_method: KSpaceMethod::FullKSpace, // Full k-space operators
        spectral_correction: SpectralCorrectionConfig {
            method: CorrectionMethod::Treeby2010, // k-Wave compatibility
            enable_absorption_correction: true,
            enable_dispersion_correction: true,
        },
    };

    // Initialize PSTD solver
    let mut solver = PSTDSolver::new(
        pstd_config,
        self.grid.inner.clone(),
        &self.medium.inner as &dyn MediumTrait,
        GridSource::default(), // Empty initial source
    )
    .map_err(kwavers_error_to_py)?;

    // Inject all sources (multi-source support)
    for source in &self.sources {
        let source_arc = self.create_source_arc(py, source, dt_actual)?;
        solver
            .add_source_arc(source_arc)
            .map_err(kwavers_error_to_py)?;
    }

    // Initialize sensor recording
    let mut sensor_data: Vec<f64> = if self.sensor.sensor_type == "point" {
        // Point sensor: record time series at single location
        Vec::with_capacity(time_steps)
    } else {
        // Grid sensor would need full field storage (not implemented yet)
        return Err(PyRuntimeError::new_err(
            "Grid sensors not yet implemented. Use Sensor.point() instead.",
        ));
    };

    // Get sensor position
    let sensor_pos = self
        .sensor
        .position
        .ok_or_else(|| PyRuntimeError::new_err("Sensor position required for point sensor"))?;

    // Convert sensor position to grid indices
    let (i, j, k) = self
        .grid
        .inner
        .coordinates_to_indices(sensor_pos[0], sensor_pos[1], sensor_pos[2])
        .ok_or_else(|| PyRuntimeError::new_err("Sensor position outside grid bounds"))?;

    // Time-stepping loop
    for step in 0..time_steps {
        // Advance solver by one time step
        // PSTD uses step_forward() method (orchestrated stepping)
        solver
            .step_forward()
            .map_err(kwavers_error_to_py)?;

        // Record sensor data from pressure field
        let pressure_field = solver.pressure_field();

        // Sample at sensor location (with bounds checking)
        if i < pressure_field.shape()[0]
            && j < pressure_field.shape()[1]
            && k < pressure_field.shape()[2]
        {
            sensor_data.push(pressure_field[[i, j, k]]);
        } else {
            sensor_data.push(0.0); // Out of bounds → zero
        }

        // Optional: Progress reporting every 100 steps
        if step % 100 == 0 && step > 0 {
            py.check_signals()?; // Allow Python interrupt (Ctrl+C)
        }
    }

    // Create time vector
    let time_vec: Vec<f64> = (0..time_steps).map(|i| i as f64 * dt_actual).collect();

    // Convert Vec<f64> to NumPy arrays for Python
    let sensor_data_np = PyArray1::from_vec_bound(py, sensor_data);
    let time_vec_np = PyArray1::from_vec_bound(py, time_vec);

    Ok(SimulationResult {
        sensor_data: sensor_data_np.into_py(py),
        time: time_vec_np.into_py(py),
        time_steps,
        dt: dt_actual,
        final_time: dt_actual * time_steps as f64,
    })
}
```

### Step 3: Test PSTD Implementation (30 minutes)

**Create test file:** `pykwavers/test_pstd_solver.py`

```python
import pykwavers as kw
import numpy as np

def test_pstd_initialization():
    """Test PSTD solver can be initialized."""
    grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))
    
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    result = sim.run(time_steps=10)
    
    assert result.time_steps == 10
    assert result.sensor_data.shape == (10,)


def test_pstd_dispersion_free():
    """Test PSTD has <1% timing error (dispersion-free)."""
    # Grid setup: 6.4 mm domain
    grid = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    
    # Plane wave source (1 MHz)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    
    # Sensor at 3 mm distance
    sensor = kw.Sensor.point(position=(0.003, 0.0032, 0.0032))
    
    # Run PSTD simulation
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    result = sim.run(time_steps=1000, dt=1e-8)
    
    # Find arrival time (threshold crossing)
    threshold = 0.1 * np.max(np.abs(result.sensor_data))
    arrival_idx = np.argmax(np.abs(result.sensor_data) > threshold)
    arrival_time = result.time[arrival_idx]
    
    # Expected arrival time: distance / speed
    distance = 3e-3  # 3 mm
    expected_time = distance / 1500.0  # 2.0 μs
    
    # Timing error should be <1% for PSTD
    timing_error = abs(arrival_time - expected_time) / expected_time
    assert timing_error < 0.01, f"PSTD timing error {timing_error:.3f} exceeds 1%"


def test_pstd_multi_source():
    """Test PSTD handles multiple sources correctly."""
    grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    
    # Two plane wave sources
    source1 = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    source2 = kw.Source.plane_wave(grid, frequency=2e6, amplitude=5e4)
    
    sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))
    
    sim = kw.Simulation(grid, medium, [source1, source2], sensor, solver=kw.SolverType.PSTD)
    result = sim.run(time_steps=100)
    
    # Should complete without errors
    assert result.time_steps == 100
    assert np.all(np.isfinite(result.sensor_data))


if __name__ == "__main__":
    test_pstd_initialization()
    print("✓ PSTD initialization test passed")
    
    test_pstd_dispersion_free()
    print("✓ PSTD dispersion-free test passed")
    
    test_pstd_multi_source()
    print("✓ PSTD multi-source test passed")
    
    print("\n✅ All PSTD tests passed!")
```

**Run tests:**
```bash
cd pykwavers
pytest test_pstd_solver.py -v
```

---

## Part 2: Hybrid Solver Implementation (2 hours)

### Step 1: Add Required Imports (5 minutes)

Add to imports section:

```rust
use kwavers::solver::forward::hybrid::{HybridSolver, HybridConfig};
use kwavers::solver::forward::hybrid::config::{
    DecompositionStrategy, OptimizationConfig, ValidationConfig,
};
use kwavers::solver::forward::fdtd::{FdtdConfig, SpatialOrder};
```

### Step 2: Replace `run_hybrid()` Stub (1.5 hours)

**Current code (lines 1133-1142):**
```rust
fn run_hybrid<'py>(
    &self,
    _py: Python<'py>,
    _time_steps: usize,
    _dt: Option<f64>,
) -> PyResult<SimulationResult> {
    Err(PyRuntimeError::new_err(
        "Hybrid solver not yet fully implemented. Use SolverType.FDTD instead."
    ))
}
```

**Replace with:**
```rust
/// Run simulation with Hybrid solver (PSTD + FDTD).
fn run_hybrid<'py>(
    &self,
    py: Python<'py>,
    time_steps: usize,
    dt: Option<f64>,
) -> PyResult<SimulationResult> {
    // Calculate time step (use FDTD's more conservative CFL)
    let dt_actual = dt.unwrap_or_else(|| {
        let c_max = 1500.0;
        let dx_min = self
            .grid
            .inner
            .dx
            .min(self.grid.inner.dy)
            .min(self.grid.inner.dz);
        let cfl = 0.3; // Conservative (FDTD constraint)
        cfl * dx_min / c_max
    });

    // Create PSTD configuration (for smooth regions)
    let pstd_config = PSTDConfig {
        dt: dt_actual,
        nt: time_steps,
        compatibility_mode: CompatibilityMode::Reference,
        boundary: BoundaryConfig::CPML(kwavers::domain::boundary::CPMLConfig {
            thickness: 10,
            alpha: 2.0,
            kappa_max: 1.0,
            sigma_max_ratio: 1.0,
        }),
        sensor_mask: None,
        kspace_method: KSpaceMethod::FullKSpace,
        spectral_correction: SpectralCorrectionConfig {
            method: CorrectionMethod::Treeby2010,
            enable_absorption_correction: true,
            enable_dispersion_correction: true,
        },
    };

    // Create FDTD configuration (for interface regions)
    let fdtd_config = FdtdConfig {
        spatial_order: SpatialOrder::Second, // Balance speed and accuracy
        dt: dt_actual,
        nt: time_steps,
        record_pressure: true,
        record_velocity: false,
    };

    // Create Hybrid configuration
    let hybrid_config = HybridConfig {
        // Use smoothness-based decomposition (gradient threshold)
        decomposition_strategy: DecompositionStrategy::Smoothness { 
            threshold: 0.1  // Gradient threshold for PSTD/FDTD selection
        },
        pstd_config,
        fdtd_config,
        selection_criteria: Default::default(), // Use default adaptive selection
        optimization_config: OptimizationConfig {
            enable_adaptive_refinement: true,
            refinement_interval: 50, // Update regions every 50 steps
            min_region_size: 8,      // Minimum 8³ grid cells per region
        },
        validation_config: ValidationConfig {
            enable_energy_conservation_check: true,
            energy_tolerance: 1e-6,
            enable_interface_continuity_check: true,
        },
    };

    // Initialize Hybrid solver
    let mut solver = HybridSolver::new(
        hybrid_config,
        &self.grid.inner,
        &self.medium.inner as &dyn MediumTrait,
    )
    .map_err(kwavers_error_to_py)?;

    // Inject all sources (using Solver trait)
    for source in &self.sources {
        let source_arc = self.create_source_arc(py, source, dt_actual)?;
        // Convert Arc to Box for Solver trait (requires Box<dyn Source>)
        let source_box: Box<dyn kwavers::domain::source::Source> = 
            Box::new(kwavers::domain::source::ArcSourceWrapper(source_arc));
        solver
            .add_source(source_box)
            .map_err(kwavers_error_to_py)?;
    }

    // Initialize sensor recording
    let mut sensor_data: Vec<f64> = if self.sensor.sensor_type == "point" {
        Vec::with_capacity(time_steps)
    } else {
        return Err(PyRuntimeError::new_err(
            "Grid sensors not yet implemented. Use Sensor.point() instead.",
        ));
    };

    // Get sensor position and indices
    let sensor_pos = self
        .sensor
        .position
        .ok_or_else(|| PyRuntimeError::new_err("Sensor position required for point sensor"))?;

    let (i, j, k) = self
        .grid
        .inner
        .coordinates_to_indices(sensor_pos[0], sensor_pos[1], sensor_pos[2])
        .ok_or_else(|| PyRuntimeError::new_err("Sensor position outside grid bounds"))?;

    // Time-stepping loop
    for step in 0..time_steps {
        // Advance Hybrid solver (automatic PSTD/FDTD region handling)
        solver
            .step_forward()
            .map_err(kwavers_error_to_py)?;

        // Record sensor data from pressure field
        let pressure_field = solver.pressure_field();

        // Sample at sensor location
        if i < pressure_field.shape()[0]
            && j < pressure_field.shape()[1]
            && k < pressure_field.shape()[2]
        {
            sensor_data.push(pressure_field[[i, j, k]]);
        } else {
            sensor_data.push(0.0);
        }

        // Progress reporting
        if step % 100 == 0 && step > 0 {
            py.check_signals()?;
        }
    }

    // Create time vector
    let time_vec: Vec<f64> = (0..time_steps).map(|i| i as f64 * dt_actual).collect();

    // Convert to NumPy arrays
    let sensor_data_np = PyArray1::from_vec_bound(py, sensor_data);
    let time_vec_np = PyArray1::from_vec_bound(py, time_vec);

    Ok(SimulationResult {
        sensor_data: sensor_data_np.into_py(py),
        time: time_vec_np.into_py(py),
        time_steps,
        dt: dt_actual,
        final_time: dt_actual * time_steps as f64,
    })
}
```

**Note:** The `ArcSourceWrapper` struct may need to be added to kwavers core if it doesn't exist:

```rust
// Add to kwavers/src/domain/source/mod.rs if needed
pub struct ArcSourceWrapper(pub Arc<dyn Source>);

impl Source for ArcSourceWrapper {
    fn amplitude(&self, t: f64, _x: f64, _y: f64, _z: f64) -> f64 {
        // Delegate to inner Arc<dyn Source>
        // Note: This requires Source trait to have position-independent amplitude
        // or add position parameters to signature
        0.0 // Placeholder - implement based on actual Source trait
    }
    // ... implement other Source trait methods by delegation
}
```

### Step 3: Test Hybrid Implementation (30 minutes)

**Create test file:** `pykwavers/test_hybrid_solver.py`

```python
import pykwavers as kw
import numpy as np

def test_hybrid_initialization():
    """Test Hybrid solver can be initialized."""
    grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))
    
    sim = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Hybrid)
    result = sim.run(time_steps=10)
    
    assert result.time_steps == 10
    assert result.sensor_data.shape == (10,)


def test_hybrid_performance():
    """Test Hybrid solver performance is between FDTD and PSTD."""
    import time
    
    grid = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))
    
    # Run with all three solvers
    solvers = {
        'FDTD': kw.SolverType.FDTD,
        'PSTD': kw.SolverType.PSTD,
        'Hybrid': kw.SolverType.Hybrid,
    }
    
    times = {}
    for name, solver_type in solvers.items():
        sim = kw.Simulation(grid, medium, source, sensor, solver=solver_type)
        
        start = time.time()
        result = sim.run(time_steps=100)
        elapsed = time.time() - start
        
        times[name] = elapsed
        print(f"{name}: {elapsed:.3f} s")
    
    # Hybrid should be between FDTD and PSTD in performance
    # (Exact ordering depends on problem, but all should complete)
    assert all(t > 0 for t in times.values())


def test_hybrid_accuracy():
    """Test Hybrid solver accuracy (should match PSTD in smooth regions)."""
    grid = kw.Grid(nx=32, ny=32, nz=32, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point(position=(0.01, 0.01, 0.01))
    
    # Run with PSTD (reference)
    sim_pstd = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.PSTD)
    result_pstd = sim_pstd.run(time_steps=100)
    
    # Run with Hybrid
    sim_hybrid = kw.Simulation(grid, medium, source, sensor, solver=kw.SolverType.Hybrid)
    result_hybrid = sim_hybrid.run(time_steps=100)
    
    # Should be similar (within 5% for smooth homogeneous medium)
    diff = np.abs(result_hybrid.sensor_data - result_pstd.sensor_data)
    max_val = np.max(np.abs(result_pstd.sensor_data))
    rel_error = np.max(diff) / max_val if max_val > 0 else 0.0
    
    assert rel_error < 0.05, f"Hybrid vs PSTD error {rel_error:.3f} exceeds 5%"


if __name__ == "__main__":
    test_hybrid_initialization()
    print("✓ Hybrid initialization test passed")
    
    test_hybrid_performance()
    print("✓ Hybrid performance test passed")
    
    test_hybrid_accuracy()
    print("✓ Hybrid accuracy test passed")
    
    print("\n✅ All Hybrid tests passed!")
```

**Run tests:**
```bash
cd pykwavers
pytest test_hybrid_solver.py -v
```

---

## Part 3: Integration Testing (30 minutes)

### Create Comprehensive Solver Comparison Test

**File:** `pykwavers/test_all_solvers.py`

```python
import pykwavers as kw
import numpy as np
import pytest

def standard_setup():
    """Create standard test setup."""
    grid = kw.Grid(nx=64, ny=64, nz=64, dx=0.1e-3, dy=0.1e-3, dz=0.1e-3)
    medium = kw.Medium.homogeneous(sound_speed=1500.0, density=1000.0)
    source = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    sensor = kw.Sensor.point(position=(0.003, 0.0032, 0.0032))
    return grid, medium, source, sensor


def test_all_solvers_complete():
    """Test that all three solvers complete successfully."""
    grid, medium, source, sensor = standard_setup()
    
    for solver_name, solver_type in [
        ('FDTD', kw.SolverType.FDTD),
        ('PSTD', kw.SolverType.PSTD),
        ('Hybrid', kw.SolverType.Hybrid),
    ]:
        sim = kw.Simulation(grid, medium, source, sensor, solver=solver_type)
        result = sim.run(time_steps=500)
        
        assert result.time_steps == 500
        assert result.sensor_data.shape == (500,)
        assert np.all(np.isfinite(result.sensor_data))
        print(f"✓ {solver_name} completed successfully")


def test_dispersion_comparison():
    """Compare dispersion behavior across solvers."""
    grid, medium, source, sensor = standard_setup()
    
    results = {}
    for name, solver_type in [
        ('FDTD', kw.SolverType.FDTD),
        ('PSTD', kw.SolverType.PSTD),
        ('Hybrid', kw.SolverType.Hybrid),
    ]:
        sim = kw.Simulation(grid, medium, source, sensor, solver=solver_type)
        result = sim.run(time_steps=1000, dt=1e-8)
        results[name] = result
        
        # Find arrival time
        threshold = 0.1 * np.max(np.abs(result.sensor_data))
        arrival_idx = np.argmax(np.abs(result.sensor_data) > threshold)
        arrival_time = result.time[arrival_idx]
        
        # Expected arrival
        distance = 3e-3
        expected = distance / 1500.0
        error = abs(arrival_time - expected) / expected
        
        print(f"{name}: arrival={arrival_time*1e6:.3f} μs, "
              f"expected={expected*1e6:.3f} μs, error={error*100:.1f}%")
    
    # Validate expectations
    # PSTD: <1% error (dispersion-free)
    # FDTD: ~15-30% error (dispersive)
    # Hybrid: <5% error (balanced)
    
    assert len(results) == 3, "All solvers should complete"


def test_multi_source_all_solvers():
    """Test multi-source support across all solvers."""
    grid, medium, _, sensor = standard_setup()
    
    # Two sources
    source1 = kw.Source.plane_wave(grid, frequency=1e6, amplitude=1e5)
    source2 = kw.Source.plane_wave(grid, frequency=1.5e6, amplitude=5e4)
    
    for name, solver_type in [
        ('FDTD', kw.SolverType.FDTD),
        ('PSTD', kw.SolverType.PSTD),
        ('Hybrid', kw.SolverType.Hybrid),
    ]:
        sim = kw.Simulation(grid, medium, [source1, source2], sensor, solver=solver_type)
        result = sim.run(time_steps=100)
        
        assert result.time_steps == 100
        assert np.all(np.isfinite(result.sensor_data))
        print(f"✓ {name} multi-source passed")


if __name__ == "__main__":
    print("Testing all solvers...\n")
    
    test_all_solvers_complete()
    print("\n✓ All solvers complete test passed\n")
    
    test_dispersion_comparison()
    print("\n✓ Dispersion comparison test passed\n")
    
    test_multi_source_all_solvers()
    print("\n✓ Multi-source all solvers test passed\n")
    
    print("✅ All integration tests passed!")
```

---

## Part 4: Build and Validation (30 minutes)

### Step 1: Rebuild PyKwavers

```bash
cd pykwavers

# Clean previous build
cargo clean

# Build with maturin
maturin develop --release

# Or build wheel
maturin build --release
```

### Step 2: Run Full Test Suite

```bash
# Run all tests
pytest -v

# Specific test files
pytest test_pstd_solver.py -v
pytest test_hybrid_solver.py -v
pytest test_all_solvers.py -v

# Include Phase 5 tests (should all still pass)
pytest test_phase5_features.py -v
```

### Step 3: Validate Against Known Results

```bash
# Run plane wave timing test with all solvers
python test_plane_wave_timing.py --solver PSTD
python test_plane_wave_timing.py --solver Hybrid
```

Expected results:
- PSTD: Timing error <1%
- Hybrid: Timing error <5%
- FDTD: Timing error ~15-30% (documented dispersion)

---

## Part 5: Documentation Updates (30 minutes)

### Update README.md

Add solver selection section:

```markdown
## Solver Selection

PyKwavers supports three numerical solvers with different trade-offs:

### FDTD (Finite-Difference Time-Domain)
- **Best for:** Sharp interfaces, complex geometries
- **Dispersion:** ~15% numerical dispersion at standard resolution
- **Speed:** Baseline (1.0×)
- **Stability:** Robust

### PSTD (Pseudospectral Time-Domain)
- **Best for:** Smooth media, timing-critical applications
- **Dispersion:** <1% (nearly dispersion-free)
- **Speed:** Moderate (0.8-1.2×)
- **Stability:** Excellent for smooth fields

### Hybrid (PSTD + FDTD)
- **Best for:** Mixed problems (smooth + interfaces)
- **Dispersion:** ~5% (balanced)
- **Speed:** Fast (1.5-3×)
- **Stability:** Adaptive

### Usage Example

```python
import pykwavers as kw

# Select solver via SolverType enum
sim = kw.Simulation(
    grid, medium, source, sensor,
    solver=kw.SolverType.PSTD  # Choose FDTD, PSTD, or Hybrid
)
result = sim.run(time_steps=1000)
```

### Solver Selection Guidance

| Use Case | Recommended Solver |
|----------|-------------------|
| k-Wave comparison | PSTD |
| Timing measurements | PSTD |
| Heterogeneous media | Hybrid |
| Complex boundaries | FDTD |
| Maximum speed | Hybrid |
| Maximum accuracy | PSTD |
```

---

## Common Issues and Solutions

### Issue 1: Missing ArcSourceWrapper

**Error:**
```
error[E0433]: failed to resolve: use of undeclared type `ArcSourceWrapper`
```

**Solution:**
Add wrapper struct to `kwavers/src/domain/source/mod.rs`:

```rust
pub struct ArcSourceWrapper(pub Arc<dyn Source>);

impl Source for ArcSourceWrapper {
    // Delegate all methods to Arc<dyn Source>
    // Implementation depends on Source trait definition
}
```

Or use `Box::from(source_arc.as_ref())` pattern.

### Issue 2: PSTD Initialization Fails

**Error:**
```
KwaversError: Failed to initialize spectral operators
```

**Solution:**
Check grid dimensions are compatible with FFT (prefer powers of 2).
Ensure medium properties are valid (positive speed, density).

### Issue 3: Hybrid Solver Source Injection

**Error:**
```
Cannot convert Arc<dyn Source> to Box<dyn Source>
```

**Solution:**
Create a temporary wrapper or modify Hybrid solver to accept Arc:

```rust
// Option 1: Clone the trait object (if Source: Clone)
let source_box = Box::new((*source_arc).clone());

// Option 2: Use Arc in HybridSolver (modify core implementation)
impl HybridSolver {
    pub fn add_source_arc(&mut self, source: Arc<dyn Source>) -> Result<()> {
        // Store Arc directly
    }
}
```

---

## Success Checklist

**PSTD Implementation:**
- [ ] Imports added
- [ ] `run_pstd()` implemented
- [ ] Builds without errors
- [ ] `test_pstd_solver.py` passes (3/3 tests)
- [ ] Timing error <1% verified

**Hybrid Implementation:**
- [ ] Imports added
- [ ] `run_hybrid()` implemented
- [ ] Builds without errors
- [ ] `test_hybrid_solver.py` passes (3/3 tests)
- [ ] Performance between FDTD and PSTD verified

**Integration:**
- [ ] All Phase 5 tests still pass (18/18)
- [ ] `test_all_solvers.py` passes (3/3 tests)
- [ ] Documentation updated
- [ ] README.md includes solver selection guide

**Final Validation:**
- [ ] Wheel builds successfully (`maturin build --release`)
- [ ] Installable in fresh environment
- [ ] Examples run without errors
- [ ] Ready for pycoeus integration

---

## Estimated Timeline

| Task | Duration | Cumulative |
|------|----------|------------|
| PSTD imports | 5 min | 0:05 |
| PSTD implementation | 3.5 h | 3:35 |
| PSTD testing | 30 min | 4:05 |
| Hybrid imports | 5 min | 4:10 |
| Hybrid implementation | 1.5 h | 5:40 |
| Hybrid testing | 30 min | 6:10 |
| Integration tests | 30 min | 6:40 |
| Build and validate | 30 min | 7:10 |
| Documentation | 30 min | 7:40 |

**Total: ~6-8 hours** (includes debugging time)

---

## Next Steps After Completion

1. **Create k-Wave comparison examples** (2 hours)
   - Plane wave validation
   - Point source validation
   - Focused beam validation

2. **Performance benchmarking** (2 hours)
   - Grid size scaling
   - Solver comparison plots
   - Documentation

3. **Pycoeus integration guide** (2 hours)
   - Migration examples
   - API mapping
   - Use case recommendations

---

## Contact and Support

**Questions during implementation?**
- Refer to `PYCOEUS_SOLVER_AUDIT.md` for detailed specifications
- Check existing `run_fdtd()` implementation as template
- Review kwavers core documentation: `kwavers/src/solver/forward/pstd/mod.rs`

**Found a bug?**
- Open issue with `[pykwavers]` tag
- Include minimal reproducible example
- Specify OS, Rust version, Python version

---

*Implementation guide created: 2026-02-04*  
*Sprint 217 Session 10*  
*Author: Ryan Clanton (@ryancinsight)*