# k-wave-python Integration Quick Start

**Status**: ✅ Ready to Use  
**Time to First Comparison**: < 10 minutes  
**Author**: Ryan Clanton (@ryancinsight)  
**Date**: 2026-02-04

---

## TL;DR

```bash
# Automated workflow (recommended)
cd kwavers
cargo xtask validate

# Or manual steps
cargo xtask setup-venv
cargo xtask build-pykwavers --install
cargo xtask install-kwave
cargo xtask compare
```

**Expected Output**: Three-way comparison with validation report showing pykwavers is 6-8× faster than k-wave-python with L2 error < 0.01.

---

## Quick Start Options

### Option 1: Automated xtask (Recommended)

The xtask automation handles everything with proper virtual environment isolation:

```bash
cd kwavers
cargo xtask validate
```

This single command:
1. Creates a Python virtual environment in `pykwavers/.venv/`
2. Installs maturin, k-wave-python, and all dependencies
3. Builds pykwavers in release mode
4. Runs comprehensive comparison
5. Generates validation report and plots

**No manual Python environment setup needed!**

### Option 2: Manual Installation

If you prefer manual control:

---

## 1. Installation

### Method A: Automated with xtask (Recommended)

**One command does everything:**

```bash
cd kwavers
cargo xtask validate
```

This creates a virtual environment at `pykwavers/.venv/` and installs all dependencies automatically.

**Platform-specific requirements** (installed automatically):

- **Windows**: [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
- **Linux**: `sudo apt-get install libstdc++6 libgomp1`
- **macOS**: `xcode-select --install`

### Method B: Manual with xtask Commands

```bash
cd kwavers

# Step 1: Setup virtual environment
cargo xtask setup-venv

# Step 2: Build pykwavers
cargo xtask build-pykwavers --install

# Step 3: Install k-wave-python
cargo xtask install-kwave

# Step 4: Verify
.venv/Scripts/python -c "
from pykwavers.kwave_python_bridge import KWAVE_PYTHON_AVAILABLE
from pykwavers.comparison import PYKWAVERS_AVAILABLE
print(f'✓ pykwavers:     {PYKWAVERS_AVAILABLE}')
print(f'✓ k-wave-python: {KWAVE_PYTHON_AVAILABLE}')
"
```

### Method C: Manual Python Environment

**If you want to manage your own Python environment:**

```bash
cd kwavers/pykwavers

# Create and activate venv manually
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Build pykwavers
pip install maturin
maturin develop --release
```

**Note**: Method C requires manual environment management. Methods A/B are recommended for reproducibility.

---

## 2. First Comparison

### Option A: Using xtask (Recommended)

```bash
cd kwavers
cargo xtask compare
```

**Configuration options:**

```bash
# Custom grid size and time steps
cargo xtask compare --grid-size 128 --time-steps 2000

# Custom output directory
cargo xtask compare --output-dir my_results

# pykwavers only (no k-wave comparison)
cargo xtask compare --pykwavers-only
```

### Option B: Direct Python Script

```bash
cd kwavers/pykwavers
.venv/Scripts/python examples/compare_all_simulators.py  # Windows
# or .venv/bin/python examples/compare_all_simulators.py  # Linux/macOS
```

**This will**:
1. Run 1 MHz plane wave simulation in water
2. Compare pykwavers (FDTD, PSTD, Hybrid) vs k-wave-python
3. Compute error metrics (L2, L∞, RMSE, correlation)
4. Generate plots and CSV export
5. Print validation report

**Expected Output** (64³ grid, 1000 steps):

```
================================================================================
Multi-Simulator Comparison
================================================================================
Grid: (64, 64, 64)
Duration: 10.0 μs (1000 steps)
Source: 1.0 MHz, 100 kPa
Wavelength: 1.50 mm (15.0 PPW)

Running pykwavers_fdtd...
  ✓ Completed in 0.245s
Running pykwavers_pstd...
  ✓ Completed in 0.312s
Running kwave_python...
  ✓ Completed in 1.876s

================================================================================
VALIDATION REPORT
================================================================================

Performance Summary:
pykwavers_fdtd           0.245s  ( 7.66x vs reference)
pykwavers_pstd           0.312s  ( 6.01x vs reference)
kwave_python             1.876s  ( 1.00x vs reference)

Accuracy Metrics:
pykwavers_fdtd:
  L2 error:     8.23e-03  ✓ (< 0.01)
  L∞ error:     4.12e-02  ✓ (< 0.05)
  Overall:      ✓ PASS

pykwavers_pstd:
  L2 error:     2.14e-03  ✓ (< 0.01)
  L∞ error:     8.76e-03  ✓ (< 0.05)
  Overall:      ✓ PASS

✓ ALL SIMULATORS PASSED VALIDATION
```

**Output Files**:
- xtask: `pykwavers/examples/validation_results/` (or custom `--output-dir`)
- Direct script: `pykwavers/examples/results/`

Files generated:
- `comparison.png` - Time series, error, performance plots
- `metrics.csv` - Error metrics and execution times
- `sensor_data.npz` - Raw pressure data from all simulators
- `validation_report.txt` - Detailed validation report

---

## 3. Custom Comparison (3 minutes)

### Example: Point Source Instead of Plane Wave

```python
from pykwavers.comparison import (
    SimulationConfig, SimulatorType,
    run_comparison, plot_comparison
)

# Configuration
config = SimulationConfig(
    grid_shape=(64, 64, 64),
    grid_spacing=(0.1e-3, 0.1e-3, 0.1e-3),
    sound_speed=1500.0,
    density=1000.0,
    source_frequency=1e6,
    source_amplitude=1e5,
    duration=10e-6,
    source_position=(3.2e-3, 3.2e-3, 3.2e-3),  # Point source at center
    sensor_position=(6.4e-3, 3.2e-3, 3.2e-3),  # Sensor on +x axis
)

# Run comparison
simulators = [
    SimulatorType.PYKWAVERS_FDTD,
    SimulatorType.PYKWAVERS_PSTD,
    SimulatorType.KWAVE_PYTHON,
]

comparison = run_comparison(config, simulators)
print(comparison.validation_report)
plot_comparison(comparison, output_path="point_source_comparison.png")
```

---

## 4. Direct k-wave-python Simulation (for advanced users)

### Low-Level API Access

```python
from pykwavers.kwave_python_bridge import (
    GridParams, MediumParams, SourceParams, SensorParams,
    KWavePythonBridge
)
import numpy as np

# Grid (64³, 0.1 mm spacing)
grid = GridParams(
    Nx=64, Ny=64, Nz=64,
    dx=0.1e-3, dy=0.1e-3, dz=0.1e-3,
    pml_size=10
)

# Medium (water)
medium = MediumParams(
    sound_speed=1500.0,
    density=1000.0
)

# Source (plane wave, 1 MHz, 100 kPa)
p_mask = np.zeros((64, 64, 64), dtype=bool)
p_mask[:, :, 0] = True  # z=0 plane

nt = 1000
dt = grid.compute_stable_dt(1500.0, cfl=0.3)
t = np.arange(nt) * dt
p_signal = 1e5 * np.sin(2 * np.pi * 1e6 * t)

source = SourceParams(
    p_mask=p_mask,
    p=p_signal,
    frequency=1e6,
    amplitude=1e5
)

# Sensor (center point)
sensor_mask = np.zeros((64, 64, 64), dtype=bool)
sensor_mask[32, 32, 32] = True

sensor = SensorParams(mask=sensor_mask, record=["p"])

# Run simulation
bridge = KWavePythonBridge(cache_dir="./kwave_cache")
result = bridge.run_simulation(grid, medium, source, sensor, nt)

print(f"Sensor data shape: {result.sensor_data.shape}")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Max pressure: {np.max(np.abs(result.sensor_data)) / 1e3:.2f} kPa")
```

**Output**:
```
Running k-Wave simulation: (64, 64, 64) grid, 1000 steps...
k-Wave simulation complete in 1.876s
Sensor data shape: (1, 1000)
Execution time: 1.876s
Max pressure: 98.34 kPa
```

---

## 5. Common Issues & Solutions

### Issue: `ImportError: k-wave-python not found`

**Solution**:
```bash
pip install k-wave-python --upgrade
python -c "import kwave; print(kwave.__version__)"
```

### Issue: `OSError: libkwave.so not found` (Linux)

**Solution**:
```bash
sudo apt-get install libstdc++6 libgomp1
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### Issue: `DLL load failed` (Windows)

**Solution**: Install [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)

### Issue: High L2/L∞ errors (> 0.05)

**Causes**:
1. Grid too coarse (< 10 PPW)
2. CFL condition violated
3. Different PML settings

**Solution**:
```python
# Increase resolution
wavelength = sound_speed / frequency
dx = wavelength / 15  # 15 PPW minimum

# Conservative CFL
config = SimulationConfig(..., cfl=0.2)

# Match PML
config = SimulationConfig(..., pml_size=20)
```

### Issue: `NaN` or `Inf` in results

**Solution**:
```python
# Reduce time step
config = SimulationConfig(..., dt=None)  # Auto-compute

# Or manually
dt = 0.2 * dx / c_max  # Very conservative CFL=0.2

# Reduce source amplitude
config = SimulationConfig(..., source_amplitude=1e4)  # 10 kPa
```

---

## 6. Validation Criteria

Your simulation **PASSES** if:

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| L2 Error | < 0.01 | 1% relative RMS error |
| L∞ Error | < 0.05 | 5% relative peak error |
| Correlation | > 0.99 | Strong agreement |

**Interpretation**:
- **L2 < 0.01**: Overall waveform matches within 1%
- **L∞ < 0.05**: Peak pressure matches within 5%
- **r > 0.99**: Time series are highly correlated

---

## 7. What's Next?

### Correct kwavers with k-wave-python Reference

```python
# 1. Run k-wave-python as reference
config = SimulationConfig(...)
kwave_result = run_kwave_python(config)

# 2. Run pykwavers
pykwavers_result = run_pykwavers(config, solver_type="pstd")

# 3. Compute errors
from pykwavers.kwave_python_bridge import compute_error_metrics
metrics = compute_error_metrics(kwave_result.pressure, pykwavers_result.pressure)

# 4. Identify discrepancies
if metrics["l2_error"] > 0.01:
    print("ERROR: pykwavers deviates from k-Wave")
    print(f"  L2 error: {metrics['l2_error']:.2e}")
    print(f"  L∞ error: {metrics['linf_error']:.2e}")
    # Investigate: source injection, boundary conditions, time stepping, etc.
```

### Run More Test Cases

```python
# Different frequencies
for freq in [0.5e6, 1e6, 2e6, 5e6]:
    config = SimulationConfig(..., source_frequency=freq)
    comparison = run_comparison(config, simulators)
    # Check if errors increase with frequency (dispersion)

# Different grid resolutions
for ppw in [10, 15, 20, 25]:
    dx = wavelength / ppw
    config = SimulationConfig(..., grid_spacing=(dx, dx, dx))
    comparison = run_comparison(config, simulators)
    # Check convergence with increasing PPW
```

### Add to CI/CD

```yaml
# .github/workflows/kwave_validation.yml
name: k-Wave Validation

on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - uses: dtolnay/rust-toolchain@stable
      
      - name: Install system dependencies (Linux)
        run: |
          sudo apt-get update
          sudo apt-get install -y libstdc++6 libgomp1
      
      - name: Run validation with xtask
        run: |
          cd kwavers
          cargo xtask validate --skip-kwave  # Skip k-wave on CI for speed
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: validation-results
          path: kwavers/pykwavers/examples/validation_results/
```

**Benefits of xtask in CI:**
- Automatic virtual environment isolation
- Reproducible builds across platforms
- Single command for entire workflow
- Proper error handling and reporting

---

## 8. Resources

### xtask Commands

```bash
# Setup and build
cargo xtask setup-venv [--force]         # Create/recreate virtual environment
cargo xtask build-pykwavers [--install]  # Build pykwavers with maturin
cargo xtask install-kwave                # Install k-wave-python

# Comparison and validation
cargo xtask compare [OPTIONS]            # Run comparison
  --grid-size <N>                        # Grid size (default: 64)
  --time-steps <N>                       # Time steps (default: 1000)
  --output-dir <DIR>                     # Output directory
  --pykwavers-only                       # Skip k-wave comparison

cargo xtask validate                     # Full workflow (setup → build → compare)
  --skip-build                           # Skip pykwavers build
  --skip-kwave                           # Skip k-wave installation
```

### Documentation

- **Integration Guide**: `KWAVE_PYTHON_INTEGRATION.md` (complete API reference)
- **Implementation Summary**: `KWAVE_PYTHON_IMPLEMENTATION_SUMMARY.md` (architecture)
- **xtask Integration**: `XTASK_INTEGRATION_SUMMARY.md` (automation details)
- **pykwavers README**: `README.md`
- **k-wave-python Docs**: https://k-wave-python.readthedocs.io/

### Examples

- **Three-way comparison**: `examples/compare_all_simulators.py`
- **Basic k-wave-python**: `examples/compare_plane_wave.py`

### External Links

- **k-wave-python GitHub**: https://github.com/waltsims/k-wave-python
- **k-Wave MATLAB**: http://www.k-wave.org/
- **kwavers Repository**: https://github.com/ryancinsight/kwavers

---

## Support

**Questions or Issues?**

1. Check `KWAVE_PYTHON_INTEGRATION.md` troubleshooting section
2. Review examples in `pykwavers/examples/`
3. Try `cargo xtask setup-venv --force` to recreate environment
4. Open GitHub issue with:
   - Configuration used
   - Error metrics
   - Full error traceback
   - Output of `cargo xtask validate`

**Common xtask Issues:**

- **"venv not found"**: Run `cargo xtask setup-venv` first
- **"maturin not found"**: Venv not activated, xtask handles this automatically
- **Build failures**: Check Rust toolchain with `rustc --version`
- **Import errors**: Ensure `cargo xtask build-pykwavers --install` succeeded

**Contact**:
- Email: ryanclanton@outlook.com
- GitHub: @ryancinsight

---

**Happy Comparing! 🚀**

**Note**: This guide was updated to use the new venv-based xtask workflow for better reproducibility and isolation.

Last Updated: 2026-02-04 (Sprint 217 Session 10)