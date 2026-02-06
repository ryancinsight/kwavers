# Virtual Environment Workflow for pykwavers

**Status**: ✅ Production Ready  
**Author**: Ryan Clanton (@ryancinsight)  
**Date**: 2026-02-04  
**Sprint**: 217 Session 10

---

## Overview

The pykwavers build and validation workflow now uses **automated virtual environment management** via `cargo xtask`. This ensures:

- **Isolation**: No conflicts with system Python packages
- **Reproducibility**: Consistent dependencies across all environments
- **Simplicity**: Single-command setup and validation
- **CI/CD Ready**: Deterministic builds in automation pipelines

---

## Architecture

### Virtual Environment Location

```
kwavers/
├── pykwavers/
│   ├── .venv/              ← Virtual environment (auto-created)
│   │   ├── Scripts/        ← Windows executables
│   │   │   ├── python.exe
│   │   │   ├── pip.exe
│   │   │   └── maturin.exe
│   │   ├── bin/            ← Linux/macOS executables
│   │   │   ├── python
│   │   │   ├── pip
│   │   │   └── maturin
│   │   └── Lib/            ← Installed packages
│   ├── requirements.txt    ← Dependencies
│   └── examples/
└── xtask/
    └── src/
        └── main.rs         ← Venv management logic
```

### xtask Venv Functions

**Core Infrastructure** (in `xtask/src/main.rs`):

```rust
/// Get venv directory path
fn venv_dir() -> PathBuf

/// Get Python executable from venv
fn venv_python() -> Result<PathBuf>

/// Get pip executable from venv
fn venv_pip() -> Result<PathBuf>

/// Get maturin executable from venv
fn venv_maturin() -> Result<PathBuf>

/// Check if venv exists and is valid
fn venv_exists() -> bool

/// Setup Python virtual environment
fn setup_venv(force: bool) -> Result<()>
```

**Integration Points**:
- `build_pykwavers()` - Uses `venv_maturin()`
- `install_kwave()` - Uses `venv_python()` and `venv_pip()`
- `run_comparison()` - Uses `venv_python()`
- `run_validation()` - Ensures venv exists before all operations

---

## Workflow Modes

### Mode 1: Fully Automated (Recommended)

**Single command for complete validation:**

```bash
cd kwavers
cargo xtask validate
```

**What happens:**
1. Checks if `.venv/` exists
2. If not, creates venv and installs dependencies
3. Builds pykwavers with maturin (from venv)
4. Installs k-wave-python (to venv)
5. Runs comparison using venv Python
6. Generates validation report

**Output:**
```
================================================================================
🎯 Running full validation workflow
================================================================================

Step 0/4: Setting up virtual environment
--------------------------------------------------------------------------------
🐍 Setting up Python virtual environment...
📦 Creating virtual environment at: D:\kwavers\pykwavers\.venv
✅ Virtual environment created
📦 Upgrading pip...
✅ pip upgraded
📦 Installing maturin...
✅ maturin installed
📦 Installing requirements.txt...
✅ requirements.txt installed

================================================================================
✅ Virtual environment ready!
================================================================================
Location: D:\kwavers\pykwavers\.venv

To activate manually:
  D:\kwavers\pykwavers\.venv\Scripts\activate

All xtask commands will now use this venv automatically.
================================================================================

Step 1/4: Building pykwavers
--------------------------------------------------------------------------------
🔨 Building pykwavers...
🔍 Using maturin from venv: D:\kwavers\pykwavers\.venv\Scripts\maturin.exe
...
✅ pykwavers built and installed

Step 2/4: Installing k-wave-python
--------------------------------------------------------------------------------
📦 Installing k-wave-python and dependencies...
✅ k-wave-python and dependencies installed

Step 3/4: Running comparison
--------------------------------------------------------------------------------
🔬 Running comparison...
...
✅ Comparison complete

================================================================================
✅ Validation workflow complete!
================================================================================
```

### Mode 2: Manual Step-by-Step

**Fine-grained control over each phase:**

```bash
cd kwavers

# Step 1: Create venv and install dependencies
cargo xtask setup-venv

# Step 2: Build pykwavers
cargo xtask build-pykwavers --release --install

# Step 3: Install k-wave-python (optional)
cargo xtask install-kwave

# Step 4: Run comparison
cargo xtask compare --grid-size 64 --time-steps 1000
```

### Mode 3: Incremental (Skip Steps)

**For development iteration:**

```bash
# Skip venv setup and build (use existing)
cargo xtask validate --skip-build

# Skip k-wave installation (pykwavers only)
cargo xtask validate --skip-kwave

# Both
cargo xtask validate --skip-build --skip-kwave
```

### Mode 4: Custom Comparison Parameters

```bash
# Larger grid
cargo xtask compare --grid-size 128 --time-steps 2000

# Custom output directory
cargo xtask compare --output-dir my_results

# pykwavers only (no k-wave reference)
cargo xtask compare --pykwavers-only

# Combine options
cargo xtask compare \
  --grid-size 256 \
  --time-steps 5000 \
  --output-dir large_grid_results
```

---

## Command Reference

### `cargo xtask setup-venv`

**Purpose**: Create Python virtual environment and install dependencies.

**Syntax**:
```bash
cargo xtask setup-venv [--force]
```

**Options**:
- `--force`: Delete and recreate venv if it exists

**What it does**:
1. Creates `pykwavers/.venv/` using `python -m venv`
2. Upgrades pip to latest version
3. Installs maturin (for building pykwavers)
4. Installs all dependencies from `requirements.txt`

**When to run**:
- First time setup
- After dependency changes in `requirements.txt`
- When venv becomes corrupted (`--force`)

**Example output**:
```
🐍 Setting up Python virtual environment...
📦 Creating virtual environment at: D:\kwavers\pykwavers\.venv
✅ Virtual environment created
📦 Upgrading pip...
✅ pip upgraded
📦 Installing maturin...
✅ maturin installed
📦 Installing requirements.txt...
✅ requirements.txt installed

================================================================================
✅ Virtual environment ready!
================================================================================
Location: D:\kwavers\pykwavers\.venv

To activate manually:
  D:\kwavers\pykwavers\.venv\Scripts\activate  (Windows)
  source D:/kwavers/pykwavers/.venv/bin/activate  (Linux/macOS)

All xtask commands will now use this venv automatically.
================================================================================
```

---

### `cargo xtask build-pykwavers`

**Purpose**: Build pykwavers Rust bindings with maturin.

**Syntax**:
```bash
cargo xtask build-pykwavers [--release] [--install]
```

**Options**:
- `--release`: Build optimized release version (default: true)
- `--install`: Install to venv after building (via `maturin develop`)

**Auto-setup**:
- If venv doesn't exist, runs `setup-venv` automatically

**What it does**:
1. Checks venv exists (creates if not)
2. Uses `venv_maturin()` to get maturin from venv
3. Runs `maturin develop --release` (if `--install`)
4. Or `maturin build --release` (wheel only)

**Example**:
```bash
# Build and install (typical)
cargo xtask build-pykwavers --install

# Build wheel only (for distribution)
cargo xtask build-pykwavers

# Debug build
cargo xtask build-pykwavers --install --no-release
```

---

### `cargo xtask install-kwave`

**Purpose**: Install k-wave-python and validation dependencies.

**Syntax**:
```bash
cargo xtask install-kwave [--skip-existing]
```

**Options**:
- `--skip-existing`: Skip if k-wave-python already installed

**Auto-setup**:
- If venv doesn't exist, runs `setup-venv` automatically

**What it does**:
1. Checks venv exists (creates if not)
2. Optionally checks for existing k-wave-python installation
3. Runs `pip install -r requirements.txt` using venv pip
4. Verifies installation by importing modules

**Example output**:
```
📦 Installing k-wave-python and dependencies...
🚀 Installing from requirements.txt...
✅ k-wave-python and dependencies installed
🔍 Verifying installation...
pykwavers: True
k-wave-python: True
```

---

### `cargo xtask compare`

**Purpose**: Run simulator comparison with configurable parameters.

**Syntax**:
```bash
cargo xtask compare [OPTIONS]
```

**Options**:
```
--grid-size <N>          Grid size (e.g., 64 for 64³) [default: 64]
--time-steps <N>         Number of time steps [default: 1000]
--output-dir <DIR>       Output directory [default: comparison_results]
--pykwavers-only         Run only pykwavers solvers (no k-wave)
```

**Auto-setup**:
- If venv doesn't exist, runs `setup-venv` automatically

**What it does**:
1. Checks venv exists (creates if not)
2. Uses venv Python to run `examples/compare_all_simulators.py`
3. Passes configuration via environment variables:
   - `KWAVERS_GRID_SIZE`
   - `KWAVERS_TIME_STEPS`
   - `KWAVERS_OUTPUT_DIR`
   - `KWAVERS_PYKWAVERS_ONLY`
4. Generates plots, CSV metrics, and validation report

**Examples**:
```bash
# Default (64³, 1000 steps)
cargo xtask compare

# Large grid
cargo xtask compare --grid-size 128 --time-steps 2000

# pykwavers only (fast, no k-wave reference)
cargo xtask compare --pykwavers-only

# Custom output
cargo xtask compare --output-dir my_experiment_results
```

---

### `cargo xtask validate`

**Purpose**: Complete validation workflow from setup to comparison.

**Syntax**:
```bash
cargo xtask validate [OPTIONS]
```

**Options**:
```
--skip-build    Skip pykwavers build if already installed
--skip-kwave    Skip k-wave-python installation
```

**What it does**:
1. **Step 0/4**: Ensure venv exists (or create)
2. **Step 1/4**: Build pykwavers (unless `--skip-build`)
3. **Step 2/4**: Install k-wave-python (unless `--skip-kwave`)
4. **Step 3/4**: Run comparison (64³, 1000 steps)
5. Print validation report

**Use cases**:
```bash
# First time full validation
cargo xtask validate

# Quick re-validation (code unchanged)
cargo xtask validate --skip-build --skip-kwave

# pykwavers validation only (no k-wave)
cargo xtask validate --skip-kwave

# After code changes (rebuild, reuse k-wave)
cargo xtask validate --skip-kwave
```

---

## Venv Management

### Check Venv Status

```bash
cd kwavers/pykwavers

# Check if venv exists
ls .venv/

# Check Python version
.venv/Scripts/python --version  # Windows
.venv/bin/python --version       # Linux/macOS

# List installed packages
.venv/Scripts/pip list  # Windows
.venv/bin/pip list       # Linux/macOS
```

### Manual Activation (Optional)

**Windows**:
```cmd
cd kwavers\pykwavers
.venv\Scripts\activate
```

**Linux/macOS**:
```bash
cd kwavers/pykwavers
source .venv/bin/activate
```

**Once activated**:
```bash
# All commands use venv automatically
python examples/compare_all_simulators.py
pip list
maturin develop --release
```

**Deactivate**:
```bash
deactivate
```

**Note**: Manual activation is **not required** for xtask commands—they handle venv automatically.

### Recreate Venv

**When to recreate**:
- Dependency changes in `requirements.txt`
- Python version upgrade
- Venv corruption or errors
- Want clean slate

**How**:
```bash
cargo xtask setup-venv --force
```

This deletes `.venv/` and recreates from scratch.

### Update Dependencies

```bash
# Edit requirements.txt, then:
cargo xtask setup-venv --force
```

Or manually:
```bash
cd kwavers/pykwavers
.venv/Scripts/activate  # or source .venv/bin/activate
pip install -r requirements.txt --upgrade
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
name: pykwavers Validation

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  validate:
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.10', '3.11', '3.12']
    
    runs-on: ${{ matrix.os }}
    
    steps:
      - uses: actions/checkout@v4
      
      - uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - uses: dtolnay/rust-toolchain@stable
      
      - name: Install system dependencies (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y libstdc++6 libgomp1
      
      - name: Run validation
        run: |
          cd kwavers
          cargo xtask validate
      
      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: validation-${{ matrix.os }}-py${{ matrix.python-version }}
          path: kwavers/pykwavers/examples/validation_results/
```

**Benefits**:
- Matrix testing across OS and Python versions
- Automatic venv isolation (no manual environment setup)
- Deterministic builds
- Artifact upload for debugging

### GitLab CI Example

```yaml
# .gitlab-ci.yml
validate:
  stage: test
  image: rust:latest
  before_script:
    - apt-get update && apt-get install -y python3 python3-venv libstdc++6 libgomp1
  script:
    - cd kwavers
    - cargo xtask validate
  artifacts:
    paths:
      - kwavers/pykwavers/examples/validation_results/
    when: always
```

---

## Troubleshooting

### Issue: "Python executable not found in venv"

**Cause**: Venv not created or corrupted.

**Solution**:
```bash
cargo xtask setup-venv --force
```

---

### Issue: "maturin executable not found in venv"

**Cause**: Maturin not installed in venv (shouldn't happen if `setup-venv` succeeded).

**Solution**:
```bash
cd kwavers/pykwavers
.venv/Scripts/pip install maturin  # Windows
# or .venv/bin/pip install maturin  # Linux/macOS
```

Or recreate venv:
```bash
cargo xtask setup-venv --force
```

---

### Issue: "Failed to create virtual environment"

**Cause**: Python `venv` module not available (some distributions omit it).

**Solution (Debian/Ubuntu)**:
```bash
sudo apt-get install python3-venv
```

**Solution (Other Linux)**:
Check your distribution's package manager for `python3-venv`.

---

### Issue: Import errors after build

**Example**:
```
ImportError: cannot import name 'pykwavers' from 'pykwavers'
```

**Cause**: pykwavers not installed to venv.

**Solution**:
```bash
cargo xtask build-pykwavers --install
```

Or manually:
```bash
cd kwavers/pykwavers
.venv/Scripts/activate
maturin develop --release
```

---

### Issue: Different Python version than expected

**Check Python version**:
```bash
cd kwavers/pykwavers
.venv/Scripts/python --version  # Windows
.venv/bin/python --version       # Linux/macOS
```

**If wrong version**:
1. Ensure correct Python is in PATH
2. Recreate venv:
   ```bash
   cargo xtask setup-venv --force
   ```

---

### Issue: Dependency conflicts

**Symptoms**: Build succeeds but imports fail, or k-wave-python errors.

**Solution**: Clean install
```bash
cargo xtask setup-venv --force
```

This installs fresh dependencies from `requirements.txt`.

---

### Issue: Permission errors (Linux/macOS)

**Cause**: Insufficient permissions to create venv.

**Solution**: Check directory permissions
```bash
ls -la kwavers/pykwavers/
chmod u+w kwavers/pykwavers/  # If needed
```

Or use different location (requires xtask modification).

---

## Best Practices

### Development Workflow

**Daily iteration**:
```bash
# 1. Edit Rust code in kwavers/
# 2. Rebuild and test
cargo xtask build-pykwavers --install
cargo xtask compare --pykwavers-only  # Fast, no k-wave

# 3. Full validation before commit
cargo xtask validate
```

**Weekly/monthly**:
```bash
# Update dependencies
cd kwavers/pykwavers
# Edit requirements.txt
cargo xtask setup-venv --force
```

---

### CI/CD Workflow

**Pull request validation**:
```yaml
- name: Fast validation (pykwavers only)
  run: cargo xtask validate --skip-kwave
```

**Nightly/weekly comprehensive validation**:
```yaml
- name: Full validation with k-wave
  run: cargo xtask validate
```

**Release validation**:
```yaml
- name: Multi-platform validation
  strategy:
    matrix:
      os: [ubuntu-latest, windows-latest, macos-latest]
  run: cargo xtask validate
```

---

### Testing Multiple Python Versions

**Manual**:
```bash
# Python 3.10
python3.10 -m venv .venv
cargo xtask setup-venv --force
cargo xtask validate

# Python 3.11
python3.11 -m venv .venv
cargo xtask setup-venv --force
cargo xtask validate

# Python 3.12
python3.12 -m venv .venv
cargo xtask setup-venv --force
cargo xtask validate
```

**Automated (tox)**:
```ini
# tox.ini
[tox]
envlist = py310,py311,py312

[testenv]
deps = -r requirements.txt
commands =
    cargo xtask build-pykwavers --install
    cargo xtask compare
```

---

## Migration from Global Python

### Old Workflow (No Venv)

```bash
# System-wide installation (not isolated)
pip install k-wave-python maturin
cd kwavers/pykwavers
pip install -r requirements.txt
maturin develop --release
python examples/compare_all_simulators.py
```

**Problems**:
- ❌ Conflicts with system packages
- ❌ Non-reproducible (different on each machine)
- ❌ Requires manual dependency management
- ❌ CI/CD environment mismatch

---

### New Workflow (Venv-based)

```bash
# Isolated, reproducible
cd kwavers
cargo xtask validate
```

**Benefits**:
- ✅ Isolated from system packages
- ✅ Reproducible across machines
- ✅ Automatic dependency management
- ✅ CI/CD parity

---

### Migrating Existing Setup

**If you have old global installation**:

```bash
# 1. Remove global pykwavers (optional but recommended)
pip uninstall pykwavers -y

# 2. Create fresh venv-based setup
cd kwavers
cargo xtask setup-venv

# 3. Build and validate
cargo xtask validate

# 4. Clean up old artifacts (optional)
rm -rf kwavers/pykwavers/target/wheels/
```

**Your old global environment is unaffected**—the venv is completely isolated.

---

## Advanced Usage

### Custom Venv Location

**Modify xtask** (in `xtask/src/main.rs`):

```rust
fn venv_dir() -> PathBuf {
    // Default: workspace_root().join("pykwavers").join(".venv")
    
    // Custom location:
    PathBuf::from("/path/to/my/venv")
}
```

**Not recommended**—standard location is best for reproducibility.

---

### Multiple Venvs (Different Python Versions)

```bash
# Create venv for each Python version
python3.10 -m venv pykwavers/.venv-py310
python3.11 -m venv pykwavers/.venv-py311
python3.12 -m venv pykwavers/.venv-py312

# Modify xtask venv_dir() to select dynamically
# Or use tox (see "Testing Multiple Python Versions")
```

---

### Shared Venv Across Projects

**Not recommended** for reproducibility, but possible:

```bash
# Create shared venv
python -m venv ~/shared-venv

# Symlink to pykwavers
ln -s ~/shared-venv kwavers/pykwavers/.venv

# Install dependencies
~/shared-venv/bin/pip install -r kwavers/pykwavers/requirements.txt
```

**Risk**: Dependency conflicts between projects.

---

## Comparison: xtask vs Manual

| Aspect | xtask Automated | Manual |
|--------|-----------------|--------|
| Venv creation | ✅ Automatic | ❌ Manual |
| Dependency install | ✅ Automatic | ❌ Manual |
| Maturin build | ✅ Automatic | ❌ Manual |
| Python selection | ✅ Venv Python | ❌ User must activate |
| Error handling | ✅ Detailed messages | ❌ Cryptic errors |
| CI/CD ready | ✅ Single command | ❌ Multi-step setup |
| Reproducibility | ✅ Guaranteed | ⚠️ Depends on user |
| Complexity | ✅ Simple | ❌ Error-prone |

**Recommendation**: Use xtask automated workflow for all use cases.

---

## Summary

### Quick Command Reference

```bash
# Setup
cargo xtask setup-venv              # Create venv and install deps
cargo xtask setup-venv --force      # Recreate venv

# Build
cargo xtask build-pykwavers --install  # Build and install pykwavers

# Install k-wave
cargo xtask install-kwave           # Install k-wave-python

# Compare
cargo xtask compare                 # Default comparison
cargo xtask compare --pykwavers-only  # Fast, no k-wave

# Validate
cargo xtask validate                # Full workflow
cargo xtask validate --skip-build   # Reuse existing build
```

### Key Principles

1. **Isolation**: Venv at `pykwavers/.venv/` for reproducibility
2. **Automation**: xtask handles venv setup automatically
3. **Simplicity**: Single command for complete validation
4. **CI/CD**: Works identically in local and CI environments
5. **Manual override**: Can still activate venv manually if needed

### When to Run What

| Scenario | Command |
|----------|---------|
| First time setup | `cargo xtask validate` |
| After code changes | `cargo xtask build-pykwavers --install` |
| Quick test | `cargo xtask compare --pykwavers-only` |
| Full validation | `cargo xtask validate` |
| After dependency update | `cargo xtask setup-venv --force` |
| Venv corrupted | `cargo xtask setup-venv --force` |
| Before commit | `cargo xtask validate` |
| CI/CD | `cargo xtask validate [--skip-kwave]` |

---

## Resources

- **xtask Source**: `kwavers/xtask/src/main.rs`
- **Requirements**: `kwavers/pykwavers/requirements.txt`
- **Quick Start**: `KWAVE_PYTHON_QUICK_START.md`
- **Integration Guide**: `KWAVE_PYTHON_INTEGRATION.md`

---

**Last Updated**: 2026-02-04 (Sprint 217 Session 10)

**Validation Status**: ✅ Production Ready

---