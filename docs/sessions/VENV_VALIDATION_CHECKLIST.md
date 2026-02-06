# Virtual Environment Fix Validation Checklist

**Purpose**: Verify that xtask properly sets up and uses Python virtual environments  
**Date**: 2026-02-04  
**Sprint**: 217 Session 10

---

## Pre-Validation Setup

### Check Current State

```bash
cd kwavers

# Check if venv exists
ls pykwavers/.venv/

# Check xtask compiles
cargo check --package xtask

# Check xtask commands available
cargo xtask --help
```

---

## Validation Tests

### ✅ Test 1: xtask Compiles

**Command**:
```bash
cargo check --package xtask
```

**Expected**:
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in X.XXs
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 2: setup-venv Command Exists

**Command**:
```bash
cargo xtask --help | grep setup-venv
```

**Expected**:
```
  setup-venv          Setup Python virtual environment for pykwavers
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 3: setup-venv Creates Venv

**Preparation**:
```bash
# Remove existing venv to test fresh creation
rm -rf pykwavers/.venv/
```

**Command**:
```bash
cargo xtask setup-venv
```

**Expected Output**:
```
🐍 Setting up Python virtual environment...
📦 Creating virtual environment at: <path>/pykwavers/.venv
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
```

**Verify Files Exist**:
```bash
# Windows
ls pykwavers/.venv/Scripts/python.exe
ls pykwavers/.venv/Scripts/pip.exe
ls pykwavers/.venv/Scripts/maturin.exe

# Linux/macOS
ls pykwavers/.venv/bin/python
ls pykwavers/.venv/bin/pip
ls pykwavers/.venv/bin/maturin
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 4: setup-venv Idempotent

**Command**:
```bash
cargo xtask setup-venv
```

**Expected**:
```
🐍 Setting up Python virtual environment...
✅ Virtual environment already exists at: <path>/pykwavers/.venv
   Use --force to recreate
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 5: setup-venv --force Recreates

**Command**:
```bash
cargo xtask setup-venv --force
```

**Expected**:
```
🐍 Setting up Python virtual environment...
⚠️  Removing existing venv (--force specified)...
📦 Creating virtual environment at: <path>/pykwavers/.venv
✅ Virtual environment created
...
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 6: build-pykwavers Uses Venv

**Command**:
```bash
cargo xtask build-pykwavers --install
```

**Expected Output Includes**:
```
🔨 Building pykwavers...
🔍 Using maturin from venv: <path>/pykwavers/.venv/Scripts/maturin.exe
```

**Verify**:
```bash
# Check pykwavers installed to venv
pykwavers/.venv/Scripts/python -c "import pykwavers; print(pykwavers.__version__)"  # Windows
pykwavers/.venv/bin/python -c "import pykwavers; print(pykwavers.__version__)"      # Linux/macOS
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 7: install-kwave Uses Venv

**Command**:
```bash
cargo xtask install-kwave
```

**Expected**:
- Uses venv pip for installation
- Verifies with venv Python

**Verify**:
```bash
# Check k-wave-python installed to venv
pykwavers/.venv/Scripts/python -c "import kwave; print(kwave.__version__)"  # Windows
pykwavers/.venv/bin/python -c "import kwave; print(kwave.__version__)"      # Linux/macOS
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 8: compare Uses Venv

**Command**:
```bash
cargo xtask compare --pykwavers-only
```

**Expected**:
- Creates venv if missing (auto-setup)
- Runs comparison script with venv Python

**Verify Output Directory**:
```bash
ls pykwavers/examples/comparison_results/
# Should contain: comparison.png, metrics.csv, sensor_data.npz, validation_report.txt
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 9: validate Auto-Creates Venv

**Preparation**:
```bash
# Remove venv to test auto-creation
rm -rf pykwavers/.venv/
```

**Command**:
```bash
cargo xtask validate --skip-kwave
```

**Expected**:
```
================================================================================
🎯 Running full validation workflow
================================================================================

Step 0/4: Setting up virtual environment
--------------------------------------------------------------------------------
🐍 Setting up Python virtual environment...
📦 Creating virtual environment at: <path>/pykwavers/.venv
...
✅ Virtual environment ready!
...

Step 1/4: Building pykwavers
...

Step 2/4: Skipping k-wave-python installation
...

Step 3/4: Running comparison
...

================================================================================
✅ Validation workflow complete!
================================================================================
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 10: validate Full Workflow

**Command**:
```bash
cargo xtask validate
```

**Expected**:
- Step 0: Venv ready (or created)
- Step 1: Build pykwavers with venv maturin
- Step 2: Install k-wave-python with venv pip
- Step 3: Run comparison with venv python
- Step 4: Validation complete

**Verify Results**:
```bash
ls pykwavers/examples/validation_results/
# Should contain: comparison.png, metrics.csv, sensor_data.npz, validation_report.txt
```

**Status**: [ ] PASS / [ ] FAIL

---

## Integration Tests

### ✅ Test 11: No Global Pollution

**Command**:
```bash
# Try to import pykwavers from system Python (should fail if not globally installed)
python -c "import pykwavers" 2>&1
```

**Expected**:
```
ModuleNotFoundError: No module named 'pykwavers'
```

**If pykwavers is found**, verify it's from a different installation, not from xtask.

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 12: Venv Isolation

**Commands**:
```bash
# List venv packages
pykwavers/.venv/Scripts/pip list  # Windows
pykwavers/.venv/bin/pip list      # Linux/macOS

# Check system packages (should be different)
pip list
```

**Expected**:
- Venv should have: pykwavers, k-wave-python, maturin, numpy, scipy, matplotlib, etc.
- System Python may have different packages
- No interference between the two

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 13: Cross-Platform Paths

**Commands**:
```bash
# Test on current platform
cargo xtask setup-venv --force

# Verify correct platform-specific paths created
# Windows: .venv/Scripts/*.exe
# Linux/macOS: .venv/bin/*
```

**Expected**:
- xtask detects platform correctly
- Uses correct executable paths
- Commands work without errors

**Status**: [ ] PASS / [ ] FAIL

---

## Error Handling Tests

### ✅ Test 14: Missing Venv Error Message

**Preparation**:
```bash
# Remove venv
rm -rf pykwavers/.venv/

# Manually try to call venv_python() by modifying xtask temporarily
# Or simulate by checking error messages
```

**Expected Error Message Should Include**:
```
Python executable not found in venv: <path>
Run: cargo xtask setup-venv
```

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 15: Invalid Python Installation

**Test** (if applicable):
- Rename python.exe temporarily
- Run xtask command

**Expected**:
- Clear error message
- Actionable recovery steps

**Status**: [ ] PASS / [ ] FAIL / [ ] N/A

---

## Documentation Tests

### ✅ Test 16: README Instructions

**Action**:
1. Open `pykwavers/README.md`
2. Follow installation instructions for "From Source (Recommended)"

**Expected**:
- Instructions mention xtask
- `cargo xtask validate` is shown as primary method
- Instructions work as written

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 17: Quick Start Guide

**Action**:
1. Open `KWAVE_PYTHON_QUICK_START.md`
2. Follow TL;DR section

**Expected**:
```bash
cd kwavers
cargo xtask validate
```

**Should complete successfully**

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 18: Venv Workflow Documentation

**Action**:
1. Open `VENV_WORKFLOW.md`
2. Verify all commands in "Quick Command Reference" section work

**Expected**:
- All commands execute successfully
- Help text matches documentation

**Status**: [ ] PASS / [ ] FAIL

---

## CI/CD Simulation Tests

### ✅ Test 19: Fresh Environment Setup

**Simulate CI** (in a fresh directory or Docker):
```bash
git clone <repo>
cd kwavers
cargo xtask validate
```

**Expected**:
- Works without any manual setup
- Creates venv automatically
- Builds and validates successfully

**Status**: [ ] PASS / [ ] FAIL / [ ] N/A (requires clean environment)

---

### ✅ Test 20: Cached Venv (CI Optimization)

**Commands**:
```bash
# First run
cargo xtask validate

# Second run (venv cached)
cargo xtask validate --skip-build
```

**Expected**:
- Second run is much faster
- Reuses existing venv
- No rebuild needed

**Status**: [ ] PASS / [ ] FAIL

---

## Performance Tests

### ✅ Test 21: Setup Time

**Command**:
```bash
time cargo xtask setup-venv --force
```

**Expected**:
- Completes in < 2 minutes (typical)
- Time depends on network speed for downloads

**Actual Time**: _____ seconds

**Status**: [ ] PASS / [ ] FAIL

---

### ✅ Test 22: Build Time with Venv

**Command**:
```bash
time cargo xtask build-pykwavers --install
```

**Expected**:
- Comparable to manual maturin build
- No significant overhead from venv

**Actual Time**: _____ seconds

**Status**: [ ] PASS / [ ] FAIL

---

## Summary

### Test Results

Total Tests: 22  
Passed: ___  
Failed: ___  
N/A: ___

### Critical Tests (Must Pass)

- [ ] Test 1: xtask compiles
- [ ] Test 2: setup-venv command exists
- [ ] Test 3: setup-venv creates venv
- [ ] Test 6: build-pykwavers uses venv
- [ ] Test 9: validate auto-creates venv
- [ ] Test 10: validate full workflow

### Overall Status

[ ] ✅ PASS - All critical tests passed  
[ ] ⚠️  PARTIAL - Some non-critical tests failed  
[ ] ❌ FAIL - Critical tests failed

---

## Issues Found

(List any issues discovered during validation)

1. 
2. 
3. 

---

## Recommendations

(List any improvements or fixes needed)

1. 
2. 
3. 

---

## Sign-Off

**Tested By**: _______________  
**Date**: _______________  
**Environment**: _______________  
**Platform**: [ ] Windows [ ] Linux [ ] macOS  
**Python Version**: _______________  
**Rust Version**: _______________

---

**Validation Status**: [ ] APPROVED [ ] NEEDS WORK

---

## Notes

(Additional observations or comments)




---

**Document Version**: 1.0  
**Last Updated**: 2026-02-04 (Sprint 217 Session 10)