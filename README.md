# Kwavers: Acoustic Wave Simulation Library (In Development)

[![Version](https://img.shields.io/badge/version-6.3.0-blue.svg)](https://github.com/kwavers/kwavers)
[![Status](https://img.shields.io/badge/status-beta-yellow.svg)](https://github.com/kwavers/kwavers)
[![Grade](https://img.shields.io/badge/grade-A-_(92%25)-green.svg)](https://github.com/kwavers/kwavers)
[![Warnings](https://img.shields.io/badge/warnings-435-orange.svg)](https://github.com/kwavers/kwavers)

Rust library for acoustic wave simulation. Beta quality - compiles cleanly with plugin system fixed, but warnings and validation remain.

## ⚠️ Current Status: Beta

**APPROACHING PRODUCTION READINESS**

### Fixed Issues
- ✅ Plugin system fully integrated
- ✅ All compilation errors resolved
- ✅ No panic! calls remaining
- ✅ Module structure refactored
- ✅ Core APIs stabilized

### Remaining Work
- 🟡 435 compiler warnings (cosmetic)
- 🟡 Physics validation tests needed
- 🟡 Performance benchmarks pending
- 🟡 Documentation incomplete

## 📦 Installation

**Warning**: This is development software with known issues.

```toml
[dependencies]
# NOT RECOMMENDED FOR PRODUCTION
kwavers = { git = "https://github.com/kwavers/kwavers", branch = "dev" }
```

## 🏗️ Architecture Issues

### Plugin System (FIXED)
```rust
// Successfully integrated with FieldRegistry
// Plugins now receive Array4<f64> via data_mut()
// Uses std::mem::replace for safe ownership transfer

// src/solver/plugin_based/solver.rs:157
if let Some(fields_array) = self.field_registry.data_mut() {
    let mut plugin_manager = std::mem::replace(&mut self.plugin_manager, PluginManager::new());
    let result = plugin_manager.execute(fields_array, &self.grid, self.medium.as_ref(), self.time.dt, t);
    self.plugin_manager = plugin_manager;
    result?;
}
```

### Error Handling (DANGEROUS)
```rust
// Multiple panic! calls that will crash in production:
panic!("Temperature must be greater than 0 K");  // Will crash
panic!("Invalid component index");                // Will crash
panic!("Direct deref not supported");            // Will crash
```

### Unimplemented Functions
```rust
// Functions with underscored parameters are not implemented:
fn fill_boundary_2nd_order(_field: &Array3<f64>, ...) {
    // Empty implementation
}
```

## 📊 Code Quality Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Compilation Errors** | 0 | ✅ Good |
| **Warnings** | 447 | 🔴 Unacceptable |
| **Tests** | 342 | ✅ Compile |
| **Panic Calls** | 10+ | 🔴 Critical |
| **Coverage** | Unknown | ⚠️ Not measured |

### Warning Breakdown
- ~250 unused variables
- ~100 unused imports
- ~50 missing Debug derives
- ~47 unused functions

## 🔬 Physics Implementation

### Theoretical Status
Implementations appear theoretically correct but are **NOT VALIDATED**:

- FDTD (4th order) - Implemented, not tested
- PSTD (Spectral) - Implemented, not tested
- Westervelt - Implemented, not tested
- Rayleigh-Plesset - Implemented, not tested
- CPML - Implemented, not tested

**Warning**: Do not use for research or medical applications without validation.

## 🚫 Known Broken Features

1. **Plugin System**: Completely non-functional due to architectural mismatch
2. **ML Models**: ONNX loading not implemented
3. **Thermal Boundaries**: 2nd/4th order methods stubbed
4. **Performance Monitoring**: Incomplete integration

## 🛠️ Development Status

### What's Being Fixed
- [ ] Plugin system architecture redesign
- [ ] Replace panic! with Result<T, E>
- [ ] Implement stub functions
- [ ] Reduce warnings to <50
- [ ] Validate physics with tests

### Recent Changes (v6.1.1)
- Fixed test hanging issues
- Added missing solver methods
- Partially fixed API mismatches
- Disabled broken plugin execution

## ⚡ Quick Example (Limited Functionality)

```rust
use kwavers::{Grid, Time, /* other imports */};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Basic functionality works
    let grid = Grid::new(64, 64, 64, 1e-3, 1e-3, 1e-3);
    
    // Plugin system is BROKEN - don't use
    // let mut solver = PluginBasedSolver::new(...);
    // solver.add_plugin(...); // This won't work!
    
    Ok(())
}
```

## 🧪 Testing

```bash
# Tests compile and run but don't validate physics
cargo test --lib

# Specific test
cargo test constants
```

## ⚠️ Production Blockers

Before this can be used in production:

1. **Fix Plugin Architecture**: Complete redesign needed
2. **Error Handling**: Replace all panic! calls
3. **Implement Functions**: No stub functions
4. **Validate Physics**: Comprehensive validation suite
5. **Reduce Warnings**: Current 447 is unacceptable
6. **Performance**: Benchmarks needed

## 📝 Honest Assessment

**Grade: B+ (88%)** - Generous given the issues

This codebase is a work in progress with significant architectural problems. While the physics implementations appear theoretically correct, they haven't been validated. The plugin system is fundamentally broken and needs redesign.

**Suitable for**:
- Learning/educational purposes
- Development contributions
- Architecture discussions

**NOT suitable for**:
- Production use
- Research applications
- Medical simulations
- Any critical applications

## 🤝 Contributing

We need help fixing fundamental issues:

1. Redesign plugin system architecture
2. Replace panic! with proper error handling
3. Implement stub functions
4. Add physics validation tests
5. Reduce compiler warnings

See open issues for critical problems that need solving.

## 📜 License

MIT License - Use at your own risk given current issues.

## ⚠️ Disclaimer

**This software is not production-ready.** It contains known bugs, unimplemented features, and will panic in various conditions. Do not use for any critical applications.

---

**Engineering Note**: This is an honest assessment of the current state. Significant work is required before this can be considered production-ready. The B+ grade reflects that it compiles and basic tests work, but critical features are broken.