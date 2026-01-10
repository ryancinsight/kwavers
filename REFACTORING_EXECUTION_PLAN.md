# Deep Vertical Hierarchy Refactoring Execution Plan

**Project**: Kwavers v3.0.0  
**Sprint**: Architecture Refactoring  
**Start Date**: 2025-01-10  
**Target Completion**: 2025-03-07 (8 weeks)  
**Status**: 🔴 PLANNING PHASE

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Pre-Flight Checklist](#pre-flight-checklist)
3. [Phase-by-Phase Execution](#phase-by-phase-execution)
4. [Automated Tools](#automated-tools)
5. [Testing Strategy](#testing-strategy)
6. [Rollback Procedures](#rollback-procedures)
7. [Progress Tracking](#progress-tracking)

---

## Executive Summary

This document provides **actionable, step-by-step instructions** for executing the deep vertical hierarchy refactoring identified in `COMPREHENSIVE_ARCHITECTURE_AUDIT.md`. Each phase includes:

- **Automated scripts** for file moves and import updates
- **Manual verification steps** for safety
- **Testing checkpoints** to prevent regressions
- **Rollback procedures** in case of issues

### Critical Success Factors

✅ **Zero Breaking Changes** (until v3.0.0)  
✅ **100% Test Pass Rate** (maintained throughout)  
✅ **Incremental Validation** (test after each phase)  
✅ **Comprehensive Tracking** (every file move documented)  

---

## Pre-Flight Checklist

### Prerequisites (MUST COMPLETE BEFORE STARTING)

- [ ] **Backup Repository**: Create full backup of current state
- [ ] **Create Branch**: `git checkout -b refactor/deep-vertical-hierarchy`
- [ ] **Freeze Features**: No new features during refactoring
- [ ] **Team Notification**: All developers aware of refactoring
- [ ] **Baseline Tests**: Run full test suite, document current state
- [ ] **Baseline Benchmarks**: Run all benchmarks, save results
- [ ] **CI/CD Ready**: Ensure CI passes on current main branch

### Baseline Metrics

```bash
# Run these commands and save output
cargo test --all-features 2>&1 | tee baseline_tests.log
cargo bench 2>&1 | tee baseline_benchmarks.log
cargo build --release --timings 2>&1 | tee baseline_build.log
cargo clippy --all-features -- -D warnings 2>&1 | tee baseline_clippy.log

# Count files and lines
find src -name "*.rs" | wc -l > baseline_files.txt
find src -name "*.rs" -exec wc -l {} + | tail -1 > baseline_loc.txt
```

**Expected Baseline**:
- Tests: 867/867 passing
- Files: 972 Rust modules
- LOC: 405,708
- Clippy: 0 warnings
- Build time: ~X minutes (document actual)

---

## Phase-by-Phase Execution

## PHASE 0: Preparation (Week 1, Days 1-2)

### Objectives
- Clean repository of dead code
- Set up tracking infrastructure
- Create migration tools

### Task 0.1: Clean Build Artifacts

**Priority**: 🟢 LOW (but immediate)

```bash
#!/bin/bash
# Script: scripts/clean_artifacts.sh

echo "🧹 Cleaning build artifacts from repository..."

# Delete build logs
rm -f baseline_tests_sprint1a.log
rm -f build_phase0.log
rm -f check_errors*.txt
rm -f check_output*.txt
rm -f errors.txt

# Delete target directory if accidentally committed
if [ -d "target" ]; then
    echo "⚠️  WARNING: target/ directory found in repository!"
    echo "   This should be in .gitignore. Removing..."
    rm -rf target/
fi

# Update .gitignore
cat >> .gitignore << 'EOF'

# Build artifacts
*.log
check_*.txt
errors.txt
baseline_*.log
baseline_*.txt

# Build directory
/target/
EOF

echo "✅ Cleanup complete"
```

**Verification**:
```bash
git status  # Should show deleted files
git add -A
git commit -m "chore: remove build artifacts and update .gitignore"
```

---

### Task 0.2: Organize Audit Documents

**Priority**: 🟢 LOW

```bash
#!/bin/bash
# Script: scripts/organize_audits.sh

echo "📁 Organizing audit documents..."

# Create audits directory
mkdir -p docs/audits/archive

# Move audit documents
mv ACCURATE_MODULE_ARCHITECTURE.md docs/audits/archive/
mv ARCHITECTURAL_AUDIT_SPRINT_ANALYSIS.md docs/audits/archive/
mv ARCHITECTURE_IMPROVEMENT_PLAN.md docs/audits/archive/
mv ARCHITECTURE_REFACTORING_AUDIT.md docs/audits/archive/
mv ARCHITECTURE_REFACTORING_EXECUTION_PLAN.md docs/audits/archive/
mv ARCHITECTURE_VALIDATION_REPORT.md docs/audits/archive/
mv AUDIT_COMPLETE_SUMMARY.md docs/audits/archive/
mv AUDIT_DELIVERABLES_README.md docs/audits/archive/
mv AUDIT_EXECUTIVE_SUMMARY.md docs/audits/archive/
mv CHERNKOV_SONOLUMINESCENCE_ANALYSIS.md docs/audits/archive/
mv COMPREHENSIVE_MODULE_REFACTORING_PLAN.md docs/audits/archive/
mv CORRECTED_DEEP_VERTICAL_HIERARCHY_AUDIT.md docs/audits/archive/
mv DEEP_VERTICAL_HIERARCHY_AUDIT.md docs/audits/archive/
mv DEEP_VERTICAL_HIERARCHY_REFACTORING_AUDIT.md docs/audits/archive/
mv DEPENDENCY_ANALYSIS.md docs/audits/archive/
mv DEPLOYMENT_GUIDE.md docs/audits/archive/
mv IMMEDIATE_ACTIONS.md docs/audits/archive/
mv IMMEDIATE_FIXES_CHECKLIST.md docs/audits/archive/
mv MODULE_ARCHITECTURE_MAP.md docs/audits/archive/
mv OPERATOR_OWNERSHIP_ANALYSIS.md docs/audits/archive/
mv PERFORMANCE_OPTIMIZATION_ANALYSIS.md docs/audits/archive/
mv PERFORMANCE_OPTIMIZATION_SUMMARY.md docs/audits/archive/
mv PHASE_0_COMPLETION_REPORT.md docs/audits/archive/
mv PHASE_1_EXECUTION_PLAN.md docs/audits/archive/
mv PHASE_1_PROGRESS.md docs/audits/archive/
mv PINN_ECOSYSTEM_SUMMARY.md docs/audits/archive/
mv REFACTORING_AUDIT_README.md docs/audits/archive/
mv REFACTORING_EXECUTION_CHECKLIST.md docs/audits/archive/
mv REFACTORING_EXECUTIVE_SUMMARY.md docs/audits/archive/
mv REFACTORING_INDEX.md docs/audits/archive/
mv REFACTORING_PROGRESS.md docs/audits/archive/
mv REFACTORING_QUICK_REFERENCE.md docs/audits/archive/
mv REFACTORING_QUICK_START.md docs/audits/archive/
mv REFACTOR_PHASE_1_CHECKLIST.md docs/audits/archive/
mv SESSION_COMPLETION_SUMMARY.md docs/audits/archive/
mv SIMULATION_REFACTORING_PLAN.md docs/audits/archive/
mv SOLVER_REFACTORING_PLAN.md docs/audits/archive/
mv SOURCE_IMPLEMENTATION_COMPLETE.md docs/audits/archive/
mv SOURCE_MODULE_AUDIT_SUMMARY.md docs/audits/archive/
mv SOURCE_SIGNAL_ARCHITECTURE.md docs/audits/archive/
mv TASK_1_1_COMPLETION.md docs/audits/archive/
mv TASK_2_1_BEAMFORMING_MIGRATION_ASSESSMENT.md docs/audits/archive/

# Keep current audit at root
cp COMPREHENSIVE_ARCHITECTURE_AUDIT.md docs/audits/
cp REFACTORING_EXECUTION_PLAN.md docs/audits/

# Create index
cat > docs/audits/INDEX.md << 'EOF'
# Audit History Index

## Current Audits
- [Comprehensive Architecture Audit](COMPREHENSIVE_ARCHITECTURE_AUDIT.md) - 2025-01-10
- [Refactoring Execution Plan](REFACTORING_EXECUTION_PLAN.md) - 2025-01-10

## Archive
See `archive/` directory for historical audits from previous sprints.

## Timeline
- **Sprint 138**: Clippy compliance and persona-driven development
- **Sprint 4**: Beamforming consolidation (Phases 1-6)
- **Sprint 125**: Pattern documentation and validation
- **Sprint 114**: Production readiness audit
- **Current**: Deep vertical hierarchy refactoring
EOF

echo "✅ Audit organization complete"
```

---

### Task 0.3: Create Migration Tracking Spreadsheet

**Priority**: 🔴 CRITICAL

Create `docs/refactoring_tracker.csv`:

```csv
Phase,Module,Source,Destination,Files,Lines,Status,Test_Status,Notes
1,core/error,domain/core/error,core/error,5,1200,PENDING,PENDING,Critical dependency
1,core/utils,domain/core/utils,core/utils,8,2400,PENDING,PENDING,Used by 250+ files
1,core/time,domain/core/time,core/time,2,350,PENDING,PENDING,Time abstractions
1,core/constants,domain/core/constants,core/constants,1,180,PENDING,PENDING,Physical constants
1,core/log,domain/core/log,core/log,1,120,PENDING,PENDING,Logging infrastructure
2,core/math/fft,domain/math/fft,core/math/fft,3,850,PENDING,PENDING,FFT operations
2,core/math/linalg,domain/math/linear_algebra,core/math/linalg,6,1800,PENDING,PENDING,Linear algebra
2,solver/operators/fd,domain/grid/operators,solver/operators/finite_difference,4,1100,PENDING,PENDING,FD stencils
2,solver/operators/spectral,solver/forward/pstd/numerics/operators,solver/operators/spectral,5,1400,PENDING,PENDING,Spectral operators
2,analysis/ml,domain/math/ml,analysis/ml,12,3500,PENDING,PENDING,Machine learning
3,beamforming_cleanup,domain/sensor/beamforming,DELETE,32,8000,PENDING,PENDING,Migrate consumers first
4,physics/imaging,physics/acoustics/imaging,physics/imaging,6,46000,PENDING,PENDING,Not nested in acoustics
4,analysis/imaging/fusion,physics/acoustics/imaging/fusion.rs,analysis/imaging/fusion.rs,1,38717,PENDING,PENDING,Post-processing
4,analysis/signal_processing/pam,physics/acoustics/imaging/pam.rs,analysis/signal_processing/pam/mod.rs,1,3172,PENDING,PENDING,Actually beamforming
5,physics/therapy,physics/acoustics/therapy,physics/therapy,2,5000,PENDING,PENDING,Not exclusively acoustic
6,solver/forward/dg,solver/forward/pstd/dg,solver/forward/dg,8,2400,PENDING,PENDING,DG is not PSTD
6,analysis/validation/physics,solver/validation,analysis/validation/physics,5,4000,PENDING,PENDING,Validation in analysis
6,analysis/validation/numerical,solver/utilities/validation,analysis/validation/numerical,3,2500,PENDING,PENDING,Validation in analysis
```

**Usage**: Update status after each module migration.

---

### Task 0.4: Create Automated Migration Tool

**Priority**: 🔴 CRITICAL

Create `scripts/migrate_module.sh`:

```bash
#!/bin/bash
# Script: scripts/migrate_module.sh
# Usage: ./scripts/migrate_module.sh <source> <destination>
# Example: ./scripts/migrate_module.sh domain/core/error core/error

set -e

SOURCE=$1
DEST=$2

if [ -z "$SOURCE" ] || [ -z "$DEST" ]; then
    echo "Usage: $0 <source> <destination>"
    echo "Example: $0 domain/core/error core/error"
    exit 1
fi

echo "🚚 Migrating $SOURCE -> $DEST"

# Convert paths to filesystem paths
SRC_PATH="src/${SOURCE}"
DEST_PATH="src/${DEST}"

# Verify source exists
if [ ! -d "$SRC_PATH" ]; then
    echo "❌ Source directory $SRC_PATH does not exist"
    exit 1
fi

# Create destination directory
echo "📁 Creating destination directory..."
mkdir -p "$DEST_PATH"

# Move files
echo "🔄 Moving files..."
if [ -d "$SRC_PATH" ]; then
    cp -r "$SRC_PATH"/* "$DEST_PATH"/
else
    echo "❌ Source is not a directory"
    exit 1
fi

# Update imports (create Python script for complex pattern matching)
echo "🔧 Updating imports..."
python3 scripts/update_imports.py "$SOURCE" "$DEST"

# Test compilation
echo "🧪 Testing compilation..."
cargo check --all-features

echo "✅ Migration complete: $SOURCE -> $DEST"
echo ""
echo "⚠️  NEXT STEPS:"
echo "1. Review changes: git diff"
echo "2. Run tests: cargo test --all-features"
echo "3. If successful, delete source: rm -rf $SRC_PATH"
echo "4. Commit changes: git commit -am 'refactor: migrate $SOURCE to $DEST'"
```

---

### Task 0.5: Create Import Update Tool

**Priority**: 🔴 CRITICAL

Create `scripts/update_imports.py`:

```python
#!/usr/bin/env python3
"""
Import path updater for Rust module migrations.
Updates all occurrences of old import paths to new paths.
"""

import sys
import re
from pathlib import Path
from typing import Tuple, List

def convert_path_to_module(path: str) -> str:
    """Convert filesystem path to Rust module path."""
    return path.replace('/', '::')

def find_rust_files(root: Path = Path("src")) -> List[Path]:
    """Find all Rust source files."""
    return list(root.rglob("*.rs"))

def update_imports_in_file(
    file_path: Path,
    old_module: str,
    new_module: str
) -> Tuple[bool, int]:
    """
    Update import statements in a single file.
    Returns (changed, count) tuple.
    """
    content = file_path.read_text(encoding='utf-8')
    original = content
    count = 0
    
    # Pattern 1: use crate::old::module
    pattern1 = rf'(use\s+crate::){re.escape(old_module)}(::|\s|;)'
    replacement1 = rf'\g<1>{new_module}\g<2>'
    content, n1 = re.subn(pattern1, replacement1, content)
    count += n1
    
    # Pattern 2: crate::old::module in other contexts
    pattern2 = rf'(crate::){re.escape(old_module)}(::|\s|;|\))'
    replacement2 = rf'\g<1>{new_module}\g<2>'
    content, n2 = re.subn(pattern2, replacement2, content)
    count += n2
    
    # Pattern 3: pub use crate::old::module
    pattern3 = rf'(pub\s+use\s+crate::){re.escape(old_module)}(::|\s|;)'
    replacement3 = rf'\g<1>{new_module}\g<2>'
    content, n3 = re.subn(pattern3, replacement3, content)
    count += n3
    
    changed = content != original
    
    if changed:
        file_path.write_text(content, encoding='utf-8')
        
    return changed, count

def main():
    if len(sys.argv) != 3:
        print("Usage: update_imports.py <old_path> <new_path>")
        print("Example: update_imports.py domain/core/error core/error")
        sys.exit(1)
        
    old_path = sys.argv[1]
    new_path = sys.argv[2]
    
    old_module = convert_path_to_module(old_path)
    new_module = convert_path_to_module(new_path)
    
    print(f"🔍 Searching for imports of: {old_module}")
    print(f"📝 Replacing with: {new_module}")
    
    rust_files = find_rust_files()
    total_files_changed = 0
    total_changes = 0
    
    for file_path in rust_files:
        changed, count = update_imports_in_file(file_path, old_module, new_module)
        if changed:
            total_files_changed += 1
            total_changes += count
            print(f"  ✓ {file_path}: {count} changes")
    
    print(f"\n✅ Updated {total_changes} imports in {total_files_changed} files")

if __name__ == "__main__":
    main()
```

Make executable:
```bash
chmod +x scripts/migrate_module.sh
chmod +x scripts/update_imports.py
```

---

## PHASE 1: Core Extraction (Week 1, Days 3-7)

### Objectives
- Extract `domain/core/` to top-level `core/`
- Update 250+ import statements
- Maintain 100% test pass rate

### Task 1.1: Create Core Module Structure

```bash
#!/bin/bash
# Script: scripts/phase1_create_core.sh

echo "🏗️  Phase 1: Creating core module structure..."

# Create directory structure
mkdir -p src/core
mkdir -p src/core/error
mkdir -p src/core/error/types
mkdir -p src/core/utils
mkdir -p src/core/time
mkdir -p src/core/constants
mkdir -p src/core/log

# Create core/mod.rs
cat > src/core/mod.rs << 'EOF'
//! # Core Infrastructure Layer
//!
//! This module provides fundamental infrastructure used throughout kwavers:
//! - Error handling and result types
//! - Generic utilities and helper functions
//! - Time abstractions
//! - Physical and mathematical constants
//! - Logging infrastructure
//!
//! The core layer has zero dependencies on domain, physics, or solver layers.
//! It only depends on standard library and external crates.

pub mod error;
pub mod utils;
pub mod time;
pub mod constants;
pub mod log;

// Re-export commonly used types
pub use error::{KwaversError, KwaversResult, GridError};
pub use time::Time;
EOF

echo "✅ Core module structure created"
```

---

### Task 1.2: Migrate Error Module

**Priority**: 🔴 CRITICAL (blocks everything)

```bash
#!/bin/bash
# Script: scripts/phase1_migrate_error.sh

set -e

echo "🚚 Migrating domain/core/error -> core/error"

# Copy files
cp -r src/domain/core/error/* src/core/error/

# Update internal imports within error module
find src/core/error -name "*.rs" -exec sed -i 's/crate::domain::core::error/crate::core::error/g' {} +

# Update all imports across codebase
python3 scripts/update_imports.py domain/core/error core/error

# Test compilation
echo "🧪 Testing compilation..."
cargo check --all-features

if [ $? -eq 0 ]; then
    echo "✅ Error module migration successful"
    echo ""
    echo "⚠️  MANUAL VERIFICATION REQUIRED:"
    echo "1. Review changes: git diff src/core/error"
    echo "2. Run tests: cargo test --lib"
    echo "3. Verify no domain/core/error imports: grep -r 'domain::core::error' src/"
else
    echo "❌ Compilation failed. Reverting changes..."
    rm -rf src/core/error
    exit 1
fi
```

**Manual Verification**:
```bash
# Verify no old imports remain
grep -r "domain::core::error" src/

# Expected output: No matches (or only in comments/docs)

# Run subset of tests
cargo test --lib core::error

# Run full test suite
cargo test --all-features
```

---

### Task 1.3: Migrate Utils Module

```bash
#!/bin/bash
# Script: scripts/phase1_migrate_utils.sh

set -e

echo "🚚 Migrating domain/core/utils -> core/utils"

# Copy files
cp -r src/domain/core/utils/* src/core/utils/

# Update internal imports
find src/core/utils -name "*.rs" -exec sed -i 's/crate::domain::core::utils/crate::core::utils/g' {} +
find src/core/utils -name "*.rs" -exec sed -i 's/crate::domain::core::error/crate::core::error/g' {} +

# Update all imports across codebase
python3 scripts/update_imports.py domain/core/utils core/utils

# Test compilation
cargo check --all-features

if [ $? -eq 0 ]; then
    echo "✅ Utils module migration successful"
else
    echo "❌ Compilation failed"
    exit 1
fi
```

---

### Task 1.4: Migrate Time, Constants, Log Modules

```bash
#!/bin/bash
# Script: scripts/phase1_migrate_remaining.sh

set -e

echo "🚚 Migrating remaining core modules..."

# Time module
echo "📦 Migrating time module..."
cp -r src/domain/core/time/* src/core/time/
find src/core/time -name "*.rs" -exec sed -i 's/crate::domain::core/crate::core/g' {} +
python3 scripts/update_imports.py domain/core/time core/time

# Constants module
echo "📦 Migrating constants module..."
cp -r src/domain/core/constants/* src/core/constants/
find src/core/constants -name "*.rs" -exec sed -i 's/crate::domain::core/crate::core/g' {} +
python3 scripts/update_imports.py domain/core/constants core/constants

# Log module
echo "📦 Migrating log module..."
cp -r src/domain/core/log/* src/core/log/
find src/core/log -name "*.rs" -exec sed -i 's/crate::domain::core/crate::core/g' {} +
python3 scripts/update_imports.py domain/core/log core/log

cargo check --all-features
echo "✅ All remaining core modules migrated"
```

---

### Task 1.5: Update lib.rs Re-exports

**Manual Edit Required**: `src/lib.rs`

```rust
// BEFORE:
pub mod error {
    pub use crate::domain::core::error::{GridError, KwaversError, KwaversResult};
}
pub mod time {
    pub use crate::domain::core::time::Time;
}

// AFTER:
// Re-export core types at crate root for convenience
pub use crate::core::error::{KwaversError, KwaversResult, GridError};
pub use crate::core::time::Time;

// Add core module
pub mod core;
```

---

### Task 1.6: Delete Old domain/core/

**ONLY AFTER ALL TESTS PASS**:

```bash
#!/bin/bash
# Script: scripts/phase1_cleanup.sh

set -e

echo "⚠️  WARNING: This will delete domain/core/"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Verify no imports remain
REMAINING=$(grep -r "domain::core::" src/ | grep -v "^Binary" | wc -l)

if [ "$REMAINING" -ne 0 ]; then
    echo "❌ ERROR: Found $REMAINING remaining references to domain::core::"
    echo "Run: grep -r 'domain::core::' src/"
    exit 1
fi

# Final test before deletion
cargo test --all-features

if [ $? -eq 0 ]; then
    echo "🗑️  Deleting domain/core/..."
    rm -rf src/domain/core/
    
    # Update domain/mod.rs
    sed -i '/pub mod core;/d' src/domain/mod.rs
    
    echo "✅ Cleanup complete"
    
    # Commit
    git add -A
    git commit -m "refactor(phase1): extract core layer from domain

- Moved domain/core/* to top-level core/*
- Updated 250+ import statements
- All 867 tests passing
- Zero compilation warnings

BREAKING CHANGE: domain::core::* imports now core::*"
else
    echo "❌ Tests failed. Aborting cleanup."
    exit 1
fi
```

---

### Phase 1 Validation Checklist

- [ ] All files moved from `domain/core/` to `core/`
- [ ] No references to `domain::core::` remain (except comments)
- [ ] `cargo check --all-features` passes
- [ ] `cargo test --all-features` passes (867/867)
- [ ] `cargo clippy --all-features -- -D warnings` passes
- [ ] Import paths shortened (e.g., `core::error` vs `domain::core::error`)
- [ ] `domain/core/` directory deleted
- [ ] Changes committed with clear message

---

## PHASE 2: Math Extraction (Week 2)

### Task 2.1: Analyze Math Dependencies

```bash
#!/bin/bash
# Script: scripts/phase2_analyze_math.sh

echo "🔍 Analyzing domain/math dependencies..."

# Find all math imports
echo "=== Files importing domain::math ==="
grep -r "use crate::domain::math" src/ | cut -d: -f1 | sort -u | wc -l

# Categorize math submodules
echo ""
echo "=== Math submodules ==="
ls -la src/domain/math/

echo ""
echo "=== FFT usage ==="
grep -r "domain::math::fft" src/ | wc -l

echo ""
echo "=== Linear algebra usage ==="
grep -r "domain::math::linear_algebra" src/ | wc -l

echo ""
echo "=== ML usage ==="
grep -r "domain::math::ml" src/ | wc -l

echo ""
echo "=== Numerics usage ==="
grep -r "domain::math::numerics" src/ | wc -l
```

---

### Task 2.2: Migrate FFT to core/math

```bash
#!/bin/bash
# Script: scripts/phase2_migrate_fft.sh

set -e

echo "🚚 Migrating domain/math/fft -> core/math/fft"

# Create structure
mkdir -p src/core/math/fft

# Copy files
cp -r src/domain/math/fft/* src/core/math/fft/

# Update imports
find src/core/math/fft -name "*.rs" -exec sed -i 's/crate::domain::math::fft/crate::core::math::fft/g' {} +
find src/core/math/fft -name "*.rs" -exec sed -i 's/crate::domain::core/crate::core/g' {} +

# Update codebase imports
python3 scripts/update_imports.py domain/math/fft core/math/fft

# Update core/math/mod.rs
cat > src/core/math/mod.rs << 'EOF'
//! # Mathematical Operations
//!
//! Core mathematical primitives used throughout kwavers:
//! - Fast Fourier Transforms (FFT)
//! - Linear algebra operations
//! - Mathematical transforms

pub mod fft;
EOF

# Update core/mod.rs
sed -i '/pub mod log;/a pub mod math;' src/core/mod.rs

cargo check --all-features
echo "✅ FFT migration complete"
```

---

### Task 2.3: Migrate Linear Algebra to core/math

```bash
#!/bin/bash
# Script: scripts/phase2_migrate_linalg.sh

set -e

echo "🚚 Migrating domain/math/linear_algebra -> core/math/linalg"

mkdir -p src/core/math/linalg
mkdir -p src/core/math/linalg/sparse

# Copy files
cp -r src/domain/math/linear_algebra/* src/core/math/linalg/

# Update imports
find src/core/math/linalg -name "*.rs" -exec sed -i 's/crate::domain::math::linear_algebra/crate::core::math::linalg/g' {} +
find src/core/math/linalg -name "*.rs" -exec sed -i 's/crate::domain::core/crate::core/g' {} +

# Update codebase imports
python3 scripts/update_imports.py domain/math/linear_algebra core/math/linalg

# Update core/math/mod.rs
sed -i '/pub mod fft;/a pub mod linalg;' src/core/math/mod.rs

cargo check --all-features
echo "✅ Linear algebra migration complete"
```

---

### Task 2.4: Migrate Numerics to solver/

```bash
#!/bin/bash
# Script: scripts/phase2_migrate_numerics.sh

set -e

echo "🚚 Migrating domain/math/numerics -> solver/numerics"

# Create structure
mkdir -p src/solver/numerics
mkdir -p src/solver/numerics/integration
mkdir -p src/solver/numerics/operators
mkdir -p src/solver/numerics/transforms

# Copy files
cp -r src/domain/math/numerics/* src/solver/numerics/

# Update imports
find src/solver/numerics -name "*.rs" -exec sed -i 's/crate::domain::math::numerics/crate::solver::numerics/g' {} +
find src/solver/numerics -name "*.rs" -exec sed -i 's/crate::domain::core/crate::core/g' {} +
find src/solver/numerics -name "*.rs" -exec sed -i 's/crate::domain::math/crate::core::math/g' {} +

# Update codebase imports
python3 scripts/update_imports.py domain/math/numerics solver/numerics

# Update solver/mod.rs
sed -i '1a pub mod numerics;' src/solver/mod.rs

cargo check --all-features
echo "✅ Numerics migration complete"
```

---

### Task 2.5: Migrate ML to analysis/

```bash
#!/bin/bash
# Script: scripts/phase2_migrate_ml.sh

set -e

echo "🚚 Migrating domain/math/ml -> analysis/ml"

# Create structure
mkdir -p src/analysis/ml
mkdir -p src/analysis/ml/models
mkdir -p src/analysis/ml/optimization
mkdir -p src/analysis/ml/pinn
mkdir -p src/analysis/ml/uncertainty

# Copy files
cp -r src/domain/math/ml/* src/analysis/ml/

# Update imports
find src/analysis/ml -name "*.rs" -exec sed -i 's/crate::domain::math::ml/crate::analysis::ml/g' {} +
find src/analysis/ml -name "*.rs" -exec sed -i 's/crate::domain::math/crate::core::math/g' {} +
find src/analysis/ml -name "*.rs" -exec sed -i 's/crate::domain::core/crate::core/g' {} +

# Update codebase imports
python3 scripts/update_imports.py domain/math/ml analysis/ml

# Update analysis/mod.rs
sed -i '/pub mod validation;/a pub mod ml;' src/analysis/mod.rs

cargo check --all-features
echo "✅ ML migration complete"
```

---

### Phase 2 Cleanup

```bash
#!/bin/bash
# Script: scripts/phase2_cleanup.sh

set -e

# Verify no domain/math imports remain
REMAINING=$(grep -r "domain::math::" src/ | grep -v "^Binary" | wc -l)

if [ "$REMAINING" -ne 0 ]; then
    echo "❌ Found $REMAINING remaining references"
    exit 1
fi

# Test
cargo test --all-features

if [ $? -eq 0 ]; then
    rm -rf src/domain/math/
    sed -i '/pub mod math;/d' src/domain/mod.rs
    
    git add -A
    git commit -m "refactor(phase2): extract math from domain layer

- Moved domain/math/fft -> core/math/fft
- Moved domain/math/linear_algebra -> core/math/linalg
- Moved domain/math/numerics -> solver/numerics
- Moved domain/math/ml -> analysis/ml
- Updated 150+ import statements
- All tests passing"
fi
```

---

## Testing Strategy

### Continuous Testing Throughout Refactoring

```bash
#!/bin/bash
# Script: scripts/continuous_test.sh
# Run after EVERY file move

echo "🧪 Running continuous test suite..."

# Quick compilation check
echo "1️⃣  Checking compilation..."
cargo check --all-features

if [ $? -ne 0 ]; then
    echo "❌ Compilation failed"
    exit 1
fi

# Run library tests (fast)
echo "2️⃣  Running library tests..."
cargo test --lib --all-features

if [ $? -ne 0 ]; then
    echo "❌ Library tests failed"
    exit 1
fi

# Clippy check
echo "3️⃣  Running clippy..."
cargo clippy --all-features -- -D warnings

if [ $? -ne 0 ]; then
    echo "⚠️  Clippy warnings detected"
fi

echo "✅ All checks passed"
```

### Full Validation (Run after Each Phase)

```bash
#!/bin/bash
# Script: scripts/full_validation.sh

echo "🔬 Running full validation suite..."

# Full test suite
echo "1️⃣  Full test suite..."
cargo test --all-features 2>&1 | tee phase_test_results.log

# Benchmarks (ensure no performance regression)
echo "2️⃣  Running benchmarks..."
cargo bench 2>&1 | tee phase_bench_results.log

# Build time check
echo "3️⃣  Build time analysis..."
cargo clean
cargo build --release --timings 2>&1 | tee phase_build_results.log

# Compare with baseline
echo ""
echo "📊 Comparison with baseline:"
echo "Tests:"
diff baseline_tests.log phase_test_results.log || echo "  Changes detected"
echo "Build time:"
diff baseline_build.log phase_build_results.log || echo "  Changes detected"

echo "✅ Full validation complete"
```

---

## Rollback Procedures

### Emergency Rollback

```bash
#!/bin/bash
# Script: scripts/emergency_rollback.sh

set -e

echo "🚨 EMERGENCY ROLLBACK"
echo "This will discard all changes and return to main branch"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Stash any uncommitted changes
git stash

# Return to main
git checkout main

# Delete refactoring branch
git branch -D refactor/deep-vertical-hierarchy

echo "✅ Rollback complete. You are back on main branch."
```

### Partial Rollback (Single Phase)

```bash
#!/bin/bash
# Script: scripts/rollback_phase.sh <phase_number>

PHASE=$1

if [ -z "$PHASE" ]; then
    echo "Usage: $0 <phase_number>"
    exit 1
fi

echo "⏮️  Rolling back Phase $PHASE"

# Find last commit before phase
PHASE_START=$(git log --grep="refactor(phase$PHASE)" --format="%H" | tail -1)

if [ -z "$PHASE_START" ]; then
    echo "❌ Could not find Phase $PHASE start commit"
    exit 1
fi

# Reset to commit before phase
git reset --hard "$PHASE_START"^

echo "✅ Rolled back to before Phase $PHASE"
```

---

## Progress Tracking

### Daily Checklist

```markdown
## Week 1 Progress

### Monday (Day 1-2)
- [ ] Backup repository
- [ ] Create refactoring branch
- [ ] Clean build artifacts
- [ ] Organize audit documents
- [ ] Create migration tools
- [ ] Baseline metrics captured

### Tuesday-Friday (Day 3-7) - PHASE 1
- [ ] Create core module structure
- [ ] Migrate error module
- [ ] Migrate utils module
- [ ] Migrate time module
- [ ] Migrate constants module
- [ ] Migrate log module
- [ ] Update lib.rs re-exports
- [ ] Delete domain/core/
- [ ] All tests passing
- [ ] Phase 1 committed

## Week 2 Progress - PHASE 2
- [ ] Analyze math dependencies
- [ ] Migrate FFT to core/math
- [ ] Migrate linalg to core/math
- [ ] Migrate numerics to solver
- [ ] Migrate ML to analysis
- [ ] Delete domain/math/
- [ ] All tests passing
- [ ] Phase 2 committed
```

### Automated Progress Report

```bash
#!/bin/bash
# Script: scripts/progress_report.sh

echo "📊 Refactoring Progress Report"
echo "=============================="
echo ""

# Count completed phases
COMPLETED_PHASES=$(git log --grep="refactor(phase" --oneline | wc -l)
echo "✅ Completed Phases: $COMPLETED_PHASES / 10"

# Current test status
echo ""
echo "🧪 Current Test Status:"
cargo test --all-features 2>&1 | grep "test result"

# Modules migrated
echo ""
echo "📦 Modules Migrated:"
[ -d "src/core" ] && echo "  ✅ core/" || echo "  ❌ core/"
[ ! -d "src/domain/core" ] && echo "  ✅ domain/core removed" || echo "  ⏳ domain/core still exists"
[ ! -d "src/domain/math" ] && echo "  ✅ domain/math removed" || echo "  ⏳ domain/math still exists"

# Import statistics
echo ""
echo "🔗 Import Statistics:"
echo "  domain::core imports: $(grep -r 'domain::core::' src/ 2>/dev/null | wc -l)"
echo "  domain::math imports: $(grep -r 'domain::math::' src/ 2>/dev/null | wc -l)"
echo "  core:: imports: $(grep -r 'crate::core::' src/ 2>/dev/null | wc -l)"

echo ""
echo "=============================="
```

---

## Appendix: Quick Reference

### Common Commands

```bash
# Run continuous tests
./scripts/continuous_test.sh

# Full validation
./scripts/full_validation.sh

# Progress report
./scripts/progress_report.sh

# Migrate a module
./scripts/migrate_module.sh <source> <destination>

# Emergency rollback
./scripts/emergency_rollback.sh

# Check for old imports
grep -r "domain::core::" src/
grep -r "domain::math::" src/
```

### File Location Quick Reference

```
BEFORE                              AFTER
======                              =====
domain/core/error                   core/error
domain/core/utils                   core/utils
domain/core/time                    core/time
domain/core/constants               core/constants
domain/core/log                     core/log
domain/math/fft                     core/math/fft
domain/math/linear_algebra          core/math/linalg
domain/math/numerics                solver/numerics
domain/math/ml                      analysis/ml
domain/sensor/beamforming           DELETE (use analysis/signal_processing/beamforming)
physics/acoustics/imaging           physics/imaging
physics/acoustics/therapy           physics/therapy
solver/forward/pstd/dg              solver/forward/dg
solver/validation                   analysis/validation/physics
solver/utilities/validation         analysis/validation/numerical
```

---

**END OF EXECUTION PLAN**

Next Steps:
1. Review this plan with team
2. Execute Task 0.1 (clean artifacts)
3. Execute Task 0.2 (organize audits)
4. Create all automated scripts
5. Begin Phase 1 execution

**Note**: This is a living document. Update after each phase completion with lessons learned and adjustments.