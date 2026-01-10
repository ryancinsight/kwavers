#!/bin/bash
# Cleanup Root Directory Documentation
#
# This script consolidates scattered architecture and refactoring documentation
# from the root directory into organized locations under docs/

set -e

echo "🧹 Cleaning up root directory documentation..."
echo ""

# Create target directories
mkdir -p docs/architecture
mkdir -p docs/archive

# Files to keep in root
KEEP_FILES=(
    "README.md"
    "LICENSE"
    "Cargo.toml"
    "Cargo.lock"
    "gap_audit.md"
    "prompt.yaml"
    "build.rs"
    "clippy.toml"
    "deny.toml"
    "docker-compose.yml"
    "Dockerfile"
    ".gitignore"
)

# Architecture documentation to consolidate
ARCH_DOCS=(
    "ACCURATE_MODULE_ARCHITECTURE.md:docs/architecture/module_structure.md"
    "ARCHITECTURE_IMPROVEMENT_PLAN.md:docs/architecture/improvement_plan.md"
    "ARCHITECTURE_REFACTORING_AUDIT.md:docs/architecture/refactoring_audit.md"
    "COMPREHENSIVE_MODULE_REFACTORING_PLAN.md:docs/architecture/module_refactoring_plan.md"
    "DEPENDENCY_ANALYSIS.md:docs/architecture/dependency_analysis.md"
    "ARCHITECTURAL_AUDIT_SPRINT_ANALYSIS.md:docs/architecture/current_audit.md"
)

# Completed/historical documentation to archive
ARCHIVE_DOCS=(
    "REFACTORING_EXECUTIVE_SUMMARY.md"
    "REFACTORING_PROGRESS.md"
    "REFACTORING_QUICK_REFERENCE.md"
    "REFACTOR_PHASE_1_CHECKLIST.md"
    "SIMULATION_REFACTORING_PLAN.md"
    "SOLVER_REFACTORING_PLAN.md"
    "SOURCE_IMPLEMENTATION_COMPLETE.md"
    "SOURCE_MODULE_AUDIT_SUMMARY.md"
    "SOURCE_SIGNAL_ARCHITECTURE.md"
)

# Performance/optimization documentation
PERF_DOCS=(
    "PERFORMANCE_OPTIMIZATION_ANALYSIS.md:docs/architecture/performance_optimization.md"
    "PERFORMANCE_OPTIMIZATION_SUMMARY.md:docs/architecture/performance_summary.md"
)

# Domain-specific analysis
DOMAIN_DOCS=(
    "CHERNKOV_SONOLUMINESCENCE_ANALYSIS.md:docs/physics/sonoluminescence_analysis.md"
    "PINN_ECOSYSTEM_SUMMARY.md:docs/ml/pinn_ecosystem.md"
)

# Build artifacts to delete
BUILD_ARTIFACTS=(
    "errors.txt"
    "ARCHITECTURE_VALIDATION_REPORT.md"
)

echo "📂 Moving architecture documentation..."
for mapping in "${ARCH_DOCS[@]}"; do
    src="${mapping%%:*}"
    dst="${mapping##*:}"
    if [ -f "$src" ]; then
        echo "  $src -> $dst"
        mv "$src" "$dst"
    else
        echo "  ⚠️  $src not found (skipping)"
    fi
done

echo ""
echo "📦 Archiving completed documentation..."
for doc in "${ARCHIVE_DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo "  $doc -> docs/archive/"
        mv "$doc" "docs/archive/"
    else
        echo "  ⚠️  $doc not found (skipping)"
    fi
done

echo ""
echo "⚡ Moving performance documentation..."
mkdir -p docs/architecture
for mapping in "${PERF_DOCS[@]}"; do
    src="${mapping%%:*}"
    dst="${mapping##*:}"
    if [ -f "$src" ]; then
        echo "  $src -> $dst"
        mv "$src" "$dst"
    else
        echo "  ⚠️  $src not found (skipping)"
    fi
done

echo ""
echo "🔬 Moving domain-specific documentation..."
mkdir -p docs/physics
mkdir -p docs/ml
for mapping in "${DOMAIN_DOCS[@]}"; do
    src="${mapping%%:*}"
    dst="${mapping##*:}"
    if [ -f "$src" ]; then
        echo "  $src -> $dst"
        mv "$src" "$dst"
    else
        echo "  ⚠️  $src not found (skipping)"
    fi
done

echo ""
echo "🗑️  Removing build artifacts..."
for artifact in "${BUILD_ARTIFACTS[@]}"; do
    if [ -f "$artifact" ]; then
        echo "  Deleting $artifact"
        rm "$artifact"
    else
        echo "  ⚠️  $artifact not found (skipping)"
    fi
done

echo ""
echo "📋 Creating architecture documentation index..."
cat > docs/architecture/README.md << 'EOF'
# Kwavers Architecture Documentation

This directory contains all architectural decision records, refactoring plans, and structural documentation for the kwavers codebase.

## Current Documentation

### Active Documents

- **[current_audit.md](current_audit.md)** - Current architectural audit with detailed cross-contamination analysis
- **[module_structure.md](module_structure.md)** - Accurate module hierarchy and organization
- **[improvement_plan.md](improvement_plan.md)** - Architectural improvement roadmap
- **[dependency_analysis.md](dependency_analysis.md)** - Module dependency graph analysis

### Refactoring Plans

- **[refactoring_audit.md](refactoring_audit.md)** - Comprehensive refactoring audit
- **[module_refactoring_plan.md](module_refactoring_plan.md)** - Module-by-module refactoring strategy

### Performance

- **[performance_optimization.md](performance_optimization.md)** - Performance analysis and optimization strategies
- **[performance_summary.md](performance_summary.md)** - Performance benchmark summaries

## Historical Documentation

Completed refactoring documentation has been moved to [`../archive/`](../archive/).

## Architecture Principles

Kwavers follows a strict **deep vertical hierarchy** with the following layer structure:

```
Layer 0: core       - Primitives (errors, constants, time)
Layer 1: infra      - Infrastructure (IO, runtime, API)
Layer 2: domain     - Domain primitives (grid, medium, boundary)
Layer 3: math       - Mathematical operations
Layer 4: physics    - Physics models
Layer 5: solver     - Numerical solvers
Layer 6: simulation - Simulation orchestration
Layer 7: clinical   - Clinical applications
Layer 8: analysis   - Post-processing
Layer 9: gpu        - Hardware acceleration (optional)
```

### Dependency Rules

1. **Bottom-up only**: Lower layers NEVER import from higher layers
2. **Adjacent layers**: Each layer primarily imports from layer N-1
3. **Core accessibility**: `core` is accessible to all layers
4. **Optional isolation**: Optional features (GPU, API) cannot be required dependencies

### Enforcement

Run architecture validation:

```bash
cd xtask
cargo run -- check-architecture
```

Generate markdown report:

```bash
cd xtask
cargo run -- check-architecture --markdown
```

Strict mode (fail on violations):

```bash
cd xtask
cargo run -- check-architecture --strict
```

## Contributing

When adding new documentation:

1. Place architecture/design docs in `docs/architecture/`
2. Place completed historical docs in `docs/archive/`
3. Update this README with links to new documents
4. Keep root directory clean - only essential files in root

## See Also

- [Main README](../../README.md) - Project overview and quick start
- [Product Requirements (PRD)](../prd.md) - Product vision and requirements
- [Software Requirements (SRS)](../srs.md) - Detailed functional requirements
- [Architecture Decisions (ADR)](../adr.md) - Architectural decision records
EOF

echo ""
echo "📋 Creating archive index..."
cat > docs/archive/README.md << 'EOF'
# Archived Documentation

This directory contains historical refactoring documentation that has been completed or superseded.

## Contents

These documents represent completed refactoring efforts and historical analysis:

- **REFACTORING_EXECUTIVE_SUMMARY.md** - Executive summary of major refactoring efforts
- **REFACTORING_PROGRESS.md** - Historical progress tracking
- **REFACTORING_QUICK_REFERENCE.md** - Quick reference guide (historical)
- **REFACTOR_PHASE_1_CHECKLIST.md** - Phase 1 refactoring checklist (completed)
- **SIMULATION_REFACTORING_PLAN.md** - Simulation module refactoring (completed)
- **SOLVER_REFACTORING_PLAN.md** - Solver module refactoring (completed)
- **SOURCE_IMPLEMENTATION_COMPLETE.md** - Source module implementation notes (completed)
- **SOURCE_MODULE_AUDIT_SUMMARY.md** - Source module audit (historical)
- **SOURCE_SIGNAL_ARCHITECTURE.md** - Signal architecture notes (historical)

## Purpose

These documents are preserved for:

1. **Historical context** - Understanding past architectural decisions
2. **Pattern learning** - Reviewing successful refactoring patterns
3. **Audit trail** - Compliance and review purposes
4. **Onboarding** - New contributors understanding project evolution

## Current Documentation

For active architecture documentation, see:

- [Architecture Documentation](../architecture/README.md)
- [Current Architectural Audit](../architecture/current_audit.md)

## Note

These documents may reference outdated module structures or naming conventions. Always refer to current architecture documentation for accurate information.
EOF

echo ""
echo "✅ Root directory cleanup complete!"
echo ""
echo "📊 Summary:"
echo "  - Architecture docs moved to: docs/architecture/"
echo "  - Historical docs moved to: docs/archive/"
echo "  - Build artifacts removed"
echo "  - Index files created"
echo ""
echo "🔍 Remaining markdown files in root:"
ls -1 *.md 2>/dev/null || echo "  (none)"
echo ""
echo "✨ Root directory is now clean and organized!"
