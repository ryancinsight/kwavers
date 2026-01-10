#!/bin/bash
# Script: scripts/progress_report.sh
# Purpose: Generate comprehensive refactoring progress report
# Usage: ./scripts/progress_report.sh

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Kwavers Deep Vertical Hierarchy Refactoring Progress      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# ============================================================================
# PHASE COMPLETION STATUS
# ============================================================================

echo "📊 PHASE COMPLETION STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Count completed phases from git commits
COMPLETED_PHASES=$(git log --grep="refactor(phase" --oneline 2>/dev/null | wc -l)
echo "  Completed Phases: $COMPLETED_PHASES / 10"
echo ""

# Check individual phase status
check_phase() {
    local phase=$1
    local description=$2
    if git log --grep="refactor(phase$phase)" --oneline 2>/dev/null | grep -q "phase$phase"; then
        echo "  ✅ Phase $phase: $description"
    else
        echo "  ⏳ Phase $phase: $description"
    fi
}

check_phase 0 "Preparation & Cleanup"
check_phase 1 "Core Extraction"
check_phase 2 "Math Extraction"
check_phase 3 "Beamforming Cleanup"
check_phase 4 "Imaging Consolidation"
check_phase 5 "Therapy Consolidation"
check_phase 6 "Solver Refactoring"
check_phase 7 "Validation Consolidation"
check_phase 8 "Hierarchy Flattening"
check_phase 9 "Documentation & Cleanup"
check_phase 10 "Final Validation"

echo ""

# ============================================================================
# MODULE MIGRATION STATUS
# ============================================================================

echo "📦 MODULE MIGRATION STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if modules exist in new locations
check_module() {
    local module=$1
    local description=$2
    if [ -d "src/$module" ]; then
        local file_count=$(find "src/$module" -name "*.rs" 2>/dev/null | wc -l)
        echo "  ✅ $description: $file_count files"
    else
        echo "  ❌ $description: Not created"
    fi
}

# Check if old modules are removed
check_removed() {
    local module=$1
    local description=$2
    if [ -d "src/$module" ]; then
        echo "  ⏳ $description: Still exists (needs removal)"
    else
        echo "  ✅ $description: Removed"
    fi
}

echo "New Modules:"
check_module "core" "core/"
check_module "core/math" "core/math/"
check_module "solver/numerics" "solver/numerics/"
check_module "analysis/ml" "analysis/ml/"
check_module "physics/imaging" "physics/imaging/"
check_module "physics/therapy" "physics/therapy/"

echo ""
echo "Old Modules (should be removed):"
check_removed "domain/core" "domain/core/"
check_removed "domain/math" "domain/math/"
check_removed "domain/sensor/beamforming" "domain/sensor/beamforming/"
check_removed "physics/acoustics/imaging" "physics/acoustics/imaging/"
check_removed "physics/acoustics/therapy" "physics/acoustics/therapy/"
check_removed "solver/forward/pstd/dg" "solver/forward/pstd/dg/"

echo ""

# ============================================================================
# IMPORT STATISTICS
# ============================================================================

echo "🔗 IMPORT PATH ANALYSIS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Count old import patterns (should decrease to zero)
OLD_CORE_IMPORTS=$(grep -r 'domain::core::' src/ 2>/dev/null | grep -v "^Binary" | wc -l)
OLD_MATH_IMPORTS=$(grep -r 'domain::math::' src/ 2>/dev/null | grep -v "^Binary" | wc -l)
OLD_BEAM_IMPORTS=$(grep -r 'domain::sensor::beamforming' src/ 2>/dev/null | grep -v "^Binary" | wc -l)

# Count new import patterns (should increase)
NEW_CORE_IMPORTS=$(grep -r 'crate::core::' src/ 2>/dev/null | grep -v "^Binary" | wc -l)
NEW_CORE_MATH_IMPORTS=$(grep -r 'crate::core::math' src/ 2>/dev/null | grep -v "^Binary" | wc -l)
NEW_BEAM_IMPORTS=$(grep -r 'analysis::signal_processing::beamforming' src/ 2>/dev/null | grep -v "^Binary" | wc -l)

echo "Old Import Patterns (target: 0):"
if [ "$OLD_CORE_IMPORTS" -eq 0 ]; then
    echo "  ✅ domain::core:: imports: $OLD_CORE_IMPORTS"
else
    echo "  ⚠️  domain::core:: imports: $OLD_CORE_IMPORTS (should be 0)"
fi

if [ "$OLD_MATH_IMPORTS" -eq 0 ]; then
    echo "  ✅ domain::math:: imports: $OLD_MATH_IMPORTS"
else
    echo "  ⚠️  domain::math:: imports: $OLD_MATH_IMPORTS (should be 0)"
fi

if [ "$OLD_BEAM_IMPORTS" -eq 0 ]; then
    echo "  ✅ domain::sensor::beamforming imports: $OLD_BEAM_IMPORTS"
else
    echo "  ⚠️  domain::sensor::beamforming imports: $OLD_BEAM_IMPORTS (should be 0)"
fi

echo ""
echo "New Import Patterns:"
echo "  📌 core:: imports: $NEW_CORE_IMPORTS"
echo "  📌 core::math imports: $NEW_CORE_MATH_IMPORTS"
echo "  📌 analysis::signal_processing::beamforming imports: $NEW_BEAM_IMPORTS"

echo ""

# ============================================================================
# BUILD & TEST STATUS
# ============================================================================

echo "🧪 BUILD & TEST STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Quick compilation check
echo "Compilation Status:"
if cargo check --all-features >/dev/null 2>&1; then
    echo "  ✅ cargo check --all-features: PASS"
else
    echo "  ❌ cargo check --all-features: FAIL"
fi

# Test status (quick check)
echo ""
echo "Test Status (run full suite for accurate results):"
TEST_RESULT=$(cargo test --lib --all-features 2>&1 | grep "test result" || echo "unknown")
if echo "$TEST_RESULT" | grep -q "ok"; then
    echo "  ✅ Library tests: PASS"
    echo "     $TEST_RESULT"
else
    echo "  ⚠️  Library tests: Status unclear (run manually)"
fi

# Clippy status
echo ""
echo "Code Quality:"
if cargo clippy --all-features -- -D warnings >/dev/null 2>&1; then
    echo "  ✅ Clippy (zero warnings): PASS"
else
    echo "  ⚠️  Clippy warnings detected"
fi

echo ""

# ============================================================================
# FILE & LINE STATISTICS
# ============================================================================

echo "📈 CODE STATISTICS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TOTAL_FILES=$(find src -name "*.rs" 2>/dev/null | wc -l)
TOTAL_LINES=$(find src -name "*.rs" -exec wc -l {} + 2>/dev/null | tail -1 | awk '{print $1}')

echo "  Total Rust Files: $TOTAL_FILES"
echo "  Total Lines of Code: $TOTAL_LINES"

# Module distribution
echo ""
echo "Module Distribution:"
for dir in core domain physics solver simulation analysis clinical infra gpu; do
    if [ -d "src/$dir" ]; then
        count=$(find "src/$dir" -name "*.rs" 2>/dev/null | wc -l)
        echo "  - src/$dir/: $count files"
    fi
done

echo ""

# ============================================================================
# REMAINING WORK
# ============================================================================

echo "📋 REMAINING WORK"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

TOTAL_ISSUES=0

# Check for old imports that need updating
if [ "$OLD_CORE_IMPORTS" -gt 0 ]; then
    echo "  ⚠️  Update $OLD_CORE_IMPORTS domain::core:: imports"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
fi

if [ "$OLD_MATH_IMPORTS" -gt 0 ]; then
    echo "  ⚠️  Update $OLD_MATH_IMPORTS domain::math:: imports"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
fi

if [ "$OLD_BEAM_IMPORTS" -gt 0 ]; then
    echo "  ⚠️  Update $OLD_BEAM_IMPORTS beamforming imports"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
fi

# Check for old directories that need removal
if [ -d "src/domain/core" ]; then
    echo "  ⚠️  Remove src/domain/core/ directory"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
fi

if [ -d "src/domain/math" ]; then
    echo "  ⚠️  Remove src/domain/math/ directory"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
fi

if [ -d "src/domain/sensor/beamforming" ]; then
    echo "  ⚠️  Remove src/domain/sensor/beamforming/ directory"
    TOTAL_ISSUES=$((TOTAL_ISSUES + 1))
fi

if [ "$TOTAL_ISSUES" -eq 0 ]; then
    echo "  ✅ No outstanding issues detected!"
fi

echo ""

# ============================================================================
# NEXT STEPS
# ============================================================================

echo "🎯 NEXT STEPS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ "$COMPLETED_PHASES" -eq 0 ]; then
    echo "  1. Execute Phase 0: Preparation & Cleanup"
    echo "     ./scripts/phase0_cleanup.sh"
    echo ""
    echo "  2. Execute Phase 1: Core Extraction"
    echo "     ./scripts/phase1_create_core.sh"
    echo "     ./scripts/phase1_migrate_error.sh"
elif [ "$COMPLETED_PHASES" -eq 1 ]; then
    echo "  1. Execute Phase 2: Math Extraction"
    echo "     ./scripts/phase2_migrate_fft.sh"
    echo "     ./scripts/phase2_migrate_linalg.sh"
    echo "     ./scripts/phase2_migrate_numerics.sh"
elif [ "$COMPLETED_PHASES" -eq 2 ]; then
    echo "  1. Execute Phase 3: Beamforming Cleanup"
    echo "     Review and migrate beamforming consumers"
else
    echo "  Continue with next phase based on refactoring plan"
fi

echo ""
echo "  Always run after changes:"
echo "    - ./scripts/continuous_test.sh"
echo "    - ./scripts/progress_report.sh"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Report Generation Complete                  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
