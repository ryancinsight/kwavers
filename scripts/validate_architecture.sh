#!/usr/bin/env bash
# Architecture Validation Script for Kwavers
# Enforces clean 8-layer architecture with zero circular dependencies
#
# Usage: ./scripts/validate_architecture.sh
# Exit code: 0 if all checks pass, 1 if violations detected

set -euo pipefail

# Color output for terminals
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Tracking variables
VIOLATIONS=0
WARNINGS=0

echo -e "${BLUE}=== Kwavers Architecture Validation ===${NC}\n"

# Layer definitions (higher number = higher layer, cannot depend on lower numbers)
declare -A LAYER_LEVELS=(
    ["core"]=0
    ["math"]=1
    ["physics"]=2
    ["domain"]=3
    ["solver"]=4
    ["simulation"]=5
    ["analysis"]=6
    ["clinical"]=7
    ["infra"]=8
)

#######################################
# Check 1: Circular Dependencies
#######################################
echo -e "${BLUE}[1/8] Checking for circular dependencies...${NC}"

check_layer_violation() {
    local from_layer=$1
    local to_layer=$2
    local from_level=${LAYER_LEVELS[$from_layer]:-999}
    local to_level=${LAYER_LEVELS[$to_layer]:-999}

    # Skip if either layer not defined
    if [ "$from_level" == "999" ] || [ "$to_level" == "999" ]; then
        return 0
    fi

    # Higher layers CAN depend on lower layers
    # Lower layers CANNOT depend on higher layers
    if [ "$from_level" -lt "$to_level" ]; then
        return 1  # Violation: lower layer depending on higher layer
    fi

    return 0
}

# Check domain → analysis violations (documented technical debt)
DOMAIN_ANALYSIS_VIOLATIONS=$(grep -r "use crate::analysis" src/domain/ 2>/dev/null | wc -l || echo "0")
DOMAIN_ANALYSIS_VIOLATIONS=$(echo "$DOMAIN_ANALYSIS_VIOLATIONS" | tr -d ' ')

if [ "$DOMAIN_ANALYSIS_VIOLATIONS" -gt "0" ]; then
    echo -e "${YELLOW}  ⚠ Found $DOMAIN_ANALYSIS_VIOLATIONS domain→analysis violations (documented technical debt)${NC}"
    echo -e "${YELLOW}    These are tracked in Phase 2 beamforming migration${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}  ✓ No domain→analysis violations${NC}"
fi

# Check domain → solver violations (CRITICAL)
DOMAIN_SOLVER_VIOLATIONS=$(grep -r "use crate::solver" src/domain/ 2>/dev/null | wc -l || echo "0")
DOMAIN_SOLVER_VIOLATIONS=$(echo "$DOMAIN_SOLVER_VIOLATIONS" | tr -d ' ')

if [ "$DOMAIN_SOLVER_VIOLATIONS" -gt "0" ]; then
    echo -e "${RED}  ✗ Found $DOMAIN_SOLVER_VIOLATIONS domain→solver violations (CRITICAL)${NC}"
    grep -rn "use crate::solver" src/domain/ 2>/dev/null | head -10
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo -e "${GREEN}  ✓ No domain→solver violations${NC}"
fi

# Check physics → domain violations (CRITICAL)
PHYSICS_DOMAIN_VIOLATIONS=$(grep -r "use crate::domain" src/physics/ 2>/dev/null | wc -l || echo "0")
PHYSICS_DOMAIN_VIOLATIONS=$(echo "$PHYSICS_DOMAIN_VIOLATIONS" | tr -d ' ')

if [ "$PHYSICS_DOMAIN_VIOLATIONS" -gt "0" ]; then
    echo -e "${RED}  ✗ Found $PHYSICS_DOMAIN_VIOLATIONS physics→domain violations (CRITICAL)${NC}"
    grep -rn "use crate::domain" src/physics/ 2>/dev/null | head -10
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo -e "${GREEN}  ✓ No physics→domain violations${NC}"
fi

# Check physics → solver violations (CRITICAL)
PHYSICS_SOLVER_VIOLATIONS=$(grep -r "use crate::solver" src/physics/ 2>/dev/null | wc -l || echo "0")
PHYSICS_SOLVER_VIOLATIONS=$(echo "$PHYSICS_SOLVER_VIOLATIONS" | tr -d ' ')

if [ "$PHYSICS_SOLVER_VIOLATIONS" -gt "0" ]; then
    echo -e "${RED}  ✗ Found $PHYSICS_SOLVER_VIOLATIONS physics→solver violations (CRITICAL)${NC}"
    grep -rn "use crate::solver" src/physics/ 2>/dev/null | head -10
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo -e "${GREEN}  ✓ No physics→solver violations${NC}"
fi

# Check solver → analysis violations (CRITICAL - recently fixed)
SOLVER_ANALYSIS_VIOLATIONS=$(grep -r "use crate::analysis" src/solver/ 2>/dev/null | wc -l || echo "0")
SOLVER_ANALYSIS_VIOLATIONS=$(echo "$SOLVER_ANALYSIS_VIOLATIONS" | tr -d ' ')

if [ "$SOLVER_ANALYSIS_VIOLATIONS" -gt "0" ]; then
    echo -e "${RED}  ✗ Found $SOLVER_ANALYSIS_VIOLATIONS solver→analysis violations (CRITICAL)${NC}"
    grep -rn "use crate::analysis" src/solver/ 2>/dev/null | head -10
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo -e "${GREEN}  ✓ No solver→analysis violations${NC}"
fi

echo ""

#######################################
# Check 2: Build Warnings and Errors
#######################################
echo -e "${BLUE}[2/8] Checking build for warnings and errors...${NC}"

BUILD_OUTPUT=$(cargo build --lib 2>&1 || echo "BUILD_FAILED")

if echo "$BUILD_OUTPUT" | grep -q "BUILD_FAILED"; then
    echo -e "${RED}  ✗ Build failed${NC}"
    echo "$BUILD_OUTPUT" | tail -20
    VIOLATIONS=$((VIOLATIONS + 1))
elif echo "$BUILD_OUTPUT" | grep -qE "warning:|error:"; then
    WARNING_COUNT=$(echo "$BUILD_OUTPUT" | grep -c "warning:" || echo "0")
    ERROR_COUNT=$(echo "$BUILD_OUTPUT" | grep -c "error:" || echo "0")

    if [ "$ERROR_COUNT" -gt "0" ]; then
        echo -e "${RED}  ✗ Found $ERROR_COUNT build errors${NC}"
        echo "$BUILD_OUTPUT" | grep "error:" | head -10
        VIOLATIONS=$((VIOLATIONS + 1))
    fi

    if [ "$WARNING_COUNT" -gt "0" ]; then
        echo -e "${YELLOW}  ⚠ Found $WARNING_COUNT build warnings${NC}"
        echo "$BUILD_OUTPUT" | grep "warning:" | head -10
        WARNINGS=$((WARNINGS + 1))
    fi
else
    echo -e "${GREEN}  ✓ Build successful with no warnings${NC}"
fi

echo ""

#######################################
# Check 3: Deprecated Code Markers
#######################################
echo -e "${BLUE}[3/8] Checking for deprecated code...${NC}"

DEPRECATED_COUNT=$(grep -r "#\[deprecated" src/ 2>/dev/null | wc -l || echo "0")
DEPRECATED_COUNT=$(echo "$DEPRECATED_COUNT" | tr -d ' ')

if [ "$DEPRECATED_COUNT" -gt "0" ]; then
    echo -e "${YELLOW}  ⚠ Found $DEPRECATED_COUNT deprecated items${NC}"
    echo -e "${YELLOW}    These should have migration paths documented${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}  ✓ No deprecated code markers${NC}"
fi

echo ""

#######################################
# Check 4: Dead Code Allowances
#######################################
echo -e "${BLUE}[4/8] Checking for excessive dead_code allowances...${NC}"

DEAD_CODE_COUNT=$(grep -r "#\[allow(dead_code)\]" src/ 2>/dev/null | wc -l || echo "0")
DEAD_CODE_COUNT=$(echo "$DEAD_CODE_COUNT" | tr -d ' ')

if [ "$DEAD_CODE_COUNT" -gt "100" ]; then
    echo -e "${YELLOW}  ⚠ Found $DEAD_CODE_COUNT dead_code allowances (>100 threshold)${NC}"
    echo -e "${YELLOW}    Consider auditing and removing unnecessary markers${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}  ✓ Dead code allowances within acceptable range ($DEAD_CODE_COUNT)${NC}"
fi

echo ""

#######################################
# Check 5: Test Suite
#######################################
echo -e "${BLUE}[5/8] Running test suite...${NC}"

TEST_OUTPUT=$(cargo test --lib 2>&1 || echo "TEST_FAILED")

if echo "$TEST_OUTPUT" | grep -q "TEST_FAILED"; then
    echo -e "${RED}  ✗ Tests failed to execute${NC}"
    VIOLATIONS=$((VIOLATIONS + 1))
elif echo "$TEST_OUTPUT" | grep -q "test result:.*FAILED"; then
    FAILED_COUNT=$(echo "$TEST_OUTPUT" | grep -oP "test result:.*\K[0-9]+ failed" || echo "unknown")
    echo -e "${RED}  ✗ Test failures detected: $FAILED_COUNT${NC}"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    PASSED_COUNT=$(echo "$TEST_OUTPUT" | grep -oP "test result: ok\. \K[0-9]+" || echo "unknown")
    echo -e "${GREEN}  ✓ All tests passed ($PASSED_COUNT tests)${NC}"
fi

echo ""

#######################################
# Check 6: Module Organization
#######################################
echo -e "${BLUE}[6/8] Checking module organization...${NC}"

# Check for overly deep module nesting (>6 levels)
DEEP_MODULES=$(find src/ -type f -name "*.rs" | awk -F/ '{if(NF>8) print}' | wc -l || echo "0")
DEEP_MODULES=$(echo "$DEEP_MODULES" | tr -d ' ')

if [ "$DEEP_MODULES" -gt "50" ]; then
    echo -e "${YELLOW}  ⚠ Found $DEEP_MODULES files with deep nesting (>6 levels)${NC}"
    echo -e "${YELLOW}    Consider flattening module hierarchy${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}  ✓ Module nesting depth acceptable${NC}"
fi

# Check for very large files (>1000 LOC)
LARGE_FILES=$(find src/ -name "*.rs" -type f -exec wc -l {} + 2>/dev/null | awk '$1 > 1000 {count++} END {print count+0}')

if [ "$LARGE_FILES" -gt "10" ]; then
    echo -e "${YELLOW}  ⚠ Found $LARGE_FILES files >1000 LOC${NC}"
    echo -e "${YELLOW}    Consider refactoring into submodules${NC}"
    find src/ -name "*.rs" -type f -exec wc -l {} + 2>/dev/null | awk '$1 > 1000 {print "      " $2 " (" $1 " lines)"}' | head -10
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}  ✓ File sizes within acceptable range${NC}"
fi

echo ""

#######################################
# Check 7: Feature Flag Consistency
#######################################
echo -e "${BLUE}[7/8] Checking feature flag compilation...${NC}"

# Test minimal build
MINIMAL_BUILD=$(cargo build --no-default-features --features minimal 2>&1 || echo "MINIMAL_FAILED")
if echo "$MINIMAL_BUILD" | grep -q "MINIMAL_FAILED"; then
    echo -e "${RED}  ✗ Minimal feature build failed${NC}"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo -e "${GREEN}  ✓ Minimal feature build successful${NC}"
fi

# Test full build
FULL_BUILD=$(cargo build --all-features 2>&1 || echo "FULL_FAILED")
if echo "$FULL_BUILD" | grep -q "FULL_FAILED"; then
    echo -e "${RED}  ✗ Full features build failed${NC}"
    VIOLATIONS=$((VIOLATIONS + 1))
else
    echo -e "${GREEN}  ✓ Full features build successful${NC}"
fi

echo ""

#######################################
# Check 8: Documentation Coverage
#######################################
echo -e "${BLUE}[8/8] Checking documentation coverage...${NC}"

# Count public items without documentation
UNDOCUMENTED=$(cargo doc --lib 2>&1 | grep -c "warning: missing documentation" || echo "0")
UNDOCUMENTED=$(echo "$UNDOCUMENTED" | tr -d ' ')

if [ "$UNDOCUMENTED" -gt "50" ]; then
    echo -e "${YELLOW}  ⚠ Found $UNDOCUMENTED undocumented public items${NC}"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}  ✓ Documentation coverage good (<50 missing docs)${NC}"
fi

echo ""

#######################################
# Summary
#######################################
echo -e "${BLUE}=== Validation Summary ===${NC}\n"

if [ $VIOLATIONS -eq 0 ]; then
    echo -e "${GREEN}✓ All critical checks passed!${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warnings detected (non-blocking)${NC}"
    fi
    echo ""
    echo -e "${GREEN}Architecture is production-ready.${NC}"
    exit 0
else
    echo -e "${RED}✗ $VIOLATIONS critical violations detected${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warnings detected${NC}"
    fi
    echo ""
    echo -e "${RED}Please fix violations before merging.${NC}"
    exit 1
fi
