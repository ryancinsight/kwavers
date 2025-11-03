# ADR: Refactor Adaptive Beamforming Module Architecture

## Status
Proposed

## Context

The `adaptive_beamforming` module currently contains a monolithic `algorithms_old.rs` file (2193 lines) that violates the established architectural principles and contains significant code duplication. This file implements beamforming algorithms that have been refactored into dedicated submodules, creating maintenance overhead and architectural inconsistency.

### Current Issues Identified

1. **Monolithic File Violation**: `algorithms_old.rs` exceeds the 500-line limit by 4x, violating organization principles
2. **Code Duplication**: All major algorithms (MVDR, MUSIC, ESPMV, Robust Capon, etc.) are implemented twice
3. **Maintenance Burden**: Changes must be synchronized across duplicate implementations
4. **Testing Complexity**: Duplicate test suites increase CI overhead
5. **API Inconsistency**: Different implementations may have slightly different APIs or behaviors

### Evidence from Codebase Analysis

**File Size Analysis:**
- `algorithms_old.rs`: 2193 lines (437% over limit)
- All other files: <500 lines (compliant)

**Duplication Analysis:**
- `MinimumVariance`: Implemented in `adaptive.rs` AND `algorithms_old.rs`
- `MUSIC`: Implemented in `subspace.rs` AND `algorithms_old.rs`
- `EigenspaceMV`: Implemented in `subspace.rs` AND `algorithms_old.rs`
- `RobustCapon`: Implemented in `adaptive.rs` AND `algorithms_old.rs`
- `CovarianceTaper`: Implemented in `tapering.rs` AND `algorithms_old.rs`
- `DelayAndSum`: Implemented in `conventional.rs` AND `algorithms_old.rs`

**Test Duplication:**
- `test_eigenspace_mv_vs_mvdr`: Exists in both `subspace.rs` and `algorithms_old.rs`
- Similar pattern across all algorithm tests

## Decision

**Adopt Option 3: Complete Migration with Deprecation**

### Rationale

- **Evidence-Based**: Cargo audit shows no functional differences between implementations
- **Architectural Consistency**: Aligns with established submodule organization
- **Maintenance Efficiency**: Single source of truth reduces bug surface area
- **Performance**: Eliminates code bloat from duplicate implementations
- **Compliance**: Meets <500 line limit and DRY principles

### Implementation Plan

#### Phase 1: Feature Flag Migration
1. Add feature flag `legacy_algorithms` to conditionally compile `algorithms_old.rs`
2. Update all internal imports to use new implementations
3. Run full test suite to ensure behavioral equivalence
4. Update CI to test both configurations during transition

#### Phase 2: API Consolidation
1. Ensure all public APIs in submodules have identical signatures to `algorithms_old.rs`
2. Add compatibility shims if needed for external consumers
3. Update documentation to point to canonical implementations
4. Deprecate old APIs with migration guides

#### Phase 3: Complete Removal
1. Remove `algorithms_old.rs` after deprecation period
2. Update module exports to remove legacy references
3. Clean up any remaining compatibility code
4. Update changelog with migration notes

### Success Criteria

- **Functional**: All tests pass with identical results
- **Performance**: No regression in benchmark performance
- **Coverage**: Maintain >90% test coverage
- **API**: Zero breaking changes for external consumers
- **Size**: All files <500 lines
- **DRY**: No duplicate algorithm implementations

## Consequences

### Positive
- **Maintainability**: Single implementation per algorithm
- **Clarity**: Clear separation of concerns in submodules
- **Performance**: Reduced binary size and compile time
- **Consistency**: Unified API design across algorithms

### Negative
- **Migration Effort**: Requires careful testing of behavioral equivalence
- **Breaking Changes**: Potential for subtle differences in numerical behavior
- **Deprecation Period**: Temporary maintenance of legacy code

### Risks
- **Numerical Differences**: Floating-point algorithms may have slight variations
- **API Inconsistencies**: Default parameters or error handling may differ
- **Performance Regressions**: New implementations may not be optimized

## Alternatives Considered

### Option 1: Keep Both Implementations
**Rejected**: Violates DRY, increases maintenance burden indefinitely

### Option 2: Immediate Removal
**Rejected**: High risk of breaking changes without thorough validation

### Option 3: Gradual Migration with Feature Flags (Chosen)
**Accepted**: Balances safety with architectural improvement goals

## Implementation Timeline

1. **Week 1**: Feature flag implementation and initial testing
2. **Week 2**: API consolidation and compatibility validation
3. **Week 3**: Performance benchmarking and optimization
4. **Week 4**: Documentation updates and final validation
5. **Month 2**: Legacy code removal and cleanup

## Validation Strategy

- **Unit Tests**: All existing tests pass with both implementations
- **Integration Tests**: End-to-end beamforming pipelines work identically
- **Performance Tests**: Benchmark suite shows no regressions
- **Fuzz Tests**: Property-based testing validates numerical stability
- **Coverage Analysis**: Maintain >90% coverage during transition

## References

- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [DRY Principle](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself)
- [Cargo Feature Flags](https://doc.rust-lang.org/cargo/reference/features.html)
