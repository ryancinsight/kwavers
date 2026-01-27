# Detailed Action Plan
## Kwavers Ultrasound and Optics Simulation Library - Audit, Optimization, Enhancement & Extension

**Date**: 2026-01-26  
**Version**: 1.0  
**Status**: Draft for Review  
**Branch**: main

---

## Executive Summary

This detailed action plan provides specific, actionable tasks for auditing, optimizing, enhancing, extending, and completing the kwavers ultrasound and optics simulation library. The plan is organized by priority (P0, P1, P2) and includes estimated effort, dependencies, and success criteria for each task.

**Total Estimated Effort**: 1,200-1,800 hours (30-45 weeks at full-time equivalent)

---

## Phase 1: Critical Infrastructure Fixes (Weeks 1-2)

### P0-1: Fix Core → Physics Dependency Violation

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 20-30 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Resolve architectural violation where Core layer imports from Physics layer.

**Tasks**:
1. [ ] Create new error type structure in Core layer
   - Define `KwaversError` enum in `src/core/error/mod.rs`
   - Create error kind enums for each layer (Core, Physics, Domain, Solver, Simulation, Analysis, Clinical)
   - Implement `From` traits for error conversion
   - **Effort**: 4-6 hours

2. [ ] Update Physics error types to use Core errors
   - Refactor `PhysicsError` to use `KwaversError`
   - Update all Physics module error returns
   - Update all Physics module tests
   - **Effort**: 6-8 hours

3. [ ] Update Domain error types to use Core errors
   - Refactor `DomainError` to use `KwaversError`
   - Update all Domain module error returns
   - Update all Domain module tests
   - **Effort**: 4-6 hours

4. [ ] Update Solver error types to use Core errors
   - Refactor `SolverError` to use `KwaversError`
   - Update all Solver module error returns
   - Update all Solver module tests
   - **Effort**: 4-6 hours

5. [ ] Update Simulation error types to use Core errors
   - Refactor `SimulationError` to use `KwaversError`
   - Update all Simulation module error returns
   - Update all Simulation module tests
   - **Effort**: 2-3 hours

6. [ ] Update Analysis error types to use Core errors
   - Refactor `AnalysisError` to use `KwaversError`
   - Update all Analysis module error returns
   - Update all Analysis module tests
   - **Effort**: 2-3 hours

7. [ ] Update Clinical error types to use Core errors
   - Refactor `ClinicalError` to use `KwaversError`
   - Update all Clinical module error returns
   - Update all Clinical module tests
   - **Effort**: 2-3 hours

8. [ ] Remove `pub use physics::*;` from Core layer
   - Remove `src/core/error/types/domain/mod.rs` line 10
   - Remove `src/core/error/mod.rs` line 41
   - **Effort**: 1-2 hours

9. [ ] Test compilation and functionality
   - Run `cargo check --lib`
   - Run `cargo test --lib`
   - Verify all tests pass
   - **Effort**: 2-3 hours

10. [ ] Update documentation
    - Update error type documentation
    - Update architecture documentation
    - Update migration guide
    - **Effort**: 2-3 hours

**Success Criteria**:
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] All tests passing (1554/1554)
- [ ] Core layer has zero dependencies on higher layers
- [ ] Documentation updated

**Deliverables**:
- New error type structure in Core layer
- All error types refactored to use Core errors
- Zero architectural violations
- Updated documentation

---

### P0-2: Fix Clinical Module Circular Import

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 2-4 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Resolve internal circular import within Clinical module.

**Tasks**:
1. [ ] Identify correct location of `ClinicalDecisionSupport`
   - Search for `ClinicalDecisionSupport` definition
   - Determine correct import path
   - **Effort**: 0.5-1 hour

2. [ ] Update import path in neural module
   - Fix import in `src/clinical/imaging/workflows/neural/mod.rs` line 35
   - Use correct path (e.g., `crate::clinical::decision_support::ClinicalDecisionSupport`)
   - **Effort**: 0.5-1 hour

3. [ ] Test compilation and functionality
   - Run `cargo check --lib`
   - Run `cargo test --lib`
   - Verify all tests pass
   - **Effort**: 0.5-1 hour

4. [ ] Verify no circular dependencies
   - Run dependency analysis
   - Verify no circular imports
   - **Effort**: 0.5-1 hour

**Success Criteria**:
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] All tests passing (1554/1554)
- [ ] No circular imports in Clinical module
- [ ] Dependency analysis confirms no circular dependencies

**Deliverables**:
- Fixed import path in neural module
- Zero circular imports
- Verified dependency structure

---

## Phase 2: Codebase Cleanup (Weeks 2-4)

### P0-3: Remove Dead Code Markers

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 40-60 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Reduce 207 `#[allow(dead_code)]` directives to <50.

**Tasks**:
1. [ ] Audit all dead code markers
   - Search for all `#[allow(dead_code)]` directives
   - Categorize by module and priority
   - Document each marker with rationale
   - **Effort**: 8-12 hours

2. [ ] Remove unnecessary dead code markers
   - Remove markers for truly dead code (delete code)
   - Remove markers for code that should be public
   - Remove markers for code used in tests
   - **Effort**: 16-24 hours

3. [ ] Document necessary dead code markers
   - Add TODO comments for code that should be removed later
   - Add feature flags for code that is conditionally used
   - Add rationale for remaining markers
   - **Effort**: 8-12 hours

4. [ ] Verify compilation after cleanup
   - Run `cargo check --lib`
   - Run `cargo clippy -- -D warnings`
   - Fix any issues
   - **Effort**: 4-6 hours

5. [ ] Update documentation
   - Document remaining dead code markers
   - Update cleanup procedures
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] Dead code markers reduced from 207 to <50
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] All tests passing (1554/1554)
- [ ] Documentation updated

**Deliverables**:
- Reduced dead code markers (<50)
- Clean codebase
- Updated documentation

---

### P0-4: Resolve TODO Markers

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 30-40 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Resolve or document all 48 TODO markers with clear priorities.

**Tasks**:
1. [ ] Audit all TODO markers
   - Search for all TODO comments
   - Categorize by priority (P0, P1, P2)
   - Estimate effort for each TODO
   - **Effort**: 6-8 hours

2. [ ] Resolve P0 TODO markers
   - Implement critical TODOs
   - Add tests for implementations
   - Update documentation
   - **Effort**: 12-16 hours

3. [ ] Document P1 TODO markers
   - Add detailed specifications
   - Add effort estimates
   - Add to backlog
   - **Effort**: 6-8 hours

4. [ ] Document P2 TODO markers
   - Add detailed specifications
   - Add effort estimates
   - Add to backlog
   - **Effort**: 6-8 hours

**Success Criteria**:
- [ ] All P0 TODO markers resolved
- [ ] All P1 TODO markers documented with specifications
- [ ] All P2 TODO markers documented with specifications
- [ ] Zero compilation errors
- [ ] All tests passing (1554/1554)

**Deliverables**:
- Resolved P0 TODO markers
- Documented P1 TODO markers
- Documented P2 TODO markers
- Updated backlog

---

### P0-5: Remove or Feature-Flag Stub Implementations

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 20-30 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Remove or properly feature-flag stub implementations.

**Tasks**:
1. [ ] Identify all stub implementations
   - Search for placeholder implementations
   - Categorize by module and priority
   - Document each stub
   - **Effort**: 4-6 hours

2. [ ] Remove unnecessary stubs
   - Remove stubs that are not needed
   - Clean up related code
   - **Effort**: 8-12 hours

3. [ ] Feature-flag necessary stubs
   - Add `#[cfg(feature = "experimental-*")]` guards
   - Add warnings for experimental features
   - Update Cargo.toml feature flags
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] All unnecessary stubs removed
- [ ] All necessary stubs feature-flagged
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] All tests passing (1554/1554)

**Deliverables**:
- Removed unnecessary stubs
- Feature-flagged necessary stubs
- Updated Cargo.toml

---

## Phase 3: Architecture Validation (Weeks 3-4)

### P0-6: Verify 8-Layer Architecture Compliance

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 30-40 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Verify 8-layer architecture compliance across all modules.

**Tasks**:
1. [ ] Create dependency analysis tool
   - Write script to analyze module dependencies
   - Generate dependency graph
   - Identify violations
   - **Effort**: 8-12 hours

2. [ ] Run dependency analysis on all modules
   - Analyze Core layer dependencies
   - Analyze Math layer dependencies
   - Analyze Physics layer dependencies
   - Analyze Domain layer dependencies
   - Analyze Solver layer dependencies
   - Analyze Simulation layer dependencies
   - Analyze Analysis layer dependencies
   - Analyze Clinical layer dependencies
   - Analyze Infrastructure layer dependencies
   - **Effort**: 8-12 hours

3. [ ] Document all dependencies
   - Create dependency matrix
   - Create dependency diagrams
   - Document any violations
   - **Effort**: 6-8 hours

4. [ ] Validate dependency flow
   - Verify unidirectional dependency flow
   - Identify any circular dependencies
   - Identify any cross-contamination
   - **Effort**: 4-6 hours

5. [ ] Create architecture compliance report
   - Document compliance status
   - Document any violations
   - Create remediation plan
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] 100% architectural compliance
- [ ] Zero circular dependencies
- [ ] Zero cross-contamination between layers
- [ ] All dependencies documented
- [ ] Dependency diagrams created

**Deliverables**:
- Dependency analysis tool
- Dependency matrix
- Dependency diagrams
- Architecture compliance report

---

### P0-7: Document SSOT Patterns

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 20-30 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Document single source of truth (SSOT) patterns for shared accessors.

**Tasks**:
1. [ ] Identify all SSOT patterns
   - Audit all shared accessors
   - Identify canonical locations
   - Document usage patterns
   - **Effort**: 6-8 hours

2. [ ] Document SSOT patterns
   - Create SSOT pattern documentation
   - Add code examples
   - Add usage guidelines
   - **Effort**: 8-12 hours

3. [ ] Validate SSOT compliance
   - Verify all layers use SSOT patterns
   - Identify any violations
   - Document any issues
   - **Effort**: 4-6 hours

4. [ ] Create SSOT compliance report
   - Document compliance status
   - Document any violations
   - Create remediation plan
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] All SSOT patterns documented
- [ ] All SSOT patterns validated
- [ ] Zero SSOT violations
- [ ] SSOT compliance report created

**Deliverables**:
- SSOT pattern documentation
- SSOT compliance report
- Usage guidelines

---

## Phase 4: Module Organization Audit (Weeks 4-5)

### P0-8: Audit Module Placement

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 40-50 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Audit all modules for correct placement in layer hierarchy.

**Tasks**:
1. [ ] Create module audit tool
   - Write script to analyze module placement
   - Check module size compliance (<500 lines)
   - Check naming conventions
   - **Effort**: 8-12 hours

2. [ ] Run module audit on all modules
   - Audit Core layer modules
   - Audit Math layer modules
   - Audit Physics layer modules
   - Audit Domain layer modules
   - Audit Solver layer modules
   - Audit Simulation layer modules
   - Audit Analysis layer modules
   - Audit Clinical layer modules
   - Audit Infrastructure layer modules
   - **Effort**: 16-20 hours

3. [ ] Identify misplaced components
   - Document modules in wrong layer
   - Document modules exceeding size limit
   - Document naming convention violations
   - **Effort**: 6-8 hours

4. [ ] Create migration plan
   - Document migration strategy
   - Estimate effort for each migration
   - Prioritize migrations
   - **Effort**: 6-8 hours

5. [ ] Create module organization report
   - Document audit findings
   - Document migration plan
   - Create remediation timeline
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] All modules audited
- [ ] All misplaced components identified
- [ ] Migration plan created
- [ ] Module organization report created

**Deliverables**:
- Module audit tool
- Module organization report
- Migration plan
- Remediation timeline

---

## Phase 5: Research Integration (Weeks 6-24)

### P0-9: Implement Differentiable Forward Solvers (jwave)

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 80-120 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Implement differentiable forward solvers inspired by jwave.

**Tasks**:
1. [ ] Design differentiable solver architecture
   - Define `DifferentiableSolver` trait
   - Design state management for autodiff
   - Design gradient computation interface
   - **Effort**: 12-16 hours

2. [ ] Implement differentiable FDTD solver
   - Refactor FDTD for autodiff
   - Implement gradient computation
   - Add tests
   - **Effort**: 24-32 hours

3. [ ] Implement differentiable PSTD solver
   - Refactor PSTD for autodiff
   - Implement gradient computation
   - Add tests
   - **Effort**: 24-32 hours

4. [ ] Integrate with Burn autodiff
   - Connect solvers to Burn backend
   - Implement backward pass
   - Add tests
   - **Effort**: 16-24 hours

5. [ ] Create functional simulation builder
   - Implement builder pattern
   - Add automatic CFL management
   - Add tests
   - **Effort**: 8-12 hours

6. [ ] Add Fourier series initial conditions
   - Implement FourierSeries type
   - Add IFFT to spatial domain
   - Add tests
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] Differentiable FDTD solver implemented
- [ ] Differentiable PSTD solver implemented
- [ ] Gradient computation working
- [ ] Integration with Burn autodiff complete
- [ ] All tests passing

**Deliverables**:
- Differentiable FDTD solver
- Differentiable PSTD solver
- Functional simulation builder
- Fourier series initial conditions

---

### P0-10: Implement Off-Grid Source/Sensor Integration (k-Wave)

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 60-80 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Implement off-grid source/sensor integration inspired by k-Wave kWaveArray.

**Tasks**:
1. [ ] Design off-grid source architecture
   - Define `ElementArray` type
   - Design surface integration interface
   - Design Fourier-space implementation
   - **Effort**: 8-12 hours

2. [ ] Implement Gaussian quadrature surface integration
   - Implement quadrature rules
   - Implement surface integration
   - Add tests
   - **Effort**: 16-20 hours

3. [ ] Implement Fourier-space distributed source
   - Implement FFT-based source computation
   - Implement off-grid interpolation
   - Add tests
   - **Effort**: 16-20 hours

4. [ ] Integrate with existing source infrastructure
   - Connect to `Source` trait
   - Update factory pattern
   - Add tests
   - **Effort**: 12-16 hours

5. [ ] Validate against k-Wave reference
   - Compare near-field patterns
   - Compare far-field patterns
   - Create validation report
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] Off-grid source integration implemented
- [ ] Surface integration working
- [ ] Fourier-space implementation working
- [ ] Validation against k-Wave complete
- [ ] All tests passing

**Deliverables**:
- Off-grid source integration
- Surface integration implementation
- Fourier-space implementation
- Validation report

---

### P0-11: Implement CT-Based Skull Modeling (BabelBrain)

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 40-60 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Implement CT-based skull modeling inspired by BabelBrain.

**Tasks**:
1. [ ] Design CT-based skull architecture
   - Define `CTSkullModel` type
   - Design HU→acoustic property mapping
   - Design cortical/trabecular classification
   - **Effort**: 8-12 hours

2. [ ] Implement HU→acoustic property conversion
   - Implement piecewise linear maps
   - Implement cortical/trabecular distinction
   - Add tests
   - **Effort**: 12-16 hours

3. [ ] Implement heterogeneous skull medium
   - Create skull medium from CT volume
   - Implement density maps
   - Add tests
   - **Effort**: 12-16 hours

4. [ ] Integrate with existing skull infrastructure
   - Connect to `SkullModel` trait
   - Update factory pattern
   - Add tests
   - **Effort**: 8-12 hours

5. [ ] Validate against BabelBrain reference
   - Compare aberration patterns
   - Compare transmission patterns
   - Create validation report
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] CT-based skull modeling implemented
- [ ] HU→acoustic property conversion working
- [ ] Heterogeneous skull medium working
- [ ] Validation against BabelBrain complete
- [ ] All tests passing

**Deliverables**:
- CT-based skull modeling
- HU→acoustic property conversion
- Heterogeneous skull medium
- Validation report

---

### P0-12: Implement Clinical Workflow Integration (BabelBrain)

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 60-80 hours  
**Dependencies**: P0-11 (CT-Based Skull Modeling)  
**Blocked By**: P0-11

**Objective**: Implement clinical workflow integration inspired by BabelBrain.

**Tasks**:
1. [ ] Design clinical workflow architecture
   - Define `TreatmentPlanner` type
   - Design three-stage workflow
   - Design coordinate system management
   - **Effort**: 12-16 hours

2. [ ] Implement medical imaging pipeline
   - Implement DICOM/NIFTI loading
   - Implement automatic segmentation
   - Implement coordinate transforms
   - Add tests
   - **Effort**: 16-20 hours

3. [ ] Implement treatment planning workflow
   - Implement acoustic simulation stage
   - Implement thermal modeling stage
   - Implement dose calculation stage
   - Add tests
   - **Effort**: 16-20 hours

4. [ ] Implement safety validation
   - Implement IEC compliance checks
   - Implement safety monitoring
   - Add tests
   - **Effort**: 8-12 hours

5. [ ] Integrate with existing clinical infrastructure
   - Connect to clinical workflows
   - Update factory pattern
   - Add tests
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] Clinical workflow integration implemented
- [ ] Medical imaging pipeline working
- [ ] Treatment planning workflow working
- [ ] Safety validation working
- [ ] All tests passing

**Deliverables**:
- Clinical workflow integration
- Medical imaging pipeline
- Treatment planning workflow
- Safety validation

---

## Phase 6: Infrastructure & Performance (Weeks 25-28)

### P1-13: Complete GPU Acceleration

**Priority**: 🟡 P1 - HIGH  
**Effort**: 60-80 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Complete GPU acceleration for critical solvers.

**Tasks**:
1. [ ] Complete GPU FDTD implementation
   - Implement GPU kernels
   - Implement memory management
   - Add tests
   - **Effort**: 20-28 hours

2. [ ] Complete GPU PSTD solver
   - Implement GPU kernels
   - Implement memory management
   - Add tests
   - **Effort**: 20-28 hours

3. [ ] Implement GPU beamforming pipeline
   - Implement GPU kernels
   - Implement delay tables
   - Add tests
   - **Effort**: 12-16 hours

4. [ ] Optimize GPU memory management
   - Implement arena allocation
   - Implement buffer pooling
   - Add tests
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] GPU FDTD implementation complete
- [ ] GPU PSTD solver complete
- [ ] GPU beamforming pipeline complete
- [ ] Memory optimization complete
- [ ] All tests passing

**Deliverables**:
- GPU FDTD implementation
- GPU PSTD solver
- GPU beamforming pipeline
- Memory optimization

---

### P1-14: Complete Cloud Deployment

**Priority**: 🟡 P1 - HIGH  
**Effort**: 40-60 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Complete cloud deployment for AWS, Azure, and GCP.

**Tasks**:
1. [ ] Complete AWS provider implementation
   - Implement AWS ML SDK integration
   - Implement authentication
   - Implement deployment operations
   - Add tests
   - **Effort**: 12-16 hours

2. [ ] Complete Azure provider implementation
   - Implement Azure ML REST API integration
   - Implement authentication
   - Implement deployment operations
   - Add tests
   - **Effort**: 12-16 hours

3. [ ] Complete GCP provider implementation
   - Implement Vertex AI REST API integration
   - Implement authentication
   - Implement deployment operations
   - Add tests
   - **Effort**: 12-16 hours

4. [ ] Add auto-scaling support
   - Implement scaling logic
   - Implement monitoring
   - Add tests
   - **Effort**: 4-6 hours

5. [ ] Add monitoring and logging
   - Implement cloud monitoring
   - Implement structured logging
   - Add tests
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] AWS provider implementation complete
- [ ] Azure provider implementation complete
- [ ] GCP provider implementation complete
- [ ] Auto-scaling support complete
- [ ] Monitoring and logging complete
- [ ] All tests passing

**Deliverables**:
- AWS provider implementation
- Azure provider implementation
- GCP provider implementation
- Auto-scaling support
- Monitoring and logging

---

## Phase 7: Testing & Validation (Weeks 29-32)

### P0-15: Achieve >95% Code Coverage

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 40-50 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Achieve >95% code coverage across all modules.

**Tasks**:
1. [ ] Run coverage analysis
   - Install coverage tool (tarpaulin/llvm-cov)
   - Run coverage analysis
   - Identify coverage gaps
   - **Effort**: 4-6 hours

2. [ ] Add tests for uncovered code
   - Add unit tests for uncovered functions
   - Add integration tests for uncovered workflows
   - Add property-based tests
   - **Effort**: 24-32 hours

3. [ ] Verify coverage >95%
   - Run coverage analysis again
   - Verify >95% coverage
   - Document coverage report
   - **Effort**: 4-6 hours

4. [ ] Add coverage to CI/CD
   - Configure coverage in CI/CD pipeline
   - Add coverage thresholds
   - Add coverage reporting
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] Code coverage >95%
- [ ] All critical paths covered
- [ ] Coverage in CI/CD pipeline
- [ ] Coverage report created

**Deliverables**:
- >95% code coverage
- Coverage report
- CI/CD integration

---

### P0-16: Validate Against Reference Implementations

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 50-60 hours  
**Dependencies**: P0-9, P0-10, P0-11, P0-12  
**Blocked By**: P0-9, P0-10, P0-11, P0-12

**Objective**: Validate kwavers against reference implementations (k-Wave, jwave, fullwave25, BabelBrain).

**Tasks**:
1. [ ] Validate against k-Wave reference results
   - Run k-Wave simulations
   - Run kwavers simulations
   - Compare results
   - Create validation report
   - **Effort**: 12-16 hours

2. [ ] Compare with jwave simulations
   - Run jwave simulations
   - Run kwavers simulations
   - Compare results
   - Create validation report
   - **Effort**: 12-16 hours

3. [ ] Benchmark against fullwave25
   - Run fullwave25 simulations
   - Run kwavers simulations
   - Compare performance
   - Create benchmark report
   - **Effort**: 12-16 hours

4. [ ] Validate clinical workflows with BabelBrain
   - Run BabelBrain workflows
   - Run kwavers workflows
   - Compare results
   - Create validation report
   - **Effort**: 12-16 hours

**Success Criteria**:
- [ ] Validation against k-Wave complete
- [ ] Comparison with jwave complete
- [ ] Benchmark against fullwave25 complete
- [ ] Validation with BabelBrain complete
- [ ] All validation reports created

**Deliverables**:
- k-Wave validation report
- jwave comparison report
- fullwave25 benchmark report
- BabelBrain validation report

---

## Phase 8: Documentation & Knowledge Transfer (Weeks 33-36)

### P1-17: Complete API Documentation

**Priority**: 🟡 P1 - HIGH  
**Effort**: 40-50 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Complete API documentation for all public APIs.

**Tasks**:
1. [ ] Document all public APIs
   - Add doc comments to all public functions
   - Add doc comments to all public types
   - Add doc comments to all public traits
   - **Effort**: 20-24 hours

2. [ ] Add usage examples
   - Add examples for all major features
   - Add examples for all workflows
   - Add examples for all solvers
   - **Effort**: 12-16 hours

3. [ ] Create tutorial documentation
   - Create getting started tutorial
   - Create advanced usage tutorial
   - Create troubleshooting guide
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] All public APIs documented
- [ ] Usage examples added
- [ ] Tutorial documentation created
- [ ] Documentation builds successfully

**Deliverables**:
- Complete API documentation
- Usage examples
- Tutorial documentation

---

### P1-18: Complete Architecture Documentation

**Priority**: 🟡 P1 - HIGH  
**Effort**: 30-40 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Complete architecture documentation.

**Tasks**:
1. [ ] Document 8-layer architecture
   - Document each layer
   - Document dependencies
   - Document design rationale
   - **Effort**: 8-12 hours

2. [ ] Create dependency diagrams
   - Create dependency flow diagrams
   - Create module structure diagrams
   - Create SSOT pattern diagrams
   - **Effort**: 8-12 hours

3. [ ] Document SSOT patterns
   - Document all SSOT patterns
   - Add usage guidelines
   - Add examples
   - **Effort**: 8-12 hours

4. [ ] Create migration guides
   - Create migration guide for error types
   - Create migration guide for SSOT patterns
   - Create migration guide for research integration
   - **Effort**: 6-8 hours

**Success Criteria**:
- [ ] 8-layer architecture documented
- [ ] Dependency diagrams created
- [ ] SSOT patterns documented
- [ ] Migration guides created

**Deliverables**:
- Architecture documentation
- Dependency diagrams
- SSOT pattern documentation
- Migration guides

---

## Success Metrics Summary

### Code Quality Metrics
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] Dead code markers reduced from 207 to <50
- [ ] TODO markers resolved or documented
- [ ] >95% code coverage
- [ ] All modules <500 lines

### Architecture Metrics
- [ ] 100% architectural compliance
- [ ] Zero circular dependencies
- [ ] Zero cross-contamination between layers
- [ ] Clear SSOT patterns for all shared accessors
- [ ] Unidirectional dependency flow

### Performance Metrics
- [ ] 10-100× speedup for critical kernels
- [ ] Real-time simulation capability for clinical workflows
- [ ] Multi-GPU scaling efficiency >80%
- [ ] Memory usage optimized with arena allocation

### Research Integration Metrics
- [ ] Differentiable forward solvers implemented
- [ ] Off-grid source/sensor integration complete
- [ ] CT-based skull modeling implemented
- [ ] Clinical workflow integration complete
- [ ] Multi-GPU domain decomposition implemented

### Validation Metrics
- [ ] Validated against k-Wave reference results
- [ ] Compared with jwave simulations
- [ ] Benchmarked against fullwave25
- [ ] Validated clinical workflows with BabelBrain
- [ ] Mathematical validation against analytical solutions

---

## Conclusion

This detailed action plan provides specific, actionable tasks for auditing, optimizing, enhancing, extending, and completing the kwavers ultrasound and optics simulation library. The plan is organized by priority (P0, P1, P2) and includes estimated effort, dependencies, and success criteria for each task.

**Key Benefits**:
- Clear, actionable tasks
- Prioritized by importance
- Estimated effort for each task
- Dependencies and blockers identified
- Success criteria defined
- Deliverables specified

**Next Steps**:
1. Review and approve this plan
2. Prioritize tasks based on business needs
3. Allocate resources and schedule
4. Begin with P0 tasks (Critical Infrastructure Fixes)

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-26  
**Status**: Draft for Review
## Kwavers Ultrasound and Optics Simulation Library - Audit, Optimization, Enhancement & Extension

**Date**: 2026-01-26  
**Version**: 1.0  
**Status**: Draft for Review  
**Branch**: main

---

## Executive Summary

This detailed action plan provides specific, actionable tasks for auditing, optimizing, enhancing, extending, and completing the kwavers ultrasound and optics simulation library. The plan is organized by priority (P0, P1, P2) and includes estimated effort, dependencies, and success criteria for each task.

**Total Estimated Effort**: 1,200-1,800 hours (30-45 weeks at full-time equivalent)

---

## Phase 1: Critical Infrastructure Fixes (Weeks 1-2)

### P0-1: Fix Core → Physics Dependency Violation

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 20-30 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Resolve architectural violation where Core layer imports from Physics layer.

**Tasks**:
1. [ ] Create new error type structure in Core layer
   - Define `KwaversError` enum in `src/core/error/mod.rs`
   - Create error kind enums for each layer (Core, Physics, Domain, Solver, Simulation, Analysis, Clinical)
   - Implement `From` traits for error conversion
   - **Effort**: 4-6 hours

2. [ ] Update Physics error types to use Core errors
   - Refactor `PhysicsError` to use `KwaversError`
   - Update all Physics module error returns
   - Update all Physics module tests
   - **Effort**: 6-8 hours

3. [ ] Update Domain error types to use Core errors
   - Refactor `DomainError` to use `KwaversError`
   - Update all Domain module error returns
   - Update all Domain module tests
   - **Effort**: 4-6 hours

4. [ ] Update Solver error types to use Core errors
   - Refactor `SolverError` to use `KwaversError`
   - Update all Solver module error returns
   - Update all Solver module tests
   - **Effort**: 4-6 hours

5. [ ] Update Simulation error types to use Core errors
   - Refactor `SimulationError` to use `KwaversError`
   - Update all Simulation module error returns
   - Update all Simulation module tests
   - **Effort**: 2-3 hours

6. [ ] Update Analysis error types to use Core errors
   - Refactor `AnalysisError` to use `KwaversError`
   - Update all Analysis module error returns
   - Update all Analysis module tests
   - **Effort**: 2-3 hours

7. [ ] Update Clinical error types to use Core errors
   - Refactor `ClinicalError` to use `KwaversError`
   - Update all Clinical module error returns
   - Update all Clinical module tests
   - **Effort**: 2-3 hours

8. [ ] Remove `pub use physics::*;` from Core layer
   - Remove `src/core/error/types/domain/mod.rs` line 10
   - Remove `src/core/error/mod.rs` line 41
   - **Effort**: 1-2 hours

9. [ ] Test compilation and functionality
   - Run `cargo check --lib`
   - Run `cargo test --lib`
   - Verify all tests pass
   - **Effort**: 2-3 hours

10. [ ] Update documentation
    - Update error type documentation
    - Update architecture documentation
    - Update migration guide
    - **Effort**: 2-3 hours

**Success Criteria**:
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] All tests passing (1554/1554)
- [ ] Core layer has zero dependencies on higher layers
- [ ] Documentation updated

**Deliverables**:
- New error type structure in Core layer
- All error types refactored to use Core errors
- Zero architectural violations
- Updated documentation

---

### P0-2: Fix Clinical Module Circular Import

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 2-4 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Resolve internal circular import within Clinical module.

**Tasks**:
1. [ ] Identify correct location of `ClinicalDecisionSupport`
   - Search for `ClinicalDecisionSupport` definition
   - Determine correct import path
   - **Effort**: 0.5-1 hour

2. [ ] Update import path in neural module
   - Fix import in `src/clinical/imaging/workflows/neural/mod.rs` line 35
   - Use correct path (e.g., `crate::clinical::decision_support::ClinicalDecisionSupport`)
   - **Effort**: 0.5-1 hour

3. [ ] Test compilation and functionality
   - Run `cargo check --lib`
   - Run `cargo test --lib`
   - Verify all tests pass
   - **Effort**: 0.5-1 hour

4. [ ] Verify no circular dependencies
   - Run dependency analysis
   - Verify no circular imports
   - **Effort**: 0.5-1 hour

**Success Criteria**:
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] All tests passing (1554/1554)
- [ ] No circular imports in Clinical module
- [ ] Dependency analysis confirms no circular dependencies

**Deliverables**:
- Fixed import path in neural module
- Zero circular imports
- Verified dependency structure

---

## Phase 2: Codebase Cleanup (Weeks 2-4)

### P0-3: Remove Dead Code Markers

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 40-60 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Reduce 207 `#[allow(dead_code)]` directives to <50.

**Tasks**:
1. [ ] Audit all dead code markers
   - Search for all `#[allow(dead_code)]` directives
   - Categorize by module and priority
   - Document each marker with rationale
   - **Effort**: 8-12 hours

2. [ ] Remove unnecessary dead code markers
   - Remove markers for truly dead code (delete code)
   - Remove markers for code that should be public
   - Remove markers for code used in tests
   - **Effort**: 16-24 hours

3. [ ] Document necessary dead code markers
   - Add TODO comments for code that should be removed later
   - Add feature flags for code that is conditionally used
   - Add rationale for remaining markers
   - **Effort**: 8-12 hours

4. [ ] Verify compilation after cleanup
   - Run `cargo check --lib`
   - Run `cargo clippy -- -D warnings`
   - Fix any issues
   - **Effort**: 4-6 hours

5. [ ] Update documentation
   - Document remaining dead code markers
   - Update cleanup procedures
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] Dead code markers reduced from 207 to <50
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] All tests passing (1554/1554)
- [ ] Documentation updated

**Deliverables**:
- Reduced dead code markers (<50)
- Clean codebase
- Updated documentation

---

### P0-4: Resolve TODO Markers

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 30-40 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Resolve or document all 48 TODO markers with clear priorities.

**Tasks**:
1. [ ] Audit all TODO markers
   - Search for all TODO comments
   - Categorize by priority (P0, P1, P2)
   - Estimate effort for each TODO
   - **Effort**: 6-8 hours

2. [ ] Resolve P0 TODO markers
   - Implement critical TODOs
   - Add tests for implementations
   - Update documentation
   - **Effort**: 12-16 hours

3. [ ] Document P1 TODO markers
   - Add detailed specifications
   - Add effort estimates
   - Add to backlog
   - **Effort**: 6-8 hours

4. [ ] Document P2 TODO markers
   - Add detailed specifications
   - Add effort estimates
   - Add to backlog
   - **Effort**: 6-8 hours

**Success Criteria**:
- [ ] All P0 TODO markers resolved
- [ ] All P1 TODO markers documented with specifications
- [ ] All P2 TODO markers documented with specifications
- [ ] Zero compilation errors
- [ ] All tests passing (1554/1554)

**Deliverables**:
- Resolved P0 TODO markers
- Documented P1 TODO markers
- Documented P2 TODO markers
- Updated backlog

---

### P0-5: Remove or Feature-Flag Stub Implementations

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 20-30 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Remove or properly feature-flag stub implementations.

**Tasks**:
1. [ ] Identify all stub implementations
   - Search for placeholder implementations
   - Categorize by module and priority
   - Document each stub
   - **Effort**: 4-6 hours

2. [ ] Remove unnecessary stubs
   - Remove stubs that are not needed
   - Clean up related code
   - **Effort**: 8-12 hours

3. [ ] Feature-flag necessary stubs
   - Add `#[cfg(feature = "experimental-*")]` guards
   - Add warnings for experimental features
   - Update Cargo.toml feature flags
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] All unnecessary stubs removed
- [ ] All necessary stubs feature-flagged
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] All tests passing (1554/1554)

**Deliverables**:
- Removed unnecessary stubs
- Feature-flagged necessary stubs
- Updated Cargo.toml

---

## Phase 3: Architecture Validation (Weeks 3-4)

### P0-6: Verify 8-Layer Architecture Compliance

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 30-40 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Verify 8-layer architecture compliance across all modules.

**Tasks**:
1. [ ] Create dependency analysis tool
   - Write script to analyze module dependencies
   - Generate dependency graph
   - Identify violations
   - **Effort**: 8-12 hours

2. [ ] Run dependency analysis on all modules
   - Analyze Core layer dependencies
   - Analyze Math layer dependencies
   - Analyze Physics layer dependencies
   - Analyze Domain layer dependencies
   - Analyze Solver layer dependencies
   - Analyze Simulation layer dependencies
   - Analyze Analysis layer dependencies
   - Analyze Clinical layer dependencies
   - Analyze Infrastructure layer dependencies
   - **Effort**: 8-12 hours

3. [ ] Document all dependencies
   - Create dependency matrix
   - Create dependency diagrams
   - Document any violations
   - **Effort**: 6-8 hours

4. [ ] Validate dependency flow
   - Verify unidirectional dependency flow
   - Identify any circular dependencies
   - Identify any cross-contamination
   - **Effort**: 4-6 hours

5. [ ] Create architecture compliance report
   - Document compliance status
   - Document any violations
   - Create remediation plan
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] 100% architectural compliance
- [ ] Zero circular dependencies
- [ ] Zero cross-contamination between layers
- [ ] All dependencies documented
- [ ] Dependency diagrams created

**Deliverables**:
- Dependency analysis tool
- Dependency matrix
- Dependency diagrams
- Architecture compliance report

---

### P0-7: Document SSOT Patterns

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 20-30 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Document single source of truth (SSOT) patterns for shared accessors.

**Tasks**:
1. [ ] Identify all SSOT patterns
   - Audit all shared accessors
   - Identify canonical locations
   - Document usage patterns
   - **Effort**: 6-8 hours

2. [ ] Document SSOT patterns
   - Create SSOT pattern documentation
   - Add code examples
   - Add usage guidelines
   - **Effort**: 8-12 hours

3. [ ] Validate SSOT compliance
   - Verify all layers use SSOT patterns
   - Identify any violations
   - Document any issues
   - **Effort**: 4-6 hours

4. [ ] Create SSOT compliance report
   - Document compliance status
   - Document any violations
   - Create remediation plan
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] All SSOT patterns documented
- [ ] All SSOT patterns validated
- [ ] Zero SSOT violations
- [ ] SSOT compliance report created

**Deliverables**:
- SSOT pattern documentation
- SSOT compliance report
- Usage guidelines

---

## Phase 4: Module Organization Audit (Weeks 4-5)

### P0-8: Audit Module Placement

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 40-50 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Audit all modules for correct placement in layer hierarchy.

**Tasks**:
1. [ ] Create module audit tool
   - Write script to analyze module placement
   - Check module size compliance (<500 lines)
   - Check naming conventions
   - **Effort**: 8-12 hours

2. [ ] Run module audit on all modules
   - Audit Core layer modules
   - Audit Math layer modules
   - Audit Physics layer modules
   - Audit Domain layer modules
   - Audit Solver layer modules
   - Audit Simulation layer modules
   - Audit Analysis layer modules
   - Audit Clinical layer modules
   - Audit Infrastructure layer modules
   - **Effort**: 16-20 hours

3. [ ] Identify misplaced components
   - Document modules in wrong layer
   - Document modules exceeding size limit
   - Document naming convention violations
   - **Effort**: 6-8 hours

4. [ ] Create migration plan
   - Document migration strategy
   - Estimate effort for each migration
   - Prioritize migrations
   - **Effort**: 6-8 hours

5. [ ] Create module organization report
   - Document audit findings
   - Document migration plan
   - Create remediation timeline
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] All modules audited
- [ ] All misplaced components identified
- [ ] Migration plan created
- [ ] Module organization report created

**Deliverables**:
- Module audit tool
- Module organization report
- Migration plan
- Remediation timeline

---

## Phase 5: Research Integration (Weeks 6-24)

### P0-9: Implement Differentiable Forward Solvers (jwave)

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 80-120 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Implement differentiable forward solvers inspired by jwave.

**Tasks**:
1. [ ] Design differentiable solver architecture
   - Define `DifferentiableSolver` trait
   - Design state management for autodiff
   - Design gradient computation interface
   - **Effort**: 12-16 hours

2. [ ] Implement differentiable FDTD solver
   - Refactor FDTD for autodiff
   - Implement gradient computation
   - Add tests
   - **Effort**: 24-32 hours

3. [ ] Implement differentiable PSTD solver
   - Refactor PSTD for autodiff
   - Implement gradient computation
   - Add tests
   - **Effort**: 24-32 hours

4. [ ] Integrate with Burn autodiff
   - Connect solvers to Burn backend
   - Implement backward pass
   - Add tests
   - **Effort**: 16-24 hours

5. [ ] Create functional simulation builder
   - Implement builder pattern
   - Add automatic CFL management
   - Add tests
   - **Effort**: 8-12 hours

6. [ ] Add Fourier series initial conditions
   - Implement FourierSeries type
   - Add IFFT to spatial domain
   - Add tests
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] Differentiable FDTD solver implemented
- [ ] Differentiable PSTD solver implemented
- [ ] Gradient computation working
- [ ] Integration with Burn autodiff complete
- [ ] All tests passing

**Deliverables**:
- Differentiable FDTD solver
- Differentiable PSTD solver
- Functional simulation builder
- Fourier series initial conditions

---

### P0-10: Implement Off-Grid Source/Sensor Integration (k-Wave)

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 60-80 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Implement off-grid source/sensor integration inspired by k-Wave kWaveArray.

**Tasks**:
1. [ ] Design off-grid source architecture
   - Define `ElementArray` type
   - Design surface integration interface
   - Design Fourier-space implementation
   - **Effort**: 8-12 hours

2. [ ] Implement Gaussian quadrature surface integration
   - Implement quadrature rules
   - Implement surface integration
   - Add tests
   - **Effort**: 16-20 hours

3. [ ] Implement Fourier-space distributed source
   - Implement FFT-based source computation
   - Implement off-grid interpolation
   - Add tests
   - **Effort**: 16-20 hours

4. [ ] Integrate with existing source infrastructure
   - Connect to `Source` trait
   - Update factory pattern
   - Add tests
   - **Effort**: 12-16 hours

5. [ ] Validate against k-Wave reference
   - Compare near-field patterns
   - Compare far-field patterns
   - Create validation report
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] Off-grid source integration implemented
- [ ] Surface integration working
- [ ] Fourier-space implementation working
- [ ] Validation against k-Wave complete
- [ ] All tests passing

**Deliverables**:
- Off-grid source integration
- Surface integration implementation
- Fourier-space implementation
- Validation report

---

### P0-11: Implement CT-Based Skull Modeling (BabelBrain)

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 40-60 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Implement CT-based skull modeling inspired by BabelBrain.

**Tasks**:
1. [ ] Design CT-based skull architecture
   - Define `CTSkullModel` type
   - Design HU→acoustic property mapping
   - Design cortical/trabecular classification
   - **Effort**: 8-12 hours

2. [ ] Implement HU→acoustic property conversion
   - Implement piecewise linear maps
   - Implement cortical/trabecular distinction
   - Add tests
   - **Effort**: 12-16 hours

3. [ ] Implement heterogeneous skull medium
   - Create skull medium from CT volume
   - Implement density maps
   - Add tests
   - **Effort**: 12-16 hours

4. [ ] Integrate with existing skull infrastructure
   - Connect to `SkullModel` trait
   - Update factory pattern
   - Add tests
   - **Effort**: 8-12 hours

5. [ ] Validate against BabelBrain reference
   - Compare aberration patterns
   - Compare transmission patterns
   - Create validation report
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] CT-based skull modeling implemented
- [ ] HU→acoustic property conversion working
- [ ] Heterogeneous skull medium working
- [ ] Validation against BabelBrain complete
- [ ] All tests passing

**Deliverables**:
- CT-based skull modeling
- HU→acoustic property conversion
- Heterogeneous skull medium
- Validation report

---

### P0-12: Implement Clinical Workflow Integration (BabelBrain)

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 60-80 hours  
**Dependencies**: P0-11 (CT-Based Skull Modeling)  
**Blocked By**: P0-11

**Objective**: Implement clinical workflow integration inspired by BabelBrain.

**Tasks**:
1. [ ] Design clinical workflow architecture
   - Define `TreatmentPlanner` type
   - Design three-stage workflow
   - Design coordinate system management
   - **Effort**: 12-16 hours

2. [ ] Implement medical imaging pipeline
   - Implement DICOM/NIFTI loading
   - Implement automatic segmentation
   - Implement coordinate transforms
   - Add tests
   - **Effort**: 16-20 hours

3. [ ] Implement treatment planning workflow
   - Implement acoustic simulation stage
   - Implement thermal modeling stage
   - Implement dose calculation stage
   - Add tests
   - **Effort**: 16-20 hours

4. [ ] Implement safety validation
   - Implement IEC compliance checks
   - Implement safety monitoring
   - Add tests
   - **Effort**: 8-12 hours

5. [ ] Integrate with existing clinical infrastructure
   - Connect to clinical workflows
   - Update factory pattern
   - Add tests
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] Clinical workflow integration implemented
- [ ] Medical imaging pipeline working
- [ ] Treatment planning workflow working
- [ ] Safety validation working
- [ ] All tests passing

**Deliverables**:
- Clinical workflow integration
- Medical imaging pipeline
- Treatment planning workflow
- Safety validation

---

## Phase 6: Infrastructure & Performance (Weeks 25-28)

### P1-13: Complete GPU Acceleration

**Priority**: 🟡 P1 - HIGH  
**Effort**: 60-80 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Complete GPU acceleration for critical solvers.

**Tasks**:
1. [ ] Complete GPU FDTD implementation
   - Implement GPU kernels
   - Implement memory management
   - Add tests
   - **Effort**: 20-28 hours

2. [ ] Complete GPU PSTD solver
   - Implement GPU kernels
   - Implement memory management
   - Add tests
   - **Effort**: 20-28 hours

3. [ ] Implement GPU beamforming pipeline
   - Implement GPU kernels
   - Implement delay tables
   - Add tests
   - **Effort**: 12-16 hours

4. [ ] Optimize GPU memory management
   - Implement arena allocation
   - Implement buffer pooling
   - Add tests
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] GPU FDTD implementation complete
- [ ] GPU PSTD solver complete
- [ ] GPU beamforming pipeline complete
- [ ] Memory optimization complete
- [ ] All tests passing

**Deliverables**:
- GPU FDTD implementation
- GPU PSTD solver
- GPU beamforming pipeline
- Memory optimization

---

### P1-14: Complete Cloud Deployment

**Priority**: 🟡 P1 - HIGH  
**Effort**: 40-60 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Complete cloud deployment for AWS, Azure, and GCP.

**Tasks**:
1. [ ] Complete AWS provider implementation
   - Implement AWS ML SDK integration
   - Implement authentication
   - Implement deployment operations
   - Add tests
   - **Effort**: 12-16 hours

2. [ ] Complete Azure provider implementation
   - Implement Azure ML REST API integration
   - Implement authentication
   - Implement deployment operations
   - Add tests
   - **Effort**: 12-16 hours

3. [ ] Complete GCP provider implementation
   - Implement Vertex AI REST API integration
   - Implement authentication
   - Implement deployment operations
   - Add tests
   - **Effort**: 12-16 hours

4. [ ] Add auto-scaling support
   - Implement scaling logic
   - Implement monitoring
   - Add tests
   - **Effort**: 4-6 hours

5. [ ] Add monitoring and logging
   - Implement cloud monitoring
   - Implement structured logging
   - Add tests
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] AWS provider implementation complete
- [ ] Azure provider implementation complete
- [ ] GCP provider implementation complete
- [ ] Auto-scaling support complete
- [ ] Monitoring and logging complete
- [ ] All tests passing

**Deliverables**:
- AWS provider implementation
- Azure provider implementation
- GCP provider implementation
- Auto-scaling support
- Monitoring and logging

---

## Phase 7: Testing & Validation (Weeks 29-32)

### P0-15: Achieve >95% Code Coverage

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 40-50 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Achieve >95% code coverage across all modules.

**Tasks**:
1. [ ] Run coverage analysis
   - Install coverage tool (tarpaulin/llvm-cov)
   - Run coverage analysis
   - Identify coverage gaps
   - **Effort**: 4-6 hours

2. [ ] Add tests for uncovered code
   - Add unit tests for uncovered functions
   - Add integration tests for uncovered workflows
   - Add property-based tests
   - **Effort**: 24-32 hours

3. [ ] Verify coverage >95%
   - Run coverage analysis again
   - Verify >95% coverage
   - Document coverage report
   - **Effort**: 4-6 hours

4. [ ] Add coverage to CI/CD
   - Configure coverage in CI/CD pipeline
   - Add coverage thresholds
   - Add coverage reporting
   - **Effort**: 4-6 hours

**Success Criteria**:
- [ ] Code coverage >95%
- [ ] All critical paths covered
- [ ] Coverage in CI/CD pipeline
- [ ] Coverage report created

**Deliverables**:
- >95% code coverage
- Coverage report
- CI/CD integration

---

### P0-16: Validate Against Reference Implementations

**Priority**: 🔴 P0 - CRITICAL  
**Effort**: 50-60 hours  
**Dependencies**: P0-9, P0-10, P0-11, P0-12  
**Blocked By**: P0-9, P0-10, P0-11, P0-12

**Objective**: Validate kwavers against reference implementations (k-Wave, jwave, fullwave25, BabelBrain).

**Tasks**:
1. [ ] Validate against k-Wave reference results
   - Run k-Wave simulations
   - Run kwavers simulations
   - Compare results
   - Create validation report
   - **Effort**: 12-16 hours

2. [ ] Compare with jwave simulations
   - Run jwave simulations
   - Run kwavers simulations
   - Compare results
   - Create validation report
   - **Effort**: 12-16 hours

3. [ ] Benchmark against fullwave25
   - Run fullwave25 simulations
   - Run kwavers simulations
   - Compare performance
   - Create benchmark report
   - **Effort**: 12-16 hours

4. [ ] Validate clinical workflows with BabelBrain
   - Run BabelBrain workflows
   - Run kwavers workflows
   - Compare results
   - Create validation report
   - **Effort**: 12-16 hours

**Success Criteria**:
- [ ] Validation against k-Wave complete
- [ ] Comparison with jwave complete
- [ ] Benchmark against fullwave25 complete
- [ ] Validation with BabelBrain complete
- [ ] All validation reports created

**Deliverables**:
- k-Wave validation report
- jwave comparison report
- fullwave25 benchmark report
- BabelBrain validation report

---

## Phase 8: Documentation & Knowledge Transfer (Weeks 33-36)

### P1-17: Complete API Documentation

**Priority**: 🟡 P1 - HIGH  
**Effort**: 40-50 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Complete API documentation for all public APIs.

**Tasks**:
1. [ ] Document all public APIs
   - Add doc comments to all public functions
   - Add doc comments to all public types
   - Add doc comments to all public traits
   - **Effort**: 20-24 hours

2. [ ] Add usage examples
   - Add examples for all major features
   - Add examples for all workflows
   - Add examples for all solvers
   - **Effort**: 12-16 hours

3. [ ] Create tutorial documentation
   - Create getting started tutorial
   - Create advanced usage tutorial
   - Create troubleshooting guide
   - **Effort**: 8-12 hours

**Success Criteria**:
- [ ] All public APIs documented
- [ ] Usage examples added
- [ ] Tutorial documentation created
- [ ] Documentation builds successfully

**Deliverables**:
- Complete API documentation
- Usage examples
- Tutorial documentation

---

### P1-18: Complete Architecture Documentation

**Priority**: 🟡 P1 - HIGH  
**Effort**: 30-40 hours  
**Dependencies**: None  
**Blocked By**: None

**Objective**: Complete architecture documentation.

**Tasks**:
1. [ ] Document 8-layer architecture
   - Document each layer
   - Document dependencies
   - Document design rationale
   - **Effort**: 8-12 hours

2. [ ] Create dependency diagrams
   - Create dependency flow diagrams
   - Create module structure diagrams
   - Create SSOT pattern diagrams
   - **Effort**: 8-12 hours

3. [ ] Document SSOT patterns
   - Document all SSOT patterns
   - Add usage guidelines
   - Add examples
   - **Effort**: 8-12 hours

4. [ ] Create migration guides
   - Create migration guide for error types
   - Create migration guide for SSOT patterns
   - Create migration guide for research integration
   - **Effort**: 6-8 hours

**Success Criteria**:
- [ ] 8-layer architecture documented
- [ ] Dependency diagrams created
- [ ] SSOT patterns documented
- [ ] Migration guides created

**Deliverables**:
- Architecture documentation
- Dependency diagrams
- SSOT pattern documentation
- Migration guides

---

## Success Metrics Summary

### Code Quality Metrics
- [ ] Zero compilation errors
- [ ] Zero compiler warnings
- [ ] Dead code markers reduced from 207 to <50
- [ ] TODO markers resolved or documented
- [ ] >95% code coverage
- [ ] All modules <500 lines

### Architecture Metrics
- [ ] 100% architectural compliance
- [ ] Zero circular dependencies
- [ ] Zero cross-contamination between layers
- [ ] Clear SSOT patterns for all shared accessors
- [ ] Unidirectional dependency flow

### Performance Metrics
- [ ] 10-100× speedup for critical kernels
- [ ] Real-time simulation capability for clinical workflows
- [ ] Multi-GPU scaling efficiency >80%
- [ ] Memory usage optimized with arena allocation

### Research Integration Metrics
- [ ] Differentiable forward solvers implemented
- [ ] Off-grid source/sensor integration complete
- [ ] CT-based skull modeling implemented
- [ ] Clinical workflow integration complete
- [ ] Multi-GPU domain decomposition implemented

### Validation Metrics
- [ ] Validated against k-Wave reference results
- [ ] Compared with jwave simulations
- [ ] Benchmarked against fullwave25
- [ ] Validated clinical workflows with BabelBrain
- [ ] Mathematical validation against analytical solutions

---

## Conclusion

This detailed action plan provides specific, actionable tasks for auditing, optimizing, enhancing, extending, and completing the kwavers ultrasound and optics simulation library. The plan is organized by priority (P0, P1, P2) and includes estimated effort, dependencies, and success criteria for each task.

**Key Benefits**:
- Clear, actionable tasks
- Prioritized by importance
- Estimated effort for each task
- Dependencies and blockers identified
- Success criteria defined
- Deliverables specified

**Next Steps**:
1. Review and approve this plan
2. Prioritize tasks based on business needs
3. Allocate resources and schedule
4. Begin with P0 tasks (Critical Infrastructure Fixes)

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-26  
**Status**: Draft for Review

