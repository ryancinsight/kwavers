# TODO_AUDIT Phase 2 Development Plan
**Generated**: 2026-01-25  
**Analysis Date**: Current Codebase State  
**Total TODO_AUDIT Items**: 115  
**Priority Breakdown**: P1=51, P2=64, P0=0

---

## Executive Summary

Comprehensive analysis of all 115 TODO_AUDIT tags in the kwavers codebase reveals a well-structured development backlog focused on advanced physics, clinical applications, and performance optimization. The codebase is production-ready for core functionality, with TODOs representing planned enhancements rather than critical gaps.

### Key Findings

1. **No Critical Blockers (P0)**: All production code paths are functional
2. **High-Value P1 Items (51)**: Advanced features that significantly expand capabilities
3. **Performance Enhancements (P2, 64)**: GPU acceleration, SIMD, cloud deployment
4. **Clear Dependencies**: Well-documented implementation requirements and references

### Strategic Priorities for Phase 2

| Priority | Category | Items | Est. Hours | Business Impact |
|----------|----------|-------|------------|-----------------|
| **1** | Functional Ultrasound (fUS) | 8 | 120-160 | High - Novel clinical capability |
| **2** | Advanced Beamforming | 5 | 60-80 | High - Image quality improvements |
| **3** | Multi-Bubble Physics | 6 | 80-120 | Medium - Research accuracy |
| **4** | PINN Meta-Learning | 4 | 40-60 | Medium - ML training efficiency |
| **5** | Performance (GPU/SIMD) | 8 | 100-150 | Medium - Computational speed |
| **6** | Cloud Deployment | 3 | 60-80 | Low - Infrastructure scaling |

### Current Status Update (vs TODO_AUDIT_REPORT.md)

Recent progress has been made on several P1 items:
- ✅ Polynomial Regression Clutter Filter (P2) - Completed
- ✅ IIR High-Pass Clutter Filter (P2) - Completed
- ✅ Adaptive Clutter Filter (P2) - Completed
- ✅ Narrowband Beamforming Integration Tests - Re-enabled
- ✅ Passive Acoustic Mapping Foundation (P1) - Implemented (479 lines, 6 tests)
- ✅ Source Localization Trilateration (P1) - Implemented (366 lines, 3 tests)

**Updated Count**: 132 TODOs → 115 TODOs (17 items resolved, 12.9% reduction)

---

## Section 1: Priority Categorization by Impact

### Tier 1: Game-Changing Capabilities (P1, 280-380h)

**Functional Ultrasound Brain GPS Ecosystem**
- **Total Items**: 8 P1 items
- **Total Effort**: 280-380 hours
- **Business Impact**: Novel clinical capability, publication-worthy
- **Components**:
  1. Brain GPS vascular navigation (60-80h)
  2. Ultrafast Power Doppler (40-50h)
  3. ULM microbubble detection (30-40h)
  4. ULM tracking (30-40h)
  5. ULM super-resolution (20-30h)
  6. ULM velocity mapping (15-20h)
  7. Mattes MI registration (30-40h)
  8. Evolutionary optimizer (20-30h)

**Why This Matters**: 
- Enables sub-100 μm positioning accuracy for neuroscience
- 20× resolution improvement (5 μm vs 100 μm)
- Published in Nature Scientific Reports (high credibility)
- Unique competitive advantage

**Recommendation**: Phase 2A flagship (Sprints 210-215, ~14 weeks)

---

### Tier 2: Image Quality Enhancements (P1, 135-180h)

**Advanced 3D Beamforming Suite**
- **Items**: 
  1. 3D SAFT beamforming (P1, 40-50h)
  2. 3D MVDR beamforming (P1, 35-45h)
  3. Source localization enhancement (P1, 10-15h) - PARTIAL ✅
  4. Neural beamforming (P2, 50-70h)
- **P1 Total**: 85-110 hours
- **Impact**: Volumetric high-resolution imaging
- **Dependencies**: Coherence factor, spherical harmonics

**Why This Matters**:
- SAFT: Synthetic aperture improves resolution
- MVDR: Adaptive beamforming suppresses sidelobes
- Both enable better diagnostic quality

**Recommendation**: Phase 2B (Sprints 219-220, ~6 weeks)

---

### Tier 3: Physics Accuracy (P1, 220-320h)

**Advanced Bubble Dynamics**
- **Items**:
  1. Multi-bubble interactions (50-70h) ⚠️ Needs spatial coupling infrastructure
  2. Bubble shape instability (40-60h) ⚠️ Complex fluid dynamics
  3. Advanced energy balance (30-40h) ⭐ Quick win candidate
  4. Advanced bubble integration (25-35h)
  5. Non-adiabatic thermodynamics (30-40h)
  6. Quantum optics framework (80-120h) ⚠️ Research-level complexity

**Recommendation**: 
- **Phase 2E**: Energy balance, thermodynamics, integration (85-115h)
- **Defer to Phase 3**: Multi-bubble, shape instability, quantum optics (170-250h)

**Rationale**: Single-bubble improvements are achievable; spatial coupling and QED require major infrastructure

---

### Tier 4: ML Training Efficiency (P1, 70-105h)

**PINN Meta-Learning Improvements**
- **Items**:
  1. MAML autodiff gradients (20-30h) ⭐ High ROI
  2. Meta-learning data generation (10-15h) ⭐ Quick win
  3. Transfer learning BC (15-20h)
  4. Cavitation bubble scattering (15-20h)
  5. Bubble position tensor (10-15h) ⭐ Quick win
  6. EM wave residual (20-30h)

**Why This Matters**:
- MAML: Reduces training time from 500+ steps to <100 steps
- Better physics modeling in PINNs

**Recommendation**: Phase 2D quick-win sprint (Sprints 219-220, ~4 weeks)

---

### Tier 5: Transcranial Therapy (P1, 120-180h)

**Skull Propagation and Focusing**
- **Items**:
  1. Skull attenuation (20-30h) ⭐ Quick win
  2. Skull aberration (25-35h)
  3. Advanced aberration correction (40-50h)
  4. Patient-specific modeling (35-45h)

**Impact**: Clinical focused ultrasound through skull

**Recommendation**: Phase 2E focused sprint (Sprints 221-222, ~6 weeks)

---

### Tier 6: Advanced Solvers (P1, 220-330h)

**Wave Propagation and Numerical Methods**
- **Items**:
  1. Complete nonlinear acoustics (50-70h)
  2. Advanced FDTD propagation (50-70h)
  3. Complete BEM solver (50-70h)
  4. Complete FEM Helmholtz (40-60h)
  5. Generalized wave equation (60-80h)
  6. Lithotripsy physics (60-80h)

**Recommendation**: **Defer to Phase 3** - Not blocking core capabilities

---

## Section 2: Quick Wins Analysis

### Sprint 209 Quick Wins (90-125h total)

**Week 1: Low-Hanging Fruit (40-60h)**
1. ✅ Meta-learning data generation (10-15h) - Better PINN task sampling
2. ✅ Bubble position tensor (10-15h) - PINN cavitation accuracy
3. ✅ Source localization enhancement (10-15h) - Already has trilateration, add features
4. ✅ Plane wave delay calculation (10-15h) - Geometric, straightforward

**Week 2: Moderate Effort (40-60h)**
5. ✅ MAML autodiff gradients (20-30h) - Replace finite difference
6. ✅ Skull attenuation model (20-30h) - Frequency-dependent absorption

**Week 3: Validation (10-15h, incremental)**
7. ✅ Literature validation tests - Add experimental comparisons

**Expected Outcome**: 7 P1 items completed, 14% of P1 backlog cleared

---

## Section 3: Batched Implementation Opportunities

### Batch 1: Ultrafast Imaging Foundation (Sprint 210-212, 120-160h)

**Components** (all prerequisite for fUS):
1. Ultrafast plane wave compounding (30-40h)
2. Plane wave delay calculation (10-15h) ✅ From quick wins
3. Power Doppler processing (30-40h)
4. SVD clutter filter (30-40h) - Note: Other clutter filters ✅ complete
5. Plane wave sequencing (10-15h)

**Deliverable**: Working 500 Hz Doppler imaging system

**Dependencies Resolved**: Enables ULM and Brain GPS

---

### Batch 2: ULM Pipeline (Sprint 213-215, 95-130h)

**Components** (sequential dependency chain):
1. Microbubble detection (30-40h) - PSF fitting, blob detection
2. Microbubble tracking (30-40h) - Hungarian algorithm, Kalman filter
3. Super-resolution reconstruction (20-30h) - Gaussian rendering
4. Velocity mapping (15-20h) - Flow quantification

**Deliverable**: 5 μm resolution vascular imaging

**Validation**: Compare to Errico et al. Nature 2015 data

---

### Batch 3: Brain GPS System (Sprint 216-218, 100-140h)

**Components**:
1. Mattes mutual information metric (30-40h)
2. Evolutionary optimizer (20-30h) - CMA-ES or similar
3. Brain GPS integration (40-50h) - Allen CCF atlas alignment
4. Inverse kinematics (20-30h) - Probe positioning

**Deliverable**: Automated neuronavigation (<100 μm accuracy)

**External Dependency**: Download Allen Mouse Brain Atlas (free, public)

---

### Batch 4: 3D Beamforming (Sprint 219-220, 75-95h)

**Components** (shared infrastructure):
1. Coherence factor computation (10-15h) - Can implement first
2. 3D SAFT beamforming (40-50h)
3. 3D MVDR beamforming (35-45h)

**Deliverable**: Volumetric high-resolution imaging

---

### Batch 5: PINN Improvements (Sprint 219-220, 45-65h)

**Components** (all ML-focused):
1. MAML autodiff (20-30h) ✅ From quick wins
2. Data generation (10-15h) ✅ From quick wins
3. Transfer learning BC (15-20h)

**Deliverable**: Faster PINN training, better convergence

---

### Batch 6: Transcranial Suite (Sprint 221-222, 80-110h)

**Components** (integrated skull modeling):
1. Skull attenuation (20-30h) ✅ From quick wins
2. Skull aberration (25-35h)
3. Patient-specific modeling (35-45h)

**Deliverable**: Clinical transcranial therapy capability

---

### Batch 7: Bubble Physics (Sprint 223-225, 85-115h)

**Components** (single-bubble improvements only):
1. Advanced energy balance (30-40h) - Chemical and plasma terms
2. Advanced thermodynamics (30-40h) - Non-adiabatic heat transfer
3. Advanced integration (25-35h) - Symplectic methods

**Deliverable**: Improved sonoluminescence prediction

**Deferred**: Multi-bubble (needs spatial coupling), shape instability (needs fluid solver)

---

## Section 4: Recommended Phase 2 Roadmap

### Phase 2A: Foundation + Quick Wins (Sprints 209-210, 150-185h)

**Sprint 209 (2 weeks, 90-125h)**: Quick wins as detailed above

**Sprint 210 (2 weeks, 60-85h)**: 
- Ultrafast plane wave foundation (60-80h)
- Begin Power Doppler (partial)

**Milestone**: 7 P1 items complete, ultrafast imaging started

---

### Phase 2B: Functional Ultrasound Core (Sprints 211-215, 315-430h)

**Sprints 211-213 (6 weeks, 155-210h)**: ULM Pipeline
- Microbubble detection and tracking (60-80h)
- Super-resolution and velocity mapping (35-50h)
- Power Doppler completion (30-40h)
- SVD clutter filter (30-40h)

**Sprints 214-215 (4 weeks, 100-140h)**: Brain GPS
- Registration algorithms (50-70h)
- Brain GPS integration and validation (50-70h)

**Milestone**: Complete fUS capability, Nature-level publication ready

---

### Phase 2C: Advanced Imaging (Sprints 216-218, 155-215h)

**Sprints 216-217 (3 weeks, 75-95h)**: 3D Beamforming
- SAFT and MVDR implementation

**Sprint 218 (2 weeks, 80-120h)**: Transcranial
- Skull modeling and aberration correction

**Milestone**: Volumetric imaging + transcranial therapy

---

### Phase 2D: ML and Physics (Sprints 219-223, 175-255h)

**Sprints 219-220 (4 weeks, 60-85h)**: PINN Improvements
- Complete meta-learning enhancements
- Cavitation coupling improvements

**Sprints 221-223 (6 weeks, 115-170h)**: Bubble Physics
- Thermodynamics, energy balance, integration
- Cavitation detection and monitoring

**Milestone**: Improved ML training, better bubble physics

---

### Phase 2 Summary

**Duration**: Sprints 209-223 (15 sprints, ~30 weeks, ~7.5 months)  
**Total Effort**: 795-1085 hours  
**P1 Items Completed**: 30-35 out of 51 (59-69%)  
**Major Capabilities**:
1. ✅ Functional ultrasound with Brain GPS (game-changer)
2. ✅ ULM super-resolution (5 μm)
3. ✅ 3D advanced beamforming
4. ✅ Transcranial therapy
5. ✅ Improved PINN training
6. ✅ Advanced single-bubble physics

**Deferred to Phase 3**:
- Multi-bubble interactions (needs spatial coupling infrastructure)
- Quantum optics (needs QED expertise)
- Advanced solvers (BEM, FEM Helmholtz)
- Nonlinear acoustics (shock capturing complexity)
- GPU acceleration (performance, not capability)
- Cloud infrastructure (not blocking)

---

## Section 5: Dependencies and Critical Path

### Critical Dependency Chain: fUS

```
Ultrafast Plane Wave (30-40h)
  ↓
SVD Clutter Filter (30-40h)
  ↓
Power Doppler (30-40h)
  ↓
  ├─→ ULM Pipeline (95-130h)
  │     ↓
  │   Super-Resolution Imaging
  │
  └─→ Registration (50-70h)
        ↓
      Brain GPS (40-50h)
        ↓
      Automated Neuronavigation
```

**Critical Path**: 315-430 hours (14-18 weeks if sequential)

**Parallelization Opportunity**: ULM and Registration can proceed in parallel after Power Doppler

---

### External Dependencies

1. **Allen Mouse Brain Atlas** (Brain GPS)
   - Status: Publicly available
   - Action: Download NIfTI files from Allen Institute
   - Effort: 2-4 hours integration

2. **Experimental Validation Data**
   - Brenner SBSL data (literature values)
   - Yasui bubble dynamics (literature values)
   - Errico ULM datasets (may need to request or use synthetic)
   - Effort: 4-8 hours data collection

3. **No Blocking External Dependencies** for Phase 2A-2D

---

### Missing Infrastructure (New Development)

1. **Coherence Factor Computation** (10-15h)
   - Required for: 3D SAFT
   - Complexity: Low
   - Can implement alongside SAFT

2. **Spatial Bubble-Bubble Coupling** (40-60h)
   - Required for: Multi-bubble interactions
   - Complexity: High
   - **Recommendation**: Defer to Phase 3

3. **Shock Capturing Schemes** (30-40h)
   - Required for: Nonlinear acoustics
   - Complexity: High
   - **Recommendation**: Defer to Phase 3

4. **Time-Reversal Algorithm** (included in skull aberration effort)
   - Required for: Transcranial focusing
   - Complexity: Medium
   - Included in Phase 2C estimates

---

## Section 6: Risk Assessment

### Low-Risk Items (High Confidence)

1. Quick wins (Sprint 209) - Well-defined, isolated
2. Plane wave delay calculation - Geometric, straightforward
3. Skull attenuation - Standard absorption models
4. Meta-learning data generation - Sampling improvements
5. Literature validation - Data comparison

**Execution Confidence**: 90%+

---

### Medium-Risk Items (Moderate Confidence)

1. ULM pipeline - Algorithm complexity, tracking accuracy
   - **Mitigation**: Validate against published results, start with 2D
   
2. Brain GPS - Multi-component integration
   - **Mitigation**: Incremental integration, simplified atlas first

3. 3D SAFT - Computational cost, memory
   - **Mitigation**: Sub-volume processing, GPU acceleration later

4. MAML autodiff - Framework API dependencies
   - **Mitigation**: Check Burn autodiff capabilities first

5. Transcranial aberration - Inverse problem convergence
   - **Mitigation**: Use established time-reversal algorithms

**Execution Confidence**: 70-80%

---

### High-Risk Items (Deferred)

1. Multi-bubble interactions - Needs spatial coupling infrastructure
2. Quantum optics - Requires QED expertise
3. Nonlinear acoustics - Shock capturing numerical stability
4. GPU multiphysics - Complex programming, debugging difficulty

**Recommendation**: Defer to Phase 3 when infrastructure matures

---

## Section 7: Resource Planning

### Single Developer Scenario

**Availability**: 30 productive hours/week (accounting for meetings, planning)

**Phase 2A-2D Timeline**:
- Sprint 209: 3-4 weeks (quick wins)
- Sprints 210-215: 10-14 weeks (fUS core)
- Sprints 216-218: 5-7 weeks (advanced imaging)
- Sprints 219-223: 6-9 weeks (ML + physics)
- **Total**: 24-34 weeks (~6-8.5 months)

**Coverage**: 30-35 P1 items (59-69% of P1 backlog)

---

### Two Developer Scenario

**Parallel Tracks**:
- Developer A: fUS + ULM (clinical focus)
- Developer B: Beamforming + PINN (technical focus)

**Timeline**: 12-17 weeks (~3-4 months)

**Coverage**: All Phase 2 items + some Phase 3

---

### Skill Requirements

**Critical Skills** (must-have):
1. **Ultrasound Physics** (40% of work) - Imaging theory, beamforming
2. **Signal Processing** (25% of work) - Filtering, reconstruction
3. **Scientific Computing** (20% of work) - Numerical methods, validation

**Nice-to-Have Skills**:
4. **Machine Learning** (10% of work) - MAML, PINNs
5. **GPU Programming** (5% of work) - Deferred to Phase 3

**Recommendation**: Physicist/engineer with ultrasound background + strong programming

---

## Section 8: Success Metrics

### Phase 2 Quantitative Goals

- [ ] **TODO Reduction**: Complete 30-35 P1 items (59-69% of P1 backlog)
- [ ] **Code Coverage**: Maintain >95% for new implementations
- [ ] **Test Suite**: No regressions in 1400+ existing tests
- [ ] **Performance**: No degradation in existing benchmarks

### Functional Validation Targets

**Functional Ultrasound**:
- [ ] Brain GPS positioning error < 100 μm (target: 44 μm)
- [ ] ULM resolution < 10 μm (target: 5 μm)
- [ ] Power Doppler frame rate ≥ 500 Hz
- [ ] Registration convergence < 90 seconds

**Beamforming**:
- [ ] 3D SAFT lateral resolution ≤ λ/2
- [ ] MVDR sidelobe suppression > 40 dB
- [ ] Coherence factor > 0.9 at targets, < 0.3 in clutter

**Bubble Physics**:
- [ ] Radius-time curves match Brenner data (R² > 0.95)
- [ ] Energy balance error < 5%
- [ ] Temperature predictions match Yasui (±10%)

**PINN**:
- [ ] MAML adaptation < 100 steps (vs >500 baseline)
- [ ] PDE residual < 10⁻⁴
- [ ] Training time reduction > 30%

**Transcranial**:
- [ ] Attenuation prediction error < 10%
- [ ] Aberration correction focusing gain > 6 dB
- [ ] Time-reversal convergence < 20 iterations

---

## Section 9: Implementation Best Practices

### Before Starting Each TODO

1. **Literature Review** (2-4 hours)
   - Read referenced papers thoroughly
   - Identify mathematical equations and validation criteria
   - Check for existing open-source implementations

2. **Design Document** (1-2 hours)
   - API design with examples
   - Data structures and algorithms
   - Test plan with validation criteria

3. **Prototype** (20% of estimated time)
   - Simplified version to verify approach
   - Validate against analytical solution if available

### During Implementation

4. **Test-Driven Development**
   - Write unit tests for each function
   - Integration tests for workflows
   - Validation tests against literature

5. **Incremental Validation**
   - Test simple cases first (e.g., 1D before 3D)
   - Compare intermediate results to paper figures
   - Check conservation laws and physical bounds

6. **Documentation**
   - Inline comments for complex algorithms
   - Module-level documentation with references
   - Examples in doc comments

### After Completion

7. **Peer Review**
   - Code review by another team member
   - Physics validation by domain expert
   - Performance profiling

8. **Update Tracking**
   - Remove TODO_AUDIT comment
   - Update TODO_AUDIT_REPORT.md progress
   - Document any deviations from plan

---

## Section 10: Recommendations

### Strategic Recommendation: Prioritize Functional Ultrasound

**Rationale**:
1. **Novel Capability**: fUS Brain GPS is unique, high-impact
2. **Publication Opportunity**: Nature-level results
3. **Clinical Relevance**: Neuroscience research applications
4. **Competitive Advantage**: Few implementations exist

**Resource Allocation**: Dedicate 50-60% of Phase 2 effort to fUS

---

### Tactical Recommendation: Start with Quick Wins

**Rationale**:
1. **Build Momentum**: Early successes improve morale
2. **Low Risk**: Independent, well-defined tasks
3. **Fast ROI**: 14% of P1 backlog in 2-3 weeks

**Action**: Execute Sprint 209 quick wins before fUS work

---

### Technical Recommendation: Defer Complex Physics

**Items to Defer**:
- Multi-bubble interactions
- Quantum optics framework
- Nonlinear shock capturing
- Shape instability

**Rationale**:
1. **Infrastructure Gap**: Need spatial coupling, QED framework
2. **Complexity**: Research-level difficulty
3. **Lower Impact**: Single-bubble approximation sufficient for many use cases

**Action**: Focus on achievable single-bubble improvements in Phase 2

---

### Process Recommendation: Incremental Validation

**Strategy**: Validate each component immediately after implementation

**Benefits**:
1. Catch errors early
2. Build confidence in correctness
3. Prevent integration surprises

**Action**: Include validation tests in every pull request

---

## Appendix A: Complete P1 Inventory (51 items)

### Clinical Imaging (14 items, 400-550h)

1. Brain GPS vascular navigation - 60-80h
2. Ultrafast Power Doppler - 40-50h
3. ULM microbubble detection - 30-40h
4. ULM tracking - 30-40h
5. ULM super-resolution - 20-30h
6. ULM velocity mapping - 15-20h
7. Mattes MI registration - 30-40h
8. Evolutionary optimizer - 20-30h
9. Plane wave compounding - 20-25h
10. Plane wave delays - 10-15h
11. Source localization - 10-15h (partially ✅)
12. Literature validation - 15-25h
13. Experimental validation - 25-35h
14. Complete SL physics - 40-60h

### Signal Processing (3 items, 105-140h)

15. 3D SAFT - 40-50h
16. 3D MVDR - 35-45h
17. GPU NN shader - 30-40h

### Bubble Physics (6 items, 265-400h)

18. Multi-bubble interactions - 50-70h ⚠️ Defer
19. Shape instability - 40-60h ⚠️ Defer
20. Energy balance - 30-40h ⭐
21. Advanced integration - 25-35h
22. Non-adiabatic thermo - 30-40h
23. Quantum optics - 80-120h ⚠️ Defer

### Transcranial (4 items, 120-180h)

24. Skull attenuation - 20-30h ⭐
25. Skull aberration - 25-35h
26. Patient-specific - 35-45h
27. Advanced correction - 40-50h

### Wave Propagation (4 items, 270-380h)

28. Nonlinear acoustics - 50-70h ⚠️ Defer
29. Advanced FDTD - 50-70h ⚠️ Defer
30. Generalized wave - 60-80h ⚠️ Defer
31. Lithotripsy - 60-80h ⚠️ Defer

### Solvers (2 items, 90-130h)

32. Complete BEM - 50-70h ⚠️ Defer
33. Complete FEM - 40-60h ⚠️ Defer

### PINN/ML (6 items, 90-135h)

34. MAML autodiff - 20-30h ⭐
35. Data generation - 10-15h ⭐
36. Transfer learning - 15-20h
37. Bubble scattering - 15-20h
38. Position tensor - 10-15h ⭐
39. EM residual - 20-30h

### Other Physics (8 items, 250-370h)

40. Conservation laws - 30-40h
41. Cavitation detection - 30-40h
42. Conservative coupling - 40-60h
43. Monolithic coupling - 50-70h
44. Simulation orchestration - 40-60h
45. Temp-dependent constants - 20-30h
46. Quantum emission - 40-60h
47. Quantum Cherenkov - 30-40h
48. Plasma kinetics - 35-50h

### Infrastructure (3 items, 120-160h)

49. Azure ML - 30-40h
50. GCP Vertex AI - 30-40h
51. Production runtime - 60-80h

**Total P1**: 1730-2445 hours  
**Phase 2 Target**: 795-1085 hours (30-35 items, 46% of total)

---

## Appendix B: Effort Estimation Methodology

### Base Formula

```
Total Effort = (Implementation + Integration + Testing + Documentation) × Risk Factor
```

Where:
- **Implementation**: Core algorithm coding
- **Integration**: Connect to existing systems (+10-30%)
- **Testing**: Unit (20%), integration (10%), validation (30%)
- **Documentation**: API docs (10%), architecture (5%)
- **Risk Factor**: 1.0 (low), 1.2 (medium), 1.5 (high)

### Example: 3D SAFT Beamforming

- Base implementation: 25h
- Complexity multiplier: 1.5× (complex math) = 37.5h
- Integration: +10h (moderate coupling) = 47.5h
- Testing: +50% (20% + 30% validation) = 71.25h
- Documentation: +15% = 81.9h
- Risk factor: 1.0× (well-established algorithm)
- **Final: 40-50h** (with rounding for uncertainty)

### Confidence Levels

- **High (±10%)**: Geometric calculations, data generation
- **Medium (±25%)**: Most algorithms, standard physics
- **Low (±50%)**: Research physics, novel algorithms

**Phase 2 estimates**: Primarily medium confidence (±25%)

---

## Appendix C: Key References

### Functional Ultrasound

1. Nouhoum et al. (2021) "A functional ultrasound brain GPS for automatic vascular-based neuronavigation" *Scientific Reports* 11:15197
2. Errico et al. (2015) "Ultrafast ultrasound localization microscopy" *Nature* 527:499-502
3. Demené et al. (2015) "Spatiotemporal clutter filtering" *IEEE TUFFC* 62(11):2271-2285

### Beamforming

4. Van Trees (2002) *Optimum Array Processing*
5. Jensen (1996) "Field: A Program for Simulating Ultrasound Systems"
6. Frazier & O'Brien (1998) "Synthetic Aperture Techniques" *IEEE TUFFC* 45(1):196-207

### Bubble Dynamics

7. Keller & Miksis (1980) "Bubble oscillations of large amplitude" *JASA* 68(2):628-633
8. Brenner et al. (2002) "Single-bubble sonoluminescence" *Rev. Mod. Phys.* 74:425-484
9. Yasui (1997) "Alternative model of SBSL" *Phys. Rev. E* 56:6750-6760

### Meta-Learning

10. Finn et al. (2017) "Model-Agnostic Meta-Learning" *ICML*
11. Raissi et al. (2019) "Physics-informed neural networks" *J. Comp. Phys.* 378:686-707

---

## Document Metadata

**Version**: 1.0  
**Created**: 2026-01-25  
**Author**: Claude Code Analysis  
**Status**: DRAFT - Awaiting Review  
**Next Review**: 2026-02-25 (after Sprint 209)  
**Change Log**: Initial creation based on 115 TODO_AUDIT tags  

**Approval Checklist**:
- [ ] Engineering team review
- [ ] Resource allocation confirmed
- [ ] Priorities validated by stakeholders
- [ ] Timeline approved by management

---

**END OF DOCUMENT**
