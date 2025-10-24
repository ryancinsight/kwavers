# Sprint 144: Shear Wave Elastography (SWE) Implementation Plan

**Status**: 🚧 IN PROGRESS  
**Priority**: P1 - HIGH  
**Duration**: 2-3 weeks (16-20 hours)  
**Dependencies**: None (infrastructure exists)

---

## Executive Summary

Sprint 144 implements Shear Wave Elastography (SWE) for non-invasive tissue characterization. SWE generates and tracks shear waves in tissue to quantify stiffness, enabling clinical applications like liver fibrosis staging and breast cancer detection.

**Key Innovation**: Acoustic Radiation Force Impulse (ARFI) + Time-of-Flight tracking + Young's modulus reconstruction

---

## Literature Foundation

### Primary References

1. **Sarvazyan et al. (1998)** - "Shear wave elasticity imaging: A new ultrasonic technology of medical diagnostics"
   - Seminal work introducing SWE for soft tissue characterization
   - Establishes shear modulus sensitivity advantage over bulk modulus
   - Foundation for all modern SWE techniques

2. **Palmeri et al. (2008)** - "Quantifying hepatic shear modulus in vivo using acoustic radiation force"
   - ARFI clinical validation for liver fibrosis
   - Methodology for in vivo shear wave generation
   - Success metrics: <10% measurement error

3. **Bercoff et al. (2004)** - "Supersonic shear imaging: A new technique for soft tissue elasticity mapping"
   - High-frame-rate tracking (>1000 fps)
   - Time-of-flight reconstruction algorithm
   - 2D elasticity mapping methodology

### Technical Principles

**Shear Wave Velocity (SWV)**:
- Faster waves → stiffer tissue
- Typical range: 0.5-5 m/s (soft tissues)
- Liver: 1-2 m/s (healthy), >2.5 m/s (fibrosis)
- Breast: 1-3 m/s (benign), >4 m/s (malignant)

**Young's Modulus (E)**:
- Relationship: E ≈ 3ρVs² (incompressible tissues)
- ρ ≈ 1000 kg/m³ (soft tissue density)
- Units: kPa (kilopascals)

**ARFI Mechanism**:
- High-intensity, short-duration acoustic pulse (>100 μs)
- Radiation force: F = 2αI/c (α=absorption, I=intensity, c=speed)
- Tissue displacement: 1-20 μm
- Shear wave frequency: 50-500 Hz

---

## Implementation Architecture

### Module Structure

```
src/physics/imaging/elastography/
├── mod.rs                    (100 lines) - Public API, feature flags
├── arfi.rs                   (150 lines) - Acoustic Radiation Force Impulse
├── shear_wave_tracking.rs    (200 lines) - Time-of-flight tracking
├── reconstruction.rs         (180 lines) - Elasticity inversion
└── tests/
    └── integration.rs        (200 lines) - End-to-end SWE tests
```

### Core Types

```rust
/// Acoustic Radiation Force Impulse configuration
pub struct ARFIConfig {
    /// Push pulse duration (100-300 μs)
    pub push_duration: Duration,
    /// Push pulse frequency (MHz)
    pub push_frequency: f64,
    /// Push pulse intensity (W/cm²)
    pub push_intensity: f64,
    /// Focal depth (mm)
    pub focal_depth: f64,
}

/// Shear wave tracking configuration
pub struct TrackingConfig {
    /// Frame rate (fps) - minimum 1000 fps for transient shear waves
    pub frame_rate: f64,
    /// Tracking duration (ms)
    pub duration: Duration,
    /// Lateral spacing between tracking locations (mm)
    pub lateral_spacing: f64,
}

/// Elasticity reconstruction result
pub struct ElasticityMap {
    /// Shear wave velocity (m/s)
    pub shear_velocity: Array2<f64>,
    /// Young's modulus (kPa)
    pub youngs_modulus: Array2<f64>,
    /// Confidence map (0-1)
    pub confidence: Array2<f64>,
    /// Grid coordinates
    pub x_coords: Array1<f64>,
    pub z_coords: Array1<f64>,
}
```

---

## Phase 1: ARFI Generation (4 hours)

### Objectives
- [x] Research ARFI physics and parameters
- [ ] Implement radiation force calculation
- [ ] Implement tissue displacement model
- [ ] Implement shear wave generation
- [ ] Add 4 tests (force, displacement, wave generation, config)

### Implementation Tasks

**1.1 Radiation Force Calculation** (1 hour)
```rust
/// Compute acoustic radiation force from ultrasound beam
/// Reference: Palmeri et al. (2008)
pub fn compute_radiation_force(
    intensity: f64,      // W/cm²
    absorption: f64,     // Np/cm
    speed_of_sound: f64, // m/s
) -> f64 {
    // F = 2αI/c
    2.0 * absorption * intensity / speed_of_sound
}
```

**1.2 Tissue Displacement Model** (2 hours)
```rust
/// Model tissue displacement from ARFI push
/// Uses Green's function solution for point force
pub fn compute_displacement(
    force: f64,
    shear_modulus: f64,
    distance: f64,
) -> f64 {
    // Simplified Green's function: u = F / (4π G r)
    force / (4.0 * std::f64::consts::PI * shear_modulus * distance)
}
```

**1.3 Shear Wave Generation** (1 hour)
```rust
/// Generate shear wave from ARFI push
pub fn generate_shear_wave(
    config: &ARFIConfig,
    grid: &Grid,
) -> KwaversResult<Array3<f64>> {
    // Returns (time, x, z) displacement field
    // Shear waves propagate perpendicular to acoustic push
}
```

---

## Phase 2: Shear Wave Tracking (6 hours)

### Objectives
- [ ] Implement time-of-flight algorithm
- [ ] Implement cross-correlation tracking
- [ ] Implement arrival time detection
- [ ] Add 4 tests (TOF, cross-correlation, arrival detection, noise robustness)

### Implementation Tasks

**2.1 Time-of-Flight Algorithm** (3 hours)
```rust
/// Estimate shear wave velocity using time-of-flight
/// Reference: Bercoff et al. (2004)
pub fn time_of_flight_velocity(
    displacement_data: &Array3<f64>, // (time, x, z)
    lateral_positions: &Array1<f64>,  // mm
    time_axis: &Array1<f64>,          // ms
) -> KwaversResult<Array1<f64>> {
    // 1. Detect peak arrival time at each lateral position
    // 2. Fit linear regression: distance vs time
    // 3. Velocity = slope of fit
}
```

**2.2 Cross-Correlation Tracking** (2 hours)
```rust
/// Track shear wave using normalized cross-correlation
pub fn cross_correlation_tracking(
    reference: &Array1<f64>,
    target: &Array1<f64>,
) -> KwaversResult<f64> {
    // Returns time delay in samples
    // Subsample precision using parabolic interpolation
}
```

**2.3 Arrival Time Detection** (1 hour)
```rust
/// Detect shear wave arrival using peak or threshold
pub enum ArrivalDetection {
    Peak,           // Maximum displacement
    Threshold(f64), // Amplitude threshold (e.g., 50%)
    CrossCorrelation,
}
```

---

## Phase 3: Elasticity Reconstruction (6 hours)

### Objectives
- [ ] Implement velocity-to-modulus conversion
- [ ] Implement 2D elasticity mapping
- [ ] Implement confidence estimation
- [ ] Add 4 tests (conversion, mapping, confidence, validation)

### Implementation Tasks

**3.1 Velocity-to-Modulus Conversion** (2 hours)
```rust
/// Convert shear wave velocity to Young's modulus
/// E ≈ 3ρVs² for incompressible tissues
pub fn velocity_to_youngs_modulus(
    shear_velocity: f64, // m/s
    density: f64,        // kg/m³ (default 1000)
) -> f64 {
    3.0 * density * shear_velocity.powi(2) / 1000.0 // Convert to kPa
}
```

**3.2 2D Elasticity Mapping** (3 hours)
```rust
/// Reconstruct 2D elasticity map from tracked shear waves
pub fn reconstruct_elasticity_map(
    displacement_data: &Array3<f64>,
    config: &TrackingConfig,
) -> KwaversResult<ElasticityMap> {
    // 1. Apply TOF at each depth
    // 2. Convert velocity to modulus
    // 3. Generate confidence map
    // 4. Apply spatial smoothing (optional)
}
```

**3.3 Confidence Estimation** (1 hour)
```rust
/// Estimate reconstruction confidence
pub fn estimate_confidence(
    correlation_coeff: f64,
    snr: f64,
) -> f64 {
    // High confidence: R² > 0.9, SNR > 10 dB
    // Returns 0-1 confidence score
}
```

---

## Phase 4: Testing & Validation (4 hours)

### Test Coverage

**Unit Tests** (12 tests minimum):
1. ARFI force calculation
2. Tissue displacement model
3. Shear wave generation
4. Configuration validation
5. Time-of-flight velocity
6. Cross-correlation tracking
7. Arrival time detection
8. Noise robustness
9. Velocity-to-modulus conversion
10. 2D elasticity mapping
11. Confidence estimation
12. End-to-end phantom validation

**Integration Tests**:
- Homogeneous phantom (constant stiffness)
- Two-layer phantom (interface detection)
- Inclusion phantom (lesion detection)
- Noisy data robustness

### Success Criteria

Per Sprint 139 strategic roadmap:
- ✅ <10% elasticity measurement error (vs known phantom)
- ✅ <1s reconstruction time for 2D map (100×100 pixels)
- ✅ Multi-layer tissue validation (detect interfaces)
- ✅ Clinical phantom accuracy (match literature values)

---

## Phase 5: Documentation (2 hours)

### Deliverables
- [ ] Comprehensive rustdoc with examples
- [ ] Usage guide for clinical applications
- [ ] Sprint completion report
- [ ] Update checklist.md and backlog.md
- [ ] Literature references in module docs

---

## Risk Assessment

### Technical Risks

1. **Shear Wave Attenuation** (Medium)
   - Risk: High-frequency shear waves attenuate quickly
   - Mitigation: Multi-frequency approach, frequency-dependent correction
   
2. **Boundary Reflections** (Low)
   - Risk: Reflections corrupt velocity estimates
   - Mitigation: Region-of-interest selection, confidence maps

3. **Tissue Heterogeneity** (Medium)
   - Risk: Complex wavefronts in layered tissues
   - Mitigation: Local velocity estimation, adaptive kernels

### Schedule Risks

1. **TOF Algorithm Complexity** (Low)
   - Contingency: Simplified peak-detection fallback
   
2. **Validation Data** (Low)
   - Mitigation: Use published phantom values (Sarvazyan 1998, Palmeri 2008)

---

## Dependencies

### Internal Dependencies
- ✅ src/physics/medium/ - Tissue properties (shear modulus, absorption)
- ✅ src/physics/time_schemes/ - Time-domain simulation
- ✅ src/grid/ - Spatial grid infrastructure

### External Dependencies
- ✅ ndarray - Multi-dimensional arrays
- ✅ ndarray-stats - Statistical operations
- ✅ approx - Floating-point comparisons

---

## Success Metrics

### Quantitative Metrics
| Metric | Target | Measurement |
|--------|--------|-------------|
| Elasticity Error | <10% | Phantom validation |
| Reconstruction Time | <1s | 100×100 pixel map |
| Test Coverage | 12+ tests | Unit + integration |
| SNR Robustness | >10 dB | Noisy data tests |

### Qualitative Metrics
- ✅ Literature compliance (Sarvazyan 1998, Bercoff 2004, Palmeri 2008)
- ✅ Clinical applicability (liver fibrosis, breast lesion detection)
- ✅ Code quality (clippy clean, rustfmt compliant)
- ✅ Documentation (comprehensive rustdoc + examples)

---

## Timeline

**Week 1** (8 hours):
- Phase 1: ARFI Generation (4 hours)
- Phase 2: Shear Wave Tracking (4 hours)

**Week 2** (8 hours):
- Phase 2 continued: Tracking algorithms (2 hours)
- Phase 3: Elasticity Reconstruction (6 hours)

**Week 3** (4 hours):
- Phase 4: Testing & Validation (4 hours)
- Phase 5: Documentation (2 hours)

**Total**: 20 hours (2-3 weeks part-time development)

---

## References

1. Sarvazyan, A. P., et al. (1998). "Shear wave elasticity imaging: A new ultrasonic technology of medical diagnostics." Ultrasound in Medicine & Biology, 24(9), 1419-1435.

2. Palmeri, M. L., et al. (2008). "Quantifying hepatic shear modulus in vivo using acoustic radiation force." Ultrasound in Medicine & Biology, 34(4), 546-558.

3. Bercoff, J., et al. (2004). "Supersonic shear imaging: A new technique for soft tissue elasticity mapping." IEEE Transactions on Ultrasonics, Ferroelectrics, and Frequency Control, 51(4), 396-409.

4. Acoustic Radiation Force Review (2021). IEEE Open Journal of Ultrasonics, Ferroelectrics, and Frequency Control, 1, 27-42.

5. Elastography 30-Year Perspective (Parker, K. J.). University of Rochester technical report.

---

## Sprint Completion Checklist

### Phase 1: ARFI Generation
- [x] Research literature and physics
- [ ] Implement radiation force calculation
- [ ] Implement tissue displacement model
- [ ] Implement shear wave generation
- [ ] Add 4 unit tests

### Phase 2: Shear Wave Tracking  
- [ ] Implement time-of-flight algorithm
- [ ] Implement cross-correlation tracking
- [ ] Implement arrival time detection
- [ ] Add 4 unit tests

### Phase 3: Elasticity Reconstruction
- [ ] Implement velocity-to-modulus conversion
- [ ] Implement 2D elasticity mapping
- [ ] Implement confidence estimation
- [ ] Add 4 unit tests

### Phase 4: Testing & Validation
- [ ] Phantom validation tests
- [ ] Integration tests
- [ ] Verify <10% error target
- [ ] Verify <1s reconstruction target

### Phase 5: Documentation
- [ ] Comprehensive rustdoc
- [ ] Usage examples
- [ ] Sprint completion report
- [ ] Update checklist and backlog

---

**Sprint Status**: Phase 1 research complete, ready for implementation
**Next Action**: Begin ARFI generation module implementation
**Expected Completion**: Sprint 145 (2-3 weeks from start)
