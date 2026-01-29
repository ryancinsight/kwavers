# Phase 4: Research-Driven Enhancements
**Priority**: 🟢 MEDIUM - Implement state-of-the-art features  
**Timeline**: 2-3 weeks  
**Status**: Ready to execute  
**Based on**: Review of 12 leading ultrasound simulation libraries

---

## Overview

Phase 4 implements five research-driven enhancements that will position kwavers as a **leading ultrasound and optics simulation library**. These features are inspired by the latest research and proven implementations in other libraries.

**Total Effort**: 70-100 hours (can be parallelized)  
**Team Size**: 1-2 engineers  
**Priority Ranking**:
1. ✅ k-Space PSTD (HIGH IMPACT - improves accuracy)
2. Autodiff Framework (HIGH IMPACT - enables inverse problems)
3. High-Order FDTD (MEDIUM - improves efficiency)
4. Clinical Workflows (MEDIUM - enables medical device)
5. Adaptive Beamforming (MEDIUM - improves imaging)

---

## 4.1: k-Space PSTD Enhancement

### Overview
Enhance the PSTD solver with k-space correction operator for improved accuracy and stability.

**Impact**: Fewer grid points needed for same accuracy (typically 2-3x improvement)  
**Reference**: k-Wave, k-Wave-Python

### Current State
- ✅ Basic PSTD implementation exists in `src/solver/forward/pstd/`
- ✅ FFT infrastructure in place
- ⚠️ Missing k-space correction operator
- ⚠️ Grid spacing calculated manually

### Implementation

#### File: `src/solver/forward/pstd/kspace_correction.rs` (NEW)
```rust
/// k-space correction operator for pseudospectral methods
pub struct KSpaceCorrection {
    /// Wavenumber grid (pre-computed)
    k_grid: Array3<f64>,
    /// Correction operator (complex exponential)
    correction_op: Array3<Complex<f64>>,
}

impl KSpaceCorrection {
    /// Create correction operator
    pub fn new(grid: &Grid, medium: &dyn Medium) -> Result<Self> {
        let k_grid = compute_wavenumber_grid(grid);
        let c = medium.sound_speed();
        let correction_op = k_grid.mapv(|k| {
            let omega = k * c;
            Complex::from_polar(1.0, omega * dt)
        });
        Ok(Self { k_grid, correction_op })
    }
    
    /// Apply correction to pressure field
    pub fn apply(&self, pressure_fft: &mut Array3<Complex<f64>>) {
        *pressure_fft *= &self.correction_op;
    }
}
```

#### File: Modify `src/solver/forward/pstd/implementation/core/stepper.rs`
```rust
// In PSTD stepping function:
pub fn time_step(&mut self, source: &Source) {
    // Existing code...
    
    // Apply k-space correction
    self.kspace_correction.apply(&mut self.pressure_fft);
    
    // Continue with FFT inverse...
}
```

### Validation Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_kspace_pstd_accuracy_improvement() {
        // Compare 2-PPW PSTD with kspace vs without
        // Expected: kspace matches 5-PPW accuracy
    }
    
    #[test]
    fn test_kspace_stability() {
        // Verify CFL conditions with kspace correction
    }
}
```

### Acceptance Criteria
- ✅ PSTD with k-space correction compiles
- ✅ Accuracy validation tests pass (matches theoretical predictions)
- ✅ Performance no worse than current PSTD
- ✅ Documentation complete with examples

**Effort**: 8-12 hours  
**Complexity**: Medium  
**Risk**: Low (localized change)

---

## 4.2: Autodiff Framework

### Overview
Add automatic differentiation support for inverse problems and gradient-based optimization.

**Impact**: Enable neural network training, parameter optimization, sensitivity analysis  
**Reference**: j-Wave, DBUA

### Current State
- ✅ Solver infrastructure in place
- ❌ No autodiff support
- ❌ No adjoint method
- ❌ No gradient computation

### Implementation

#### File: `src/solver/inverse/autodiff/mod.rs` (NEW - Feature-gated)
```rust
#[cfg(feature = "autodiff")]
pub mod autodiff {
    use ndarray::Array3;
    use num_complex::Complex;
    
    /// Differentiable wave equation solver
    pub struct DifferentiableFdtd {
        /// Forward solver state
        forward: FdtdSolver,
        /// Gradient state (adjoint method)
        gradient: GradientState,
    }
    
    /// Gradient computation via adjoint method
    impl DifferentiableFdtd {
        pub fn forward(&self, params: &Parameters) -> Array3<f64> {
            // Standard FDTD forward pass
        }
        
        pub fn backward(&self, grad_output: &Array3<f64>) -> Gradients {
            // Time-reversed adjoint computation
            // Compute sensitivity to: density, sound speed, etc.
        }
    }
}
```

#### File: `Cargo.toml` feature
```toml
[features]
autodiff = []  # Automatic differentiation support
```

#### File: `src/solver/inverse/optimization/mod.rs` (NEW)
```rust
/// Gradient-based parameter optimization
pub struct ParameterOptimizer {
    learning_rate: f64,
    solver: DifferentiableFdtd,
}

impl ParameterOptimizer {
    /// Optimize parameters to match target field
    pub fn optimize(&mut self, target: &Array3<f64>, iterations: u32) {
        for iter in 0..iterations {
            let output = self.solver.forward(&self.params);
            let loss = compute_loss(&output, target);
            
            let gradients = self.solver.backward(&output);
            self.params.update(&gradients, self.learning_rate);
        }
    }
}
```

### Validation Tests

```rust
#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "autodiff")]
    fn test_gradient_computation() {
        // Verify gradients via finite differences
    }
    
    #[test]
    #[cfg(feature = "autodiff")]
    fn test_parameter_optimization() {
        // Optimize sound speed and density
        // Verify convergence to target
    }
}
```

### Acceptance Criteria
- ✅ Autodiff feature compiles cleanly
- ✅ Gradient computation matches finite differences
- ✅ Optimization converges on simple problems
- ✅ Benchmarks show acceptable overhead

**Effort**: 16-20 hours  
**Complexity**: High  
**Risk**: Medium (new feature, requires validation)

---

## 4.3: High-Order FDTD

### Overview
Extend FDTD to 4th and 8th-order spatial accuracy for improved efficiency.

**Impact**: Same accuracy with 3-5x fewer grid points  
**Reference**: Fullwave25, acoustic simulation literature

### Current State
- ✅ 2nd-order FDTD working
- ❌ No 4th-order option
- ❌ No 8th-order option
- ❌ Stencil coefficients hardcoded

### Implementation

#### File: `src/solver/forward/fdtd/stencils.rs` (NEW)
```rust
/// Spatial stencil coefficients for different orders
pub struct Stencil {
    order: u32,
    coefficients: Vec<f64>,  // Central difference coefficients
}

impl Stencil {
    pub fn standard_2nd_order() -> Self {
        Self {
            order: 2,
            coefficients: vec![1.0, -2.0, 1.0],  // [f(x+h), f(x), f(x-h)]
        }
    }
    
    pub fn central_4th_order() -> Self {
        Self {
            order: 4,
            coefficients: vec![-1.0, 16.0, -30.0, 16.0, -1.0],
        }
    }
    
    pub fn central_8th_order() -> Self {
        Self {
            order: 8,
            coefficients: vec![
                1.0, -8.0, 28.0, -56.0, 70.0, -56.0, 28.0, -8.0, 1.0
            ],
        }
    }
}
```

#### File: Modify `src/solver/forward/fdtd/implementation/core/stepper.rs`
```rust
pub struct FdtdStepper {
    stencil: Stencil,
    // ... other fields
}

impl FdtdStepper {
    /// Apply spatial derivative using selected stencil
    fn apply_laplacian(&self, field: &Array3<f64>) -> Array3<f64> {
        match self.stencil.order {
            2 => self.apply_2nd_order(field),
            4 => self.apply_4th_order(field),
            8 => self.apply_8th_order(field),
            _ => panic!("Unsupported order"),
        }
    }
}
```

### CFL Stability Analysis

```rust
/// Updated CFL condition for high-order schemes
pub fn compute_cfl_limit(order: u32, dx: f64, c: f64) -> f64 {
    match order {
        2 => 1.0 / (c * (1.0 / (2.0*dx)).sqrt()),      // ~0.5
        4 => 1.0 / (c * (10.0 / (12.0*dx)).sqrt()),    // ~0.37
        8 => 1.0 / (c * (840.0 / (840.0*dx)).sqrt()),  // ~0.26
        _ => panic!("Unsupported order"),
    }
}
```

### Validation Tests

```rust
#[test]
fn test_4th_order_fdtd_accuracy() {
    // Gaussian pulse propagation
    // Verify 4th-order convergence rate
}

#[test]
fn test_8th_order_fdtd_efficiency() {
    // Compare memory usage and speed vs 2nd-order
    // 3-5x fewer points for same accuracy?
}
```

### Acceptance Criteria
- ✅ 4th and 8th-order FDTD compiles
- ✅ Convergence rates match theory
- ✅ Stability limits respected
- ✅ Performance benchmarks completed

**Effort**: 12-16 hours  
**Complexity**: Medium  
**Risk**: Low (well-established methods)

---

## 4.4: Clinical Workflow Integration

### Overview
Complete treatment planning pipeline with safety validation and DICOM I/O.

**Impact**: Enable medical device development, clinical adoption  
**Reference**: BabelBrain, Kranion

### Current State
- ⚠️ Basic clinical module exists
- ❌ No DICOM import/export
- ❌ No safety metric calculation
- ❌ No multi-stage pipeline
- ❌ No treatment parameter export

### Implementation

#### File: `src/clinical/workflow/pipeline.rs` (NEW)
```rust
/// Multi-stage clinical treatment planning pipeline
pub struct TreatmentPipeline {
    stage_prep: PreparationStage,
    stage_acoustic: AcousticSimulation,
    stage_thermal: ThermalAssessment,
}

impl TreatmentPipeline {
    pub fn execute(&self, patient: &PatientData) -> Result<TreatmentPlan> {
        // Stage 1: Preparation (DICOM import, registration, segmentation)
        let prep = self.stage_prep.execute(patient)?;
        
        // Stage 2: Acoustic simulation
        let acoustic = self.stage_acoustic.execute(&prep)?;
        
        // Stage 3: Thermal assessment
        let thermal = self.stage_thermal.execute(&acoustic)?;
        
        Ok(TreatmentPlan { prep, acoustic, thermal })
    }
}
```

#### File: `src/clinical/safety/metrics.rs` (NEW)
```rust
/// Safety metrics per FDA regulations
pub struct SafetyMetrics {
    pub mechanical_index: f64,      // MI ≤ 1.9
    pub thermal_index_bone: f64,    // TIB ≤ 6.0
    pub thermal_index_soft: f64,    // TIS ≤ 3.0
    pub ispta: f64,                 // ISPTA ≤ 720 mW/cm²
}

impl SafetyMetrics {
    pub fn validate(&self) -> Result<()> {
        if self.mechanical_index > 1.9 {
            return Err(SafetyError::MIExceeded(self.mechanical_index));
        }
        if self.thermal_index_bone > 6.0 {
            return Err(SafetyError::TIBExceeded(self.thermal_index_bone));
        }
        // ... more checks
        Ok(())
    }
}
```

#### File: `src/infra/io/dicom_handler.rs` (NEW)
```rust
/// DICOM import/export for clinical integration
pub struct DicomHandler;

impl DicomHandler {
    pub fn import_ct(path: &Path) -> Result<ImageVolume> {
        // Use dicom crate to read CT data
    }
    
    pub fn import_mri(path: &Path) -> Result<ImageVolume> {
        // Read MRI data
    }
    
    pub fn export_treatment_plan(path: &Path, plan: &TreatmentPlan) -> Result<()> {
        // Export as DICOM Radiotherapy Plan
    }
}
```

### Acceptance Criteria
- ✅ Pipeline executes full workflow
- ✅ Safety metrics computed per FDA standards
- ✅ DICOM I/O functional
- ✅ Treatment parameters exportable to clinical systems
- ✅ Multi-patient workflow tested

**Effort**: 20-24 hours  
**Complexity**: High  
**Risk**: Medium (FDA compliance required)

---

## 4.5: Adaptive Beamforming

### Overview
Neural network-based beamforming with coherence analysis and uncertainty quantification.

**Impact**: Improved image quality, reduced artifacts, adaptive to patient anatomy  
**Reference**: DBUA, Sound-Speed-Estimation, DBUA

### Current State
- ✅ Basic DAS and MVDR beamforming exists
- ❌ No neural network optimization
- ❌ No coherence analysis
- ❌ No sound speed estimation
- ❌ No uncertainty maps

### Implementation

#### File: `src/analysis/signal_processing/beamforming/neural_adaptive.rs` (NEW)
```rust
/// Adaptive beamforming with neural optimization
pub struct AdaptiveBeamformer {
    delays: Array2<f32>,  // Learnable delay parameters
    config: AdaptiveConfig,
}

impl AdaptiveBeamformer {
    /// Optimize delays via gradient descent
    pub fn optimize(&mut self, rf_data: &Array3<f32>, target: &Array2<f32>) {
        for iteration in 0..self.config.iterations {
            let output = self.beamform(rf_data);
            let loss = self.compute_loss(&output, target);
            let gradients = self.compute_gradients(&output, target);
            
            self.delays -= self.config.learning_rate * gradients;
        }
    }
}
```

#### File: `src/analysis/signal_processing/beamforming/coherence.rs` (NEW)
```rust
/// Coherence-based sound speed estimation
pub struct CoherenceEstimator;

impl CoherenceEstimator {
    /// Estimate sound speed from coherence measurements
    pub fn estimate_sound_speed(
        &self,
        iq_data: &Array3<Complex<f32>>,
        speed_range: (f32, f32),
    ) -> f32 {
        let mut best_speed = speed_range.0;
        let mut best_coherence = 0.0;
        
        for speed in (speed_range.0 as i32..=speed_range.1 as i32).map(|s| s as f32) {
            let coherence = self.compute_slsc(iq_data, speed);
            if coherence > best_coherence {
                best_coherence = coherence;
                best_speed = speed;
            }
        }
        best_speed
    }
    
    /// Short-lag spatial coherence (SLSC) metric
    fn compute_slsc(&self, iq_data: &Array3<Complex<f32>>, speed: f32) -> f32 {
        // Beamform with trial speed, compute coherence
    }
}
```

#### File: `src/analysis/signal_processing/beamforming/uncertainty.rs` (NEW)
```rust
/// Uncertainty quantification for beamforming
pub struct UncertaintyMap {
    pub confidence: Array2<f32>,     // Per-pixel confidence [0,1]
    pub variance: Array2<f32>,       // Per-pixel variance
}

impl UncertaintyMap {
    pub fn compute_from_ensemble(
        ensemble: &[Array2<f32>],    // Multiple beamforming passes
    ) -> Self {
        // Compute mean and variance across ensemble
    }
}
```

### Acceptance Criteria
- ✅ Adaptive beamforming learns optimal delays
- ✅ Sound speed estimation converges
- ✅ Uncertainty maps computed and validated
- ✅ Integration tests with real beam data
- ✅ Performance acceptable for real-time use

**Effort**: 16-20 hours  
**Complexity**: High  
**Risk**: Medium (requires neural network training)

---

## Execution Strategy

### Week 1: Foundation (4.1 + 4.3)
- **Days 1-2**: k-Space PSTD (8-12 hrs)
  - Implement correction operator
  - Add validation tests
  - Commit and verify

- **Days 3-5**: High-Order FDTD (12-16 hrs)
  - Implement stencils and CFL analysis
  - Add convergence tests
  - Performance benchmarking

### Week 2: AI Features (4.2 + 4.5)
- **Days 1-3**: Autodiff Framework (16-20 hrs)
  - Implement gradient computation
  - Create optimizer
  - Validation tests

- **Days 4-5**: Adaptive Beamforming (16-20 hrs)
  - Neural optimizer
  - Sound speed estimation
  - Uncertainty quantification

### Week 3: Clinical (4.4)
- **Days 1-5**: Clinical Workflows (20-24 hrs)
  - Treatment pipeline
  - Safety metrics
  - DICOM I/O
  - Validation and testing

---

## Success Criteria

Each enhancement must meet:

1. ✅ **Compilation**: Zero errors, <5 warnings
2. ✅ **Testing**: 95%+ tests passing
3. ✅ **Validation**: Comparison with reference implementations
4. ✅ **Documentation**: Complete with examples
5. ✅ **Integration**: No breaking changes to existing API

---

## Expected Impact

After Phase 4 complete:

- **kwavers** becomes **world-leading** simulation library
- **k-space PSTD**: 2-3x accuracy improvement over standard PSTD
- **Autodiff**: Enables inverse problems and parameter optimization
- **High-Order FDTD**: 3-5x computational efficiency gains
- **Clinical Workflows**: FDA-compliant treatment planning
- **Adaptive Beamforming**: Neural network-based imaging improvements

---

## References

All enhancements based on published research:

- **k-Wave**: Treeby & Cox (2010), IEEE Trans. Biomed. Eng.
- **j-Wave**: Stanziola et al. (2022), arXiv preprint
- **Fullwave25**: Full-Wave Simulator, GitHub
- **BabelBrain**: MRI-guided HIFU planning, GitHub
- **DBUA**: Deep Learning Beamforming, GitHub
- **Sound-Speed-Estimation**: Adaptive imaging, GitHub

---

**Status**: Ready to execute  
**Next**: Start Phase 4.1 (k-Space PSTD)  
**Estimated Completion**: 2-3 weeks
