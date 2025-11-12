# kwavers Ultrasound Simulation Platform - API Guide

## Overview

The `kwavers` platform provides a comprehensive Rust-based framework for ultrasound simulation with physics-informed neural networks, advanced wave propagation algorithms, and clinical applications. This guide covers the complete API surface for building ultrasound simulation workflows.

## Core Concepts

### Grid System
All simulations operate on structured computational grids:

```rust
use kwavers::grid::Grid;

// Create a 3D computational grid (256x256x128, 0.1mm resolution)
let grid = Grid::new(256, 256, 128, 0.0001, 0.0001, 0.0001)?;
```

### Medium Models
Support for homogeneous, heterogeneous, and specialized tissue models:

```rust
use kwavers::medium::{HomogeneousMedium, HeterogeneousMedium};

// Homogeneous tissue (water-like)
let water = HomogeneousMedium::new(1000.0, 1500.0, 0.5, 1.0, &grid);

// Heterogeneous tissue with spatial variation
let liver = HeterogeneousMedium::create_from_ct_scan(ct_data, &grid)?;
```

## Wave Propagation APIs

### 1. Finite Difference Time Domain (FDTD)

```rust
use kwavers::solver::fdtd::{FdtdSolver, FdtdConfig};

let config = FdtdConfig {
    cfl_safety_factor: 0.5,
    boundary_condition: BoundaryCondition::PML { thickness: 10 },
    ..Default::default()
};

let mut solver = FdtdSolver::new(&grid, &medium, config)?;

// Add excitation source
let source = PointSource::new(grid.nx/2, grid.ny/2, grid.nz/2, 1e5, 5e6, 2);
solver.add_source(source);

// Run simulation
let result = solver.simulate(1e-4)?; // 100 μs simulation

// Access pressure field
let pressure_field = result.pressure_field();
```

### 2. Pseudospectral Time Domain (PSTD)

```rust
use kwavers::solver::pstd::{PstdSolver, PstdConfig};

let config = PstdConfig {
    kspace_epsilon: 1e-6,
    absorption_correction: true,
    dispersion_compensation: true,
};

let mut solver = PstdSolver::new(&grid, &medium, config)?;

// PSTD provides higher accuracy for smooth media
let result = solver.simulate_with_kspace(1e-4)?;
```

### 3. Hybrid Angular Spectrum (HAS)

```rust
use kwavers::solver::angular_spectrum::{HybridAngularSpectrumSolver, HASConfig, CorrectionType};

let config = HASConfig {
    max_angle: std::f64::consts::PI / 3.0, // 60° max angle
    correction_threshold: 0.1, // 10% inhomogeneity threshold
    correction_layers: 3,
    separable: true,
};

let mut solver = HybridAngularSpectrumSolver::new(config, &grid, &medium)?;

// Add local corrections for inhomogeneities
solver.add_correction((100, 100, 50), CorrectionType::BornApproximation, &grid)?;

let result = solver.propagate(&initial_field, 0.0, 0.01)?; // 10mm propagation
```

## Advanced Physics APIs

### Shear Wave Elastography (SWE)

```rust
use kwavers::physics::imaging::elastography::{ShearWaveElastography, NonlinearSWEConfig};
use kwavers::physics::imaging::elastography::HyperelasticModel;

// Configure nonlinear SWE
let config = NonlinearSWEConfig {
    nonlinearity_parameter: 0.3,
    harmonic_detection: true,
    material_model: HyperelasticModel::mooney_rivlin_liver(),
    ..Default::default()
};

let mut swe = ShearWaveElastography::new(&grid, &liver_tissue, config)?;

// Generate shear wave via ARFI
let push_location = [0.01, 0.01, 0.01]; // 10mm position
let displacement_history = swe.generate_shear_wave(push_location)?;

// Estimate stiffness with nonlinear effects
let stiffness_map = swe.estimate_nonlinear_stiffness(&displacement_history)?;

// Analyze harmonic content
let harmonics = swe.extract_harmonics(&displacement_history)?;
```

### Contrast-Enhanced Ultrasound (CEUS)

```rust
use kwavers::physics::imaging::ceus::{ContrastEnhancedUltrasound, MicrobubbleModel};

// Configure microbubble model
let bubble_model = MicrobubbleModel::definity(); // Or custom parameters
let ceus = ContrastEnhancedUltrasound::new(&grid, &liver_tissue, bubble_model)?;

// Simulate bolus injection
let injection_profile = ceus.simulate_bolus_injection(1e7)?; // 10^7 bubbles

// Generate contrast signal over time
let time_points = (0..100).map(|i| i as f64 * 0.1).collect::<Vec<_>>(); // 0-10 seconds
let contrast_signal = ceus.simulate_contrast_sequence(&injection_profile, &time_points)?;

// Perform perfusion analysis
let perfusion_model = PerfusionModel::gamma_variate_model();
let perfusion_map = ceus.estimate_perfusion(&contrast_signal, &perfusion_model)?;
```

### Transcranial Focused Ultrasound (tFUS)

```rust
use kwavers::physics::transcranial::{TreatmentPlanner, BBBOpening, SafetyMonitor};

// Plan transcranial treatment
let patient_ct = load_patient_ct_scan()?;
let planner = TreatmentPlanner::new(&brain_grid, &patient_ct)?;

let targets = vec![
    TargetVolume {
        center: [0.0, 0.02, 0.06], // 20mm lateral, 60mm depth
        dimensions: [0.004, 0.004, 0.004], // 4mm x 4mm x 4mm
        shape: TargetShape::Ellipsoidal,
        priority: 9,
        max_temperature: 43.0,
        required_intensity: 200.0, // W/cm²
    }
];

let spec = TransducerSpecification {
    num_elements: 1024,
    frequency: 650e3, // 650 kHz for brain
    ..Default::default()
};

let treatment_plan = planner.generate_plan("patient_001", &targets, &spec)?;

// Simulate BBB opening
let bbb_config = BBBParameters {
    frequency: 1.0e6,
    target_mi: 0.3, // Low MI for BBB opening
    duration: 120.0, // 2 minutes
    ..Default::default()
};

let mut bbb_opening = BBBOpening::new(treatment_plan.acoustic_field.clone(),
                                    microbubble_concentration, bbb_config)?;
bbb_opening.simulate_opening()?;

let permeability_enhancement = bbb_opening.permeability();
```

## Physics-Informed Neural Networks (PINNs)

### 1D Wave PINN

```rust
use kwavers::ml::pinn::{BurnPINN1DWave, BurnPINNConfig, BurnLossWeights, BurnTrainingMetrics};

let config = BurnPINNConfig {
    hidden_layers: vec![50, 50, 50],
    learning_rate: 1e-3,
    weight_decay: 1e-4,
    ..Default::default()
};

let loss_weights = BurnLossWeights {
    pde_weight: 1.0,
    boundary_weight: 10.0,
    initial_weight: 10.0,
    data_weight: 100.0,
};

let mut pinn = BurnPINN1DWave::new(config, loss_weights)?;

// Add training data
let training_data = generate_wave_training_data()?;
pinn.add_collocation_points(&training_data.collocation_points)?;
pinn.add_boundary_conditions(&training_data.boundary_conditions)?;
pinn.add_initial_conditions(&training_data.initial_conditions)?;

// Train the PINN
let metrics = pinn.train(10000, Some(validation_data))?;

// Make predictions
let test_positions = Array1::linspace(0.0, 1.0, 100);
let predictions = pinn.predict(&test_positions.into_shape((100, 1))?)?;
```

### 2D Heterogeneous Wave PINN

```rust
use kwavers::ml::pinn::BurnPINN2DWave;

let config = BurnPINNConfig {
    hidden_layers: vec![100, 100, 100, 100],
    learning_rate: 5e-4,
    ..Default::default()
};

let mut pinn_2d = BurnPINN2DWave::new(config, loss_weights)?;

// Add heterogeneous medium information
pinn_2d.set_medium_properties(&speed_of_sound_map, &attenuation_map)?;

// Train on 2D wave problems
let metrics_2d = pinn_2d.train(20000, Some(validation_2d))?;
```

### 3D Wave PINN with Interface Conditions

```rust
use kwavers::ml::pinn::burn_wave_equation_3d::{BurnPINN3DWave, BurnPINN3DConfig, InterfaceCondition3D};

let config_3d = BurnPINN3DConfig {
    hidden_layers: vec![150, 150, 150, 150, 150],
    learning_rate: 1e-4,
    use_gpu: true,
    ..Default::default()
};

let mut pinn_3d = BurnPINN3DWave::new(config_3d, loss_weights)?;

// Add interface conditions for multi-region domains
let interface_conditions = vec![
    InterfaceCondition3D {
        interface_position: 0.5, // z = 0.5
        normal_vector: [0.0, 0.0, 1.0],
        continuity_type: ContinuityType::PressureAndNormalVelocity,
    }
];

pinn_3d.add_interface_conditions(&interface_conditions)?;
```

## Uncertainty Quantification

### Bayesian Neural Networks

```rust
use kwavers::uncertainty::{UncertaintyQuantifier, UncertaintyConfig, UncertaintyMethod};
use kwavers::uncertainty::bayesian_networks::BayesianPINN;

let uncertainty_config = UncertaintyConfig {
    method: UncertaintyMethod::MonteCarloDropout,
    num_samples: 50,
    confidence_level: 0.95,
    dropout_rate: 0.1,
    ..Default::default()
};

let uncertainty_analyzer = UncertaintyQuantifier::new(uncertainty_config)?;

// Quantify uncertainty in PINN predictions
let uncertainty_result = uncertainty_analyzer.quantify_pinn_uncertainty(
    &pinn,
    &test_inputs,
    Some(&ground_truth)
)?;

// Check prediction confidence
if uncertainty_analyzer.is_confident(&uncertainty_result, 0.8) {
    println!("High confidence prediction: {:.3} ± {:.3}",
             uncertainty_result.mean_prediction.mean().unwrap(),
             uncertainty_result.uncertainty.std());
}
```

### Conformal Prediction

```rust
use kwavers::uncertainty::conformal_prediction::{ConformalPredictor, ConformalConfig};

let conformal_config = ConformalConfig {
    confidence_level: 0.9,
    calibration_size: 1000,
};

let mut conformal_predictor = ConformalPredictor::new(conformal_config)?;

// Calibrate on training data
conformal_predictor.calibrate(&training_predictions, &training_targets)?;

// Generate prediction intervals
let prediction_intervals = conformal_predictor.predict_intervals(&test_predictions)?;

// Validate coverage
let validation_metrics = conformal_predictor.validate_performance(
    &test_predictions,
    &test_targets
)?;
```

## GPU Acceleration

### Multi-GPU Context Management

```rust
use kwavers::gpu::multi_gpu::MultiGpuContext;
use kwavers::gpu::memory::{UnifiedMemoryManager, MemoryPoolType};

// Initialize multi-GPU context
let gpu_context = MultiGpuContext::new()?;
gpu_context.detect_available_gpus()?;
gpu_context.initialize_peer_access()?;

// Configure unified memory
let memory_manager = UnifiedMemoryManager::new();
let handle = memory_manager.allocate(0, MemoryPoolType::FFT, 1024 * 1024)?; // 1MB FFT workspace

// Transfer data between GPUs
memory_manager.transfer(&handle_gpu0, &handle_gpu1, data_size)?;
```

### Real-Time Imaging Pipeline

```rust
use kwavers::gpu::pipeline::{RealtimeImagingPipeline, RealtimePipelineConfig};

let pipeline_config = RealtimePipelineConfig {
    target_fps: 30.0,
    max_latency_ms: 33.0, // 1 frame at 30fps
    buffer_size: 10,
    gpu_accelerated: true,
    adaptive_processing: true,
    streaming_mode: true,
};

let mut pipeline = RealtimeImagingPipeline::new(pipeline_config)?;

// Start real-time processing
pipeline.start()?;

// Submit RF data frames
let rf_frame = load_rf_data_from_scanner()?;
pipeline.submit_rf_data(rf_frame)?;

// Retrieve processed images
while let Some(processed_image) = pipeline.get_processed_frame() {
    display_image(&processed_image);
    update_clinical_interface(&processed_image);
}

// Monitor performance
let stats = pipeline.metrics();
println!("Average processing time: {:.2}ms", stats.average_latency.as_millis());
```

## Clinical Validation Framework

### Comprehensive Validation Suite

```rust
use kwavers::validation::clinical_validation::ClinicalValidationFramework;

let validator = ClinicalValidationFramework::new();

// Run full clinical validation
let validation_report = validator.run_full_validation()?;

println!("Validation Results:");
println!("  Tests Passed: {}/{}", validation_report.passed_tests, validation_report.total_tests);
println!("  Overall Pass Rate: {:.1}%", validation_report.overall_pass_rate * 100.0);
println!("  Safety Critical: {}", if validation_report.safety_critical_passed { "PASS" } else { "FAIL" });

for recommendation in &validation_report.recommendations {
    println!("  Recommendation: {}", recommendation);
}
```

### Custom Validation Test

```rust
use kwavers::validation::clinical_validation::{ValidationTestCase, TestType};

// Define custom validation test
let custom_test = ValidationTestCase {
    name: "Custom Tissue Characterization".to_string(),
    test_type: TestType::ElastographyAccuracy,
    description: "Validate stiffness estimation in custom tissue phantom".to_string(),
    clinical_standard: "Quantitative ultrasound elastography".to_string(),
    acceptance_criteria: "MAE < 0.3 kPa, R² > 0.95".to_string(),
};

// Run validation
let result = validator.run_test_case(&custom_test)?;
println!("Custom test result: {}", if result.passed { "PASSED" } else { "FAILED" });
```

## Performance Benchmarking

### Automated Benchmark Suite

```rust
use kwavers::benches::performance_benchmark::PerformanceBenchmarkSuite;

let mut benchmark_suite = PerformanceBenchmarkSuite::new();
benchmark_suite.run_full_suite()?;

// Results are automatically printed and can be exported
let results = benchmark_suite.get_results();
export_benchmark_results(&results, "benchmark_results.json")?;
```

### Custom Performance Test

```rust
use criterion::{criterion_group, criterion_main, Criterion};

// Define custom benchmark
fn custom_simulation_benchmark(c: &mut Criterion) {
    c.bench_function("my_custom_simulation", |b| {
        b.iter(|| {
            // Your simulation code here
            let result = run_custom_ultrasound_simulation();
            criterion::black_box(result);
        });
    });
}

criterion_group!(benches, custom_simulation_benchmark);
criterion_main!(benches);
```

## Complete Clinical Workflow Example

Here's a comprehensive example integrating multiple APIs:

```rust
use kwavers::error::KwaversResult;

fn comprehensive_liver_assessment() -> KwaversResult<()> {
    // 1. Patient setup and grid initialization
    let patient_id = "LIVER_PATIENT_001";
    let liver_grid = Grid::new(200, 150, 100, 0.0005, 0.0005, 0.0005)?;

    // 2. Load patient-specific tissue properties
    let ct_data = load_patient_ct_scan(patient_id)?;
    let liver_tissue = HeterogeneousMedium::from_ct_with_tissue_properties(
        &ct_data, &liver_grid, TissueType::Liver
    )?;

    // 3. B-mode imaging for anatomical reference
    let b_mode_result = perform_b_mode_scan(&liver_grid, &liver_tissue)?;

    // 4. Shear wave elastography for fibrosis assessment
    let swe_config = NonlinearSWEConfig {
        nonlinearity_parameter: 0.3,
        material_model: HyperelasticModel::mooney_rivlin_liver(),
        harmonic_detection: true,
        ..Default::default()
    };

    let mut swe_system = ShearWaveElastography::new(&liver_grid, &liver_tissue, swe_config)?;
    let swe_result = perform_swe_examination(&mut swe_system)?;

    // 5. Contrast-enhanced ultrasound for perfusion
    let ceus_system = ContrastEnhancedUltrasound::new(&liver_grid, &liver_tissue)?;
    let ceus_result = perform_ceus_examination(&ceus_system)?;

    // 6. Uncertainty quantification
    let uncertainty_config = UncertaintyConfig {
        method: UncertaintyMethod::Hybrid,
        num_samples: 30,
        confidence_level: 0.95,
    };
    let uncertainty_analyzer = UncertaintyQuantifier::new(uncertainty_config)?;
    let uncertainty_assessment = assess_diagnostic_uncertainty(
        &swe_result, &ceus_result, &uncertainty_analyzer
    )?;

    // 7. Clinical decision support
    let diagnosis = generate_clinical_diagnosis(
        &swe_result, &ceus_result, &uncertainty_assessment
    )?;

    // 8. Treatment planning (if indicated)
    let treatment_plan = if diagnosis.requires_treatment {
        plan_liver_treatment(&diagnosis, &liver_grid, &liver_tissue)?
    } else {
        TreatmentPlan::monitoring_only()
    };

    // 9. Safety validation
    let safety_assessment = validate_treatment_safety(&treatment_plan)?;

    // 10. Generate comprehensive report
    let report = generate_clinical_report(
        patient_id,
        &b_mode_result,
        &swe_result,
        &ceus_result,
        &diagnosis,
        &treatment_plan,
        &safety_assessment,
        &uncertainty_assessment
    )?;

    println!("Comprehensive liver assessment completed for {}", patient_id);
    println!("Fibrosis Stage: {}", diagnosis.fibrosis_stage);
    println!("Confidence Level: {:.1}%", diagnosis.confidence_level * 100.0);

    Ok(())
}
```

## API Organization

The kwavers API is organized into the following main modules:

- **`grid`**: Computational grid management
- **`medium`**: Tissue and material property models
- **`solver`**: Wave propagation algorithms (FDTD, PSTD, HAS, etc.)
- **`physics`**: Specialized physics modules (elastography, CEUS, tFUS, etc.)
- **`ml::pinn`**: Physics-informed neural networks
- **`uncertainty`**: Uncertainty quantification methods
- **`gpu`**: GPU acceleration and memory management
- **`validation`**: Clinical validation and quality assurance
- **`source`**: Excitation source definitions
- **`sensor`**: Transducer and sensor models

## Error Handling

All APIs return `KwaversResult<T>` which is an alias for `Result<T, KwaversError>`. Common error types include:

- `InvalidInput`: Invalid parameters or data
- `NumericalError`: Convergence or stability issues
- `GpuError`: GPU-related failures
- `IoError`: File I/O issues
- `ValidationError`: Clinical validation failures

## Performance Optimization

### GPU Acceleration
Enable GPU features with `--features gpu` for significant performance improvements.

### Memory Management
Use the unified memory manager for optimal memory utilization across GPUs.

### Parallel Processing
The platform automatically parallelizes computations where beneficial.

### Benchmarking
Regular benchmarking helps identify performance bottlenecks and optimization opportunities.

This API guide provides a comprehensive overview of the kwavers platform capabilities. For detailed documentation of specific modules, refer to the individual module documentation and examples.





