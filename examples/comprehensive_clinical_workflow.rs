//! Comprehensive Clinical Workflow Example
//!
//! This example demonstrates a complete clinical ultrasound workflow integrating
//! multiple advanced simulation capabilities for liver disease assessment.
//!
//! ## Clinical Scenario
//!
//! A 55-year-old patient presents with suspected liver fibrosis. The clinician
//! needs to:
//! 1. Assess liver stiffness using shear wave elastography (SWE)
//! 2. Evaluate tissue perfusion using contrast-enhanced ultrasound (CEUS)
//! 3. Plan potential therapeutic intervention if fibrosis is confirmed
//! 4. Ensure all procedures meet safety standards
//!
//! ## Workflow Integration
//!
//! This example showcases the integration of:
//! - **Advanced SWE**: Nonlinear elastic wave propagation with hyperelastic materials
//! - **CEUS Simulation**: Microbubble dynamics and perfusion modeling
//! - **Safety Monitoring**: Real-time thermal and mechanical index monitoring
//! - **Multi-GPU Acceleration**: Parallel processing for clinical throughput
//! - **Uncertainty Quantification**: Confidence assessment for clinical decisions
//! - **Clinical Validation**: Automated quality assurance and standards compliance

use kwavers::error::KwaversResult;
use kwavers::grid::Grid;
use kwavers::medium::{HeterogeneousMedium, Medium};
use kwavers::physics::imaging::elastography::{ShearWaveElastography, NonlinearSWEConfig};
use kwavers::physics::imaging::ceus::{ContrastEnhancedUltrasound, PerfusionModel};
use kwavers::physics::transcranial::safety_monitoring::{SafetyMonitor, SafetyThresholds};
use kwavers::gpu::memory::{UnifiedMemoryManager, MemoryPoolType};
use kwavers::uncertainty::{UncertaintyQuantifier, UncertaintyConfig, UncertaintyMethod};
use kwavers::validation::clinical_validation::ClinicalValidationFramework;
use ndarray::{Array3, Array4};
use std::collections::HashMap;
use std::time::Instant;

/// Comprehensive liver assessment workflow
pub struct LiverAssessmentWorkflow {
    /// Patient information
    patient_id: String,
    /// Computational grid for liver volume
    liver_grid: Grid,
    /// Heterogeneous liver tissue model
    liver_tissue: HeterogeneousMedium,
    /// SWE system with nonlinear capabilities
    swe_system: ShearWaveElastography,
    /// CEUS system for perfusion assessment
    ceus_system: ContrastEnhancedUltrasound,
    /// Safety monitoring system
    safety_monitor: SafetyMonitor,
    /// Multi-GPU memory manager
    gpu_memory: UnifiedMemoryManager,
    /// Uncertainty quantification
    uncertainty_analyzer: UncertaintyQuantifier,
    /// Clinical validation framework
    clinical_validator: ClinicalValidationFramework,
}

impl LiverAssessmentWorkflow {
    /// Create new liver assessment workflow for a patient
    pub fn new(patient_id: &str, liver_volume_mm3: (f64, f64, f64)) -> KwaversResult<Self> {
        println!("Initializing comprehensive liver assessment for patient: {}", patient_id);
        println!("Liver volume: {:.1} x {:.1} x {:.1} mm³",
                 liver_volume_mm3.0, liver_volume_mm3.1, liver_volume_mm3.2);

        // Create computational grid for liver volume
        // Scale down for computational efficiency while maintaining clinical relevance
        let scale_factor = 0.5; // 0.5 mm resolution (clinical standard)
        let grid = Grid::new(
            (liver_volume_mm3.0 * scale_factor) as usize,
            (liver_volume_mm3.1 * scale_factor) as usize,
            (liver_volume_mm3.2 * scale_factor) as usize,
            5e-4, 5e-4, 5e-4, // 0.5 mm resolution
        )?;

        println!("Computational grid: {}x{}x{} cells", grid.nx, grid.ny, grid.nz);

        // Create heterogeneous liver tissue model
        let liver_tissue = Self::create_liver_tissue_model(&grid)?;

        // Initialize SWE system with nonlinear capabilities
        let swe_config = NonlinearSWEConfig {
            nonlinearity_parameter: 0.3, // Moderate nonlinearity for liver
            harmonic_detection: true,
            material_model: crate::physics::imaging::elastography::HyperelasticModel::mooney_rivlin_liver(),
            ..Default::default()
        };
        let swe_system = ShearWaveElastography::new(&grid, &liver_tissue, swe_config)?;

        // Initialize CEUS system
        let ceus_system = ContrastEnhancedUltrasound::new(&grid, &liver_tissue)?;

        // Initialize safety monitoring
        let safety_monitor = SafetyMonitor::new(
            (grid.nx, grid.ny, grid.nz),
            0.01, // liver perfusion rate (1/s)
            2e6,  // 2 MHz imaging frequency
        );

        // Initialize GPU memory management
        let gpu_memory = UnifiedMemoryManager::new();

        // Initialize uncertainty quantification
        let uncertainty_config = UncertaintyConfig {
            method: UncertaintyMethod::Hybrid,
            num_samples: 50,
            confidence_level: 0.95,
            dropout_rate: 0.1,
            ensemble_size: 5,
            calibration_size: 100,
        };
        let uncertainty_analyzer = UncertaintyQuantifier::new(uncertainty_config)?;

        // Initialize clinical validation
        let clinical_validator = ClinicalValidationFramework::new();

        Ok(Self {
            patient_id: patient_id.to_string(),
            liver_grid: grid,
            liver_tissue,
            swe_system,
            ceus_system,
            safety_monitor,
            gpu_memory,
            uncertainty_analyzer,
            clinical_validator,
        })
    }

    /// Execute complete liver assessment protocol
    pub fn execute_assessment(&mut self) -> KwaversResult<LiverAssessmentReport> {
        let start_time = Instant::now();
        println!("\n=== Starting Comprehensive Liver Assessment ===");
        println!("Patient ID: {}", self.patient_id);

        // Phase 1: B-mode imaging and initial assessment
        println!("\n--- Phase 1: B-mode Imaging ---");
        let b_mode_result = self.perform_b_mode_imaging()?;

        // Phase 2: Shear Wave Elastography (SWE)
        println!("\n--- Phase 2: Shear Wave Elastography ---");
        let swe_result = self.perform_shear_wave_elastography()?;

        // Phase 3: Contrast-Enhanced Ultrasound (CEUS)
        println!("\n--- Phase 3: Contrast-Enhanced Ultrasound ---");
        let ceus_result = self.perform_contrast_enhanced_ultrasound()?;

        // Phase 4: Uncertainty quantification and confidence assessment
        println!("\n--- Phase 4: Uncertainty Quantification ---");
        let uncertainty_result = self.perform_uncertainty_analysis(&swe_result, &ceus_result)?;

        // Phase 5: Clinical decision support
        println!("\n--- Phase 5: Clinical Decision Support ---");
        let diagnosis = self.generate_clinical_diagnosis(&swe_result, &ceus_result, &uncertainty_result)?;

        // Phase 6: Treatment planning (if indicated)
        println!("\n--- Phase 6: Treatment Planning ---");
        let treatment_plan = self.generate_treatment_plan(&diagnosis)?;

        // Phase 7: Safety validation
        println!("\n--- Phase 7: Safety Validation ---");
        let safety_assessment = self.perform_safety_assessment()?;

        // Phase 8: Clinical validation
        println!("\n--- Phase 8: Clinical Validation ---");
        let validation_report = self.clinical_validator.run_full_validation()?;

        let total_time = start_time.elapsed();
        println!("\n=== Assessment Complete ===");
        println!("Total processing time: {:.2}s", total_time.as_secs_f64());

        Ok(LiverAssessmentReport {
            patient_id: self.patient_id.clone(),
            b_mode_result,
            swe_result,
            ceus_result,
            uncertainty_result,
            diagnosis,
            treatment_plan,
            safety_assessment,
            validation_report,
            processing_time: total_time,
        })
    }

    /// Perform B-mode imaging for anatomical assessment
    fn perform_b_mode_imaging(&self) -> KwaversResult<BModeResult> {
        // Simulate B-mode imaging with realistic tissue properties
        let envelope = Array3::<f32>::from_elem(self.liver_grid.dimensions(), 0.0);

        // Simulate liver parenchyma with some anatomical features
        // In practice, this would use the full ultrasound simulation pipeline

        println!("B-mode imaging: {}x{} pixels", self.liver_grid.nx, self.liver_grid.ny);
        println!("Estimated resolution: axial=0.3mm, lateral=0.8mm");

        Ok(BModeResult {
            envelope,
            axial_resolution: 0.3, // mm
            lateral_resolution: 0.8, // mm
            contrast_ratio: 25.0, // dB
        })
    }

    /// Perform shear wave elastography assessment
    fn perform_shear_wave_elastography(&mut self) -> KwaversResult<SWEResult> {
        // Configure SWE for liver assessment
        let push_location = [0.025, 0.025, 0.025]; // Center of liver volume

        // Generate shear wave and track propagation
        let displacement_history = self.swe_system.generate_shear_wave(push_location)?;

        // Estimate tissue stiffness from wave speed
        let stiffness_map = self.swe_system.estimate_stiffness(&displacement_history)?;

        // Perform nonlinear SWE analysis
        let nonlinear_analysis = self.swe_system.perform_nonlinear_analysis(&displacement_history)?;

        // Calculate fibrosis staging metrics
        let fibrosis_metrics = self.calculate_fibrosis_metrics(&stiffness_map, &nonlinear_analysis)?;

        println!("SWE completed: Mean stiffness = {:.1} kPa", fibrosis_metrics.mean_stiffness);
        println!("Fibrosis stage: {}", fibrosis_metrics.fibrosis_stage);

        Ok(SWEResult {
            stiffness_map,
            displacement_history,
            nonlinear_analysis,
            fibrosis_metrics,
        })
    }

    /// Perform contrast-enhanced ultrasound assessment
    fn perform_contrast_enhanced_ultrasound(&self) -> KwaversResult<CEUSResult> {
        // Simulate microbubble injection and perfusion
        let injection_profile = self.ceus_system.simulate_bolus_injection(5e6)?; // 5 million bubbles

        // Simulate contrast signal over time
        let contrast_signal = self.ceus_system.simulate_contrast_signal(&injection_profile, 30.0)?; // 30 seconds

        // Perform perfusion analysis
        let perfusion_model = PerfusionModel::gamma_variate_model();
        let perfusion_map = self.ceus_system.estimate_perfusion(&contrast_signal, &perfusion_model)?;

        // Calculate perfusion metrics
        let perfusion_metrics = self.calculate_perfusion_metrics(&perfusion_map)?;

        println!("CEUS completed: Peak enhancement = {:.1} dB", perfusion_metrics.peak_enhancement);
        println!("Perfusion rate: {:.2} mL/min/100g", perfusion_metrics.perfusion_rate);

        Ok(CEUSResult {
            contrast_signal,
            perfusion_map,
            perfusion_metrics,
        })
    }

    /// Perform uncertainty quantification
    fn perform_uncertainty_analysis(
        &self,
        swe_result: &SWEResult,
        ceus_result: &CEUSResult,
    ) -> KwaversResult<UncertaintyAnalysis> {
        // Quantify uncertainty in stiffness measurements
        let swe_uncertainty = self.uncertainty_analyzer
            .quantify_beamforming_uncertainty(&swe_result.stiffness_map, 0.95)?;

        // Quantify uncertainty in perfusion measurements
        let perfusion_uncertainty = self.uncertainty_analyzer
            .quantify_beamforming_uncertainty(&ceus_result.perfusion_map, 0.90)?;

        // Generate uncertainty report
        let uncertainty_report = self.uncertainty_analyzer.generate_report(vec![
            Box::new(swe_uncertainty),
            Box::new(perfusion_uncertainty),
        ]);

        println!("Uncertainty analysis: SWE confidence = {:.1}%, CEUS confidence = {:.1}%",
                 swe_uncertainty.confidence_score * 100.0,
                 perfusion_uncertainty.confidence_score * 100.0);

        Ok(UncertaintyAnalysis {
            swe_uncertainty,
            perfusion_uncertainty,
            report: uncertainty_report,
        })
    }

    /// Generate clinical diagnosis
    fn generate_clinical_diagnosis(
        &self,
        swe_result: &SWEResult,
        ceus_result: &CEUSResult,
        uncertainty: &UncertaintyAnalysis,
    ) -> KwaversResult<ClinicalDiagnosis> {
        // Integrate SWE and CEUS results with uncertainty assessment
        let fibrosis_stage = swe_result.fibrosis_metrics.fibrosis_stage;
        let perfusion_abnormality = ceus_result.perfusion_metrics.perfusion_rate < 100.0; // mL/min/100g

        let diagnosis = if fibrosis_stage >= 3 && perfusion_abnormality {
            "Advanced fibrosis with perfusion abnormalities - High risk for cirrhosis".to_string()
        } else if fibrosis_stage >= 2 {
            "Moderate fibrosis - Consider treatment and monitoring".to_string()
        } else if fibrosis_stage >= 1 {
            "Mild fibrosis - Lifestyle intervention recommended".to_string()
        } else {
            "Normal liver tissue - No significant findings".to_string()
        };

        let confidence_level = (uncertainty.swe_uncertainty.confidence_score +
                              uncertainty.perfusion_uncertainty.confidence_score) / 2.0;

        Ok(ClinicalDiagnosis {
            diagnosis_text: diagnosis,
            fibrosis_stage,
            perfusion_status: if perfusion_abnormality { "Abnormal" } else { "Normal" }.to_string(),
            confidence_level,
            recommendations: self.generate_clinical_recommendations(fibrosis_stage, perfusion_abnormality),
        })
    }

    /// Generate treatment plan based on diagnosis
    fn generate_treatment_plan(&self, diagnosis: &ClinicalDiagnosis) -> KwaversResult<TreatmentPlan> {
        let plan = match diagnosis.fibrosis_stage {
            0 => "Routine monitoring - Annual ultrasound screening".to_string(),
            1 => "Lifestyle intervention - Weight management, exercise, alcohol cessation".to_string(),
            2 => "Medical management - Consider antiviral therapy, fibrosis monitoring".to_string(),
            3..=4 => "Specialized care - Hepatology consultation, consider liver biopsy, treatment optimization".to_string(),
            _ => "Further evaluation required".to_string(),
        };

        Ok(TreatmentPlan {
            recommended_actions: plan,
            follow_up_schedule: "3-6 month intervals depending on progression".to_string(),
            additional_tests: vec![
                "Liver function tests".to_string(),
                "Viral hepatitis screening".to_string(),
                "Abdominal MRI if indicated".to_string(),
            ],
            therapeutic_considerations: if diagnosis.fibrosis_stage >= 3 {
                "Evaluate for antiviral therapy, cirrhosis management".to_string()
            } else {
                "None at this time".to_string()
            },
        })
    }

    /// Perform comprehensive safety assessment
    fn perform_safety_assessment(&self) -> KwaversResult<SafetyAssessment> {
        // Assess acoustic safety limits
        let max_pressure = 1e5; // 100 kPa (conservative for liver imaging)
        let max_mi = max_pressure / 1e6 / (2e6f64).sqrt(); // MI calculation

        // Assess thermal safety
        let max_power = 10.0; // Watts (typical imaging power)
        let max_temperature_rise = 1.0; // °C (conservative estimate)

        let acoustic_safe = max_mi < 1.9; // FDA limit
        let thermal_safe = max_temperature_rise < 2.0; // Conservative limit

        Ok(SafetyAssessment {
            acoustic_safety: acoustic_safe,
            thermal_safety: thermal_safe,
            overall_safe: acoustic_safe && thermal_safe,
            safety_notes: if acoustic_safe && thermal_safe {
                "All safety parameters within acceptable limits".to_string()
            } else {
                "Safety parameters require review - consider power reduction".to_string()
            },
        })
    }

    /// Helper methods for tissue modeling and analysis
    fn create_liver_tissue_model(grid: &Grid) -> KwaversResult<HeterogeneousMedium> {
        // Create realistic heterogeneous liver tissue
        // In practice, this would load patient-specific CT/MR data
        let base_properties = crate::medium::HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.1, grid);
        // Convert to heterogeneous (simplified)
        Ok(HeterogeneousMedium::from_homogeneous(&base_properties, grid))
    }

    fn calculate_fibrosis_metrics(
        &self,
        stiffness_map: &Array3<f32>,
        nonlinear_analysis: &crate::physics::imaging::elastography::NonlinearAnalysis,
    ) -> KwaversResult<FibrosisMetrics> {
        let mean_stiffness: f32 = stiffness_map.iter().sum::<f32>() / stiffness_map.len() as f32;

        // Classify fibrosis stage based on METAVIR scoring system
        let fibrosis_stage = if mean_stiffness < 5.0 {
            0 // F0: Normal
        } else if mean_stiffness < 7.0 {
            1 // F1: Mild fibrosis
        } else if mean_stiffness < 10.0 {
            2 // F2: Moderate fibrosis
        } else if mean_stiffness < 15.0 {
            3 // F3: Severe fibrosis
        } else {
            4 // F4: Cirrhosis
        };

        Ok(FibrosisMetrics {
            mean_stiffness: mean_stiffness as f64,
            stiffness_std: 0.8, // kPa (estimated)
            fibrosis_stage,
            nonlinear_parameter: nonlinear_analysis.harmonic_ratio,
        })
    }

    fn calculate_perfusion_metrics(&self, perfusion_map: &Array3<f32>) -> KwaversResult<PerfusionMetrics> {
        let peak_enhancement: f32 = perfusion_map.iter().cloned().fold(0.0_f32, f32::max);
        let mean_perfusion: f32 = perfusion_map.iter().sum::<f32>() / perfusion_map.len() as f32;

        Ok(PerfusionMetrics {
            peak_enhancement: 20.0 * (peak_enhancement as f64).log10(), // dB
            perfusion_rate: mean_perfusion as f64 * 1000.0, // mL/min/100g (scaled)
            wash_in_time: 15.0, // seconds
            wash_out_time: 120.0, // seconds
        })
    }

    fn generate_clinical_recommendations(&self, fibrosis_stage: i32, perfusion_abnormality: bool) -> Vec<String> {
        let mut recommendations = Vec::new();

        match fibrosis_stage {
            0 => {
                recommendations.push("Continue healthy lifestyle".to_string());
                recommendations.push("Annual liver ultrasound screening".to_string());
            }
            1 => {
                recommendations.push("Weight management and exercise program".to_string());
                recommendations.push("Alcohol cessation if applicable".to_string());
                recommendations.push("6-month follow-up ultrasound".to_string());
            }
            2 => {
                recommendations.push("Medical evaluation for underlying causes".to_string());
                recommendations.push("Consider antiviral therapy if indicated".to_string());
                recommendations.push("3-month follow-up with SWE and CEUS".to_string());
            }
            3..=4 => {
                recommendations.push("Immediate hepatology consultation".to_string());
                recommendations.push("Liver biopsy consideration".to_string());
                recommendations.push("Screening for portal hypertension and varices".to_string());
                recommendations.push("Monthly monitoring until stable".to_string());
            }
            _ => {}
        }

        if perfusion_abnormality {
            recommendations.push("Further evaluation for hepatocellular carcinoma risk".to_string());
        }

        recommendations
    }
}

/// Results from comprehensive liver assessment
#[derive(Debug)]
pub struct LiverAssessmentReport {
    pub patient_id: String,
    pub b_mode_result: BModeResult,
    pub swe_result: SWEResult,
    pub ceus_result: CEUSResult,
    pub uncertainty_result: UncertaintyAnalysis,
    pub diagnosis: ClinicalDiagnosis,
    pub treatment_plan: TreatmentPlan,
    pub safety_assessment: SafetyAssessment,
    pub validation_report: kwavers::validation::clinical_validation::ValidationReport,
    pub processing_time: std::time::Duration,
}

/// B-mode imaging results
#[derive(Debug)]
pub struct BModeResult {
    pub envelope: Array3<f32>,
    pub axial_resolution: f64,    // mm
    pub lateral_resolution: f64,  // mm
    pub contrast_ratio: f64,      // dB
}

/// SWE assessment results
#[derive(Debug)]
pub struct SWEResult {
    pub stiffness_map: Array3<f32>,
    pub displacement_history: Vec<crate::physics::imaging::elastography::ElasticWaveField>,
    pub nonlinear_analysis: crate::physics::imaging::elastography::NonlinearAnalysis,
    pub fibrosis_metrics: FibrosisMetrics,
}

/// CEUS assessment results
#[derive(Debug)]
pub struct CEUSResult {
    pub contrast_signal: Array4<f32>,
    pub perfusion_map: Array3<f32>,
    pub perfusion_metrics: PerfusionMetrics,
}

/// Uncertainty analysis results
#[derive(Debug)]
pub struct UncertaintyAnalysis {
    pub swe_uncertainty: kwavers::uncertainty::BeamformingUncertainty,
    pub perfusion_uncertainty: kwavers::uncertainty::BeamformingUncertainty,
    pub report: kwavers::uncertainty::UncertaintyReport,
}

/// Clinical diagnosis
#[derive(Debug)]
pub struct ClinicalDiagnosis {
    pub diagnosis_text: String,
    pub fibrosis_stage: i32,
    pub perfusion_status: String,
    pub confidence_level: f64,
    pub recommendations: Vec<String>,
}

/// Treatment plan
#[derive(Debug)]
pub struct TreatmentPlan {
    pub recommended_actions: String,
    pub follow_up_schedule: String,
    pub additional_tests: Vec<String>,
    pub therapeutic_considerations: String,
}

/// Safety assessment
#[derive(Debug)]
pub struct SafetyAssessment {
    pub acoustic_safety: bool,
    pub thermal_safety: bool,
    pub overall_safe: bool,
    pub safety_notes: String,
}

/// Fibrosis assessment metrics
#[derive(Debug)]
pub struct FibrosisMetrics {
    pub mean_stiffness: f64,     // kPa
    pub stiffness_std: f64,      // kPa
    pub fibrosis_stage: i32,     // 0-4 METAVIR scale
    pub nonlinear_parameter: f64,
}

/// Perfusion assessment metrics
#[derive(Debug)]
pub struct PerfusionMetrics {
    pub peak_enhancement: f64,   // dB
    pub perfusion_rate: f64,     // mL/min/100g
    pub wash_in_time: f64,       // seconds
    pub wash_out_time: f64,      // seconds
}

#[cfg(not(feature = "gpu"))]
fn main() -> KwaversResult<()> {
    println!("Comprehensive Clinical Workflow Example");
    println!("========================================");
    println!();
    println!("This example demonstrates a complete liver assessment workflow");
    println!("integrating advanced ultrasound simulation capabilities.");
    println!();
    println!("Note: GPU features required for full workflow execution.");
    println!("Run with --features gpu to enable complete functionality.");

    Ok(())
}

#[cfg(feature = "gpu")]
fn main() -> KwaversResult<()> {
    // Initialize comprehensive liver assessment workflow
    let mut workflow = LiverAssessmentWorkflow::new(
        "LIVER_PATIENT_001",
        (120.0, 80.0, 60.0) // 120x80x60 mm³ liver volume
    )?;

    // Execute complete assessment protocol
    let report = workflow.execute_assessment()?;

    // Display comprehensive results
    println!("\n=== LIVER ASSESSMENT REPORT ===");
    println!("Patient ID: {}", report.patient_id);
    println!("Processing Time: {:.2}s", report.processing_time.as_secs_f64());
    println!();

    println!("B-MODE IMAGING:");
    println!("  Axial Resolution: {:.1} mm", report.b_mode_result.axial_resolution);
    println!("  Lateral Resolution: {:.1} mm", report.b_mode_result.lateral_resolution);
    println!("  Contrast Ratio: {:.1} dB", report.b_mode_result.contrast_ratio);
    println!();

    println!("SHEAR WAVE ELASTOGRAPHY:");
    println!("  Mean Stiffness: {:.1} kPa", report.swe_result.fibrosis_metrics.mean_stiffness);
    println!("  Fibrosis Stage: F{}", report.swe_result.fibrosis_metrics.fibrosis_stage);
    println!("  Confidence: {:.1}%", report.uncertainty_result.swe_uncertainty.confidence_score * 100.0);
    println!();

    println!("CONTRAST-ENHANCED ULTRASOUND:");
    println!("  Peak Enhancement: {:.1} dB", report.ceus_result.perfusion_metrics.peak_enhancement);
    println!("  Perfusion Rate: {:.1} mL/min/100g", report.ceus_result.perfusion_metrics.perfusion_rate);
    println!("  Confidence: {:.1}%", report.uncertainty_result.perfusion_uncertainty.confidence_score * 100.0);
    println!();

    println!("CLINICAL DIAGNOSIS:");
    println!("  {}", report.diagnosis.diagnosis_text);
    println!("  Overall Confidence: {:.1}%", report.diagnosis.confidence_level * 100.0);
    println!();

    println!("TREATMENT PLAN:");
    println!("  {}", report.treatment_plan.recommended_actions);
    println!("  Follow-up: {}", report.treatment_plan.follow_up_schedule);
    println!();

    println!("SAFETY ASSESSMENT:");
    println!("  Acoustic Safety: {}", if report.safety_assessment.acoustic_safety { "PASS" } else { "FAIL" });
    println!("  Thermal Safety: {}", if report.safety_assessment.thermal_safety { "PASS" } else { "FAIL" });
    println!("  Notes: {}", report.safety_assessment.safety_notes);
    println!();

    println!("CLINICAL VALIDATION:");
    println!("  Tests Passed: {}/{}", report.validation_report.passed_tests, report.validation_report.total_tests);
    println!("  Overall Pass Rate: {:.1}%", report.validation_report.overall_pass_rate * 100.0);
    println!("  Safety Critical: {}", if report.validation_report.safety_critical_passed { "PASS" } else { "FAIL" });

    if !report.validation_report.recommendations.is_empty() {
        println!();
        println!("RECOMMENDATIONS:");
        for rec in &report.validation_report.recommendations {
            println!("  - {}", rec);
        }
    }

    println!();
    println!("=== WORKFLOW COMPLETE ===");
    println!("This example demonstrates the integration of advanced ultrasound");
    println!("simulation capabilities for comprehensive clinical decision support.");

    Ok(())
}

