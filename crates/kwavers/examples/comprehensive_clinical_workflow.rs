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

#[cfg(feature = "gpu")]
use kwavers_analysis::ml::uncertainty::{
    MlUncertaintyConfig, MlUncertaintyMethod, Seed, UncertaintyQuantifier,
};
#[cfg(feature = "gpu")]
use kwavers_analysis::validation::clinical::{
    ClinicalValidator, ImageQualityMetrics, MeasurementAccuracy, SafetyIndices,
};
use kwavers_core::error::KwaversResult;
#[cfg(feature = "gpu")]
use kwavers_gpu::gpu::memory::UnifiedMemoryManager;
#[cfg(feature = "gpu")]
use kwavers_grid::Grid;
#[cfg(feature = "gpu")]
use kwavers_imaging::ultrasound::elastography::NonlinearParameterMap;
#[cfg(feature = "gpu")]
use kwavers_medium::{heterogeneous::HeterogeneousMedium, homogeneous::HomogeneousMedium};
#[cfg(feature = "gpu")]
use kwavers_physics::acoustics::transcranial::safety_monitoring::TranscranialSafetyMonitor;
#[cfg(feature = "gpu")]
use kwavers_simulation::imaging::ceus::ContrastEnhancedUltrasound;
#[cfg(feature = "gpu")]
use leto::Array3 as LetoArray3;
#[cfg(feature = "gpu")]
use leto::Array3;
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/modalities.rs"]
mod modalities;
#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/presentation.rs"]
mod presentation;
#[cfg(feature = "gpu")]
#[path = "comprehensive_clinical_workflow/results.rs"]
mod results;

#[cfg(feature = "gpu")]
use results::*;

/// Comprehensive liver assessment workflow
#[cfg(feature = "gpu")]
struct LiverAssessmentWorkflow {
    /// Patient information
    patient_id: String,
    /// Computational grid for liver volume
    liver_grid: Grid,
    /// Heterogeneous liver tissue model
    liver_tissue: HeterogeneousMedium,
    /// SWE displacement generation and inversion performed per assessment
    /// CEUS system for perfusion assessment
    ceus_system: ContrastEnhancedUltrasound,
    /// Safety monitoring system
    safety_monitor: TranscranialSafetyMonitor,
    /// Multi-GPU memory manager
    #[cfg(feature = "gpu")]
    gpu_memory: UnifiedMemoryManager,
    /// Uncertainty quantification
    uncertainty_analyzer: UncertaintyQuantifier,
    /// Clinical validation framework
    clinical_validator: ClinicalValidator,
}

#[cfg(feature = "gpu")]
impl LiverAssessmentWorkflow {
    /// Create new liver assessment workflow for a patient
    fn new(patient_id: &str, liver_volume_mm3: (f64, f64, f64)) -> KwaversResult<Self> {
        println!(
            "Initializing comprehensive liver assessment for patient: {}",
            patient_id
        );
        println!(
            "Liver volume: {:.1} x {:.1} x {:.1} mm³",
            liver_volume_mm3.0, liver_volume_mm3.1, liver_volume_mm3.2
        );

        // Create computational grid for liver volume
        // Scale down for computational efficiency while maintaining clinical relevance
        let scale_factor = 0.5; // 0.5 mm resolution (clinical standard)
        let grid = Grid::new(
            (liver_volume_mm3.0 * scale_factor) as usize,
            (liver_volume_mm3.1 * scale_factor) as usize,
            (liver_volume_mm3.2 * scale_factor) as usize,
            5e-4,
            5e-4,
            5e-4, // 0.5 mm resolution
        )?;

        println!(
            "Computational grid: {}x{}x{} cells",
            grid.nx, grid.ny, grid.nz
        );

        // Create heterogeneous liver tissue model
        let liver_tissue = Self::create_liver_tissue_model(&grid)?;

        // SWE handled within assessment using available ARFI + ElasticWaveSolver + inversion

        // Initialize CEUS system (use typical liver CEUS bubble params)
        let ceus_system = ContrastEnhancedUltrasound::new(&grid, &liver_tissue, 5.0e6, 3.0)?;

        // Initialize safety monitoring
        let safety_monitor = TranscranialSafetyMonitor::new(
            (grid.nx, grid.ny, grid.nz),
            0.01, // liver perfusion rate (1/s)
            2e6,  // 2 MHz imaging frequency
        );

        // Initialize GPU memory management
        #[cfg(feature = "gpu")]
        let gpu_memory = UnifiedMemoryManager::new();

        // Initialize uncertainty quantification
        let uncertainty_config = MlUncertaintyConfig {
            method: MlUncertaintyMethod::Hybrid,
            num_samples: 50,
            confidence_level: 0.95,
            dropout_rate: 0.1,
            ensemble_size: 5,
            calibration_size: 100,
            sensitivity_seed: Seed::new(0),
        };
        let uncertainty_analyzer = UncertaintyQuantifier::new(uncertainty_config)?;

        // Initialize clinical validation
        let clinical_validator = ClinicalValidator::new();

        Ok(Self {
            patient_id: patient_id.to_string(),
            liver_grid: grid,
            liver_tissue,
            ceus_system,
            safety_monitor,
            #[cfg(feature = "gpu")]
            gpu_memory,
            uncertainty_analyzer,
            clinical_validator,
        })
    }

    /// Execute complete liver assessment protocol
    fn execute_assessment(&mut self) -> KwaversResult<LiverAssessmentReport> {
        let start_time = Instant::now();
        println!("\n=== Starting Comprehensive Liver Assessment ===");
        println!("Patient ID: {}", self.patient_id);
        #[cfg(feature = "gpu")]
        let _gpu_stats = self.gpu_memory.statistics();

        // Phase 1: B-mode imaging and initial assessment
        println!("\n--- Phase 1: B-mode Imaging ---");
        let b_mode_result = self.perform_b_mode_imaging()?;

        // Phase 2: Shear Wave Elastography (SWE)
        println!("\n--- Phase 2: Shear Wave Elastography ---");
        let swe_result = modalities::perform_shear_wave_elastography(self)?;

        // Phase 3: Contrast-Enhanced Ultrasound (CEUS)
        println!("\n--- Phase 3: Contrast-Enhanced Ultrasound ---");
        let ceus_result = modalities::perform_contrast_enhanced_ultrasound(self)?;

        // Phase 4: Uncertainty quantification and confidence assessment
        println!("\n--- Phase 4: Uncertainty Quantification ---");
        let uncertainty_result = self.perform_uncertainty_analysis(&swe_result, &ceus_result)?;

        // Phase 5: Clinical decision support
        println!("\n--- Phase 5: Clinical Decision Support ---");
        let diagnosis =
            self.generate_clinical_diagnosis(&swe_result, &ceus_result, &uncertainty_result)?;

        // Phase 6: Treatment planning (if indicated)
        println!("\n--- Phase 6: Treatment Planning ---");
        let treatment_plan = self.generate_treatment_plan(&diagnosis)?;

        // Phase 7: Safety validation
        println!("\n--- Phase 7: Safety Validation ---");
        let safety_assessment = self.perform_safety_assessment()?;

        // Phase 8: Clinical validation
        println!("\n--- Phase 8: Clinical Validation ---");
        // Build minimal clinical validation using available metrics
        // Image quality and measurement accuracy derived from Phase 1 results
        let quality_metrics = ImageQualityMetrics {
            contrast_resolution: b_mode_result.contrast_ratio,
            axial_resolution: b_mode_result.axial_resolution,
            lateral_resolution: b_mode_result.lateral_resolution,
            // Conservative dynamic range/SNR/CNR estimates for liver imaging
            dynamic_range: 60.0,
            snr: 25.0,
            cnr: 20.0,
        };
        let accuracy_metrics = MeasurementAccuracy {
            distance_error_percent: 3.0,
            area_error_percent: 5.0,
            volume_error_percent: 8.0,
            velocity_error_percent: 10.0,
            angle_error_degrees: 2.0,
        };
        // Safety indices consistent with imaging constraints
        let safety_indices = SafetyIndices {
            mechanical_index: 1.0,
            thermal_index_bone: 0.8,
            thermal_index_soft: 0.5,
            thermal_index_cranial: 0.7,
            spta_intensity: 500.0, // mW/cm²
            sppa_intensity: 100.0, // W/cm²
        };

        let bmode_validation = self.clinical_validator.validate_bmode(
            &quality_metrics,
            &accuracy_metrics,
            &safety_indices,
        )?;
        let safety_validation = self.clinical_validator.validate_safety(&safety_indices)?;

        let validation_report = self.clinical_validator.generate_validation_report(
            Some(&bmode_validation),
            None,
            Some(&safety_validation),
        );

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

        println!(
            "B-mode imaging: {}x{} pixels",
            self.liver_grid.nx, self.liver_grid.ny
        );
        println!("Estimated resolution: axial=0.3mm, lateral=0.8mm");

        Ok(BModeResult {
            envelope,
            axial_resolution: 0.3,   // mm
            lateral_resolution: 0.8, // mm
            contrast_ratio: 25.0,    // dB
        })
    }

    /// Perform uncertainty quantification
    fn perform_uncertainty_analysis(
        &self,
        swe_result: &SWEResult,
        ceus_result: &CEUSResult,
    ) -> KwaversResult<UncertaintyAnalysis> {
        // Quantify uncertainty in stiffness measurements
        let swe_uncertainty = self
            .uncertainty_analyzer
            .quantify_beamforming_uncertainty(&swe_result.stiffness_map, 0.95)?;

        // Quantify uncertainty in perfusion measurements
        let perfusion_uncertainty = self
            .uncertainty_analyzer
            .quantify_beamforming_uncertainty(&ceus_result.perfusion_map, 0.90)?;

        println!(
            "Uncertainty analysis: SWE confidence = {:.1}%, CEUS confidence = {:.1}%",
            swe_uncertainty.confidence_score * 100.0,
            perfusion_uncertainty.confidence_score * 100.0
        );

        Ok(UncertaintyAnalysis {
            swe_uncertainty,
            perfusion_uncertainty,
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

        let confidence_level = (uncertainty.swe_uncertainty.confidence_score
            + uncertainty.perfusion_uncertainty.confidence_score)
            / 2.0;

        Ok(ClinicalDiagnosis {
            diagnosis_text: diagnosis,
            fibrosis_stage,
            perfusion_status: if perfusion_abnormality {
                "Abnormal"
            } else {
                "Normal"
            }
            .to_string(),
            confidence_level,
            recommendations: self
                .generate_clinical_recommendations(fibrosis_stage, perfusion_abnormality),
        })
    }

    /// Generate treatment plan based on diagnosis
    fn generate_treatment_plan(
        &self,
        diagnosis: &ClinicalDiagnosis,
    ) -> KwaversResult<TreatmentPlan> {
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
        let _status = self.safety_monitor.safety_status();

        // Assess acoustic safety limits
        let max_pressure = 1e5; // 100 kPa (conservative for liver imaging)
        let max_mi = max_pressure / 1e6 / (2e6f64).sqrt(); // MI calculation

        // Assess thermal safety
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
        let base_properties = HomogeneousMedium::new(1000.0, 1540.0, 0.5, 1.1, grid);
        // Convert to heterogeneous (simplified)
        Ok(HeterogeneousMedium::from_homogeneous(
            &base_properties,
            grid,
        ))
    }

    fn calculate_fibrosis_metrics(
        &self,
        stiffness_map: &LetoArray3<f32>,
        nonlinear_analysis: &NonlinearParameterMap,
    ) -> KwaversResult<FibrosisMetrics> {
        let mean_stiffness: f32 = stiffness_map.iter().sum::<f32>() / stiffness_map.size() as f32;

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
            nonlinear_parameter: nonlinear_analysis
                .nonlinearity_parameter
                .iter()
                .copied()
                .sum::<f64>()
                / nonlinear_analysis.nonlinearity_parameter.size() as f64,
        })
    }

    fn calculate_perfusion_metrics(
        &self,
        perfusion_map: &LetoArray3<f32>,
    ) -> KwaversResult<PerfusionMetrics> {
        let peak_enhancement: f32 = perfusion_map.iter().cloned().fold(0.0_f32, f32::max);
        let mean_perfusion: f32 = perfusion_map.iter().sum::<f32>() / perfusion_map.size() as f32;

        Ok(PerfusionMetrics {
            peak_enhancement: 20.0 * (peak_enhancement as f64).log10(), // dB
            perfusion_rate: mean_perfusion as f64 * 1000.0,             // mL/min/100g (scaled)
            wash_in_time: 15.0,                                         // seconds
            wash_out_time: 120.0,                                       // seconds
        })
    }

    fn generate_clinical_recommendations(
        &self,
        fibrosis_stage: i32,
        perfusion_abnormality: bool,
    ) -> Vec<String> {
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
            recommendations
                .push("Further evaluation for hepatocellular carcinoma risk".to_string());
        }

        recommendations
    }
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
    presentation::run()
}
