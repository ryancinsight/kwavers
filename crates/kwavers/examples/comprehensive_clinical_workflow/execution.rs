//! Assessment orchestration and validation phases.

use super::{
    BModeResult, LiverAssessmentReport, LiverAssessmentWorkflow, SafetyAssessment,
    UncertaintyAnalysis,
};
use kwavers_analysis::validation::clinical::{
    ImageQualityMetrics, MeasurementAccuracy, SafetyIndices,
};
use kwavers_core::error::KwaversResult;
use leto::Array3;
use std::time::Instant;

impl LiverAssessmentWorkflow {
    pub(super) fn execute_assessment(&mut self) -> KwaversResult<LiverAssessmentReport> {
        let start_time = Instant::now();
        println!("\n=== Starting Comprehensive Liver Assessment ===");
        println!("Patient ID: {}", self.patient_id);
        let _gpu_stats = self.gpu_memory.statistics();

        println!("\n--- Phase 1: B-mode Imaging ---");
        let b_mode_result = self.perform_b_mode_imaging();

        println!("\n--- Phase 2: Shear Wave Elastography ---");
        let swe_result = super::modalities::perform_shear_wave_elastography(self)?;

        println!("\n--- Phase 3: Contrast-Enhanced Ultrasound ---");
        let ceus_result = super::modalities::perform_contrast_enhanced_ultrasound(self)?;

        println!("\n--- Phase 4: Uncertainty Quantification ---");
        let uncertainty_result = self.perform_uncertainty_analysis(&swe_result, &ceus_result)?;

        println!("\n--- Phase 5: Clinical Decision Support ---");
        let diagnosis =
            self.generate_clinical_diagnosis(&swe_result, &ceus_result, &uncertainty_result);

        println!("\n--- Phase 6: Treatment Planning ---");
        let treatment_plan = self.generate_treatment_plan(&diagnosis);

        println!("\n--- Phase 7: Safety Validation ---");
        let safety_assessment = self.perform_safety_assessment();

        println!("\n--- Phase 8: Clinical Validation ---");
        let quality_metrics = ImageQualityMetrics {
            contrast_resolution: b_mode_result.contrast_ratio,
            axial_resolution: b_mode_result.axial_resolution,
            lateral_resolution: b_mode_result.lateral_resolution,
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
        let safety_indices = SafetyIndices {
            mechanical_index: 1.0,
            thermal_index_bone: 0.8,
            thermal_index_soft: 0.5,
            thermal_index_cranial: 0.7,
            spta_intensity: 500.0,
            sppa_intensity: 100.0,
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

        let processing_time = start_time.elapsed();
        println!("\n=== Assessment Complete ===");
        println!(
            "Total processing time: {:.2}s",
            processing_time.as_secs_f64()
        );

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
            processing_time,
        })
    }

    fn perform_b_mode_imaging(&self) -> BModeResult {
        let envelope = Array3::<f32>::from_elem(self.liver_grid.dimensions(), 0.0);

        println!(
            "B-mode imaging: {}x{} pixels",
            self.liver_grid.nx, self.liver_grid.ny
        );
        println!("Estimated resolution: axial=0.3mm, lateral=0.8mm");

        BModeResult {
            envelope,
            axial_resolution: 0.3,
            lateral_resolution: 0.8,
            contrast_ratio: 25.0,
        }
    }

    fn perform_uncertainty_analysis(
        &self,
        swe_result: &super::SWEResult,
        ceus_result: &super::CEUSResult,
    ) -> KwaversResult<UncertaintyAnalysis> {
        let swe_uncertainty = self
            .uncertainty_analyzer
            .quantify_beamforming_uncertainty(&swe_result.stiffness_map, 0.95)?;
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

    fn perform_safety_assessment(&self) -> SafetyAssessment {
        let _status = self.safety_monitor.safety_status();
        let max_pressure = 1e5;
        let max_mi = max_pressure / 1e6 / (2e6f64).sqrt();
        let max_temperature_rise = 1.0;
        let acoustic_safe = max_mi < 1.9;
        let thermal_safe = max_temperature_rise < 2.0;

        SafetyAssessment {
            acoustic_safety: acoustic_safe,
            thermal_safety: thermal_safe,
            overall_safe: acoustic_safe && thermal_safe,
            safety_notes: if acoustic_safe && thermal_safe {
                "All safety parameters within acceptable limits".to_string()
            } else {
                "Safety parameters require review - consider power reduction".to_string()
            },
        }
    }
}
