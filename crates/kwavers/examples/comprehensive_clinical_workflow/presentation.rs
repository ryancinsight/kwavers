//! Console presentation for a completed liver assessment.

use super::LiverAssessmentWorkflow;
use kwavers_core::error::KwaversResult;

pub(super) fn run() -> KwaversResult<()> {
    let mut workflow = LiverAssessmentWorkflow::new("LIVER_PATIENT_001", (120.0, 80.0, 60.0))?;
    let report = workflow.execute_assessment()?;

    println!("\n=== LIVER ASSESSMENT REPORT ===");
    println!("Patient ID: {}", report.patient_id);
    println!(
        "Processing Time: {:.2}s",
        report.processing_time.as_secs_f64()
    );
    println!();

    println!("B-MODE IMAGING:");
    println!(
        "  Axial Resolution: {:.1} mm",
        report.b_mode_result.axial_resolution
    );
    println!(
        "  Lateral Resolution: {:.1} mm",
        report.b_mode_result.lateral_resolution
    );
    println!(
        "  Contrast Ratio: {:.1} dB",
        report.b_mode_result.contrast_ratio
    );
    println!("  Voxels: {}", report.b_mode_result.envelope.size());
    println!();

    println!("SHEAR WAVE ELASTOGRAPHY:");
    println!(
        "  Mean Stiffness: {:.1} kPa",
        report.swe_result.fibrosis_metrics.mean_stiffness
    );
    println!(
        "  Fibrosis Stage: F{}",
        report.swe_result.fibrosis_metrics.fibrosis_stage
    );
    println!(
        "  Stiffness Standard Deviation: {:.1} kPa",
        report.swe_result.fibrosis_metrics.stiffness_std
    );
    println!(
        "  Nonlinear Parameter: {:.3}",
        report.swe_result.fibrosis_metrics.nonlinear_parameter
    );
    println!(
        "  Displacement Frames: {}",
        report.swe_result.displacement_history.len()
    );
    println!(
        "  Nonlinear Voxels: {}",
        report
            .swe_result
            .nonlinear_analysis
            .nonlinearity_parameter
            .size()
    );
    println!(
        "  Confidence: {:.1}%",
        report.uncertainty_result.swe_uncertainty.confidence_score * 100.0
    );
    println!();

    println!("CONTRAST-ENHANCED ULTRASOUND:");
    println!(
        "  Peak Enhancement: {:.1} dB",
        report.ceus_result.perfusion_metrics.peak_enhancement
    );
    println!(
        "  Perfusion Rate: {:.1} mL/min/100g",
        report.ceus_result.perfusion_metrics.perfusion_rate
    );
    println!(
        "  Wash-in / Wash-out: {:.1}s / {:.1}s",
        report.ceus_result.perfusion_metrics.wash_in_time,
        report.ceus_result.perfusion_metrics.wash_out_time
    );
    println!(
        "  Contrast Samples: {}",
        report.ceus_result.contrast_signal.size()
    );
    println!(
        "  Confidence: {:.1}%",
        report
            .uncertainty_result
            .perfusion_uncertainty
            .confidence_score
            * 100.0
    );
    println!();

    println!("CLINICAL DIAGNOSIS:");
    println!("  {}", report.diagnosis.diagnosis_text);
    println!("  Perfusion Status: {}", report.diagnosis.perfusion_status);
    println!(
        "  Overall Confidence: {:.1}%",
        report.diagnosis.confidence_level * 100.0
    );
    for recommendation in &report.diagnosis.recommendations {
        println!("  Recommendation: {recommendation}");
    }
    println!();

    println!("TREATMENT PLAN:");
    println!("  {}", report.treatment_plan.recommended_actions);
    println!("  Follow-up: {}", report.treatment_plan.follow_up_schedule);
    println!(
        "  Therapeutic Considerations: {}",
        report.treatment_plan.therapeutic_considerations
    );
    for test in &report.treatment_plan.additional_tests {
        println!("  Additional Test: {test}");
    }
    println!();

    println!("SAFETY ASSESSMENT:");
    println!(
        "  Acoustic Safety: {}",
        if report.safety_assessment.acoustic_safety {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "  Thermal Safety: {}",
        if report.safety_assessment.thermal_safety {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!(
        "  Overall Safety: {}",
        if report.safety_assessment.overall_safe {
            "PASS"
        } else {
            "FAIL"
        }
    );
    println!("  Notes: {}", report.safety_assessment.safety_notes);
    println!();

    println!("CLINICAL VALIDATION REPORT:");
    println!("{}", report.validation_report);
    println!();
    println!("=== WORKFLOW COMPLETE ===");
    println!("This example demonstrates the integration of advanced ultrasound");
    println!("simulation capabilities for comprehensive clinical decision support.");

    Ok(())
}
