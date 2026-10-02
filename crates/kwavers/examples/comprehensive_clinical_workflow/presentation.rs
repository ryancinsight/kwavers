//! Console presentation for a completed liver assessment.

use super::LiverAssessmentWorkflow;
use kwavers_core::error::KwaversResult;
use std::io::Write;

pub(super) fn run() -> KwaversResult<()> {
    let mut workflow = LiverAssessmentWorkflow::new("LIVER_PATIENT_001", (120.0, 80.0, 60.0))?;
    let report = workflow.execute_assessment()?;

    let _ = writeln!(
        std::io::stdout().lock(),
        "\n=== LIVER ASSESSMENT REPORT ==="
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Patient ID: {}",
        report.patient_id
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "Processing Time: {:.2}s",
        report.processing_time.as_secs_f64()
    );
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "B-MODE IMAGING:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Axial Resolution: {:.1} mm",
        report.b_mode_result.axial_resolution
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Lateral Resolution: {:.1} mm",
        report.b_mode_result.lateral_resolution
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Contrast Ratio: {:.1} dB",
        report.b_mode_result.contrast_ratio
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Voxels: {}",
        report.b_mode_result.envelope.size()
    );
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "SHEAR WAVE ELASTOGRAPHY:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Mean Stiffness: {:.1} kPa",
        report.swe_result.fibrosis_metrics.mean_stiffness
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Fibrosis Stage: F{}",
        report.swe_result.fibrosis_metrics.fibrosis_stage
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Stiffness Standard Deviation: {:.1} kPa",
        report.swe_result.fibrosis_metrics.stiffness_std
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Nonlinear Parameter: {:.3}",
        report.swe_result.fibrosis_metrics.nonlinear_parameter
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Displacement Frames: {}",
        report.swe_result.displacement_history.len()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Nonlinear Voxels: {}",
        report
            .swe_result
            .nonlinear_analysis
            .nonlinearity_parameter
            .size()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Confidence: {:.1}%",
        report.uncertainty_result.swe_uncertainty.confidence_score * 100.0
    );
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "CONTRAST-ENHANCED ULTRASOUND:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Peak Enhancement: {:.1} dB",
        report.ceus_result.perfusion_metrics.peak_enhancement
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Perfusion Rate: {:.1} mL/min/100g",
        report.ceus_result.perfusion_metrics.perfusion_rate
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Wash-in / Wash-out: {:.1}s / {:.1}s",
        report.ceus_result.perfusion_metrics.wash_in_time,
        report.ceus_result.perfusion_metrics.wash_out_time
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Contrast Samples: {}",
        report.ceus_result.contrast_signal.size()
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Confidence: {:.1}%",
        report
            .uncertainty_result
            .perfusion_uncertainty
            .confidence_score
            * 100.0
    );
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "CLINICAL DIAGNOSIS:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  {}",
        report.diagnosis.diagnosis_text
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Perfusion Status: {}",
        report.diagnosis.perfusion_status
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Overall Confidence: {:.1}%",
        report.diagnosis.confidence_level * 100.0
    );
    for recommendation in &report.diagnosis.recommendations {
        let _ = writeln!(
            std::io::stdout().lock(),
            "  Recommendation: {recommendation}"
        );
    }
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "TREATMENT PLAN:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  {}",
        report.treatment_plan.recommended_actions
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Follow-up: {}",
        report.treatment_plan.follow_up_schedule
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Therapeutic Considerations: {}",
        report.treatment_plan.therapeutic_considerations
    );
    for test in &report.treatment_plan.additional_tests {
        let _ = writeln!(std::io::stdout().lock(), "  Additional Test: {test}");
    }
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "SAFETY ASSESSMENT:");
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Acoustic Safety: {}",
        if report.safety_assessment.acoustic_safety {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Thermal Safety: {}",
        if report.safety_assessment.thermal_safety {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Overall Safety: {}",
        if report.safety_assessment.overall_safe {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "  Notes: {}",
        report.safety_assessment.safety_notes
    );
    let _ = writeln!(std::io::stdout().lock());

    let _ = writeln!(std::io::stdout().lock(), "CLINICAL VALIDATION REPORT:");
    let _ = writeln!(std::io::stdout().lock(), "{}", report.validation_report);
    let _ = writeln!(std::io::stdout().lock());
    let _ = writeln!(std::io::stdout().lock(), "=== WORKFLOW COMPLETE ===");
    let _ = writeln!(
        std::io::stdout().lock(),
        "This example demonstrates the integration of advanced ultrasound"
    );
    let _ = writeln!(
        std::io::stdout().lock(),
        "simulation capabilities for comprehensive clinical decision support."
    );

    Ok(())
}
