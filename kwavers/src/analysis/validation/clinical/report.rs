//! Report generation for `ClinicalValidator`.
//!
//! SRP: changes when the report format or content structure changes.

use super::types::ClinicalValidationResult;
use super::validator::ClinicalValidator;

impl ClinicalValidator {
    /// Generate comprehensive clinical validation report (Markdown).
    pub fn generate_validation_report(
        &self,
        bmode_result: Option<&ClinicalValidationResult>,
        doppler_result: Option<&ClinicalValidationResult>,
        safety_result: Option<&ClinicalValidationResult>,
    ) -> String {
        let mut report = String::new();
        report.push_str("# Clinical Validation Report - Kwavers Ultrasound System\n\n");
        report.push_str(&format!(
            "**Generated**: {}\n\n",
            chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        ));

        let all_passed = [bmode_result, doppler_result, safety_result]
            .iter()
            .all(|r| r.map(|res| res.passed).unwrap_or(true));

        report.push_str(&format!(
            "## Overall Status: {}\n\n",
            if all_passed {
                "✅ PASSED"
            } else {
                "❌ FAILED"
            }
        ));

        if let Some(result) = bmode_result {
            report.push_str(&format!(
                "## B-Mode Imaging Validation {}\n\n",
                if result.passed { "✅" } else { "❌" }
            ));
            report.push_str(&format!(
                "**Clinical Score**: {:.1}/100\n\n",
                result.clinical_score
            ));
            if !result.issues.is_empty() {
                report.push_str("### Issues Identified\n\n");
                for issue in &result.issues {
                    report.push_str(&format!("- {}\n", issue));
                }
                report.push('\n');
            }
            if !result.recommendations.is_empty() {
                report.push_str("### Recommendations\n\n");
                for rec in &result.recommendations {
                    report.push_str(&format!("- {}\n", rec));
                }
                report.push('\n');
            }
        }

        if let Some(result) = doppler_result {
            report.push_str(&format!(
                "## Doppler Imaging Validation {}\n\n",
                if result.passed { "✅" } else { "❌" }
            ));
            report.push_str(&format!(
                "**Clinical Score**: {:.1}/100\n\n",
                result.clinical_score
            ));
            if !result.issues.is_empty() {
                report.push_str("### Issues Identified\n\n");
                for issue in &result.issues {
                    report.push_str(&format!("- {}\n", issue));
                }
                report.push('\n');
            }
        }

        if let Some(result) = safety_result {
            report.push_str(&format!(
                "## Safety Validation (IEC 60601-2-37) {}\n\n",
                if result.passed { "✅" } else { "❌" }
            ));
            report.push_str(&format!(
                "**Safety Compliance**: {}\n\n",
                if result.regulatory_compliant {
                    "REGULATORY COMPLIANT"
                } else {
                    "REQUIRES CORRECTION"
                }
            ));
            if !result.issues.is_empty() {
                report.push_str("### Safety Issues\n\n");
                for issue in &result.issues {
                    report.push_str(&format!("- **CRITICAL**: {}\n", issue));
                }
                report.push('\n');
            }
        }

        report.push_str("## Regulatory Compliance Summary\n\n");
        report.push_str("| Standard | Status | Notes |\n");
        report.push_str("|----------|--------|-------|\n");

        if let Some(result) = bmode_result {
            report.push_str(&format!(
                "| FDA 510(k) | {} | B-mode imaging requirements |\n",
                if result.regulatory_compliant {
                    "✅ Compliant"
                } else {
                    "❌ Non-compliant"
                }
            ));
        }
        if let Some(result) = safety_result {
            report.push_str(&format!(
                "| IEC 60601-2-37 | {} | Ultrasound safety standards |\n",
                if result.regulatory_compliant {
                    "✅ Compliant"
                } else {
                    "❌ Non-compliant"
                }
            ));
        }

        report.push_str("\n## Next Steps\n\n");
        if !all_passed {
            report.push_str("1. **Address critical safety issues** immediately\n");
            report.push_str("2. **Improve image quality metrics** to meet clinical requirements\n");
            report.push_str("3. **Calibrate measurement accuracy** for regulatory compliance\n");
            report.push_str("4. **Re-validate** after implementing corrections\n");
            report.push_str("5. **Generate updated clinical validation report**\n");
        } else {
            report.push_str(
                "1. **Proceed with clinical trials** - system meets regulatory requirements\n",
            );
            report.push_str("2. **Document validation results** for FDA submission\n");
            report.push_str("3. **Monitor performance** in clinical environment\n");
            report.push_str("4. **Plan post-market surveillance** and updates\n");
        }

        report
    }
}
