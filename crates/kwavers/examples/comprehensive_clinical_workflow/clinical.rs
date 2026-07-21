//! Clinical interpretation and treatment planning.

use super::{
    CEUSResult, ClinicalDiagnosis, LiverAssessmentWorkflow, SWEResult, TreatmentPlan,
    UncertaintyAnalysis,
};

impl LiverAssessmentWorkflow {
    pub(super) fn generate_clinical_diagnosis(
        &self,
        swe_result: &SWEResult,
        ceus_result: &CEUSResult,
        uncertainty: &UncertaintyAnalysis,
    ) -> ClinicalDiagnosis {
        let fibrosis_stage = swe_result.fibrosis_metrics.fibrosis_stage;
        let perfusion_abnormality = ceus_result.perfusion_metrics.perfusion_rate < 100.0;

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

        ClinicalDiagnosis {
            diagnosis_text: diagnosis,
            fibrosis_stage,
            perfusion_status: if perfusion_abnormality {
                "Abnormal"
            } else {
                "Normal"
            }
            .to_string(),
            confidence_level,
            recommendations: Self::clinical_recommendations(fibrosis_stage, perfusion_abnormality),
        }
    }

    pub(super) fn generate_treatment_plan(&self, diagnosis: &ClinicalDiagnosis) -> TreatmentPlan {
        let plan = match diagnosis.fibrosis_stage {
            0 => "Routine monitoring - Annual ultrasound screening".to_string(),
            1 => {
                "Lifestyle intervention - Weight management, exercise, alcohol cessation"
                    .to_string()
            }
            2 => "Medical management - Consider antiviral therapy, fibrosis monitoring".to_string(),
            3..=4 => "Specialized care - Hepatology consultation, consider liver biopsy, treatment optimization".to_string(),
            _ => "Further evaluation required".to_string(),
        };

        TreatmentPlan {
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
        }
    }

    fn clinical_recommendations(fibrosis_stage: i32, perfusion_abnormality: bool) -> Vec<String> {
        let mut recommendations = match fibrosis_stage {
            0 => vec![
                "Continue healthy lifestyle".to_string(),
                "Annual liver ultrasound screening".to_string(),
            ],
            1 => vec![
                "Weight management and exercise program".to_string(),
                "Alcohol cessation if applicable".to_string(),
                "6-month follow-up ultrasound".to_string(),
            ],
            2 => vec![
                "Medical evaluation for underlying causes".to_string(),
                "Consider antiviral therapy if indicated".to_string(),
                "3-month follow-up with SWE and CEUS".to_string(),
            ],
            3..=4 => vec![
                "Immediate hepatology consultation".to_string(),
                "Liver biopsy consideration".to_string(),
                "Screening for portal hypertension and varices".to_string(),
                "Monthly monitoring until stable".to_string(),
            ],
            _ => Vec::new(),
        };

        if perfusion_abnormality {
            recommendations
                .push("Further evaluation for hepatocellular carcinoma risk".to_string());
        }

        recommendations
    }
}
