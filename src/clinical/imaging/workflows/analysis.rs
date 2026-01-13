use super::results::{DiagnosticRecommendation, DiagnosticUrgency};
use crate::core::error::KwaversResult;
use crate::physics::imaging::fusion::FusedImageResult;
use ndarray::Array3;
use std::collections::HashMap;

pub fn generate_diagnostic_recommendations(
    tissue_properties: &HashMap<String, Array3<f64>>,
) -> KwaversResult<Vec<DiagnosticRecommendation>> {
    let mut recommendations = Vec::new();

    // Advanced multi-parameter diagnostic analysis
    let mut diagnostic_score = 0.0;
    let mut evidence = Vec::new();

    // Tissue classification analysis
    if let Some(classification) = tissue_properties.get("tissue_classification") {
        let high_risk_voxels = classification.iter().filter(|&&x| x >= 2.0).count();
        let moderate_risk_voxels = classification
            .iter()
            .filter(|&&x| (1.0..2.0).contains(&x))
            .count();
        let borderline_voxels = classification
            .iter()
            .filter(|&&x| (0.5..1.0).contains(&x))
            .count();
        let total_voxels = classification.len();

        let high_risk_ratio = high_risk_voxels as f64 / total_voxels as f64;
        let moderate_risk_ratio = moderate_risk_voxels as f64 / total_voxels as f64;

        if high_risk_ratio > 0.05 {
            // >5% high-risk tissue
            diagnostic_score += 30.0;
            evidence.push(format!(
                "{:.1}% high-risk tissue regions detected",
                high_risk_ratio * 100.0
            ));
        } else if moderate_risk_ratio > 0.15 {
            // >15% moderate-risk tissue
            diagnostic_score += 20.0;
            evidence.push(format!(
                "{:.1}% moderate-risk tissue regions detected",
                moderate_risk_ratio * 100.0
            ));
        }

        if borderline_voxels > 0 {
            evidence.push(format!(
                "{} borderline regions require monitoring",
                borderline_voxels
            ));
        }
    }

    // Oxygenation analysis
    if let Some(oxygenation) = tissue_properties.get("oxygenation_index") {
        let low_oxygenation_voxels = oxygenation.iter().filter(|&&x| x < 0.6).count();
        let high_oxygenation_voxels = oxygenation.iter().filter(|&&x| x > 0.9).count();
        let total_voxels = oxygenation.len();

        let hypoxia_ratio = low_oxygenation_voxels as f64 / total_voxels as f64;
        let hyperoxia_ratio = high_oxygenation_voxels as f64 / total_voxels as f64;

        if hypoxia_ratio > 0.2 {
            // >20% hypoxic regions
            diagnostic_score += 25.0;
            evidence.push(format!(
                "{:.1}% hypoxic tissue regions (potential malignancy)",
                hypoxia_ratio * 100.0
            ));
        }

        if hyperoxia_ratio > 0.3 {
            // >30% hyperoxic regions
            diagnostic_score += 10.0;
            evidence.push(format!(
                "{:.1}% hypervascular regions detected",
                hyperoxia_ratio * 100.0
            ));
        }
    }

    // Stiffness analysis
    if let Some(stiffness) = tissue_properties.get("composite_stiffness") {
        let high_stiffness_voxels = stiffness.iter().filter(|&&x| x > 40.0).count(); // >40 kPa
        let total_voxels = stiffness.len();

        let stiff_ratio = high_stiffness_voxels as f64 / total_voxels as f64;
        if stiff_ratio > 0.25 {
            // >25% stiff tissue
            diagnostic_score += 20.0;
            evidence.push(format!(
                "{:.1}% stiff tissue regions (fibrosis/carcinoma)",
                stiff_ratio * 100.0
            ));
        }
    }

    // Generate recommendations based on diagnostic score
    if diagnostic_score >= 40.0 {
        // High suspicion case
        recommendations.push(DiagnosticRecommendation {
            finding: "High suspicion of tissue abnormality requiring immediate attention"
                .to_string(),
            confidence: f64::min(75.0 + diagnostic_score * 0.5, 98.0),
            recommendations: vec![
                "Urgent biopsy recommended within 1-2 weeks".to_string(),
                "Consider MRI or PET-CT for staging".to_string(),
                "Schedule follow-up imaging within 1 month".to_string(),
                "Consultation with oncology specialist advised".to_string(),
                "Consider molecular/genetic testing".to_string(),
            ],
            urgency: DiagnosticUrgency::Urgent,
            evidence,
        });
    } else if diagnostic_score >= 20.0 {
        // Moderate suspicion case
        recommendations.push(DiagnosticRecommendation {
            finding: "Moderate tissue abnormalities detected - requires monitoring".to_string(),
            confidence: f64::min(65.0 + diagnostic_score * 0.75, 85.0),
            recommendations: vec![
                "Biopsy recommended within 4-6 weeks".to_string(),
                "Schedule follow-up imaging in 3 months".to_string(),
                "Consider additional molecular imaging".to_string(),
                "Regular clinical monitoring advised".to_string(),
            ],
            urgency: DiagnosticUrgency::Urgent,
            evidence,
        });
    } else if diagnostic_score >= 5.0 {
        // Low suspicion case
        recommendations.push(DiagnosticRecommendation {
            finding: "Minor tissue variations detected - low suspicion".to_string(),
            confidence: f64::min(80.0 + diagnostic_score, 92.0),
            recommendations: vec![
                "Continue routine screening schedule".to_string(),
                "Annual follow-up imaging recommended".to_string(),
                "Monitor for any symptom changes".to_string(),
            ],
            urgency: DiagnosticUrgency::Normal,
            evidence,
        });
    } else {
        // Normal case
        recommendations.push(DiagnosticRecommendation {
            finding: "No significant abnormalities detected - normal findings".to_string(),
            confidence: 95.0,
            recommendations: vec![
                "Continue routine screening schedule".to_string(),
                "Annual follow-up as per standard protocol".to_string(),
            ],
            urgency: DiagnosticUrgency::Normal,
            evidence: vec![
                "Homogeneous tissue appearance across all modalities".to_string(),
                "All quantitative parameters within normal ranges".to_string(),
                "No concerning patterns detected".to_string(),
            ],
        });
    }

    Ok(recommendations)
}

pub fn calculate_confidence_score(
    fused_result: &FusedImageResult,
    tissue_properties: &HashMap<String, Array3<f64>>,
) -> f64 {
    // Calculate overall confidence based on multiple factors
    let mut confidence = 80.0; // Base confidence

    // Quality factor - handle empty collections
    if !fused_result.modality_quality.is_empty() {
        let avg_quality = fused_result.modality_quality.values().sum::<f64>()
            / fused_result.modality_quality.len() as f64;
        if avg_quality.is_finite() {
            confidence += (avg_quality - 0.5) * 10.0; // ±10 based on quality
        }
    }

    // Fusion confidence factor - handle empty collections
    if !fused_result.confidence_map.is_empty() {
        let avg_confidence = fused_result.confidence_map.iter().sum::<f64>()
            / fused_result.confidence_map.len() as f64;
        if avg_confidence.is_finite() {
            confidence += avg_confidence * 5.0; // ±5 based on fusion confidence
        }
    }

    // Tissue property consistency factor
    if tissue_properties.contains_key("tissue_classification") {
        confidence += 5.0; // Bonus for having tissue classification
    }

    confidence.clamp(0.0, 100.0)
}
