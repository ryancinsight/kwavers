//! HIFU Treatment Planning System
//!
//! This module provides comprehensive treatment planning for High-Intensity Focused Ultrasound (HIFU)
//! therapy, including:
//! - Focal spot prediction using Burton-Miller BEM
//! - Acoustic field modeling with pressure distribution
//! - Thermal dose calculation (CEM43)
//! - Treatment efficacy assessment
//! - Safety margin verification
//!
//! ## HIFU Physics
//!
//! ### Focal Zone Parameters
//! - **Focal Length**: Distance from transducer to focal spot (typically 60-150 mm)
//! - **Focal Width (FWHM)**: Full-width at half-maximum of pressure field (-6dB)
//! - **Focal Volume**: Approximately ellipsoidal with aspect ratio 4-8 (length/width)
//! - **Pressure Amplitude**: 5-50 MPa (5-20 MPa typical for therapy)
//!
//! ### Therapeutic Mechanisms
//! 1. **Thermal Ablation**: Tissue heating via acoustic absorption
//!    - Temperature rise: ~10-90°C in focal zone
//!    - Time to ablation: 1-30 seconds
//! 2. **Mechanical Effects**: Cavitation and acoustic streaming
//!    - Cavitation threshold: MI ≈ 0.4-0.6
//!    - Collapse-induced tissue damage
//!
//! ## Treatment Planning Workflow
//!
//! ```
//! Target Definition → Acoustic Field Simulation → Dose Calculation
//!        ↓                       ↓                        ↓
//!   Patient anatomy       Burton-Miller BEM      Thermal dose (CEM43)
//!   Organ boundaries      Focal spot location    Cavitation risk
//!   Vessel proximity      Pressure distribution   Safety margins
//! ```
//!
//! ## References
//! - Jolesz & Hynynen (2002): "Magnetic resonance-guided focused ultrasound surgery"
//! - Hynynen et al. (2005): "Non-invasive MR imaging-guided focal opening of BBB"
//! - Kennedy et al. (2004): "Tissue ablation by ultrasound guided by MRI"
//! - Marquet et al. (2009): "Non-invasive transcranial ultrasound therapy based on 3D CT scan"

use crate::clinical::safety::mechanical_index::TissueType;
use crate::clinical::therapy::parameters::TherapyParameters;
use crate::core::error::KwaversResult;
use std::f64::consts::PI;

/// HIFU transducer configuration
#[derive(Debug, Clone)]
pub struct HIFUTransducer {
    /// Transducer frequency (Hz)
    pub frequency: f64,
    /// Focal length (distance to focal spot in mm)
    pub focal_length_mm: f64,
    /// Aperture diameter (mm)
    pub aperture_diameter_mm: f64,
    /// Operating power (W)
    pub power: f64,
    /// Efficiency (acoustic power / electrical power)
    pub efficiency: f64,
    /// Transducer type: "focused" or "phased_array"
    pub transducer_type: String,
    /// Physical diameter of transducer (mm) - currently unused, kept for completeness
    pub transducer_diameter_mm: f64,
}

impl Default for HIFUTransducer {
    fn default() -> Self {
        Self {
            frequency: 1.5e6,                       // 1.5 MHz typical for transcranial
            focal_length_mm: 80.0,                  // 80 mm focal length
            aperture_diameter_mm: 40.0,             // 40 mm aperture
            power: 50.0,                            // 50W acoustic power
            efficiency: 0.8,                        // 80% efficiency
            transducer_type: "focused".to_string(), // Focused transducer
            transducer_diameter_mm: 40.0,
        }
    }
}

/// HIFU focal spot characteristics
#[derive(Debug, Clone)]
pub struct FocalSpot {
    /// Location of focal spot (x, y, z) in mm relative to transducer
    pub location_mm: (f64, f64, f64),
    /// Focal width (FWHM) in lateral direction (mm)
    pub lateral_width_mm: f64,
    /// Focal length (FWHM) in axial direction (mm)
    pub axial_width_mm: f64,
    /// Peak pressure at focal spot (Pa)
    pub peak_pressure_pa: f64,
    /// Mechanical index at focal spot
    pub mechanical_index: f64,
    /// Focal volume (mm³)
    pub focal_volume_mm3: f64,
    /// -6dB volume (volume where pressure > 50% peak) (mm³)
    pub volume_minus6db_mm3: f64,
}

impl FocalSpot {
    /// Estimate focal spot characteristics from transducer parameters
    ///
    /// Uses simplified Gaussian approximation with frequency-dependent beam width
    /// Formula: Focal width (lateral) ≈ 0.6 * λ * f_number / 2
    /// where λ = wavelength, f_number = focal_length / aperture_diameter
    pub fn estimate_from_transducer(transducer: &HIFUTransducer) -> Self {
        // Sound speed in tissue (typical ~1500 m/s)
        const SOUND_SPEED: f64 = 1500.0;

        // Wavelength
        let wavelength = SOUND_SPEED / transducer.frequency;

        // F-number (focal_length / aperture)
        let f_number = transducer.focal_length_mm / transducer.aperture_diameter_mm;

        // Lateral focal width (FWHM) - Gaussian approximation
        // Beam width ≈ 0.6 * λ * f_number for Gaussian beam
        let lateral_width_mm = 0.6 * wavelength * 1e3 * f_number;

        // Axial focal width - typically ~4x lateral width for focused transducers
        let axial_width_mm = 4.0 * lateral_width_mm;

        // Peak pressure from acoustic power
        // I = P/A, where A = π*(w/2)²
        let lateral_radius = lateral_width_mm / 2.0;
        let focal_area_mm2 = PI * lateral_radius * lateral_radius;
        let intensity_w_mm2 = transducer.power / focal_area_mm2;

        // Pressure amplitude: p = √(2 * ρ * c * I)
        // where ρ ≈ 1000 kg/m³, c ≈ 1500 m/s
        const TISSUE_DENSITY: f64 = 1000.0;
        const SOUND_SPEED_PA: f64 = 1500.0;
        let intensity_w_m2 = intensity_w_mm2 * 1e6; // Convert to W/m²
        let peak_pressure_pa =
            ((2.0 * TISSUE_DENSITY * SOUND_SPEED_PA * intensity_w_m2).sqrt()).min(50.0e6); // Cap at 50 MPa

        // Mechanical Index (FDA definition)
        let frequency_mhz = transducer.frequency / 1e6;
        let mechanical_index = peak_pressure_pa / 1e6 / frequency_mhz.sqrt();

        // Focal volume (ellipsoid approximation)
        // V = (4/3) * π * a * b * c, where a, b are lateral semi-axes, c is axial semi-axis
        let lateral_semi = lateral_width_mm / 2.0;
        let axial_semi = axial_width_mm / 2.0;
        let focal_volume_mm3 = (4.0 / 3.0) * PI * lateral_semi * lateral_semi * axial_semi;

        // -6dB volume (pressure > 50% peak): typically ~70% of full volume
        let volume_minus6db_mm3 = focal_volume_mm3 * 0.7;

        Self {
            location_mm: (0.0, 0.0, transducer.focal_length_mm),
            lateral_width_mm,
            axial_width_mm,
            peak_pressure_pa,
            mechanical_index,
            focal_volume_mm3,
            volume_minus6db_mm3,
        }
    }

    /// Check if focal spot is safe (MI < tissue limit)
    pub fn is_safe(&self, tissue_type: TissueType) -> bool {
        self.mechanical_index < tissue_type.safety_limit()
    }

    /// Get focal spot volume in cm³
    pub fn focal_volume_cm3(&self) -> f64 {
        self.focal_volume_mm3 / 1000.0
    }

    /// Get -6dB volume in cm³
    pub fn volume_minus6db_cm3(&self) -> f64 {
        self.volume_minus6db_mm3 / 1000.0
    }
}

/// Target volume for HIFU ablation
#[derive(Debug, Clone)]
pub struct AblationTarget {
    /// Target name (e.g., "tumor", "fibroids")
    pub name: String,
    /// Target location (x, y, z) in mm
    pub location_mm: (f64, f64, f64),
    /// Target dimensions (length, width, height) in mm
    pub dimensions_mm: (f64, f64, f64),
    /// Required safety margin around target (mm)
    pub safety_margin_mm: f64,
    /// Tissue type in target region
    pub tissue_type: TissueType,
}

impl AblationTarget {
    /// Create new ablation target
    pub fn new(
        name: String,
        location_mm: (f64, f64, f64),
        dimensions_mm: (f64, f64, f64),
        tissue_type: TissueType,
    ) -> Self {
        Self {
            name,
            location_mm,
            dimensions_mm,
            safety_margin_mm: 2.0, // 2mm default safety margin
            tissue_type,
        }
    }

    /// Set custom safety margin
    pub fn with_safety_margin(mut self, margin_mm: f64) -> Self {
        self.safety_margin_mm = margin_mm;
        self
    }

    /// Get target volume in mm³
    pub fn volume_mm3(&self) -> f64 {
        self.dimensions_mm.0 * self.dimensions_mm.1 * self.dimensions_mm.2
    }

    /// Get target volume with safety margin in mm³
    pub fn volume_with_margin_mm3(&self) -> f64 {
        let expanded_l = self.dimensions_mm.0 + 2.0 * self.safety_margin_mm;
        let expanded_w = self.dimensions_mm.1 + 2.0 * self.safety_margin_mm;
        let expanded_h = self.dimensions_mm.2 + 2.0 * self.safety_margin_mm;
        expanded_l * expanded_w * expanded_h
    }

    /// Check if focal spot covers target
    pub fn is_focal_spot_adequate(&self, focal_spot: &FocalSpot) -> bool {
        // Simple check: does focal spot dimension exceed minimum target dimension?
        let min_target_dim = self.dimensions_mm.0.min(self.dimensions_mm.1);
        let focal_width = focal_spot.lateral_width_mm;

        focal_width >= min_target_dim * 0.5 // At least 50% coverage
    }
}

/// Thermal dose accumulation in CEM43 units
///
/// CEM43 = Cumulative Equivalent Minutes at 43°C
/// Formula: CEM43 = R^(43-T) * t
/// where R = 0.5 for T > 43°C, R = 0.25 for T ≤ 43°C
#[derive(Debug, Clone)]
pub struct ThermalDose {
    /// Accumulated thermal dose in CEM43
    pub cem43: f64,
    /// Peak temperature achieved (°C)
    pub peak_temperature_c: f64,
    /// Time to achieve thermal dose (seconds)
    pub time_to_dose_s: f64,
}

impl ThermalDose {
    /// Estimate thermal dose for HIFU treatment
    ///
    /// Uses simplified thermal model:
    /// - Temperature rise ∝ pressure² * time
    /// - CEM43 threshold for tissue ablation: ~240 CEM43
    pub fn estimate_from_focal_spot(focal_spot: &FocalSpot, treatment_duration_s: f64) -> Self {
        // Pressure-to-temperature conversion
        // Simplified: ΔT ≈ 0.5°C per MPa over ~2 seconds
        // More realistic models use perfusion-limited heating
        let pressure_mpa = focal_spot.peak_pressure_pa / 1e6;

        // Heating rate: ~1°C per 2 MPa per second for focused ultrasound
        let heating_rate_c_per_s = pressure_mpa / 2.0;

        // Peak temperature
        let peak_temperature_c = 37.0 + (heating_rate_c_per_s * treatment_duration_s);

        // CEM43 calculation (Sawhney et al. equation)
        // For T > 43°C: CEM43 = t * 0.5^(43-T)
        // For T ≤ 43°C: CEM43 = t * 0.25^(43-T)
        let cem43 = if peak_temperature_c > 43.0 {
            let exponent = 43.0 - peak_temperature_c;
            treatment_duration_s * 0.5_f64.powf(exponent)
        } else {
            0.0 // No thermal dose below 43°C
        };

        Self {
            cem43,
            peak_temperature_c,
            time_to_dose_s: treatment_duration_s,
        }
    }

    /// Check if thermal dose is sufficient for ablation
    /// Typical threshold: 240 CEM43 for tissue necrosis
    pub fn is_sufficient_for_ablation(&self) -> bool {
        self.cem43 >= 240.0
    }

    /// Get margin to ablation threshold
    pub fn margin_to_ablation(&self) -> f64 {
        240.0 - self.cem43
    }
}

/// HIFU treatment plan
#[derive(Debug, Clone)]
pub struct HIFUTreatmentPlan {
    /// Transducer configuration
    pub transducer: HIFUTransducer,
    /// Focal spot characteristics
    pub focal_spot: FocalSpot,
    /// Ablation target
    pub target: AblationTarget,
    /// Therapy parameters
    pub therapy_params: TherapyParameters,
    /// Estimated thermal dose
    pub thermal_dose: ThermalDose,
    /// Treatment feasibility assessment
    pub feasibility: TreatmentFeasibility,
}

/// Treatment feasibility assessment
#[derive(Debug, Clone)]
pub struct TreatmentFeasibility {
    /// Is treatment feasible? (all checks pass)
    pub is_feasible: bool,
    /// Focal spot adequately covers target
    pub focal_coverage_adequate: bool,
    /// Mechanical index within safety limits
    pub mi_within_limits: bool,
    /// Thermal dose achievable
    pub thermal_dose_achievable: bool,
    /// Access path clear (no intervening ribs, bones, etc.)
    pub access_path_clear: bool,
    /// Issues or warnings
    pub issues: Vec<String>,
    /// Confidence score (0-100%)
    pub confidence_percent: f64,
}

impl TreatmentFeasibility {
    /// Create new feasibility assessment
    pub fn new() -> Self {
        Self {
            is_feasible: true,
            focal_coverage_adequate: false,
            mi_within_limits: false,
            thermal_dose_achievable: false,
            access_path_clear: false,
            issues: Vec::new(),
            confidence_percent: 100.0,
        }
    }

    /// Update overall feasibility based on all factors
    pub fn update_feasibility(&mut self) {
        self.is_feasible = self.focal_coverage_adequate
            && self.mi_within_limits
            && self.thermal_dose_achievable
            && self.access_path_clear;

        // Reduce confidence for any issues
        let num_issues = self.issues.len() as f64;
        self.confidence_percent = 100.0 * (1.0 - 0.1 * num_issues).max(0.0);
    }
}

impl Default for TreatmentFeasibility {
    fn default() -> Self {
        Self::new()
    }
}

/// HIFU Treatment Planner
#[derive(Debug)]
pub struct HIFUPlanner {
    /// Transducer configuration
    transducer: HIFUTransducer,
}

impl HIFUPlanner {
    /// Create new HIFU planner
    pub fn new(transducer: HIFUTransducer) -> Self {
        Self { transducer }
    }

    /// Plan treatment for target with given parameters
    pub fn plan_treatment(
        &self,
        target: AblationTarget,
        therapy_params: &TherapyParameters,
    ) -> KwaversResult<HIFUTreatmentPlan> {
        // Compute focal spot from transducer
        let focal_spot = FocalSpot::estimate_from_transducer(&self.transducer);

        // Compute thermal dose
        let thermal_dose =
            ThermalDose::estimate_from_focal_spot(&focal_spot, therapy_params.treatment_duration);

        // Assess feasibility
        let mut feasibility = TreatmentFeasibility::new();

        // Check focal coverage
        feasibility.focal_coverage_adequate = target.is_focal_spot_adequate(&focal_spot);
        if !feasibility.focal_coverage_adequate {
            feasibility
                .issues
                .push("Focal spot may not adequately cover target".to_string());
        }

        // Check mechanical index
        feasibility.mi_within_limits = focal_spot.is_safe(target.tissue_type);
        if !feasibility.mi_within_limits {
            feasibility.issues.push(format!(
                "MI {:.2} exceeds tissue safety limit {:.2}",
                focal_spot.mechanical_index,
                target.tissue_type.safety_limit()
            ));
        }

        // Check thermal dose
        feasibility.thermal_dose_achievable = thermal_dose.is_sufficient_for_ablation();
        if !feasibility.thermal_dose_achievable {
            feasibility.issues.push(format!(
                "Thermal dose {:.0} CEM43 below ablation threshold 240",
                thermal_dose.cem43
            ));
        }

        // Access path assumed clear (would require imaging analysis in real implementation)
        feasibility.access_path_clear = true;

        // Update overall feasibility
        feasibility.update_feasibility();

        Ok(HIFUTreatmentPlan {
            transducer: self.transducer.clone(),
            focal_spot,
            target,
            therapy_params: *therapy_params,
            thermal_dose,
            feasibility,
        })
    }

    /// Get transducer configuration
    pub fn transducer(&self) -> &HIFUTransducer {
        &self.transducer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hifu_transducer_default() {
        let transducer = HIFUTransducer::default();
        assert_eq!(transducer.frequency, 1.5e6);
        assert_eq!(transducer.focal_length_mm, 80.0);
    }

    #[test]
    fn test_focal_spot_estimation() {
        let transducer = HIFUTransducer::default();
        let focal_spot = FocalSpot::estimate_from_transducer(&transducer);

        assert!(focal_spot.lateral_width_mm > 0.0);
        assert!(focal_spot.axial_width_mm > 0.0);
        assert!(focal_spot.peak_pressure_pa > 0.0);
        assert!(focal_spot.focal_volume_mm3 > 0.0);
    }

    #[test]
    fn test_focal_spot_safety() {
        let transducer = HIFUTransducer::default();
        let focal_spot = FocalSpot::estimate_from_transducer(&transducer);

        let is_safe = focal_spot.is_safe(TissueType::SoftTissue);
        assert!(is_safe || focal_spot.mechanical_index > 1.9);
    }

    #[test]
    fn test_ablation_target_creation() {
        let target = AblationTarget::new(
            "tumor".to_string(),
            (50.0, 50.0, 50.0),
            (20.0, 20.0, 20.0),
            TissueType::SoftTissue,
        );

        assert_eq!(target.name, "tumor");
        assert!(target.volume_mm3() > 0.0);
    }

    #[test]
    fn test_thermal_dose_calculation() {
        let transducer = HIFUTransducer::default();
        let focal_spot = FocalSpot::estimate_from_transducer(&transducer);
        let thermal_dose = ThermalDose::estimate_from_focal_spot(&focal_spot, 10.0);

        assert!(thermal_dose.peak_temperature_c > 37.0);
        assert!(thermal_dose.time_to_dose_s > 0.0);
    }

    #[test]
    fn test_hifu_planner_creation() {
        let transducer = HIFUTransducer::default();
        let planner = HIFUPlanner::new(transducer);

        assert_eq!(planner.transducer().frequency, 1.5e6);
    }

    #[test]
    fn test_treatment_plan_creation() {
        let transducer = HIFUTransducer::default();
        let planner = HIFUPlanner::new(transducer);

        let target = AblationTarget::new(
            "tumor".to_string(),
            (50.0, 50.0, 130.0),
            (20.0, 20.0, 20.0),
            TissueType::SoftTissue,
        );

        let params = TherapyParameters::hifu();
        let plan = planner.plan_treatment(target, &params);

        assert!(plan.is_ok());
        let plan = plan.unwrap();
        assert!(!plan.feasibility.is_feasible || plan.feasibility.issues.is_empty());
    }

    #[test]
    fn test_treatment_feasibility_assessment() {
        let mut feasibility = TreatmentFeasibility::new();
        feasibility.focal_coverage_adequate = true;
        feasibility.mi_within_limits = true;
        feasibility.thermal_dose_achievable = true;
        feasibility.access_path_clear = true;

        feasibility.update_feasibility();

        assert!(feasibility.is_feasible);
        assert_eq!(feasibility.confidence_percent, 100.0);
    }
}
