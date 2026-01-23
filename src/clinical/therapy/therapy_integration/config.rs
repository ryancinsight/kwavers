//! Therapy Session Configuration
//!
//! This module contains all configuration types and enums for clinical therapy sessions,
//! following AAPM TG-166 recommendations for treatment planning and IEC 62359:2010
//! standards for safety monitoring.
//!
//! ## References
//!
//! - AAPM TG-166: "Quality assurance for ultrasound-guided interventions"
//! - IEC 62359:2010: "Ultrasonics - Field characterization - Test methods"
//! - FDA Guidance: "510(k) Submissions for Ultrasound Devices"

use super::tissue::TissuePropertyMap;

/// Unified therapy session configuration
///
/// This struct contains all parameters required for a clinical therapy session,
/// following AAPM TG-166 recommendations for treatment planning and
/// IEC 62359:2010 standards for safety monitoring.
///
/// ## Clinical Guidelines
///
/// - **Treatment Duration**: Should follow ALARA (As Low As Reasonably Achievable) principle
/// - **Acoustic Parameters**: Must comply with FDA 510(k) guidance for the specific modality
/// - **Safety Limits**: Should be set according to patient-specific risk assessment
/// - **Patient Parameters**: Must be obtained from pre-treatment imaging and assessment
///
/// ## References
///
/// - AAPM TG-166: "Quality assurance for ultrasound-guided interventions"
/// - IEC 62359:2010: "Ultrasonics - Field characterization - Test methods"
/// - FDA Guidance: "510(k) Submissions for Ultrasound Devices"
#[derive(Debug, Clone)]
pub struct TherapySessionConfig {
    /// Primary therapy modality
    ///
    /// The main therapeutic approach to be used in the session.
    /// Should be selected based on clinical indication and patient assessment.
    pub primary_modality: TherapyModality,

    /// Secondary modalities for combination therapy
    ///
    /// Additional therapeutic approaches to enhance efficacy or provide
    /// synergistic effects. Should be carefully evaluated for potential
    /// interactions and safety considerations.
    pub secondary_modalities: Vec<TherapyModality>,

    /// Treatment duration (s)
    ///
    /// Total duration of the therapy session. Should be determined based on
    /// clinical protocol, treatment volume, and patient tolerance.
    /// Must comply with safety limits and ALARA principles.
    pub duration: f64,

    /// Acoustic parameters
    ///
    /// Acoustic parameters for the therapy session. Must be set according to
    /// clinical guidelines for the specific modality and patient condition.
    /// Should be validated against safety standards before treatment.
    pub acoustic_params: AcousticTherapyParams,

    /// Safety limits
    ///
    /// Patient-specific safety limits based on pre-treatment assessment.
    /// Must comply with IEC 62359:2010 and FDA guidance.
    /// Should be monitored continuously during treatment.
    pub safety_limits: SafetyLimits,

    /// Patient-specific parameters
    ///
    /// Individual patient characteristics obtained from medical imaging
    /// and clinical assessment. Critical for treatment planning and
    /// safety monitoring.
    pub patient_params: PatientParameters,

    /// Path to medical imaging data (CT, MRI)
    ///
    /// Optional path to patient imaging data in NIFTI (.nii, .nii.gz) or
    /// DICOM (.dcm) format. When provided, enables patient-specific
    /// treatment planning with actual anatomical data.
    ///
    /// - NIFTI files: Single-file CT/MRI volumes (supported with 'nifti' feature)
    /// - DICOM files: Individual slices or series (future implementation)
    ///
    /// If not provided, synthetic phantom data will be used for planning.
    pub imaging_data_path: Option<String>,
}

/// Therapy modality enumeration
///
/// Defines the different ultrasound therapy approaches supported by the framework.
/// Each modality has specific clinical applications and safety considerations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TherapyModality {
    /// Microbubble-enhanced therapy
    ///
    /// Uses ultrasound contrast agents to enhance drug delivery and imaging.
    /// References: Konofagou et al. (2012) "Focused ultrasound-mediated brain drug delivery"
    Microbubble,

    /// Transcranial focused ultrasound
    ///
    /// Focused ultrasound through the skull for brain applications.
    /// Requires aberration correction due to skull heterogeneity.
    Transcranial,

    /// Sonodynamic therapy
    ///
    /// ROS-mediated cancer therapy using ultrasound activation.
    /// References: Umemura et al. (1996) "Sonodynamic therapy"
    Sonodynamic,

    /// Histotripsy (mechanical ablation)
    ///
    /// Mechanical tissue ablation using cavitation.
    /// References: Hall et al. (2010) "Histotripsy: minimally invasive tissue ablation"
    Histotripsy,

    /// Lithotripsy (stone fragmentation)
    ///
    /// Kidney stone fragmentation using shock waves.
    /// References: Chaussy et al. (1980) "Extracorporeally induced destruction of kidney stones"
    Lithotripsy,

    /// Oncotripsy (tumor-specific histotripsy)
    ///
    /// Tumor-specific mechanical ablation.
    /// References: Xu et al. (2016) "Oncotripsy: targeted cancer therapy"
    Oncotripsy,

    /// Combined therapy approaches
    ///
    /// Multiple modalities used in combination for synergistic effects.
    Combined,
}

/// Acoustic parameters for therapy
///
/// Defines the ultrasound exposure parameters for therapeutic applications.
/// All parameters must comply with FDA 510(k) guidance and IEC 62359:2010 standards.
#[derive(Debug, Clone)]
pub struct AcousticTherapyParams {
    /// Center frequency (Hz)
    ///
    /// Primary frequency of ultrasound exposure.
    /// Typical range: 0.5-3 MHz for therapy, 0.25-1 MHz for transcranial
    pub frequency: f64,

    /// Peak negative pressure (Pa)
    ///
    /// Maximum negative pressure amplitude.
    /// Critical for mechanical index calculations and cavitation prediction.
    pub pnp: f64,

    /// Pulse repetition frequency (Hz)
    ///
    /// Rate at which ultrasound pulses are delivered.
    /// Affects duty cycle and thermal effects.
    pub prf: f64,

    /// Duty cycle (0-1)
    ///
    /// Fraction of time ultrasound is active.
    /// Lower duty cycles reduce thermal effects.
    pub duty_cycle: f64,

    /// Focal depth (m)
    ///
    /// Distance from transducer to acoustic focus.
    /// Critical for treatment planning and targeting.
    pub focal_depth: f64,

    /// Treatment volume (cm³)
    ///
    /// Volume of tissue to be treated.
    /// Used for dose calculations and treatment time estimation.
    pub treatment_volume: f64,
}

/// Safety limits for therapy
///
/// Patient-specific safety constraints based on clinical guidelines and
/// pre-treatment risk assessment.
///
/// ## References
///
/// - IEC 62359:2010: Safety indices and exposure limits
/// - FDA 510(k) Guidance: Ultrasound device safety requirements
#[derive(Debug, Clone)]
pub struct SafetyLimits {
    /// Maximum thermal index
    ///
    /// TI < 6.0 is generally safe for most applications.
    /// IEC 62359:2010 compliant.
    pub thermal_index_max: f64,

    /// Maximum mechanical index
    ///
    /// MI < 1.9 is generally safe for diagnostic applications.
    /// MI < 0.7 is recommended for fetal imaging.
    pub mechanical_index_max: f64,

    /// Maximum cavitation dose
    ///
    /// Integrated cavitation activity limit.
    /// Based on Apfel & Holland (1991) cavitation threshold models.
    pub cavitation_dose_max: f64,

    /// Maximum treatment time (s)
    ///
    /// Total duration limit based on patient tolerance and ALARA principles.
    pub max_treatment_time: f64,
}

/// Patient-specific parameters
///
/// Individual patient characteristics obtained from medical imaging (CT, MRI, ultrasound)
/// and clinical assessment. These parameters are critical for accurate treatment planning
/// and real-time safety monitoring.
#[derive(Debug, Clone)]
pub struct PatientParameters {
    /// Skull thickness map (for transcranial applications)
    ///
    /// 3D map of skull thickness from CT imaging.
    /// Used for transcranial aberration correction.
    /// None if not applicable (non-transcranial therapy).
    pub skull_thickness: Option<ndarray::Array3<f64>>,

    /// Tissue properties map
    ///
    /// Spatially-varying acoustic tissue properties.
    /// Derived from pre-treatment imaging and tissue characterization.
    pub tissue_properties: TissuePropertyMap,

    /// Target volume definition
    ///
    /// Geometric definition of treatment target.
    /// Should be validated against imaging data.
    pub target_volume: TargetVolume,

    /// Risk organs to avoid
    ///
    /// Critical structures that must be protected during therapy.
    /// Used for treatment planning constraints and real-time monitoring.
    pub risk_organs: Vec<RiskOrgan>,
}

/// Target volume definition
///
/// Geometric definition of the treatment target based on medical imaging
/// and clinical assessment.
#[derive(Debug, Clone)]
pub struct TargetVolume {
    /// Center coordinates (x, y, z) in meters
    ///
    /// Anatomical center of the treatment target in patient coordinates.
    pub center: (f64, f64, f64),

    /// Dimensions (width, height, depth) in meters
    ///
    /// Bounding box dimensions of the treatment volume.
    pub dimensions: (f64, f64, f64),

    /// Target tissue type
    ///
    /// Classification of the target tissue for parameter selection.
    pub tissue_type: TissueType,
}

/// Tissue type enumeration
///
/// Classification of tissue types for treatment planning parameter selection
/// and safety monitoring.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TissueType {
    /// Brain tissue
    Brain,
    /// Liver tissue
    Liver,
    /// Kidney tissue
    Kidney,
    /// Prostate tissue
    Prostate,
    /// Tumor tissue
    Tumor,
    /// Muscle tissue
    Muscle,
}

/// Risk organ definition
///
/// Critical anatomical structures that must be protected during therapy.
/// Dose constraints are applied to ensure these structures remain within
/// safe exposure limits.
#[derive(Debug, Clone)]
pub struct RiskOrgan {
    /// Organ name
    ///
    /// Clinical identifier for the organ (e.g., "heart", "kidney", "spinal_cord").
    pub name: String,

    /// Organ volume bounds
    ///
    /// 3D bounding box in patient coordinates: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
    pub bounds: ((f64, f64), (f64, f64), (f64, f64)),

    /// Maximum allowed dose
    ///
    /// Dose constraint for this organ based on clinical guidelines and patient risk assessment.
    pub max_dose: f64,
}
