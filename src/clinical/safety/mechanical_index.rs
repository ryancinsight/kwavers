//! Mechanical Index (MI) Calculations for Ultrasound Safety
//!
//! The Mechanical Index is a critical safety metric for diagnostic and therapeutic
//! ultrasound, quantifying the likelihood of mechanical bioeffects such as cavitation.
//!
//! # Definition
//!
//! MI = P_r / √f_c
//!
//! Where:
//! - P_r: Peak rarefactional pressure (MPa)
//! - f_c: Center frequency (MHz)
//!
//! # Safety Limits
//!
//! - **FDA Limit (Diagnostic)**: MI < 1.9
//! - **FDA Limit (Ophthalmic)**: MI < 0.23
//! - **WFUMB Recommendation**: MI < 0.7 for lung/bowel
//! - **Cavitation Threshold**: MI ≈ 0.4-0.6 (tissue-dependent)
//!
//! # References
//!
//! - FDA (2008). "Information for Manufacturers Seeking Marketing Clearance
//!   of Diagnostic Ultrasound Systems and Transducers"
//! - AIUM/NEMA (2004). "Standard for Real-Time Display of Thermal and
//!   Mechanical Acoustic Output Indices on Diagnostic Ultrasound Equipment"
//! - Duck, F. A. (2002). "Nonlinear acoustics in diagnostic ultrasound"
//!   *Ultrasound in Medicine & Biology*, 28(1), 1-18.

use crate::core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

/// Mechanical Index calculator for ultrasound safety assessment
#[derive(Debug, Clone)]
pub struct MechanicalIndexCalculator {
    /// Center frequency in MHz
    center_frequency_mhz: f64,
    /// Attenuation coefficient in dB/cm/MHz
    attenuation_coeff: f64,
    /// Tissue type for safety limit selection
    tissue_type: TissueType,
}

/// Tissue types with different MI safety thresholds
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TissueType {
    /// General soft tissue (MI < 1.9)
    SoftTissue,
    /// Ophthalmic tissue (MI < 0.23)
    Ophthalmic,
    /// Lung tissue with gas bodies (MI < 0.7)
    Lung,
    /// Bowel tissue with gas bodies (MI < 0.7)
    Bowel,
    /// Fetal tissue (MI < 1.0 recommended)
    Fetal,
    /// Brain tissue (MI < 1.5)
    Brain,
}

impl TissueType {
    /// Get FDA/WFUMB recommended MI limit for this tissue type
    #[must_use]
    pub fn safety_limit(&self) -> f64 {
        match self {
            Self::SoftTissue => 1.9,  // FDA diagnostic limit
            Self::Ophthalmic => 0.23, // FDA ophthalmic limit
            Self::Lung => 0.7,        // WFUMB gas-body tissue
            Self::Bowel => 0.7,       // WFUMB gas-body tissue
            Self::Fetal => 1.0,       // Conservative fetal limit
            Self::Brain => 1.5,       // Transcranial limit
        }
    }

    /// Get cavitation threshold estimate for this tissue
    #[must_use]
    pub fn cavitation_threshold(&self) -> f64 {
        match self {
            Self::SoftTissue => 0.6,
            Self::Ophthalmic => 0.3,
            Self::Lung => 0.4,   // Lower due to gas bodies
            Self::Bowel => 0.4,  // Lower due to gas bodies
            Self::Fetal => 0.5,  // Conservative estimate
            Self::Brain => 0.55, // Slightly lower than soft tissue
        }
    }
}

/// Mechanical Index calculation result
#[derive(Debug, Clone)]
pub struct MechanicalIndexResult {
    /// Calculated MI value
    pub mi: f64,
    /// Peak rarefactional pressure (MPa)
    pub peak_rarefactional_pressure_mpa: f64,
    /// Center frequency (MHz)
    pub center_frequency_mhz: f64,
    /// Safety assessment
    pub safety_status: SafetyStatus,
    /// Distance from transducer where MI is calculated (cm)
    pub focal_distance_cm: f64,
    /// Tissue-specific safety limit
    pub safety_limit: f64,
}

/// Safety status based on MI value
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SafetyStatus {
    /// MI well below safety limits
    Safe,
    /// MI approaching safety limits (>80% of limit)
    Caution,
    /// MI exceeds recommended limits
    Unsafe,
    /// MI at cavitation threshold
    CavitationRisk,
}

impl MechanicalIndexCalculator {
    /// Create a new mechanical index calculator
    ///
    /// # Arguments
    ///
    /// * `center_frequency_mhz` - Center frequency in MHz
    /// * `attenuation_coeff` - Attenuation coefficient in dB/cm/MHz (typical: 0.3-0.6)
    /// * `tissue_type` - Type of tissue for safety limit selection
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::clinical::safety::mechanical_index::{MechanicalIndexCalculator, TissueType};
    ///
    /// let mi_calc = MechanicalIndexCalculator::new(5.0, 0.5, TissueType::SoftTissue);
    /// ```
    #[must_use]
    pub fn new(center_frequency_mhz: f64, attenuation_coeff: f64, tissue_type: TissueType) -> Self {
        Self {
            center_frequency_mhz,
            attenuation_coeff,
            tissue_type,
        }
    }

    /// Calculate mechanical index from pressure field
    ///
    /// # Arguments
    ///
    /// * `pressure_field` - 3D pressure field in Pascals
    /// * `focal_distance_cm` - Distance from transducer in cm
    ///
    /// # Returns
    ///
    /// MI calculation result with safety assessment
    ///
    /// # Example
    ///
    /// ```
    /// use kwavers::clinical::safety::mechanical_index::{MechanicalIndexCalculator, TissueType};
    /// use ndarray::Array3;
    ///
    /// let mi_calc = MechanicalIndexCalculator::new(5.0, 0.5, TissueType::SoftTissue);
    /// let pressure = Array3::from_elem((10, 10, 10), 1e5); // 0.1 MPa
    /// let result = mi_calc.calculate(&pressure, 5.0).unwrap();
    /// assert!(result.mi > 0.0);
    /// ```
    pub fn calculate(
        &self,
        pressure_field: &Array3<f64>,
        focal_distance_cm: f64,
    ) -> KwaversResult<MechanicalIndexResult> {
        // Find peak negative (rarefactional) pressure
        let peak_rarefactional_pa = pressure_field
            .iter()
            .filter(|&&p| p < 0.0)
            .map(|&p| p.abs())
            .fold(0.0, f64::max);

        if peak_rarefactional_pa == 0.0 {
            return Err(KwaversError::Validation(
                crate::core::error::ValidationError::InvalidValue {
                    parameter: "pressure_field".to_string(),
                    value: 0.0,
                    reason: "No negative pressure found in field".to_string(),
                },
            ));
        }

        // Convert to MPa
        let peak_rarefactional_mpa = peak_rarefactional_pa / 1e6;

        // Apply attenuation for in-situ pressure
        // P_in_situ = P_measured × 10^(-α × f × z / 20)
        let attenuation_factor = 10.0_f64
            .powf(-(self.attenuation_coeff * self.center_frequency_mhz * focal_distance_cm) / 20.0);
        let peak_rarefactional_in_situ = peak_rarefactional_mpa * attenuation_factor;

        // Calculate MI: MI = P_r / √f_c
        let mi = peak_rarefactional_in_situ / self.center_frequency_mhz.sqrt();

        // Assess safety
        let safety_limit = self.tissue_type.safety_limit();
        let cavitation_threshold = self.tissue_type.cavitation_threshold();

        let safety_status = if mi > safety_limit {
            SafetyStatus::Unsafe
        } else if mi >= cavitation_threshold {
            SafetyStatus::CavitationRisk
        } else if mi > safety_limit * 0.8 {
            SafetyStatus::Caution
        } else {
            SafetyStatus::Safe
        };

        Ok(MechanicalIndexResult {
            mi,
            peak_rarefactional_pressure_mpa: peak_rarefactional_in_situ,
            center_frequency_mhz: self.center_frequency_mhz,
            safety_status,
            focal_distance_cm,
            safety_limit,
        })
    }

    /// Calculate MI at multiple focal depths
    ///
    /// Useful for assessing safety along the entire beam path
    ///
    /// # Arguments
    ///
    /// * `pressure_field` - 3D pressure field in Pascals
    /// * `focal_distances_cm` - Array of distances from transducer in cm
    ///
    /// # Returns
    ///
    /// Vector of MI results at each focal distance
    pub fn calculate_depth_profile(
        &self,
        pressure_field: &Array3<f64>,
        focal_distances_cm: &[f64],
    ) -> KwaversResult<Vec<MechanicalIndexResult>> {
        focal_distances_cm
            .iter()
            .map(|&distance| self.calculate(pressure_field, distance))
            .collect()
    }

    /// Calculate maximum MI along beam axis
    ///
    /// Returns the worst-case MI value for safety assessment
    pub fn calculate_max_mi(
        &self,
        pressure_field: &Array3<f64>,
        max_depth_cm: f64,
        num_points: usize,
    ) -> KwaversResult<MechanicalIndexResult> {
        let depths: Vec<f64> = (0..num_points)
            .map(|i| (i as f64) * max_depth_cm / (num_points as f64 - 1.0))
            .collect();

        let results = self.calculate_depth_profile(pressure_field, &depths)?;

        results
            .into_iter()
            .max_by(|a, b| a.mi.partial_cmp(&b.mi).unwrap())
            .ok_or_else(|| {
                KwaversError::System(crate::core::error::SystemError::InvalidOperation {
                    operation: "calculate_max_mi".to_string(),
                    reason: "No valid MI values computed".to_string(),
                })
            })
    }
}

impl MechanicalIndexResult {
    /// Check if MI is within safety limits
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self.safety_status, SafetyStatus::Safe)
    }

    /// Get safety margin as percentage below limit
    #[must_use]
    pub fn safety_margin_percent(&self) -> f64 {
        ((self.safety_limit - self.mi) / self.safety_limit) * 100.0
    }

    /// Format result for display
    #[must_use]
    pub fn format_report(&self) -> String {
        format!(
            "Mechanical Index Report\n\
             ========================\n\
             MI Value: {:.3}\n\
             Peak Rarefactional Pressure: {:.3} MPa\n\
             Center Frequency: {:.2} MHz\n\
             Focal Distance: {:.1} cm\n\
             Safety Limit: {:.2}\n\
             Safety Status: {:?}\n\
             Safety Margin: {:.1}%\n",
            self.mi,
            self.peak_rarefactional_pressure_mpa,
            self.center_frequency_mhz,
            self.focal_distance_cm,
            self.safety_limit,
            self.safety_status,
            self.safety_margin_percent()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mi_calculation_soft_tissue() {
        let mi_calc = MechanicalIndexCalculator::new(5.0, 0.5, TissueType::SoftTissue);

        // Create pressure field with 1 MPa peak negative pressure
        let mut pressure = Array3::zeros((10, 10, 10));
        pressure[[5, 5, 5]] = -1e6; // -1 MPa in Pascals

        let result = mi_calc.calculate(&pressure, 5.0).unwrap();

        let expected_mi = (1.0 * 10.0_f64.powf(-(0.5 * 5.0 * 5.0) / 20.0)) / 5.0_f64.sqrt();
        assert!((result.mi - expected_mi).abs() < 1e-6);
        assert_eq!(result.center_frequency_mhz, 5.0);
        assert!(result.is_safe()); // Well below 1.9 limit
    }

    #[test]
    fn test_mi_safety_limits() {
        assert_eq!(TissueType::SoftTissue.safety_limit(), 1.9);
        assert_eq!(TissueType::Ophthalmic.safety_limit(), 0.23);
        assert_eq!(TissueType::Lung.safety_limit(), 0.7);
    }

    #[test]
    fn test_mi_safety_status() {
        let mi_calc = MechanicalIndexCalculator::new(1.0, 0.3, TissueType::Ophthalmic);

        // Create field that exceeds ophthalmic limit (0.23)
        let mut pressure = Array3::zeros((10, 10, 10));
        pressure[[5, 5, 5]] = -0.5e6; // -0.5 MPa

        let result = mi_calc.calculate(&pressure, 3.0).unwrap();

        // MI = 0.5 × 10^(-0.3*1*3/20) / sqrt(1.0), exceeds 0.23 limit
        assert!(result.mi > 0.23);
        assert_eq!(result.safety_status, SafetyStatus::Unsafe);
    }

    #[test]
    fn test_mi_depth_profile() {
        let mi_calc = MechanicalIndexCalculator::new(3.0, 0.5, TissueType::Brain);

        let mut pressure = Array3::zeros((10, 10, 10));
        pressure[[5, 5, 5]] = -0.8e6; // -0.8 MPa

        let depths = vec![1.0, 3.0, 5.0, 7.0];
        let results = mi_calc.calculate_depth_profile(&pressure, &depths).unwrap();

        assert_eq!(results.len(), 4);
        // MI should decrease with depth due to attenuation
        assert!(results[3].mi < results[0].mi);
    }

    #[test]
    fn test_cavitation_threshold() {
        assert!(TissueType::SoftTissue.cavitation_threshold() > 0.4);
        assert!(TissueType::Lung.cavitation_threshold() < 0.5); // Lower for gas-body tissue
    }

    #[test]
    fn test_mi_report_format() {
        let result = MechanicalIndexResult {
            mi: 0.8,
            peak_rarefactional_pressure_mpa: 1.5,
            center_frequency_mhz: 3.5,
            safety_status: SafetyStatus::Safe,
            focal_distance_cm: 5.0,
            safety_limit: 1.9,
        };

        let report = result.format_report();
        assert!(report.contains("MI Value: 0.800"));
        assert!(report.contains("Safety Status: Safe"));
    }
}
