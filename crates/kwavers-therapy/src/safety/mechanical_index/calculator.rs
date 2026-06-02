//! MechanicalIndexCalculator implementation.

use kwavers_core::constants::numerical::MPA_TO_PA;
use kwavers_core::error::{KwaversError, KwaversResult};
use ndarray::Array3;

use super::types::{MechanicalIndexResult, MechanicalIndexSafetyStatus, MechanicalIndexTissueType};

/// Mechanical Index calculator for ultrasound safety assessment.
#[derive(Debug, Clone)]
pub struct MechanicalIndexCalculator {
    /// Center frequency in MHz
    center_frequency_mhz: f64,
    /// Attenuation coefficient in dB/cm/MHz
    attenuation_coeff: f64,
    /// Tissue type for safety limit selection
    tissue_type: MechanicalIndexTissueType,
}

impl MechanicalIndexCalculator {
    fn invalid_value(parameter: &str, value: f64, reason: &str) -> KwaversError {
        KwaversError::Validation(kwavers_core::error::ValidationError::InvalidValue {
            parameter: parameter.to_owned(),
            value,
            reason: reason.to_owned(),
        })
    }

    fn validate_calculation_domain(&self, focal_distance_cm: f64) -> KwaversResult<()> {
        if !self.center_frequency_mhz.is_finite() || self.center_frequency_mhz <= 0.0 {
            return Err(Self::invalid_value(
                "center_frequency_mhz",
                self.center_frequency_mhz,
                "Mechanical index denominator sqrt(f_c) requires finite positive MHz frequency",
            ));
        }

        if !self.attenuation_coeff.is_finite() || self.attenuation_coeff < 0.0 {
            return Err(Self::invalid_value(
                "attenuation_coeff",
                self.attenuation_coeff,
                "Attenuation coefficient must be finite and nonnegative in dB/cm/MHz",
            ));
        }

        if !focal_distance_cm.is_finite() || focal_distance_cm < 0.0 {
            return Err(Self::invalid_value(
                "focal_distance_cm",
                focal_distance_cm,
                "Focal distance must be finite and nonnegative in centimeters",
            ));
        }

        Ok(())
    }

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
    /// use kwavers_therapy::safety::mechanical_index::{MechanicalIndexCalculator, MechanicalIndexTissueType};
    ///
    /// let mi_calc = MechanicalIndexCalculator::new(5.0, 0.5, MechanicalIndexTissueType::SoftTissue);
    /// ```
    #[must_use]
    pub fn new(
        center_frequency_mhz: f64,
        attenuation_coeff: f64,
        tissue_type: MechanicalIndexTissueType,
    ) -> Self {
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
    /// use kwavers_therapy::safety::mechanical_index::{MechanicalIndexCalculator, MechanicalIndexTissueType};
    /// use ndarray::Array3;
    ///
    /// let mi_calc = MechanicalIndexCalculator::new(5.0, 0.5, MechanicalIndexTissueType::SoftTissue);
    /// let pressure = Array3::from_elem((10, 10, 10), -1e5);
    /// let result = mi_calc.calculate(&pressure, 5.0).unwrap();
    /// assert!(result.mi > 0.0);
    /// ```
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn calculate(
        &self,
        pressure_field: &Array3<f64>,
        focal_distance_cm: f64,
    ) -> KwaversResult<MechanicalIndexResult> {
        self.validate_calculation_domain(focal_distance_cm)?;

        // Find peak negative (rarefactional) pressure
        let peak_rarefactional_pa = pressure_field
            .iter()
            .filter(|&&p| p < 0.0)
            .map(|&p| p.abs())
            .fold(0.0, f64::max);

        if peak_rarefactional_pa == 0.0 {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::InvalidValue {
                    parameter: "pressure_field".to_owned(),
                    value: 0.0,
                    reason: "No negative pressure found in field".to_owned(),
                },
            ));
        }

        // Convert to MPa (1 MPa = MPA_TO_PA Pa)
        let peak_rarefactional_mpa = peak_rarefactional_pa / MPA_TO_PA;

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
            MechanicalIndexSafetyStatus::Unsafe
        } else if mi >= cavitation_threshold {
            MechanicalIndexSafetyStatus::CavitationRisk
        } else if mi > safety_limit * 0.8 {
            MechanicalIndexSafetyStatus::Caution
        } else {
            MechanicalIndexSafetyStatus::Safe
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn calculate_max_mi(
        &self,
        pressure_field: &Array3<f64>,
        max_depth_cm: f64,
        num_points: usize,
    ) -> KwaversResult<MechanicalIndexResult> {
        if num_points < 2 {
            return Err(Self::invalid_value(
                "num_points",
                num_points as f64,
                "At least two depth samples are required to define a closed [0, max_depth] profile",
            ));
        }

        if !max_depth_cm.is_finite() || max_depth_cm < 0.0 {
            return Err(Self::invalid_value(
                "max_depth_cm",
                max_depth_cm,
                "Maximum depth must be finite and nonnegative in centimeters",
            ));
        }

        let depths: Vec<f64> = (0..num_points)
            .map(|i| (i as f64) * max_depth_cm / (num_points as f64 - 1.0))
            .collect();

        let results = self.calculate_depth_profile(pressure_field, &depths)?;

        results
            .into_iter()
            // NaN-safe: treat NaN as equal so the comparison never panics.
            .max_by(|a, b| a.mi.total_cmp(&b.mi))
            .ok_or_else(|| {
                KwaversError::System(kwavers_core::error::SystemError::InvalidOperation {
                    operation: "calculate_max_mi".to_owned(),
                    reason: "No valid MI values computed".to_owned(),
                })
            })
    }
}
