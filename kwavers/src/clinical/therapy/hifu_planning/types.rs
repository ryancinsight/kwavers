use crate::clinical::safety::mechanical_index::MechanicalIndexTissueType;
use crate::clinical::therapy::parameters::ClinicalTherapyParameters;
use crate::core::constants::fundamental::{
    ACOUSTIC_ABSORPTION_TISSUE, DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM,
};
use crate::core::constants::medical::{
    THERMAL_DOSE_REFERENCE_TEMP_C, THERMAL_DOSE_R_ABOVE_43C, THERMAL_DOSE_R_BELOW_43C,
};
use crate::core::constants::thermodynamic::{BODY_TEMPERATURE_C, SPECIFIC_HEAT_TISSUE};
use crate::core::constants::MHZ_TO_HZ;
use crate::core::error::{KwaversError, KwaversResult};
use crate::physics::acoustics::analysis::calculate_mechanical_index;
use std::f64::consts::PI;

/// HIFU transducer configuration.
#[derive(Debug, Clone)]
pub struct ClinicalHIFUTransducer {
    pub frequency: f64,
    pub focal_length_mm: f64,
    pub aperture_diameter_mm: f64,
    pub power: f64,
    pub efficiency: f64,
    pub transducer_type: String,
    pub transducer_diameter_mm: f64,
}

impl Default for ClinicalHIFUTransducer {
    fn default() -> Self {
        Self {
            frequency: 1.5e6,
            focal_length_mm: 80.0,
            aperture_diameter_mm: 40.0,
            power: 50.0,
            efficiency: 0.8,
            transducer_type: "focused".to_owned(),
            transducer_diameter_mm: 40.0,
        }
    }
}

/// HIFU focal spot characteristics.
#[derive(Debug, Clone)]
pub struct FocalSpot {
    pub location_mm: (f64, f64, f64),
    /// Lateral FWHM (mm): 1.02·λ·F# (O'Neil 1949)
    pub lateral_width_mm: f64,
    /// Axial FWHM (mm): (8/π)·λ·F#² (Gaussian beam approximation)
    pub axial_width_mm: f64,
    pub peak_pressure_pa: f64,
    pub mechanical_index: f64,
    pub focal_volume_mm3: f64,
    pub volume_minus6db_mm3: f64,
}

impl FocalSpot {
    /// Estimate the FWHM ellipsoid and focal pressure from a focused aperture.
    ///
    /// Theorem: for a circular focused source with f-number `F# = F/D`,
    /// O'Neil's harmonic piston approximation gives
    /// `FWHM_lat = 1.02 lambda F#` and the Gaussian axial approximation gives
    /// `FWHM_ax = (8/pi) lambda F#^2`. Acoustic output power is
    /// `P_ac = P_drive eta`; the harmonic peak pressure follows
    /// `p = sqrt(2 rho c P_ac / A_f)`. No empirical pressure ceiling is applied.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when the transducer frequency,
    /// focal length, aperture, power, or efficiency is outside its physical
    /// domain.
    pub fn estimate_from_transducer(transducer: &ClinicalHIFUTransducer) -> KwaversResult<Self> {
        validate_positive_finite("transducer.frequency", transducer.frequency)?;
        validate_positive_finite("transducer.focal_length_mm", transducer.focal_length_mm)?;
        validate_positive_finite(
            "transducer.aperture_diameter_mm",
            transducer.aperture_diameter_mm,
        )?;
        validate_nonnegative_finite("transducer.power", transducer.power)?;
        validate_unit_interval("transducer.efficiency", transducer.efficiency)?;

        let wavelength = SOUND_SPEED_WATER_SIM / transducer.frequency;
        let f_number = transducer.focal_length_mm / transducer.aperture_diameter_mm;
        let lateral_width_mm = 1.02 * wavelength * 1e3 * f_number;
        let axial_width_mm = (8.0 / PI) * f_number * f_number * wavelength * 1e3;
        let lateral_radius = lateral_width_mm / 2.0;
        let focal_area_mm2 = PI * lateral_radius * lateral_radius;
        let acoustic_power_w = transducer.power * transducer.efficiency;
        let intensity_w_mm2 = acoustic_power_w / focal_area_mm2;
        let intensity_w_m2 = intensity_w_mm2 * 1e6;
        let peak_pressure_pa =
            (2.0 * DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM * intensity_w_m2).sqrt();
        let mechanical_index = calculate_mechanical_index(peak_pressure_pa, transducer.frequency);
        let lateral_semi = lateral_width_mm / 2.0;
        let axial_semi = axial_width_mm / 2.0;
        let focal_volume_mm3 = (4.0 / 3.0) * PI * lateral_semi * lateral_semi * axial_semi;
        let volume_minus6db_mm3 = focal_volume_mm3 * 0.7;
        Ok(Self {
            location_mm: (0.0, 0.0, transducer.focal_length_mm),
            lateral_width_mm,
            axial_width_mm,
            peak_pressure_pa,
            mechanical_index,
            focal_volume_mm3,
            volume_minus6db_mm3,
        })
    }

    #[must_use]
    pub fn is_safe(&self, tissue_type: MechanicalIndexTissueType) -> bool {
        self.mechanical_index < tissue_type.safety_limit()
    }

    #[must_use]
    pub fn focal_volume_cm3(&self) -> f64 {
        self.focal_volume_mm3 / 1000.0
    }
    #[must_use]
    pub fn volume_minus6db_cm3(&self) -> f64 {
        self.volume_minus6db_mm3 / 1000.0
    }
}

/// Target volume for HIFU ablation.
#[derive(Debug, Clone)]
pub struct AblationTarget {
    pub name: String,
    pub location_mm: (f64, f64, f64),
    pub dimensions_mm: (f64, f64, f64),
    pub safety_margin_mm: f64,
    pub tissue_type: MechanicalIndexTissueType,
}

impl AblationTarget {
    #[must_use]
    pub fn new(
        name: String,
        location_mm: (f64, f64, f64),
        dimensions_mm: (f64, f64, f64),
        tissue_type: MechanicalIndexTissueType,
    ) -> Self {
        Self {
            name,
            location_mm,
            dimensions_mm,
            safety_margin_mm: 2.0,
            tissue_type,
        }
    }

    #[must_use]
    pub fn with_safety_margin(mut self, margin_mm: f64) -> Self {
        self.safety_margin_mm = margin_mm;
        self
    }

    #[must_use]
    pub fn volume_mm3(&self) -> f64 {
        self.dimensions_mm.0 * self.dimensions_mm.1 * self.dimensions_mm.2
    }

    #[must_use]
    pub fn volume_with_margin_mm3(&self) -> f64 {
        let (m, d) = (self.safety_margin_mm, self.dimensions_mm);
        2.0f64.mul_add(m, d.0) * 2.0f64.mul_add(m, d.1) * 2.0f64.mul_add(m, d.2)
    }

    #[must_use]
    pub fn is_focal_spot_adequate(&self, focal_spot: &FocalSpot) -> bool {
        let min_target_dim = self.dimensions_mm.0.min(self.dimensions_mm.1);
        focal_spot.lateral_width_mm >= min_target_dim * 0.5
    }
}

/// Thermal dose in CEM43: CEM43 = R^(43−T)·t, R=0.5 for T>43°C else 0.25.
#[derive(Debug, Clone)]
pub struct FocalSpotDoseEstimate {
    pub cem43: f64,
    pub peak_temperature_c: f64,
    pub time_to_dose_s: f64,
}

impl FocalSpotDoseEstimate {
    /// Estimate single-focus heating and cumulative equivalent minutes at 43 C.
    ///
    /// The temperature model uses a one-compartment Pennes response with
    /// `DeltaT(t) = (q_dot / (rho c_p perfusion)) (1 - exp(-perfusion t))`.
    /// The dose model follows Sapareto-Dewey:
    /// `CEM43 = t_min R^(43 - T)`, with `R = 0.5` at or above 43 C and
    /// `R = 0.25` below 43 C. The returned `cem43` is in equivalent minutes.
    ///
    /// # Errors
    /// Returns [`KwaversError::InvalidInput`] when the pressure, frequency,
    /// duty cycle, or duration is outside its physical domain.
    pub fn estimate_from_focal_spot(
        focal_spot: &FocalSpot,
        frequency_hz: f64,
        duty_cycle: f64,
        treatment_duration_s: f64,
    ) -> KwaversResult<Self> {
        let specific_heat = SPECIFIC_HEAT_TISSUE;
        const PERFUSION_RATE: f64 = 0.01;
        const SECONDS_PER_MINUTE: f64 = 60.0;
        const ABLATION_DOSE_CEM43_MIN: f64 = 240.0;

        validate_nonnegative_finite("focal_spot.peak_pressure_pa", focal_spot.peak_pressure_pa)?;
        validate_positive_finite("frequency_hz", frequency_hz)?;
        validate_unit_interval("duty_cycle", duty_cycle)?;
        validate_nonnegative_finite("treatment_duration_s", treatment_duration_s)?;

        let frequency_mhz = frequency_hz / MHZ_TO_HZ;
        let alpha_np_per_m =
            ACOUSTIC_ABSORPTION_TISSUE * frequency_mhz * 100.0 * (std::f64::consts::LN_10 / 20.0);
        let intensity_w_m2 = focal_spot.peak_pressure_pa.powi(2)
            / (2.0 * DENSITY_WATER_NOMINAL * SOUND_SPEED_WATER_SIM);
        let heating_w_m3 = 2.0 * alpha_np_per_m * intensity_w_m2 * duty_cycle;
        let heating_rate_c_per_s = heating_w_m3 / (DENSITY_WATER_NOMINAL * specific_heat);
        let delta_t = (heating_rate_c_per_s / PERFUSION_RATE)
            * (1.0 - (-PERFUSION_RATE * treatment_duration_s).exp());
        let peak_temperature_c = BODY_TEMPERATURE_C + delta_t;
        let r: f64 = if peak_temperature_c >= THERMAL_DOSE_REFERENCE_TEMP_C {
            THERMAL_DOSE_R_ABOVE_43C
        } else {
            THERMAL_DOSE_R_BELOW_43C
        };
        let dose_rate_cem43_per_min = r.powf(THERMAL_DOSE_REFERENCE_TEMP_C - peak_temperature_c);
        let cem43 = (treatment_duration_s / SECONDS_PER_MINUTE) * dose_rate_cem43_per_min;
        let time_to_dose_s = if peak_temperature_c >= THERMAL_DOSE_REFERENCE_TEMP_C {
            if dose_rate_cem43_per_min > 0.0 {
                ABLATION_DOSE_CEM43_MIN * SECONDS_PER_MINUTE / dose_rate_cem43_per_min
            } else {
                f64::INFINITY
            }
        } else {
            f64::INFINITY
        };
        Ok(Self {
            cem43,
            peak_temperature_c,
            time_to_dose_s,
        })
    }

    #[must_use]
    pub fn is_sufficient_for_ablation(&self) -> bool {
        self.cem43 >= 240.0
    }
    #[must_use]
    pub fn margin_to_ablation(&self) -> f64 {
        240.0 - self.cem43
    }
}

/// HIFU treatment plan.
#[derive(Debug, Clone)]
pub struct ClinicalHIFUTreatmentPlan {
    pub transducer: ClinicalHIFUTransducer,
    pub focal_spot: FocalSpot,
    pub target: AblationTarget,
    pub therapy_params: ClinicalTherapyParameters,
    pub thermal_dose: FocalSpotDoseEstimate,
    pub feasibility: TreatmentFeasibility,
}

/// Treatment feasibility assessment.
#[derive(Debug, Clone)]
pub struct TreatmentFeasibility {
    pub is_feasible: bool,
    pub focal_coverage_adequate: bool,
    pub mi_within_limits: bool,
    pub thermal_dose_achievable: bool,
    pub access_path_clear: bool,
    pub issues: Vec<String>,
    pub confidence_percent: f64,
}

impl TreatmentFeasibility {
    #[must_use]
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

    pub fn update_feasibility(&mut self) {
        self.is_feasible = self.focal_coverage_adequate
            && self.mi_within_limits
            && self.thermal_dose_achievable
            && self.access_path_clear;
        let num_issues = self.issues.len() as f64;
        self.confidence_percent = 100.0 * 0.1f64.mul_add(-num_issues, 1.0).max(0.0);
    }
}

impl Default for TreatmentFeasibility {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_positive_finite(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and positive, got {value}"
        )))
    }
}

fn validate_nonnegative_finite(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && value >= 0.0 {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and nonnegative, got {value}"
        )))
    }
}

fn validate_unit_interval(name: &str, value: f64) -> KwaversResult<()> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(KwaversError::InvalidInput(format!(
            "{name} must be finite and within [0, 1], got {value}"
        )))
    }
}
