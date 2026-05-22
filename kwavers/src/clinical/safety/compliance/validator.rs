use std::collections::VecDeque;
use std::time::Instant;

use crate::clinical::therapy::parameters::ClinicalTherapyParameters;
use crate::core::error::{KwaversError, KwaversResult};

use super::{
    ComplianceAudit, ComplianceCheck, ComplianceConfig, ComplianceStatus,
    EnhancedComplianceValidator, SafetyComplianceReport, SessionMetrics,
};

impl EnhancedComplianceValidator {
    /// New.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn new(config: ComplianceConfig) -> KwaversResult<Self> {
        config.validate()?;

        Ok(Self {
            config,
            audit_trail: VecDeque::with_capacity(100),
            session_start: None,
            accumulated_time: 0.0,
            accumulated_dose: 0.0,
        })
    }
    /// Audit parameters.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn audit_parameters(
        &mut self,
        params: &ClinicalTherapyParameters,
    ) -> KwaversResult<ComplianceAudit> {
        let mut checks = Vec::new();
        let mut alerts = Vec::new();

        // Frequency range check
        let freq_warning = self.config.frequency_range.1 * 0.8;
        checks.push(ComplianceCheck::new(
            "Frequency Range".to_owned(),
            params.frequency,
            self.config.frequency_range.1,
            "Hz".to_owned(),
            freq_warning,
        ));

        if params.frequency < self.config.frequency_range.0
            || params.frequency > self.config.frequency_range.1
        {
            alerts.push(format!(
                "Frequency {:.2e} Hz outside valid range [{:.2e}, {:.2e}] Hz",
                params.frequency, self.config.frequency_range.0, self.config.frequency_range.1
            ));
        }

        // Mechanical Index check
        let tissue_mi_limit = self.config.tissue_type.safety_limit();
        let estimated_mi = params.mechanical_index;

        checks.push(ComplianceCheck::new(
            "Mechanical Index".to_owned(),
            estimated_mi,
            tissue_mi_limit,
            "MI".to_owned(),
            tissue_mi_limit * 0.8,
        ));

        if estimated_mi > tissue_mi_limit {
            alerts.push(format!(
                "Mechanical Index {:.2} exceeds tissue limit {:.2}",
                estimated_mi, tissue_mi_limit
            ));
        }

        // Duty cycle validation
        if params.duty_cycle < 0.0 || params.duty_cycle > 1.0 {
            alerts.push(format!(
                "Invalid duty cycle: {:.2}. Must be 0-1",
                params.duty_cycle
            ));
        }

        // PRF validation
        if params.prf <= 0.0 {
            alerts.push(format!(
                "Invalid pulse repetition frequency: {:.2} Hz",
                params.prf
            ));
        }

        // Temperature rise check
        let estimated_temp_rise = self.estimate_temperature_rise(params);
        checks.push(ComplianceCheck::new(
            "Maximum Temperature Rise".to_owned(),
            estimated_temp_rise,
            self.config.max_temp_rise,
            "°C".to_owned(),
            self.config.max_temp_rise * 0.8,
        ));

        if estimated_temp_rise > self.config.max_temp_rise {
            alerts.push(format!(
                "Estimated temperature rise {:.1}°C exceeds limit {:.1}°C",
                estimated_temp_rise, self.config.max_temp_rise
            ));
        }

        // Session duration check
        checks.push(ComplianceCheck::new(
            "Session Duration".to_owned(),
            params.duration,
            self.config.max_session_time,
            "s".to_owned(),
            self.config.max_session_time * 0.8,
        ));

        let overall_status = if checks
            .iter()
            .any(|c| c.status == ComplianceStatus::NonCompliant)
        {
            ComplianceStatus::NonCompliant
        } else if checks.iter().any(|c| c.status == ComplianceStatus::Warning) {
            ComplianceStatus::Warning
        } else {
            ComplianceStatus::Compliant
        };

        let audit = ComplianceAudit {
            timestamp: Instant::now(),
            checks,
            overall_status,
            alerts,
        };

        self.audit_trail.push_back(audit.clone());
        if self.audit_trail.len() > 1000 {
            self.audit_trail.pop_front();
        }

        Ok(audit)
    }

    /// Estimate tissue temperature rise from acoustic therapy parameters.
    ///
    /// Derived from Pennes bioheat equation (worst-case, no perfusion):
    /// `ΔT = Q · t_eff / (ρ · c_p)` where `Q = 2α · I_SPTA` [W/m³].
    /// Reference: Nyborg WL (1988), *Phys. Med. Biol.* 33(7):785–792.
    fn estimate_temperature_rise(&self, params: &ClinicalTherapyParameters) -> f64 {
        use crate::core::constants::fundamental::{
            DENSITY_BLOOD, DENSITY_WATER_NOMINAL as RHO_W, SOUND_SPEED_WATER,
        };

        // IEC 62127 tissue model uses 1060 kg/m³; equals SSOT DENSITY_BLOOD by value.
        const TISSUE_DENSITY: f64 = DENSITY_BLOOD;
        const TISSUE_HEAT_CAPACITY: f64 = 3500.0;

        // IEC 62127 absorption model: α [Np/m] = 0.3 dB/cm/MHz × f_MHz × 100/8.686
        let f_mhz = (params.frequency / 1e6).max(1e-3);
        let alpha_np_per_m = 0.3 * f_mhz * 100.0 / 8.686;

        let p_rms = params.pressure / std::f64::consts::SQRT_2;
        let i_spta = (p_rms * p_rms) / (RHO_W * SOUND_SPEED_WATER);

        let q_vol = 2.0 * alpha_np_per_m * i_spta;
        let t_eff = params.treatment_duration * params.duty_cycle;

        let delta_t = q_vol * t_eff / (TISSUE_DENSITY * TISSUE_HEAT_CAPACITY);
        delta_t.min(self.config.max_temp_rise * 2.0)
    }
    /// Start session.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn start_session(&mut self) {
        self.session_start = Some(Instant::now());
        self.accumulated_time = 0.0;
    }
    /// End session.
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    ///
    pub fn end_session(&mut self) -> KwaversResult<SessionMetrics> {
        let elapsed = if let Some(start) = self.session_start {
            start.elapsed().as_secs_f64()
        } else {
            return Err(KwaversError::InvalidInput(
                "No session in progress".to_owned(),
            ));
        };

        self.accumulated_time += elapsed;
        self.session_start = None;

        Ok(SessionMetrics {
            session_duration: elapsed,
            accumulated_time: self.accumulated_time,
            accumulated_dose: self.accumulated_dose,
            session_compliant: true,
        })
    }

    #[must_use]
    pub fn audit_history(&self) -> Vec<ComplianceAudit> {
        self.audit_trail.iter().cloned().collect()
    }

    #[must_use]
    pub fn latest_audit(&self) -> Option<&ComplianceAudit> {
        self.audit_trail.back()
    }

    #[must_use]
    pub fn generate_report(&self) -> SafetyComplianceReport {
        let total_audits = self.audit_trail.len();
        let compliant_audits = self
            .audit_trail
            .iter()
            .filter(|a| a.overall_status == ComplianceStatus::Compliant)
            .count();

        let warning_audits = self
            .audit_trail
            .iter()
            .filter(|a| a.overall_status == ComplianceStatus::Warning)
            .count();

        let non_compliant_audits = total_audits - compliant_audits - warning_audits;

        let compliance_percentage = if total_audits > 0 {
            (compliant_audits as f64 / total_audits as f64) * 100.0
        } else {
            100.0
        };

        SafetyComplianceReport {
            total_audits,
            compliant_audits,
            warning_audits,
            non_compliant_audits,
            compliance_percentage,
            system_status: if non_compliant_audits > 0 {
                "UNSAFE - Non-compliant audits detected".to_owned()
            } else if warning_audits > 0 {
                "CAUTION - Warnings detected".to_owned()
            } else {
                "SAFE - All audits compliant".to_owned()
            },
        }
    }

    pub fn clear_history(&mut self) {
        self.audit_trail.clear();
    }
}
