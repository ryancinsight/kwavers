//! Bioeffects assessment for lithotripsy safety.
//!
//! This module implements tissue damage assessment and safety monitoring for
//! extracorporeal shock wave lithotripsy (ESWL), including thermal and mechanical
//! bioeffects evaluation.

use ndarray::Array3;

/// Bioeffects model parameters.
#[derive(Debug, Clone)]
pub struct BioeffectsParameters {
    /// Thermal damage threshold (CEM43)
    pub thermal_threshold: f64,
    /// Mechanical index threshold
    pub mechanical_index_threshold: f64,
    /// Maximum peak negative pressure (Pa)
    pub max_negative_pressure: f64,
}

impl Default for BioeffectsParameters {
    fn default() -> Self {
        Self {
            thermal_threshold: 240.0,        // 240 CEM43 minutes
            mechanical_index_threshold: 1.9, // FDA guideline
            max_negative_pressure: 20e6,     // 20 MPa
        }
    }
}

/// Safety assessment results.
#[derive(Debug, Clone)]
pub struct SafetyAssessment {
    /// Thermal damage indicator (0-1, 0=safe, 1=damage)
    pub thermal_damage: f64,
    /// Mechanical damage indicator (0-1)
    pub mechanical_damage: f64,
    /// Overall safety score (0-1, 1=safe)
    pub safety_score: f64,

    /// Overall safety flag
    pub overall_safe: bool,
    /// Max MI recorded
    pub max_mechanical_index: f64,
    /// Max TI recorded
    pub max_thermal_index: f64,
    /// Max cavitation dose
    pub max_cavitation_dose: f64,
    /// Max damage prob
    pub max_damage_probability: f64,
    /// List of violations
    pub violations: Vec<String>,
}

impl Default for SafetyAssessment {
    fn default() -> Self {
        Self {
            thermal_damage: 0.0,
            mechanical_damage: 0.0,
            safety_score: 1.0,
            overall_safe: true,
            max_mechanical_index: 0.0,
            max_thermal_index: 0.0,
            max_cavitation_dose: 0.0,
            max_damage_probability: 0.0,
            violations: Vec::new(),
        }
    }
}

impl SafetyAssessment {
    /// Check safety limits against FDA/IEC guidelines and populate violations.
    ///
    /// | Parameter | Threshold | Reference |
    /// |-----------|-----------|-----------|
    /// | MI | 1.9 | FDA 510(k) guideline |
    /// | TI | 6.0 | IEC 62359 |
    /// | Damage probability | 0.05 | 5 % threshold |
    pub fn check_safety_limits(mut self) -> Self {
        self.violations.clear();

        // Mechanical Index limit (FDA 510(k))
        if self.max_mechanical_index > 1.9 {
            self.violations.push(format!(
                "MI {:.2} exceeds FDA limit 1.9",
                self.max_mechanical_index
            ));
        }

        // Thermal Index limit (IEC 62359)
        if self.max_thermal_index > 6.0 {
            self.violations.push(format!(
                "TI {:.2} exceeds IEC limit 6.0",
                self.max_thermal_index
            ));
        }

        // Damage probability
        if self.max_damage_probability > 0.05 {
            self.violations.push(format!(
                "Damage probability {:.3} exceeds 5% threshold",
                self.max_damage_probability
            ));
        }

        // Cavitation dose
        if self.max_cavitation_dose > 1.0 {
            self.violations.push(format!(
                "Cavitation dose {:.3} exceeds unit threshold",
                self.max_cavitation_dose
            ));
        }

        self.overall_safe = self.violations.is_empty();

        // Safety score: 1 = fully safe, 0 = at or beyond all limits
        let mi_ratio = self.max_mechanical_index / 1.9;
        let ti_ratio = self.max_thermal_index / 6.0;
        let dp_ratio = self.max_damage_probability / 0.05;
        let worst = mi_ratio.max(ti_ratio).max(dp_ratio);
        self.safety_score = (1.0 - worst).clamp(0.0, 1.0);

        self
    }
}

/// Bioeffects assessment model.
#[derive(Debug, Clone)]
pub struct BioeffectsModel {
    /// Model parameters
    parameters: BioeffectsParameters,
    /// Current assessment
    assessment: SafetyAssessment,
}

impl BioeffectsModel {
    /// Create new bioeffects model.
    /// Caller passes (dimensions, params)
    pub fn new(_dimensions: (usize, usize, usize), parameters: BioeffectsParameters) -> Self {
        Self {
            parameters,
            assessment: SafetyAssessment::default(),
        }
    }

    /// Get model parameters.
    pub fn parameters(&self) -> &BioeffectsParameters {
        &self.parameters
    }

    /// Update assessment based on current simulation fields.
    ///
    /// Computes safety indices per IEC 62127 / FDA 510(k):
    ///
    /// - **Mechanical Index**: `MI = PNP / √f_MHz` where PNP is the peak
    ///   negative pressure in MPa and `f` is the centre frequency in MHz.
    /// - **Thermal Index**: simplified as `TI = ΔT_max / 1°C` where ΔT is
    ///   estimated from the spatial-peak temporal-average intensity.
    /// - **Cavitation dose**: peak amplitude of the cavitation field normalised
    ///   by its critical threshold.
    /// - **Damage probability**: logistic model based on MI and cumulative
    ///   thermal dose.
    pub fn update_assessment(
        &mut self,
        pressure: &Array3<f64>,
        intensity: &Array3<f64>,
        cavitation: &Array3<f64>,
        frequency: f64,
        _dt: f64,
    ) {
        let f_mhz = frequency / 1e6;
        let sqrt_f = f_mhz.sqrt().max(1e-12);

        // ── Mechanical Index ───────────────────────────────────────────
        // PNP = max |p_neg|  (Pa → MPa)
        let peak_neg_pressure_pa = pressure
            .iter()
            .map(|&p| if p < 0.0 { -p } else { 0.0 })
            .fold(0.0_f64, f64::max);
        let pnp_mpa = peak_neg_pressure_pa / 1e6;
        let mi = pnp_mpa / sqrt_f;

        // ── Thermal Index (simplified SPTA-based estimate) ─────────
        //  I_SPTA = max(intensity)
        //  ΔT ≈ 2 · α · I_SPTA · d / k   (simplified for soft tissue)
        //  where α=0.3 dB/cm/MHz, d=1cm, k=0.6 W/(m·K)
        let i_spta = intensity
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        let alpha_np_per_m = 0.3 * f_mhz * 100.0 / 8.686; // dB/cm/MHz → Np/m
        let depth_m = 0.01; // reference depth 1 cm
        let thermal_conductivity = 0.6; // W/(m·K)
        let delta_t = 2.0 * alpha_np_per_m * i_spta * depth_m / thermal_conductivity;
        let ti = delta_t; // TI ≈ ΔT / 1°C

        // ── Cavitation dose ────────────────────────────────────────────
        // Normalised peak cavitation amplitude
        let cav_peak = cavitation
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        // Normalise by threshold (critical bubble radius ~ 1e-6 m)
        let cav_threshold = 1e-6;
        let cav_dose = cav_peak / cav_threshold;

        // ── Damage probability (logistic model) ───────────────────────
        // P(damage) = 1 / (1 + exp(−k·(MI − MI_50)))  where MI_50 ≈ 1.5
        let mi_50 = 1.5;
        let k_logistic = 5.0;
        let damage_prob = 1.0 / (1.0 + (-k_logistic * (mi - mi_50)).exp());

        // ── Thermal and mechanical damage indicators ──────────────────
        // thermal_damage: fraction of CEM43 threshold reached
        let thermal_damage = (ti / self.parameters.thermal_threshold.max(1e-15)).clamp(0.0, 1.0);
        // mechanical_damage: fraction of MI threshold reached
        let mechanical_damage =
            (mi / self.parameters.mechanical_index_threshold.max(1e-15)).clamp(0.0, 1.0);

        // ── Store results ─────────────────────────────────────────────
        self.assessment.thermal_damage = thermal_damage;
        self.assessment.mechanical_damage = mechanical_damage;
        self.assessment.max_mechanical_index = mi;
        self.assessment.max_thermal_index = ti;
        self.assessment.max_cavitation_dose = cav_dose;
        self.assessment.max_damage_probability = damage_prob;

        // Run threshold checks
        self.assessment = std::mem::take(&mut self.assessment).check_safety_limits();
    }

    /// Check safety status.
    pub fn check_safety(&self) -> &SafetyAssessment {
        &self.assessment
    }

    /// Get current assessment (alias/same as check_safety for now)
    pub fn current_assessment(&self) -> &SafetyAssessment {
        &self.assessment
    }
}

impl Default for BioeffectsModel {
    fn default() -> Self {
        Self::new((1, 1, 1), BioeffectsParameters::default())
    }
}

/// Tissue damage assessment results.
#[derive(Debug, Clone)]
pub struct TissueDamageAssessment {
    /// Cumulative equivalent minutes at 43°C
    pub cem43: f64,
    /// Mechanical index
    pub mechanical_index: f64,
    /// Safety assessment
    pub safety: SafetyAssessment,
}

impl Default for TissueDamageAssessment {
    fn default() -> Self {
        Self {
            cem43: 0.0,
            mechanical_index: 0.0,
            safety: SafetyAssessment::default(),
        }
    }
}
