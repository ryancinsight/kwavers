use super::{ClinicalSafetyLevel, ClinicalSafetyLimits};
use kwavers_core::error::KwaversResult;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

/// Predicate evaluating one compliance requirement against a system configuration.
/// `Send + Sync` so the validator registry can be shared across threads.
type ComplianceValidationFn =
    Arc<dyn Fn(&SystemConfiguration) -> KwaversResult<ComplianceResult> + Send + Sync>;

/// IEC 60601-2-37 compliance validator
#[derive(Debug)]
pub struct ComplianceValidator {
    standard_version: String,
    compliance_checks: Vec<ComplianceCheck>,
    validation_results: HashMap<String, ComplianceResult>,
}

impl ComplianceValidator {
    /// Create new compliance validator for IEC 60601-2-37.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new() -> Self {
        Self {
            standard_version: "IEC 60601-2-37:2007+A1:2010".to_owned(),
            compliance_checks: Self::create_compliance_checks(),
            validation_results: HashMap::new(),
        }
    }

    /// Run all compliance checks against `system_config`.
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    pub fn validate_compliance(
        &mut self,
        system_config: &SystemConfiguration,
    ) -> KwaversResult<ValidatorComplianceReport> {
        self.validation_results.clear();

        for check in &self.compliance_checks {
            let result = (check.validation_function)(system_config)?;
            self.validation_results.insert(check.id.clone(), result);
        }

        let report = ValidatorComplianceReport {
            standard_version: self.standard_version.clone(),
            timestamp: Instant::now(),
            results: self.validation_results.clone(),
            overall_compliant: self.is_overall_compliant(),
        };

        Ok(report)
    }

    /// Return `true` if all checks passed.
    #[must_use]
    pub fn is_overall_compliant(&self) -> bool {
        self.validation_results.values().all(|r| r.passed)
    }

    /// Get detailed result for a specific check ID.
    #[must_use]
    pub fn get_check_result(&self, check_id: &str) -> Option<&ComplianceResult> {
        self.validation_results.get(check_id)
    }

    fn create_compliance_checks() -> Vec<ComplianceCheck> {
        vec![
            ComplianceCheck {
                id: "safety_limits".to_owned(),
                name: "Safety Limits Implementation".to_owned(),
                requirement: "Clause 201.12 - Essential performance: acoustic output control"
                    .to_owned(),
                validation_function: Arc::new(|config| {
                    let passed = config.safety_limits.max_intensity <= 3.0
                        && config.safety_limits.max_power <= 100.0
                        && config.safety_limits.max_temperature_rise <= 5.0;
                    Ok(ComplianceResult {
                        passed,
                        details: if passed {
                            "Safety limits meet IEC requirements".to_owned()
                        } else {
                            "Safety limits exceed IEC maximum values".to_owned()
                        },
                        severity: if passed {
                            ClinicalSafetyLevel::Normal
                        } else {
                            ClinicalSafetyLevel::Critical
                        },
                    })
                }),
            },
            ComplianceCheck {
                id: "monitoring".to_owned(),
                name: "Real-time Monitoring".to_owned(),
                requirement: "Clause 201.12.4 - Acoustic output measurement accuracy".to_owned(),
                validation_function: Arc::new(|config| {
                    let passed = config.monitoring_enabled && config.interlocks_enabled;
                    Ok(ComplianceResult {
                        passed,
                        details: if passed {
                            "Real-time monitoring and interlocks enabled".to_owned()
                        } else {
                            "Monitoring or interlocks not properly configured".to_owned()
                        },
                        severity: if passed {
                            ClinicalSafetyLevel::Normal
                        } else {
                            ClinicalSafetyLevel::Critical
                        },
                    })
                }),
            },
            ComplianceCheck {
                id: "emergency_stop".to_owned(),
                name: "Emergency Stop Functionality".to_owned(),
                requirement: "Clause 201.9 - Emergency stop accessible within 1 second".to_owned(),
                validation_function: Arc::new(|config| {
                    let passed = config.emergency_stop_tested && config.interlocks_enabled;
                    Ok(ComplianceResult {
                        passed,
                        details: if passed {
                            "Emergency stop tested and interlocks enabled".to_owned()
                        } else {
                            format!(
                                "Emergency stop: tested={}, interlocks={}",
                                config.emergency_stop_tested, config.interlocks_enabled
                            )
                        },
                        severity: if passed {
                            ClinicalSafetyLevel::Normal
                        } else {
                            ClinicalSafetyLevel::Critical
                        },
                    })
                }),
            },
        ]
    }
}

impl Default for ComplianceValidator {
    fn default() -> Self {
        Self::new()
    }
}

struct ComplianceCheck {
    id: String,
    name: String,
    requirement: String,
    validation_function: ComplianceValidationFn,
}

impl std::fmt::Debug for ComplianceCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplianceCheck")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("requirement", &self.requirement)
            .finish()
    }
}

/// Result of a single compliance check
#[derive(Clone, Debug)]
pub struct ComplianceResult {
    pub passed: bool,
    pub details: String,
    pub severity: ClinicalSafetyLevel,
}

/// Full compliance validation report
#[derive(Debug)]
pub struct ValidatorComplianceReport {
    pub standard_version: String,
    pub timestamp: Instant,
    pub results: HashMap<String, ComplianceResult>,
    pub overall_compliant: bool,
}

/// System configuration for compliance validation
#[derive(Debug)]
pub struct SystemConfiguration {
    pub safety_limits: ClinicalSafetyLimits,
    pub monitoring_enabled: bool,
    pub interlocks_enabled: bool,
    pub emergency_stop_tested: bool,
}