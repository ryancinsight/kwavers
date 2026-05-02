//! IEC 60601 Safety Compliance Framework
//!
//! Implements comprehensive compliance checking for IEC 60601-2-37 (therapeutic ultrasound).
//!
//! ## IEC 60601-2-37 Safety Requirements
//!
//! - **Maximum Intensity**: 3 W/cm² (FDA limit)
//! - **Maximum Temp Rise**: 5°C above baseline
//! - **Frequency Range**: 0.5-10 MHz
//! - **Mechanical Index**: tissue-type dependent
//!
//! ## References
//! - IEC 60601-1, IEC 60601-2-37, FDA (2008), ISO 14971

use super::mechanical_index::TissueType;
use std::collections::VecDeque;
use std::time::Instant;

/// Compliance audit result for a single check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComplianceStatus {
    Compliant,
    Warning,
    NonCompliant,
    NotApplicable,
}

impl std::fmt::Display for ComplianceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compliant => write!(f, "Compliant"),
            Self::Warning => write!(f, "Warning"),
            Self::NonCompliant => write!(f, "Non-Compliant"),
            Self::NotApplicable => write!(f, "N/A"),
        }
    }
}

/// Individual compliance check result
#[derive(Debug, Clone)]
pub struct ComplianceCheck {
    pub name: String,
    pub measured: f64,
    pub limit: f64,
    pub unit: String,
    pub status: ComplianceStatus,
    pub warning_threshold: f64,
}

/// Enhanced safety compliance validator with IEC 60601 audit trail
#[derive(Debug)]
pub struct EnhancedComplianceValidator {
    pub(super) config: ComplianceConfig,
    pub(super) audit_trail: VecDeque<ComplianceAudit>,
    pub(super) session_start: Option<Instant>,
    pub(super) accumulated_time: f64,
    pub(super) accumulated_dose: f64,
}

/// Compliance validation configuration
#[derive(Debug, Clone)]
pub struct ComplianceConfig {
    pub max_power: f64,
    pub max_intensity: f64,
    pub max_temp_rise: f64,
    pub max_session_time: f64,
    pub max_total_dose: f64,
    pub tissue_type: TissueType,
    pub frequency_range: (f64, f64),
    pub max_bnur: f64,
    pub enable_monitoring: bool,
    pub history_window: f64,
}

/// Compliance audit record
#[derive(Debug, Clone)]
pub struct ComplianceAudit {
    pub timestamp: Instant,
    pub checks: Vec<ComplianceCheck>,
    pub overall_status: ComplianceStatus,
    pub alerts: Vec<String>,
}

/// Session metrics
#[derive(Debug, Clone)]
pub struct SessionMetrics {
    pub session_duration: f64,
    pub accumulated_time: f64,
    pub accumulated_dose: f64,
    pub session_compliant: bool,
}

/// Compliance report
#[derive(Debug, Clone)]
pub struct ComplianceReport {
    pub total_audits: usize,
    pub compliant_audits: usize,
    pub warning_audits: usize,
    pub non_compliant_audits: usize,
    pub compliance_percentage: f64,
    pub system_status: String,
}

mod check;
mod config;
#[cfg(test)]
mod tests;
mod validator;
