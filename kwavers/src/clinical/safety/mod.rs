//! Clinical Safety Framework - IEC 60601-2-37 Compliance
//!
//! This module implements the safety and regulatory compliance framework for clinical
//! ultrasound therapy systems, following IEC 60601-2-37 standards for therapeutic ultrasound equipment.
//!
//! ## Safety Metrics
//!
//! - **Mechanical Index (MI)**: Cavitation risk assessment
//! - **Thermal Index (TI)**: Tissue heating risk assessment
//! - **Dose Monitoring**: Treatment dose verification
//!
//! ## IEC 60601-2-37 Requirements
//!
//! ### Essential Performance
//! - **Treatment accuracy**: ±10% of prescribed dose
//! - **Safety margins**: Automatic shutdown if parameters exceed limits
//! - **Monitoring**: Real-time acoustic output verification
//! - **Alarms**: Audible/visual alerts for system faults
//!
//! ### Risk Management
//! - **Hazard analysis**: Systematic identification of potential harms
//! - **Risk control**: Mitigation measures for identified hazards
//! - **Residual risk**: Acceptability evaluation
//! - **Verification**: Testing of safety measures
//!
//! ## Architecture
//!
//! The safety framework consists of:
//! - **SafetyMonitor**: Real-time parameter monitoring and validation
//! - **InterlockSystem**: Hardware/software interlocks for emergency stops
//! - **DoseController**: Treatment dose calculation and control
//! - **ComplianceValidator**: IEC standard compliance checking
//! - **AuditLogger**: Comprehensive safety event logging

pub mod audit;
pub mod compliance;
pub mod interlocks;
pub mod mechanical_index;

mod compliance_validator;
mod dose;
mod monitor;
#[cfg(test)]
mod tests;

pub use audit::{AuditEntry, SafetyAuditLogger, SafetyEventType};
pub use compliance::{
    ComplianceAudit, ComplianceCheck as EnhancedComplianceCheck, ComplianceConfig,
    ComplianceStatus, EnhancedComplianceValidator, SessionMetrics,
};
pub use compliance_validator::{
    ComplianceReport, ComplianceResult, ComplianceValidator, SystemConfiguration,
};
pub use dose::{DoseController, TreatmentRecord};
pub use interlocks::{Interlock, InterlockSystem};
pub use monitor::{SafetyLevel, SafetyLimits, SafetyMonitor, SafetyViolation};
