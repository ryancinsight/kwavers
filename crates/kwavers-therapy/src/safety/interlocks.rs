//! Hardware/Software Interlock System — IEC 60601-2-37 Compliance
//!
//! Implements the interlock mechanism for clinical therapy systems.
//! Interlocks are safety gates that must all be satisfied before the
//! system can be enabled for operation.
//!
//! # IEC 60601-2-37 Requirements
//!
//! - **Clause 201.12.4.1**: Equipment shall include means to prevent unintended
//!   operation (hardware and software interlocks)
//! - **Clause 201.12.4.3**: Emergency stop shall be provided and shall override
//!   all other controls
//!
//! # Architecture
//!
//! Each `Interlock` encapsulates a condition check function that returns
//! `KwaversResult<bool>`. The `InterlockSystem` evaluates all registered
//! interlocks and only permits system operation when all conditions pass.

use kwavers_core::error::{KwaversError, KwaversResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Hardware/software interlock system.
///
/// Manages a set of named interlock conditions that must all be satisfied
/// before the therapy system can operate. Supports emergency stop with
/// manual reset requirement.
#[derive(Debug)]
pub struct InterlockSystem {
    interlocks: HashMap<String, Interlock>,
    system_enabled: bool,
    emergency_stop_active: bool,
}

impl InterlockSystem {
    /// Create new interlock system (disabled by default).
    #[must_use]
    pub fn new() -> Self {
        Self {
            interlocks: HashMap::new(),
            system_enabled: false,
            emergency_stop_active: false,
        }
    }

    /// Register an interlock condition.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn add_interlock(&mut self, name: String, interlock: Interlock) {
        self.interlocks.insert(name, interlock);
    }

    /// Evaluate all interlock conditions.
    ///
    /// Returns `Ok(true)` only if all interlocks pass and no emergency stop is active.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn check_interlocks(&mut self) -> KwaversResult<bool> {
        if self.emergency_stop_active {
            return Ok(false);
        }

        for (name, interlock) in &self.interlocks {
            if !interlock.check_condition()? {
                log::warn!("Interlock '{}' failed: {}", name, interlock.description);
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Enable system operation (requires all interlocks to pass).
    /// # Errors
    /// - Returns [`KwaversError::InvalidInput`] if the precondition for invalid or out-of-range input parameters is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn enable_system(&mut self) -> KwaversResult<()> {
        if !self.check_interlocks()? {
            return Err(KwaversError::InvalidInput(
                "Cannot enable system: interlock conditions not satisfied".to_owned(),
            ));
        }

        self.system_enabled = true;
        log::info!("Therapy system enabled - all safety interlocks satisfied");
        Ok(())
    }

    /// Activate emergency stop — immediate system shutdown.
    ///
    /// Overrides all other controls per IEC 60601-2-37 clause 201.12.4.3.
    pub fn emergency_stop(&mut self) {
        self.emergency_stop_active = true;
        self.system_enabled = false;
        log::error!("EMERGENCY STOP ACTIVATED - System shutdown");
    }

    /// Reset emergency stop (requires manual verification).
    pub fn reset_emergency_stop(&mut self) {
        self.emergency_stop_active = false;
        log::warn!("Emergency stop reset - manual verification required");
    }

    /// Check if system is enabled for operation.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn is_system_enabled(&self) -> bool {
        self.system_enabled && !self.emergency_stop_active
    }
}

impl Default for InterlockSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Individual interlock condition with a named check function.
#[derive(Clone)]
pub struct Interlock {
    /// Human-readable description of the interlock condition
    pub description: String,
    /// Check function that returns `Ok(true)` if condition is satisfied
    pub check_function: Arc<dyn Fn() -> KwaversResult<bool> + Send + Sync>,
}

impl std::fmt::Debug for Interlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Interlock")
            .field("description", &self.description)
            .field("check_function", &"<function>")
            .finish()
    }
}

impl Interlock {
    /// Create new interlock with a condition check function.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new<F>(description: String, check_function: F) -> Self
    where
        F: Fn() -> KwaversResult<bool> + Send + Sync + 'static,
    {
        Self {
            description,
            check_function: Arc::new(check_function),
        }
    }

    /// Evaluate the interlock condition.
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn check_condition(&self) -> KwaversResult<bool> {
        (self.check_function)()
    }
}
