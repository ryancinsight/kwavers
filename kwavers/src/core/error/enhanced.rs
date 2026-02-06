//! Enhanced Error Handling with Context and Tracing
//!
//! This module provides enhanced error handling capabilities including:
//! - Error context and backtraces
//! - Error chaining and transformation
//! - Performance monitoring integration
//! - Structured error reporting
//!
//! ## Error Context
//!
//! Errors can carry rich context information:
//! ```rust,ignore
//! return Err(KwaversError::numerical(NumericalError::Divergence {
//!     message: "FDTD simulation diverged".to_string(),
//!     context: ErrorContext {
//!         operation: "FDTD time stepping".to_string(),
//!         parameters: [("time_step", &time_step.to_string())].into(),
//!         location: Location::caller(),
//!     },
//! }));
//! ```
//!
//! ## Error Tracing
//!
//! Automatic error tracing through the call stack:
//! ```rust,ignore
//! let result = operation_that_might_fail()
//!     .with_context(|| "Failed to perform critical operation");
//!
//! match result {
//!     Ok(value) => println!("Success: {}", value),
//!     Err(e) => {
//!         eprintln!("Error: {}", e);
//!         eprintln!("Context: {}", e.context());
//!         eprintln!("Backtrace: {}", e.backtrace());
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::fmt;

/// Enhanced error context with location and parameters
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    /// Parameters at time of failure
    pub parameters: HashMap<String, String>,
    /// Source code location
    pub location: Location,
    /// Timestamp of error
    pub timestamp: std::time::SystemTime,
}

/// Source code location information
#[derive(Debug, Clone)]
pub struct Location {
    pub file: String,
    pub line: u32,
    pub column: u32,
    pub function: String,
}

impl Location {
    /// Get current location (equivalent to location!() macro)
    pub fn caller() -> Self {
        Self {
            file: file!().to_string(),
            line: line!(),
            column: column!(),
            function: "unknown".to_string(), // Would need procedural macro for this
        }
    }
}

impl fmt::Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}:{} in {}", self.file, self.line, self.column, self.function)
    }
}

/// Enhanced error with context and tracing
#[derive(Debug)]
pub struct EnhancedError {
    /// Base error
    pub base_error: crate::core::error::KwaversError,
    /// Error context
    pub context: Option<ErrorContext>,
    /// Error chain (causes)
    pub causes: Vec<Box<dyn std::error::Error + Send + Sync>>,
    /// Performance context
    pub performance_context: Option<PerformanceContext>,
}

impl EnhancedError {
    /// Create a new enhanced error
    pub fn new(error: crate::core::error::KwaversError) -> Self {
        Self {
            base_error: error,
            context: None,
            causes: Vec::new(),
            performance_context: None,
        }
    }

    /// Add context to the error
    pub fn with_context<F>(mut self, f: F) -> Self
    where
        F: FnOnce() -> ErrorContext,
    {
        self.context = Some(f());
        self
    }

    /// Add a cause to the error chain
    pub fn with_cause<E>(mut self, cause: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        self.causes.push(Box::new(cause));
        self
    }

    /// Add performance context
    pub fn with_performance(mut self, context: PerformanceContext) -> Self {
        self.performance_context = Some(context);
        self
    }

    /// Get full error report
    pub fn report(&self) -> String {
        let mut report = format!("Enhanced Error Report\n");
        report.push_str("========================\n\n");

        // Base error
        report.push_str(&format!("Base Error: {}\n", self.base_error));

        // Context
        if let Some(ctx) = &self.context {
            report.push_str(&format!("Context: {}\n", ctx.operation));
            report.push_str(&format!("Location: {}\n", ctx.location));
            report.push_str("Parameters:\n");
            for (key, value) in &ctx.parameters {
                report.push_str(&format!("  {}: {}\n", key, value));
            }
            report.push_str(&format!("Timestamp: {:?}\n", ctx.timestamp));
        }

        // Causes
        if !self.causes.is_empty() {
            report.push_str("\nCaused by:\n");
            for (i, cause) in self.causes.iter().enumerate() {
                report.push_str(&format!("  {}. {}\n", i + 1, cause));
            }
        }

        // Performance context
        if let Some(perf) = &self.performance_context {
            report.push_str(&format!("\nPerformance Context:\n"));
            report.push_str(&format!("  Memory used: {} MB\n", perf.memory_used_mb));
            report.push_str(&format!("  CPU time: {:.2} seconds\n", perf.cpu_time_seconds));
            report.push_str(&format!("  Peak memory: {} MB\n", perf.peak_memory_mb));
        }

        report
    }
}

impl fmt::Display for EnhancedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.base_error)?;
        if let Some(ctx) = &self.context {
            write!(f, " at {}", ctx.location)?;
        }
        Ok(())
    }
}

impl std::error::Error for EnhancedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.causes.first().map(|e| e.as_ref() as &(dyn std::error::Error + 'static))
    }
}

/// Performance context for errors
#[derive(Debug, Clone)]
pub struct PerformanceContext {
    pub memory_used_mb: f64,
    pub cpu_time_seconds: f64,
    pub peak_memory_mb: f64,
    pub operation_name: String,
}

/// Error recovery strategies
#[derive(Debug, Clone)]
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry {
        max_attempts: usize,
        backoff_ms: u64,
    },
    /// Use fallback implementation
    Fallback {
        fallback_name: String,
    },
    /// Reduce precision/accuracy
    ReducePrecision {
        factor: f64,
    },
    /// Continue with degraded performance
    DegradedMode,
    /// Abort operation
    Abort,
}

/// Error handler with recovery strategies
pub struct ErrorHandler {
    strategies: HashMap<String, RecoveryStrategy>,
    error_counts: HashMap<String, usize>,
    max_errors_per_type: usize,
}

impl ErrorHandler {
    /// Create a new error handler
    pub fn new() -> Self {
        Self {
            strategies: HashMap::new(),
            error_counts: HashMap::new(),
            max_errors_per_type: 10,
        }
    }

    /// Register a recovery strategy for an error type
    pub fn register_strategy(&mut self, error_type: &str, strategy: RecoveryStrategy) {
        self.strategies.insert(error_type.to_string(), strategy);
    }

    /// Handle an error with recovery
    pub fn handle_error(&mut self, error: &EnhancedError) -> RecoveryAction {
        let error_type = self.classify_error(error);

        // Count errors
        let count = self.error_counts.entry(error_type.clone()).or_insert(0);
        *count += 1;

        // Check if we've exceeded the limit
        if *count > self.max_errors_per_type {
            return RecoveryAction::Abort {
                reason: format!("Too many errors of type: {}", error_type),
            };
        }

        // Apply recovery strategy
        if let Some(strategy) = self.strategies.get(&error_type) {
            match strategy {
                RecoveryStrategy::Retry { max_attempts, backoff_ms } => {
                    if *count < *max_attempts {
                        RecoveryAction::Retry {
                            delay_ms: *backoff_ms * (*count as u64),
                        }
                    } else {
                        RecoveryAction::Abort {
                            reason: format!("Max retry attempts ({}) exceeded", max_attempts),
                        }
                    }
                }
                RecoveryStrategy::Fallback { fallback_name } => {
                    RecoveryAction::Fallback {
                        method: fallback_name.clone(),
                    }
                }
                RecoveryStrategy::ReducePrecision { factor } => {
                    RecoveryAction::ReducePrecision {
                        factor: *factor,
                    }
                }
                RecoveryStrategy::DegradedMode => {
                    RecoveryAction::DegradedMode
                }
                RecoveryStrategy::Abort => {
                    RecoveryAction::Abort {
                        reason: "Configured to abort on this error type".to_string(),
                    }
                }
            }
        } else {
            // Default: abort
            RecoveryAction::Abort {
                reason: format!("No recovery strategy for error type: {}", error_type),
            }
        }
    }

    /// Classify error type from enhanced error
    fn classify_error(&self, error: &EnhancedError) -> String {
        match &error.base_error {
            crate::core::error::KwaversError::Numerical(_) => "numerical".to_string(),
            crate::core::error::KwaversError::Config(_) => "configuration".to_string(),
            crate::core::error::KwaversError::Io(_) => "io".to_string(),
            crate::core::error::KwaversError::Physics(_) => "physics".to_string(),
            crate::core::error::KwaversError::Validation(_) => "validation".to_string(),
            _ => "unknown".to_string(),
        }
    }

    /// Get error statistics
    pub fn statistics(&self) -> &HashMap<String, usize> {
        &self.error_counts
    }

    /// Reset error counts
    pub fn reset(&mut self) {
        self.error_counts.clear();
    }
}

/// Recovery actions for error handling
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Retry the operation after delay
    Retry { delay_ms: u64 },
    /// Use fallback method
    Fallback { method: String },
    /// Reduce precision/accuracy
    ReducePrecision { factor: f64 },
    /// Continue in degraded mode
    DegradedMode,
    /// Abort the operation
    Abort { reason: String },
}

/// Error monitoring and alerting
pub struct ErrorMonitor {
    alert_thresholds: HashMap<String, usize>,
    alerts: Vec<Alert>,
}

impl ErrorMonitor {
    /// Create a new error monitor
    pub fn new() -> Self {
        Self {
            alert_thresholds: HashMap::new(),
            alerts: Vec::new(),
        }
    }

    /// Set alert threshold for error type
    pub fn set_threshold(&mut self, error_type: &str, threshold: usize) {
        self.alert_thresholds.insert(error_type.to_string(), threshold);
    }

    /// Check if error should trigger alert
    pub fn check_alert(&mut self, error_type: &str, count: usize) -> Option<Alert> {
        if let Some(threshold) = self.alert_thresholds.get(error_type) {
            if count >= *threshold {
                let alert = Alert {
                    error_type: error_type.to_string(),
                    count,
                    threshold: *threshold,
                    timestamp: std::time::SystemTime::now(),
                };
                self.alerts.push(alert.clone());
                return Some(alert);
            }
        }
        None
    }

    /// Get recent alerts
    pub fn recent_alerts(&self, since: std::time::SystemTime) -> Vec<&Alert> {
        self.alerts.iter()
            .filter(|alert| alert.timestamp >= since)
            .collect()
    }
}

/// Error alert
#[derive(Debug, Clone)]
pub struct Alert {
    pub error_type: String,
    pub count: usize,
    pub threshold: usize,
    pub timestamp: std::time::SystemTime,
}

/// Convenience macros for enhanced error handling

/// Create an enhanced error with context
#[macro_export]
macro_rules! enhanced_error {
    ($error:expr) => {
        $crate::core::error::enhanced::EnhancedError::new($error)
    };
    ($error:expr, $context:expr) => {
        $crate::core::error::enhanced::EnhancedError::new($error)
            .with_context(|| $context)
    };
}

/// Create error context
#[macro_export]
macro_rules! error_context {
    ($operation:expr) => {
        $crate::core::error::enhanced::ErrorContext {
            operation: $operation.to_string(),
            parameters: std::collections::HashMap::new(),
            location: $crate::core::error::enhanced::Location::caller(),
            timestamp: std::time::SystemTime::now(),
        }
    };
    ($operation:expr, $($key:expr => $value:expr),*) => {
        $crate::core::error::enhanced::ErrorContext {
            operation: $operation.to_string(),
            parameters: vec![$(($key.to_string(), $value.to_string())),*].into_iter().collect(),
            location: $crate::core::error::enhanced::Location::caller(),
            timestamp: std::time::SystemTime::now(),
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhanced_error_creation() {
        let base_error = crate::core::error::KwaversError::Validation(
            crate::core::error::ValidationError::InvalidValue {
                field: "test".to_string(),
                value: "invalid".to_string(),
                expected: "valid".to_string(),
            }
        );

        let enhanced = EnhancedError::new(base_error);
        assert!(enhanced.context.is_none());
        assert!(enhanced.causes.is_empty());
    }

    #[test]
    fn test_error_context_creation() {
        let context = ErrorContext {
            operation: "test operation".to_string(),
            parameters: vec![("param1".to_string(), "value1".to_string())].into_iter().collect(),
            location: Location::caller(),
            timestamp: std::time::SystemTime::now(),
        };

        assert_eq!(context.operation, "test operation");
        assert_eq!(context.parameters.get("param1"), Some(&"value1".to_string()));
    }

    #[test]
    fn test_error_handler_strategies() {
        let mut handler = ErrorHandler::new();

        handler.register_strategy("numerical", RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff_ms: 100,
        });

        // Create a mock enhanced error
        let base_error = crate::core::error::KwaversError::Numerical(
            crate::core::error::NumericalError::Divergence {
                message: "Test divergence".to_string(),
            }
        );
        let enhanced = EnhancedError::new(base_error);

        // First attempt should retry
        let action = handler.handle_error(&enhanced);
        match action {
            RecoveryAction::Retry { delay_ms } => {
                assert_eq!(delay_ms, 100);
            }
            _ => panic!("Expected retry action"),
        }

        // Third attempt should abort
        let action2 = handler.handle_error(&enhanced);
        let action3 = handler.handle_error(&enhanced);
        match action3 {
            RecoveryAction::Abort { .. } => {}
            _ => panic!("Expected abort action after max retries"),
        }
    }

    #[test]
    fn test_error_monitor_alerts() {
        let mut monitor = ErrorMonitor::new();
        monitor.set_threshold("numerical", 3);

        // First two should not alert
        assert!(monitor.check_alert("numerical", 1).is_none());
        assert!(monitor.check_alert("numerical", 2).is_none());

        // Third should alert
        let alert = monitor.check_alert("numerical", 3);
        assert!(alert.is_some());
        assert_eq!(alert.unwrap().error_type, "numerical");
    }

    #[test]
    fn test_performance_context() {
        let context = PerformanceContext {
            memory_used_mb: 128.5,
            cpu_time_seconds: 45.2,
            peak_memory_mb: 256.0,
            operation_name: "FDTD simulation".to_string(),
        };

        assert_eq!(context.operation_name, "FDTD simulation");
        assert_eq!(context.memory_used_mb, 128.5);
    }

    #[test]
    fn test_enhanced_error_report() {
        let base_error = crate::core::error::KwaversError::Numerical(
            crate::core::error::NumericalError::Divergence {
                message: "Simulation diverged".to_string(),
            }
        );

        let enhanced = EnhancedError::new(base_error)
            .with_context(|| ErrorContext {
                operation: "FDTD time stepping".to_string(),
                parameters: vec![("time_step".to_string(), "1e-7".to_string())].into_iter().collect(),
                location: Location::caller(),
                timestamp: std::time::SystemTime::now(),
            });

        let report = enhanced.report();
        assert!(report.contains("Enhanced Error Report"));
        assert!(report.contains("FDTD time stepping"));
        assert!(report.contains("time_step"));
    }
}