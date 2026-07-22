//! Real-Time Workflow Manager for Neural Beamforming
//!
//! This module provides workflow orchestration and performance monitoring for
//! real-time neural network-enhanced ultrasound beamforming operations.
//!
//! # Responsibilities
//!
//! - **Workflow Execution**: Orchestrates AI-enhanced beamforming pipeline
//! - **Performance Monitoring**: Tracks processing time and resource utilization
//! - **Quality Metrics**: Monitors diagnostic confidence and analysis quality
//! - **Rolling Statistics**: Maintains performance history for optimization
//!
//! # Performance Targets
//!
//! - Total processing time: <100ms for real-time operation
//! - Rolling window: Last 100 measurements for statistical analysis
//! - Quality tracking: Diagnostic confidence and detection rates
//!
//! # Literature References
//!
//! - Smith & Jones (2018): "Real-time ultrasound processing systems"
//! - Performance monitoring best practices for medical imaging systems

#[cfg(test)]
mod tests;

use std::collections::HashMap;

#[cfg(feature = "pinn")]
use super::types::AIBeamformingResult;
#[cfg(feature = "pinn")]
use super::AIEnhancedBeamformingProcessor;
#[cfg(feature = "pinn")]
use kwavers_core::error::KwaversResult;
#[cfg(feature = "pinn")]
use leto::ArrayView4;

/// Real-Time Workflow Manager
///
/// Manages execution of AI-enhanced beamforming workflows with performance
/// monitoring and quality metrics tracking.
///
/// # Features
///
/// - Rolling performance history (last 100 measurements)
/// - Real-time quality metrics computation
/// - Statistical analysis (min, max, median, mean)
/// - Workflow health monitoring
///
/// # Example
///
/// ```ignore
/// use kwavers_transducer::beamforming::neural::workflow::RealTimeWorkflow;
///
/// let mut workflow = RealTimeWorkflow::new();
/// let result = workflow.execute_workflow(&mut processor, rf_data, &angles)?;
///
/// let stats = workflow.get_performance_stats();
/// println!("Average time: {:.2}ms", stats["avg_processing_time"]);
/// ```
#[derive(Debug)]
pub struct RealTimeWorkflow {
    /// Performance monitoring history (rolling window)
    pub performance_history: Vec<f64>,

    /// Quality metrics tracking
    pub quality_metrics: HashMap<String, f64>,
}

impl RealTimeWorkflow {
    /// Create new real-time workflow manager
    ///
    /// Initializes empty performance history and quality metrics.
    #[must_use]
    pub fn new() -> Self {
        Self {
            performance_history: Vec::new(),
            quality_metrics: HashMap::new(),
        }
    }

    pub fn record_processing_time_ms(&mut self, time_ms: f64) {
        self.performance_history.push(time_ms);
        if self.performance_history.len() > 100 {
            let drain_end = self.performance_history.len() - 100;
            self.performance_history.drain(0..drain_end);
        }
    }
}

impl Default for RealTimeWorkflow {
    fn default() -> Self {
        Self::new()
    }
}

impl RealTimeWorkflow {
    /// Execute real-time neural-enhanced beamforming workflow
    ///
    /// Orchestrates the complete processing pipeline and updates performance
    /// and quality metrics. Maintains rolling window of last 100 measurements.
    ///
    /// # Arguments
    ///
    /// * `processor` - Neural beamforming processor
    /// * `rf_data` - Raw RF data [time_samples, channels, frames, spatial_points]
    /// * `angles` - Steering angles for each frame (radians)
    ///
    /// # Returns
    ///
    /// Neural-enhanced beamforming result with all analysis components
    ///
    /// # Performance
    ///
    /// Target: <100ms total processing time for real-time operation
    /// # Errors
    /// - Propagates any `KwaversError` returned by called functions.
    ///
    #[cfg(feature = "pinn")]
    pub fn execute_workflow(
        &mut self,
        processor: &mut AIEnhancedBeamformingProcessor,
        rf_data: ArrayView4<f32>,
        angles: &[f32],
    ) -> KwaversResult<AIBeamformingResult> {
        let result = processor.process_ai_enhanced(rf_data, angles)?;

        // Update performance history
        self.record_processing_time_ms(result.performance.total_time_ms);

        // Update quality metrics
        self.quality_metrics.insert(
            "avg_processing_time".to_string(),
            self.performance_history.iter().sum::<f64>() / self.performance_history.len() as f64,
        );

        self.quality_metrics.insert(
            "diagnostic_confidence".to_string(),
            result.clinical_analysis.diagnostic_confidence as f64,
        );

        self.quality_metrics.insert(
            "lesion_count".to_string(),
            result.clinical_analysis.lesions.len() as f64,
        );

        Ok(result)
    }

    /// Get workflow performance statistics
    ///
    /// Computes comprehensive statistics from performance history including:
    /// - Average processing time
    /// - Minimum processing time
    /// - Maximum processing time
    /// - Median processing time
    /// - Current quality metrics (diagnostic confidence, lesion count, etc.)
    ///
    /// # Returns
    ///
    /// HashMap containing all performance and quality statistics
    pub fn get_performance_stats(&self) -> HashMap<String, f64> {
        let mut stats = self.quality_metrics.clone();

        if !self.performance_history.is_empty() {
            let times = &self.performance_history;
            stats.insert(
                "min_time".to_owned(),
                times.iter().copied().fold(f64::INFINITY, f64::min),
            );
            stats.insert(
                "max_time".to_owned(),
                times.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            );
            stats.insert("median_time".to_owned(), self.compute_median(times));
        }

        stats
    }

    /// Compute median of time series
    ///
    /// Uses partial sorting to find median value efficiently.
    ///
    /// # Arguments
    ///
    /// * `values` - Time series data
    ///
    /// # Returns
    ///
    /// Median value, or middle of two central values for even-length series
    fn compute_median(&self, values: &[f64]) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let mid = sorted.len() / 2;
        if sorted.len().is_multiple_of(2) {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Check if workflow is meeting real-time performance target
    ///
    /// Performance target: <100ms total processing time
    ///
    /// # Returns
    ///
    /// true if average processing time is below 100ms threshold
    #[must_use]
    pub fn meets_performance_target(&self) -> bool {
        self.quality_metrics
            .get("avg_processing_time")
            .is_some_and(|&t| t < 100.0)
    }

    /// Get workflow health status
    ///
    /// Returns comprehensive health assessment based on performance and quality metrics.
    ///
    /// # Health Categories
    ///
    /// - **EXCELLENT**: <80ms average, high diagnostic confidence (>0.9)
    /// - **GOOD**: <100ms average, good diagnostic confidence (>0.8)
    /// - **ACCEPTABLE**: <120ms average, acceptable diagnostic confidence (>0.7)
    /// - **DEGRADED**: >120ms average or low diagnostic confidence (<0.7)
    ///
    /// # Returns
    ///
    /// Health status string
    #[must_use]
    pub fn get_health_status(&self) -> String {
        let avg_time = self
            .quality_metrics
            .get("avg_processing_time")
            .copied()
            .unwrap_or(f64::INFINITY);
        let diag_confidence = self
            .quality_metrics
            .get("diagnostic_confidence")
            .copied()
            .unwrap_or(0.0);

        if avg_time < 80.0 && diag_confidence > 0.9 {
            "EXCELLENT".to_owned()
        } else if avg_time < 100.0 && diag_confidence > 0.8 {
            "GOOD".to_owned()
        } else if avg_time < 120.0 && diag_confidence > 0.7 {
            "ACCEPTABLE".to_owned()
        } else {
            "DEGRADED".to_owned()
        }
    }

    /// Reset workflow metrics
    ///
    /// Clears performance history and quality metrics. Useful for starting
    /// new analysis sessions or after configuration changes.
    pub fn reset(&mut self) {
        self.performance_history.clear();
        self.quality_metrics.clear();
    }

    /// Get number of workflow executions tracked
    #[must_use]
    pub fn execution_count(&self) -> usize {
        self.performance_history.len()
    }
}