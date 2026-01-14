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

use std::collections::HashMap;

#[cfg(feature = "pinn")]
use crate::core::error::KwaversResult;
#[cfg(feature = "pinn")]
use crate::domain::sensor::beamforming::AIBeamformingResult;
#[cfg(feature = "pinn")]
use crate::domain::sensor::beamforming::neural::processor::AIEnhancedBeamformingProcessor;
#[cfg(feature = "pinn")]
use ndarray::ArrayView4;

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
/// use kwavers::domain::sensor::beamforming::neural::workflow::RealTimeWorkflow;
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
                "min_time".to_string(),
                times.iter().cloned().fold(f64::INFINITY, f64::min),
            );
            stats.insert(
                "max_time".to_string(),
                times.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            );
            stats.insert("median_time".to_string(), self.compute_median(times));
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
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
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
    pub fn meets_performance_target(&self) -> bool {
        self.quality_metrics
            .get("avg_processing_time")
            .map(|&t| t < 100.0)
            .unwrap_or(false)
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
            "EXCELLENT".to_string()
        } else if avg_time < 100.0 && diag_confidence > 0.8 {
            "GOOD".to_string()
        } else if avg_time < 120.0 && diag_confidence > 0.7 {
            "ACCEPTABLE".to_string()
        } else {
            "DEGRADED".to_string()
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
    pub fn execution_count(&self) -> usize {
        self.performance_history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workflow_creation() {
        let workflow = RealTimeWorkflow::new();
        assert!(workflow.performance_history.is_empty());
        assert!(workflow.quality_metrics.is_empty());

        let default_workflow = RealTimeWorkflow::default();
        assert!(default_workflow.performance_history.is_empty());
    }

    #[test]
    fn test_median_computation() {
        let workflow = RealTimeWorkflow::new();

        // Odd number of elements
        let values_odd = vec![5.0, 3.0, 7.0, 1.0, 9.0];
        assert_eq!(workflow.compute_median(&values_odd), 5.0);

        // Even number of elements
        let values_even = vec![2.0, 4.0, 6.0, 8.0];
        assert_eq!(workflow.compute_median(&values_even), 5.0); // (4.0 + 6.0) / 2.0
    }

    #[test]
    fn test_performance_stats() {
        let mut workflow = RealTimeWorkflow::new();

        // Add performance data
        workflow
            .performance_history
            .extend_from_slice(&[50.0, 60.0, 55.0, 65.0, 70.0]);
        workflow
            .quality_metrics
            .insert("avg_processing_time".to_string(), 60.0);
        workflow
            .quality_metrics
            .insert("diagnostic_confidence".to_string(), 0.9);

        let stats = workflow.get_performance_stats();
        assert_eq!(stats.get("min_time"), Some(&50.0));
        assert_eq!(stats.get("max_time"), Some(&70.0));
        assert_eq!(stats.get("median_time"), Some(&60.0));
        assert_eq!(stats.get("diagnostic_confidence"), Some(&0.9));
    }

    #[test]
    fn test_rolling_window() {
        let mut workflow = RealTimeWorkflow::new();

        // Add 150 measurements
        for i in 0..150 {
            workflow.record_processing_time_ms(i as f64);
        }

        // Should maintain only last 100
        assert_eq!(workflow.performance_history.len(), 100);
        assert_eq!(workflow.performance_history[0], 50.0); // First element is 50th measurement
        assert_eq!(workflow.performance_history[99], 149.0); // Last element is 149th measurement
    }

    #[test]
    fn test_performance_target() {
        let mut workflow = RealTimeWorkflow::new();

        // Meets target
        workflow
            .quality_metrics
            .insert("avg_processing_time".to_string(), 85.0);
        assert!(workflow.meets_performance_target());

        // Exceeds target
        workflow
            .quality_metrics
            .insert("avg_processing_time".to_string(), 110.0);
        assert!(!workflow.meets_performance_target());
    }

    #[test]
    fn test_health_status() {
        let mut workflow = RealTimeWorkflow::new();

        // EXCELLENT
        workflow
            .quality_metrics
            .insert("avg_processing_time".to_string(), 75.0);
        workflow
            .quality_metrics
            .insert("diagnostic_confidence".to_string(), 0.95);
        assert_eq!(workflow.get_health_status(), "EXCELLENT");

        // GOOD
        workflow
            .quality_metrics
            .insert("avg_processing_time".to_string(), 90.0);
        workflow
            .quality_metrics
            .insert("diagnostic_confidence".to_string(), 0.85);
        assert_eq!(workflow.get_health_status(), "GOOD");

        // ACCEPTABLE
        workflow
            .quality_metrics
            .insert("avg_processing_time".to_string(), 110.0);
        workflow
            .quality_metrics
            .insert("diagnostic_confidence".to_string(), 0.75);
        assert_eq!(workflow.get_health_status(), "ACCEPTABLE");

        // DEGRADED
        workflow
            .quality_metrics
            .insert("avg_processing_time".to_string(), 150.0);
        workflow
            .quality_metrics
            .insert("diagnostic_confidence".to_string(), 0.6);
        assert_eq!(workflow.get_health_status(), "DEGRADED");
    }

    #[test]
    fn test_reset() {
        let mut workflow = RealTimeWorkflow::new();

        workflow.performance_history.push(50.0);
        workflow
            .quality_metrics
            .insert("avg_time".to_string(), 50.0);

        workflow.reset();

        assert!(workflow.performance_history.is_empty());
        assert!(workflow.quality_metrics.is_empty());
    }

    #[test]
    fn test_execution_count() {
        let mut workflow = RealTimeWorkflow::new();
        assert_eq!(workflow.execution_count(), 0);

        workflow
            .performance_history
            .extend_from_slice(&[50.0, 60.0, 70.0]);
        assert_eq!(workflow.execution_count(), 3);
    }
}
