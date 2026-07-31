//! Performance monitoring for clinical workflows.

use aequitas::systems::si::quantities::{Dimensionless, Time};
use std::collections::HashMap;
use std::time::Instant;

/// Performance monitoring for clinical workflows.
#[derive(Debug)]
pub struct WorkflowPerformanceMonitor {
    pub(super) start_time: Instant,
    pub(super) stage_times: HashMap<String, Time<f64>>,
}

impl WorkflowPerformanceMonitor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            stage_times: HashMap::new(),
        }
    }

    pub fn start_monitoring(&mut self) {
        self.start_time = Instant::now();
        self.stage_times.clear();
    }

    pub fn record_stage(&mut self, stage: &str, duration: Time<f64>) {
        self.stage_times.insert(stage.to_owned(), duration);
    }

    #[must_use]
    pub fn get_stage_times(&self) -> HashMap<String, Time<f64>> {
        self.stage_times.clone()
    }

    #[must_use]
    pub fn get_total_time(&self) -> Time<f64> {
        Time::from_base(self.start_time.elapsed().as_secs_f64())
    }

    #[must_use]
    pub fn get_gpu_utilization(&self) -> Option<Dimensionless<f64>> {
        None
    }

    #[must_use]
    pub fn get_memory_usage_bytes(&self) -> Option<u64> {
        None
    }
}

impl Default for WorkflowPerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aequitas::systems::si::units::Millisecond;

    #[test]
    fn records_typed_stage_duration_without_synthetic_telemetry() {
        let mut monitor = WorkflowPerformanceMonitor::new();
        monitor.start_monitoring();
        monitor.record_stage("acquisition", Time::from_unit::<Millisecond>(12.5));

        let stages = monitor.get_stage_times();
        assert_eq!(stages["acquisition"].in_unit::<Millisecond>(), 12.5);
        assert!(monitor.get_gpu_utilization().is_none());
        assert!(monitor.get_memory_usage_bytes().is_none());
    }
}
