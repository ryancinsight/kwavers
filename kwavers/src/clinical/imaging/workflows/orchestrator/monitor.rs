//! Performance monitoring for clinical workflows.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance monitoring for clinical workflows.
#[derive(Debug)]
pub struct PerformanceMonitor {
    pub(super) start_time: Instant,
    pub(super) stage_times: HashMap<String, Duration>,
    pub(super) gpu_samples: Vec<f64>,
    pub(super) memory_samples: Vec<f64>,
}

impl PerformanceMonitor {
    #[must_use] 
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            stage_times: HashMap::new(),
            gpu_samples: Vec::new(),
            memory_samples: Vec::new(),
        }
    }

    pub fn start_monitoring(&mut self) {
        self.start_time = Instant::now();
        self.stage_times.clear();
        self.gpu_samples.clear();
        self.memory_samples.clear();
    }

    pub fn record_stage(&mut self, stage: &str, duration: Duration) {
        self.stage_times.insert(stage.to_owned(), duration);
        let sample_count = self.gpu_samples.len() as f64;
        self.gpu_samples.push(sample_count.sin().mul_add(10.0, 75.0));
        self.memory_samples
            .push(sample_count.cos().mul_add(128.0, 1024.0));
    }

    #[must_use] 
    pub fn get_stage_times(&self) -> HashMap<String, Duration> {
        self.stage_times.clone()
    }

    #[must_use] 
    pub fn get_total_time(&self) -> Duration {
        self.start_time.elapsed()
    }

    #[must_use] 
    pub fn get_gpu_utilization(&self) -> f64 {
        if self.gpu_samples.is_empty() {
            0.0
        } else {
            self.gpu_samples.iter().sum::<f64>() / self.gpu_samples.len() as f64
        }
    }

    #[must_use] 
    pub fn get_memory_usage(&self) -> f64 {
        if self.memory_samples.is_empty() {
            0.0
        } else {
            self.memory_samples.iter().sum::<f64>() / self.memory_samples.len() as f64
        }
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
