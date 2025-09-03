//! Adaptive control algorithms

use super::super::detection::CavitationMetrics;
use super::types::FeedbackConfig;

/// Adaptive controller for parameter tuning
pub struct AdaptiveController {
    learning_rate: f64,
    history_size: usize,
    performance_history: Vec<f64>,
}

impl AdaptiveController {
    #[must_use]
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            history_size: 100,
            performance_history: Vec::with_capacity(100),
        }
    }

    pub fn adapt_parameters(
        &mut self,
        config: &mut FeedbackConfig,
        metrics: &CavitationMetrics,
        target: f64,
    ) {
        if !config.enable_adaptive {
            return;
        }

        let error = target - metrics.intensity;
        self.performance_history.push(error.abs());

        if self.performance_history.len() > self.history_size {
            self.performance_history.remove(0);
        }

        // Calculate performance trend
        if self.performance_history.len() >= 10 {
            let recent_avg = self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0;

            let old_avg = self.performance_history.iter().take(10).sum::<f64>() / 10.0;

            // If performance is degrading, adjust response time
            if recent_avg > old_avg * 1.1 {
                config.response_time *= 0.95; // Faster response
            } else if recent_avg < old_avg * 0.9 {
                config.response_time *= 1.05; // Slower response for stability
            }

            // Clamp response time
            config.response_time = config.response_time.clamp(0.01, 1.0);
        }
    }

    #[must_use]
    pub fn suggest_gain_adjustment(&self, error_trend: f64) -> f64 {
        // Suggest gain adjustment based on error trend
        if error_trend.abs() < 0.01 {
            1.0 // No adjustment needed
        } else if error_trend > 0.0 {
            1.0 + self.learning_rate // Increase gain
        } else {
            1.0 - self.learning_rate // Decrease gain
        }
    }

    pub fn reset(&mut self) {
        self.performance_history.clear();
    }
}
