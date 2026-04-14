use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};

use super::severity::ErrorSeverity;

/// Alert threshold configuration.
#[derive(Debug, Clone)]
pub struct AlertThreshold {
    pub severity: ErrorSeverity,
    pub max_errors_per_minute: f64,
    pub consecutive_breaches: u32,
    pub window_seconds: u64,
}

impl Default for AlertThreshold {
    fn default() -> Self {
        Self {
            severity: ErrorSeverity::High,
            max_errors_per_minute: 5.0,
            consecutive_breaches: 3,
            window_seconds: 60,
        }
    }
}

#[derive(Debug)]
pub(crate) struct BreachTracker {
    streaks: HashMap<ErrorSeverity, AtomicU32>,
}

impl Default for BreachTracker {
    fn default() -> Self {
        let streaks = ErrorSeverity::all()
            .into_iter()
            .map(|severity| (severity, AtomicU32::new(0)))
            .collect();
        Self { streaks }
    }
}

impl BreachTracker {
    pub(crate) fn update(&self, severity: ErrorSeverity, breached: bool) -> u32 {
        let Some(streak) = self.streaks.get(&severity) else {
            return 0;
        };

        if breached {
            streak.fetch_add(1, Ordering::Relaxed) + 1
        } else {
            streak.store(0, Ordering::Relaxed);
            0
        }
    }

    pub(crate) fn streak(&self, severity: ErrorSeverity) -> u32 {
        self.streaks
            .get(&severity)
            .map(|streak| streak.load(Ordering::Relaxed))
            .unwrap_or(0)
    }
}
