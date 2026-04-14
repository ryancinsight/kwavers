use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{info, instrument, warn};

/// Quality adaptation threshold levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityLevel {
    /// Maximum quality (1.0).
    Maximum,
    /// High quality (0.8).
    High,
    /// Medium quality (0.5).
    Medium,
    /// Low quality (0.3).
    Low,
    /// Minimal quality (0.1).
    Minimal,
}

impl QualityLevel {
    /// Get quality factor for this level.
    pub fn factor(&self) -> f32 {
        match self {
            QualityLevel::Maximum => 1.0,
            QualityLevel::High => 0.8,
            QualityLevel::Medium => 0.5,
            QualityLevel::Low => 0.3,
            QualityLevel::Minimal => 0.1,
        }
    }

    /// Get next lower quality level.
    pub fn downgrade(&self) -> Self {
        match self {
            QualityLevel::Maximum => QualityLevel::High,
            QualityLevel::High => QualityLevel::Medium,
            QualityLevel::Medium => QualityLevel::Low,
            QualityLevel::Low => QualityLevel::Minimal,
            QualityLevel::Minimal => QualityLevel::Minimal,
        }
    }

    /// Get next higher quality level.
    pub fn upgrade(&self) -> Self {
        match self {
            QualityLevel::Maximum => QualityLevel::Maximum,
            QualityLevel::High => QualityLevel::Maximum,
            QualityLevel::Medium => QualityLevel::High,
            QualityLevel::Low => QualityLevel::Medium,
            QualityLevel::Minimal => QualityLevel::Low,
        }
    }
}

/// Adaptive quality controller based on performance metrics.
pub struct QualityController {
    /// Current quality level.
    current_level: RwLock<QualityLevel>,
    /// Adaptation threshold (latency at which to downgrade).
    downgrade_threshold_ms: f64,
    /// Upgrade threshold (latency at which to upgrade).
    upgrade_threshold_ms: f64,
    /// History of quality changes.
    history: Mutex<VecDeque<(Instant, QualityLevel, f64)>>,
    /// Maximum history size.
    max_history: usize,
    /// Minimum time between quality changes.
    min_change_interval: Duration,
    /// Last change timestamp.
    last_change: Mutex<Instant>,
}

impl QualityController {
    /// Create a new quality controller.
    pub fn new(target_latency_ms: f64) -> Self {
        Self {
            current_level: RwLock::new(QualityLevel::Maximum),
            downgrade_threshold_ms: target_latency_ms * 0.95,
            upgrade_threshold_ms: target_latency_ms * 0.6,
            history: Mutex::new(VecDeque::with_capacity(100)),
            max_history: 100,
            min_change_interval: Duration::from_millis(500),
            last_change: Mutex::new(Instant::now() - Duration::from_secs(1)),
        }
    }

    /// Evaluate performance and adjust quality if needed.
    #[instrument(skip(self, latency_ms))]
    pub fn evaluate(&self, latency_ms: f64) -> QualityLevel {
        let current = *self.current_level.read();
        let last_change = *self.last_change.lock();
        let now = Instant::now();

        // Debounce quality changes
        if now.duration_since(last_change) < self.min_change_interval {
            return current;
        }

        let new_level = if latency_ms > self.downgrade_threshold_ms {
            // Downgrade quality
            let downgraded = current.downgrade();
            if downgraded != current {
                warn!(
                    latency_ms,
                    threshold = self.downgrade_threshold_ms,
                    from = ?current,
                    to = ?downgraded,
                    "Downgrading quality due to latency"
                );
                downgraded
            } else {
                current
            }
        } else if latency_ms < self.upgrade_threshold_ms && current != QualityLevel::Maximum {
            // Upgrade quality (gradually)
            let upgraded = current.upgrade();
            info!(
                latency_ms,
                threshold = self.upgrade_threshold_ms,
                from = ?current,
                to = ?upgraded,
                "Upgrading quality due to performance headroom"
            );
            upgraded
        } else {
            current
        };

        if new_level != current {
            let mut level = self.current_level.write();
            *level = new_level;
            *self.last_change.lock() = now;

            // Record change
            let mut history = self.history.lock();
            if history.len() >= self.max_history {
                history.pop_front();
            }
            history.push_back((now, new_level, latency_ms));
        }

        new_level
    }

    /// Get current quality factor.
    pub fn current_factor(&self) -> f32 {
        self.current_level.read().factor()
    }

    /// Get current quality level.
    pub fn current_level(&self) -> QualityLevel {
        *self.current_level.read()
    }

    /// Force quality to a specific level.
    pub fn force_level(&self, level: QualityLevel) {
        let now = Instant::now();
        let current = *self.current_level.read();

        if level != current {
            let mut l = self.current_level.write();
            *l = level;
            *self.last_change.lock() = now;

            let mut history = self.history.lock();
            if history.len() >= self.max_history {
                history.pop_front();
            }
            history.push_back((now, level, 0.0));

            info!(from = ?current, to = ?level, "Quality level forced");
        }
    }

    /// Get quality change history.
    pub fn history(&self) -> Vec<(Instant, QualityLevel, f64)> {
        self.history.lock().iter().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quality_level_factors() {
        assert_eq!(QualityLevel::Maximum.factor(), 1.0);
        assert_eq!(QualityLevel::High.factor(), 0.8);
        assert_eq!(QualityLevel::Medium.factor(), 0.5);
        assert_eq!(QualityLevel::Low.factor(), 0.3);
        assert_eq!(QualityLevel::Minimal.factor(), 0.1);
    }

    #[test]
    fn test_quality_downgrade() {
        assert_eq!(QualityLevel::Maximum.downgrade(), QualityLevel::High);
        assert_eq!(QualityLevel::High.downgrade(), QualityLevel::Medium);
        assert_eq!(QualityLevel::Medium.downgrade(), QualityLevel::Low);
        assert_eq!(QualityLevel::Low.downgrade(), QualityLevel::Minimal);
        assert_eq!(QualityLevel::Minimal.downgrade(), QualityLevel::Minimal); // Can't go below minimal
    }

    #[test]
    fn test_quality_upgrade() {
        assert_eq!(QualityLevel::Maximum.upgrade(), QualityLevel::Maximum); // Can't go above max
        assert_eq!(QualityLevel::High.upgrade(), QualityLevel::Maximum);
        assert_eq!(QualityLevel::Medium.upgrade(), QualityLevel::High);
        assert_eq!(QualityLevel::Low.upgrade(), QualityLevel::Medium);
        assert_eq!(QualityLevel::Minimal.upgrade(), QualityLevel::Low);
    }

    #[test]
    fn test_quality_controller_evaluation() {
        let controller = QualityController::new(16.7);
        controller.force_level(QualityLevel::Maximum);

        // Should downgrade if latency too high
        assert_eq!(controller.evaluate(20.0), QualityLevel::High);

        // Should stay at high if still above threshold
        assert_eq!(controller.evaluate(18.0), QualityLevel::High);

        // Should eventually downgrade to minimal with sustained high latency
        let _ = controller.evaluate(17.0);
        let _ = controller.evaluate(17.0);
        let _ = controller.evaluate(17.0);
        let final_level = controller.evaluate(17.0);
        assert!(final_level.factor() <= 0.3);
    }

    #[test]
    fn test_quality_controller_history() {
        let controller = QualityController::new(16.7);
        controller.force_level(QualityLevel::Medium);
        controller.force_level(QualityLevel::Low);

        let history = controller.history();
        assert_eq!(history.len(), 2);
    }
}
