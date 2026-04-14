use parking_lot::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tracing::{info, warn};

use super::budget::LatencyBudget;
use super::pacer::FramePacer;
use super::quality::{QualityController, QualityLevel};
use super::state::{PacingStrategy, SyncState, SyncStatistics};

/// Complete synchronization coordinator.
pub struct SyncCoordinator {
    /// Frame pacer for timing control.
    pacer: Arc<FramePacer>,
    /// Latency budget manager.
    budget: Arc<LatencyBudget>,
    /// Quality controller.
    quality: Arc<QualityController>,
    /// Current sync state.
    state: RwLock<SyncState>,
    /// Pause/resume control.
    paused: AtomicBool,
    /// Frame sync notification.
    sync_notify: Arc<Notify>,
    /// Frames dropped counter.
    frames_dropped: AtomicU64,
    /// Frames rendered counter.
    frames_rendered: AtomicU64,
}

impl SyncCoordinator {
    /// Create a new sync coordinator.
    pub fn new(target_fps: f64) -> Self {
        let target_interval = Duration::from_secs_f64(1.0 / target_fps);
        let pacer = Arc::new(FramePacer::new(PacingStrategy::Adaptive));
        pacer.set_target_interval(target_interval);

        let budget = Arc::new(LatencyBudget::new(target_interval.as_secs_f64() * 1000.0));
        // Standard allocations for visualization pipeline
        budget.allocate("extract", 5.0);
        budget.allocate("encode", 5.0);
        budget.allocate("render", 6.7); // Target 60fps = 16.7ms, leaving headroom

        let quality = Arc::new(QualityController::new(
            target_interval.as_secs_f64() * 1000.0,
        ));

        let mut state = SyncState::default();
        state.target_interval = target_interval;

        Self {
            pacer,
            budget,
            quality,
            state: RwLock::new(state),
            paused: AtomicBool::new(false),
            sync_notify: Arc::new(Notify::new()),
            frames_dropped: AtomicU64::new(0),
            frames_rendered: AtomicU64::new(0),
        }
    }

    /// Wait for next frame synchronization point.
    pub async fn wait_for_sync(&self) -> SyncState {
        if self.paused.load(Ordering::Relaxed) {
            // Wait until unpaused
            loop {
                self.sync_notify.notified().await;
                if !self.paused.load(Ordering::Relaxed) {
                    break;
                }
            }
        }

        self.pacer.pace().await;

        self.state.read().clone()
    }

    /// Signal frame completion with latency measurement.
    pub fn complete_frame(&self, latency: Duration, sim_time: f64) {
        self.pacer.record_frame_time(latency);
        self.frames_rendered.fetch_add(1, Ordering::Relaxed);

        // Update sync state
        let mut state = self.state.write();
        state.last_display_time = Instant::now();
        state.last_sim_time = sim_time;
        state.pipeline_latency = latency;

        // Evaluate quality
        let latency_ms = latency.as_secs_f64() * 1000.0;
        let level = self.quality.evaluate(latency_ms);
        state.quality_level = level.factor();

        // Check if within budget
        let within = self.budget.check("total", latency_ms);
        state.within_budget = within;
    }

    /// Report a dropped frame.
    pub fn report_drop(&self, reason: &str) {
        self.frames_dropped.fetch_add(1, Ordering::Relaxed);
        warn!(
            reason,
            total_drops = self.frames_dropped.load(Ordering::Relaxed),
            "Frame dropped"
        );
    }

    /// Pause synchronization.
    pub fn pause(&self) {
        self.paused.store(true, Ordering::Relaxed);
        info!("Sync coordinator paused");
    }

    /// Resume synchronization.
    pub fn resume(&self) {
        self.paused.store(false, Ordering::Relaxed);
        self.sync_notify.notify_one();
        info!("Sync coordinator resumed");
    }

    /// Check if currently paused.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    /// Get current statistics.
    pub fn statistics(&self) -> SyncStatistics {
        let state = self.state.read();
        let quality_history = self.quality.history();

        SyncStatistics {
            target_fps: 1.0 / state.target_interval.as_secs_f64(),
            actual_fps: self
                .pacer
                .average_frame_time()
                .map(|d| 1.0 / d.as_secs_f64()),
            latency_ms: state.pipeline_latency.as_secs_f64() * 1000.0,
            quality_level: state.quality_level,
            within_budget: state.within_budget,
            frames_rendered: self.frames_rendered.load(Ordering::Relaxed),
            frames_dropped: self.frames_dropped.load(Ordering::Relaxed),
            drop_rate_percent: self.calculate_drop_rate(),
            budget_utilization: self.budget.utilization_percent(),
            frame_jitter_ms: self.pacer.frame_jitter().map(|d| d.as_secs_f64() * 1000.0),
            quality_history: quality_history
                .into_iter()
                .map(|(t, level, lat)| (t, level.factor(), lat))
                .collect(),
        }
    }

    /// Calculate drop rate as percentage.
    fn calculate_drop_rate(&self) -> f64 {
        let rendered = self.frames_rendered.load(Ordering::Relaxed) as f64;
        let dropped = self.frames_dropped.load(Ordering::Relaxed) as f64;
        let total = rendered + dropped;

        if total > 0.0 {
            (dropped / total) * 100.0
        } else {
            0.0
        }
    }

    /// Get budget summary.
    pub fn budget_summary(&self) -> String {
        self.budget.summary()
    }

    /// Force quality level.
    pub fn set_quality(&self, level: QualityLevel) {
        self.quality.force_level(level);
        let mut state = self.state.write();
        state.quality_level = level.factor();
    }

    /// Get current quality factor.
    pub fn quality_factor(&self) -> f32 {
        self.quality.current_factor()
    }

    /// Update target FPS (affects pacing).
    pub fn set_target_fps(&self, fps: f64) {
        let interval = Duration::from_secs_f64(1.0 / fps);
        self.pacer.set_target_interval(interval);
        let mut state = self.state.write();
        state.target_interval = interval;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sync_coordinator_creation() {
        let coordinator = SyncCoordinator::new(60.0);
        assert!(!coordinator.is_paused());
        assert_eq!(coordinator.quality_factor(), 1.0);
    }

    #[test]
    fn test_sync_coordinator_pause_resume() {
        let coordinator = SyncCoordinator::new(60.0);
        assert!(!coordinator.is_paused());

        coordinator.pause();
        assert!(coordinator.is_paused());

        coordinator.resume();
        assert!(!coordinator.is_paused());
    }
}
