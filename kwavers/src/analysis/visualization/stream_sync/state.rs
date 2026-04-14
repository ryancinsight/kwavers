use std::time::{Duration, Instant};
use tracing::info;

/// Frame pacing strategy for synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacingStrategy {
    /// Adaptive pacing based on measured latency.
    Adaptive,
    /// Fixed interval pacing.
    FixedInterval(Duration),
    /// VSync-based pacing (wait for display refresh).
    VSync,
    /// Variable rate based on simulation speed.
    Variable {
        /// Maximum frame interval.
        max_interval_ms: u64,
        /// Minimum frame interval.
        min_interval_ms: u64,
    },
}

/// Synchronization state between simulation and visualization.
#[derive(Debug, Clone)]
pub struct SyncState {
    /// Last simulation timestamp processed.
    pub last_sim_time: f64,
    /// Last display timestamp.
    pub last_display_time: Instant,
    /// Current target frame interval.
    pub target_interval: Duration,
    /// Measured pipeline latency.
    pub pipeline_latency: Duration,
    /// Current quality level (0.0 to 1.0).
    pub quality_level: f32,
    /// Whether sync is currently within budget.
    pub within_budget: bool,
}

impl Default for SyncState {
    fn default() -> Self {
        Self {
            last_sim_time: 0.0,
            last_display_time: Instant::now(),
            target_interval: Duration::from_millis(33), // ~30 FPS default
            pipeline_latency: Duration::from_millis(0),
            quality_level: 1.0,
            within_budget: true,
        }
    }
}

/// Synchronization statistics.
#[derive(Debug, Clone)]
pub struct SyncStatistics {
    /// Target frames per second.
    pub target_fps: f64,
    /// Actual measured FPS (if available).
    pub actual_fps: Option<f64>,
    /// Current latency in milliseconds.
    pub latency_ms: f64,
    /// Current quality factor (0.0 to 1.0).
    pub quality_level: f32,
    /// Whether current performance is within budget.
    pub within_budget: bool,
    /// Total frames rendered.
    pub frames_rendered: u64,
    /// Total frames dropped.
    pub frames_dropped: u64,
    /// Drop rate as percentage.
    pub drop_rate_percent: f64,
    /// Budget utilization percentage.
    pub budget_utilization: f64,
    /// Frame timing jitter in milliseconds.
    pub frame_jitter_ms: Option<f64>,
    /// Quality adjustment history.
    pub quality_history: Vec<(Instant, f32, f64)>,
}

/// Report sync metrics to console/log.
pub fn report_sync_metrics(stats: &SyncStatistics) {
    info!("=== Synchronization Metrics ===");
    info!("Target FPS: {:.1}", stats.target_fps);
    info!("Actual FPS: {:.1}", stats.actual_fps.unwrap_or(0.0));
    info!("Latency: {:.2} ms", stats.latency_ms);
    info!("Quality Level: {:.1}%", stats.quality_level * 100.0);
    info!("Within Budget: {}", stats.within_budget);
    info!("Frames Rendered: {}", stats.frames_rendered);
    info!("Frames Dropped: {}", stats.frames_dropped);
    info!("Drop Rate: {:.2}%", stats.drop_rate_percent);
    info!("Budget Utilization: {:.1}%", stats.budget_utilization);
    if let Some(jitter) = stats.frame_jitter_ms {
        info!("Frame Jitter: {:.2} ms", jitter);
    }
    info!("Quality Adjustments: {}", stats.quality_history.len());
    info!("===============================");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_sync_state_default() {
        let state = SyncState::default();
        assert_eq!(state.quality_level, 1.0);
        assert!(state.within_budget);
        assert_eq!(state.last_sim_time, 0.0);
    }

    #[test]
    fn test_sync_statistics_drop_rate() {
        let stats = SyncStatistics {
            target_fps: 60.0,
            actual_fps: Some(58.0),
            latency_ms: 17.2,
            quality_level: 0.8,
            within_budget: true,
            frames_rendered: 100,
            frames_dropped: 2,
            drop_rate_percent: 2.0,
            budget_utilization: 97.0,
            frame_jitter_ms: Some(0.5),
            quality_history: vec![],
        };

        assert_eq!(stats.drop_rate_percent, 2.0);
        assert_eq!(stats.quality_level, 0.8);
    }

    #[test]
    fn test_pacing_strategy_variants() {
        let adaptive = PacingStrategy::Adaptive;
        let fixed = PacingStrategy::FixedInterval(Duration::from_millis(33));
        let vsync = PacingStrategy::VSync;
        let variable = PacingStrategy::Variable {
            max_interval_ms: 33,
            min_interval_ms: 16,
        };

        // Just verify they can be created
        let _ = format!("{:?}", adaptive);
        let _ = format!("{:?}", fixed);
        let _ = format!("{:?}", vsync);
        let _ = format!("{:?}", variable);
    }
}
