use super::state::PacingStrategy;
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{info, instrument, trace};

/// Frame pacing controller.
pub struct FramePacer {
    /// Pacing strategy.
    strategy: PacingStrategy,
    /// Target frame interval.
    target_interval: RwLock<Duration>,
    /// Last frame time.
    last_frame_time: Mutex<Instant>,
    /// Frame history for adaptive pacing.
    frame_times: Mutex<VecDeque<Duration>>,
    /// Adaptive smoothing factor.
    alpha: f64,
    /// Maximum history size.
    max_history: usize,
}

impl FramePacer {
    /// Create a new frame pacer with specified strategy.
    pub fn new(strategy: PacingStrategy) -> Self {
        let initial_interval = match strategy {
            PacingStrategy::Adaptive => Duration::from_millis(33),
            PacingStrategy::FixedInterval(d) => d,
            PacingStrategy::VSync => Duration::from_millis(16), // Assume 60Hz
            PacingStrategy::Variable {
                max_interval_ms, ..
            } => Duration::from_millis(max_interval_ms),
        };

        Self {
            strategy,
            target_interval: RwLock::new(initial_interval),
            last_frame_time: Mutex::new(Instant::now()),
            frame_times: Mutex::new(VecDeque::with_capacity(60)),
            alpha: 0.1,
            max_history: 60,
        }
    }

    /// Wait for the appropriate time before next frame.
    #[instrument(skip(self))]
    pub async fn pace(&self) {
        let target = *self.target_interval.read();
        let last = *self.last_frame_time.lock();
        let elapsed = last.elapsed();

        match self.strategy {
            PacingStrategy::Adaptive => {
                self.adaptive_pace(target, elapsed).await;
            }
            PacingStrategy::FixedInterval(interval) => {
                if elapsed < interval {
                    let wait = interval - elapsed;
                    trace!(wait_ms = wait.as_millis(), "Fixed interval wait");
                    tokio::time::sleep(wait).await;
                }
            }
            PacingStrategy::VSync => {
                // VSync waits handled by display system
                // Here we just ensure minimum interval
                if elapsed < Duration::from_millis(16) {
                    tokio::time::sleep(Duration::from_millis(16) - elapsed).await;
                }
            }
            PacingStrategy::Variable {
                min_interval_ms,
                max_interval_ms,
            } => {
                let min = Duration::from_millis(min_interval_ms);
                let max = Duration::from_millis(max_interval_ms);
                let adaptive = self.calculate_adaptive_interval();
                let final_interval = adaptive.clamp(min, max);
                if elapsed < final_interval {
                    tokio::time::sleep(final_interval - elapsed).await;
                }
            }
        }

        *self.last_frame_time.lock() = Instant::now();
    }

    /// Adaptive pacing based on measured performance.
    async fn adaptive_pace(&self, target: Duration, elapsed: Duration) {
        // Calculate jitter (deviation from target)
        let jitter = if elapsed > target {
            elapsed - target
        } else {
            Duration::from_millis(0)
        };

        // If we're ahead of schedule, wait
        if elapsed < target {
            let wait = target - elapsed;
            // Adjust wait based on recent jitter (be more conservative if jitter is high)
            let adjusted_wait = if jitter > Duration::from_millis(5) {
                wait / 2 // Don't wait full interval if jitter is high
            } else {
                wait
            };
            trace!(
                target_ms = target.as_millis(),
                elapsed_ms = elapsed.as_millis(),
                wait_ms = adjusted_wait.as_millis(),
                jitter_ms = jitter.as_millis(),
                "Adaptive pacing wait"
            );
            tokio::time::sleep(adjusted_wait).await;
        } else {
            trace!(
                target_ms = target.as_millis(),
                elapsed_ms = elapsed.as_millis(),
                "No wait needed (behind schedule)"
            );
        }
    }

    /// Calculate adaptive interval based on recent frame times.
    fn calculate_adaptive_interval(&self) -> Duration {
        let times = self.frame_times.lock();
        if times.len() < 10 {
            return *self.target_interval.read();
        }

        // Calculate average and variance
        let avg: f64 = times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;
        let variance: f64 = times
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - avg;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;
        let std_dev = variance.sqrt();

        // Add some margin based on variance
        let margin = std_dev * 0.5;
        let target_secs = avg + margin;

        Duration::from_secs_f64(target_secs)
    }

    /// Record a frame completion time for adaptive calculations.
    pub fn record_frame_time(&self, duration: Duration) {
        let mut times = self.frame_times.lock();
        if times.len() >= self.max_history {
            times.pop_front();
        }
        times.push_back(duration);
    }

    /// Update target frame interval.
    pub fn set_target_interval(&self, interval: Duration) {
        *self.target_interval.write() = interval;
        info!(
            interval_ms = interval.as_millis(),
            "Updated target frame interval"
        );
    }

    /// Get current target interval.
    pub fn target_interval(&self) -> Duration {
        *self.target_interval.read()
    }

    /// Get measured average frame time.
    pub fn average_frame_time(&self) -> Option<Duration> {
        let times = self.frame_times.lock();
        if times.is_empty() {
            return None;
        }
        let avg = times.iter().sum::<Duration>() / times.len() as u32;
        Some(avg)
    }

    /// Get frame time jitter (standard deviation).
    pub fn frame_jitter(&self) -> Option<Duration> {
        let times = self.frame_times.lock();
        if times.len() < 2 {
            return None;
        }

        let avg: f64 = times.iter().map(|d| d.as_secs_f64()).sum::<f64>() / times.len() as f64;
        let variance: f64 = times
            .iter()
            .map(|d| {
                let diff = d.as_secs_f64() - avg;
                diff * diff
            })
            .sum::<f64>()
            / times.len() as f64;
        Some(Duration::from_secs_f64(variance.sqrt()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test(flavor = "multi_thread")]
    async fn test_frame_pacer_creation() {
        let pacer = FramePacer::new(PacingStrategy::FixedInterval(Duration::from_millis(33)));
        assert_eq!(pacer.target_interval().as_millis(), 33);
    }

    #[test]
    fn test_frame_pacer_average_calculation() {
        let pacer = FramePacer::new(PacingStrategy::Adaptive);
        assert!(pacer.average_frame_time().is_none());

        pacer.record_frame_time(Duration::from_millis(20));
        pacer.record_frame_time(Duration::from_millis(30));
        pacer.record_frame_time(Duration::from_millis(25));

        let avg = pacer.average_frame_time().unwrap();
        assert!(avg.as_millis() >= 24 && avg.as_millis() <= 26);
    }

    #[test]
    fn test_frame_pacer_jitter_calculation() {
        let pacer = FramePacer::new(PacingStrategy::Adaptive);
        assert!(pacer.frame_jitter().is_none());

        // Add many samples to get variance
        for i in 0..20 {
            let val = 20 + (i % 5) as u64; // Some variation
            pacer.record_frame_time(Duration::from_millis(val));
        }

        let jitter = pacer.frame_jitter().unwrap();
        assert!(jitter.as_nanos() > 0);
    }
}
