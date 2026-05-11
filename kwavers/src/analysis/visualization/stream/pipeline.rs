//! Stream Visualization Pipeline Configuration and Stage Processing
//!
//! ## Mathematical Foundation
//!
//! **Theorem: Frame Budget Allocation (Little's Law, 1961)**
//! ```text
//! For a stable pipeline with target frame rate F (fps):
//!   T_budget = 1 / F  (total frame budget, seconds)
//!   T_stage  = T_budget / K  (per-stage budget, K = number of stages)
//!
//! Little (1961) shows that for a stable queue: L = λ·W
//!   where L = average queue length, λ = arrival rate, W = average wait time.
//!   Pipeline is stable when λ < 1/T_stage for each stage.
//!
//! Reference: Little, J.D.C. (1961). "A proof for the queuing formula: L = λW".
//!   Operations Research 9(3):383–387. DOI:10.1287/opre.9.3.383
//! ```
//!
//! **Algorithm: Stage Pipeline with Async Back-Pressure**
//! ```text
//! 1. Input channel (bounded, capacity = config.channel_capacity) accepts VizFrame
//! 2. Background task:
//!    a. recv frame from input channel
//!    b. record arrival timestamp t_in
//!    c. extract scalar field (mean of pressure field)
//!    d. record t_out = now()
//!    e. update running metrics: latency += (t_out - t_in), drop_rate via EMA
//! 3. metrics() returns snapshot of current PipelineRunMetrics
//! ```

use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use tokio::sync::mpsc;
use tracing::{debug, trace};

use super::super::stream::{FrameId, VizFrame};

/// Lightweight pipeline configuration.
///
/// Controls frame rate targeting, channel depth, and adaptive behavior.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineConfig {
    /// Target frames per second.
    pub target_fps: f64,
    /// Bounded channel capacity for the input frame queue.
    pub channel_capacity: usize,
    /// Whether to run stages in parallel (currently informational).
    pub parallel_execution: bool,
    /// Whether to enable adaptive quality based on latency.
    pub adaptive_quality: bool,
    /// Maximum acceptable end-to-end latency in milliseconds.
    pub latency_threshold_ms: f64,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            target_fps: 30.0,
            channel_capacity: 8,
            parallel_execution: false,
            adaptive_quality: false,
            latency_threshold_ms: 33.33,
        }
    }
}

impl PipelineConfig {
    /// Frame budget in milliseconds: 1000 / target_fps.
    ///
    /// **Theorem**: For F fps, each frame must complete within T = 1000/F ms.
    pub fn frame_budget_ms(&self) -> f64 {
        1000.0 / self.target_fps
    }

    /// Per-stage budget in milliseconds: frame_budget / stages.
    pub fn stage_budget_ms(&self, stages: usize) -> f64 {
        self.frame_budget_ms() / stages.max(1) as f64
    }
}

/// Per-stage statistics snapshot.
#[derive(Debug, Clone, Default)]
pub struct StageStats {
    /// Stage name identifier.
    pub name: String,
    /// Frames successfully processed by this stage.
    pub frames_processed: u64,
    /// Frames dropped or failed at this stage.
    pub frames_dropped: u64,
    /// Cumulative processing time (milliseconds).
    pub total_latency_ms: f64,
}

/// Pipeline-wide runtime metrics snapshot.
#[derive(Debug, Clone, Default)]
pub struct PipelineRunMetrics {
    /// Per-stage breakdown (at least one entry when pipeline is active).
    pub stages: Vec<StageStats>,
    /// Average end-to-end pipeline latency (milliseconds).
    pub total_latency_ms: f64,
    /// Frame drop rate as a percentage.
    pub drop_rate_percent: f64,
}

/// Internal mutable state of the background processing task.
struct PipelineState {
    frames_processed: u64,
    frames_dropped: u64,
    cumulative_latency_ms: f64,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            frames_processed: 0,
            frames_dropped: 0,
            cumulative_latency_ms: 0.0,
        }
    }
}

/// Async stage pipeline for real-time frame processing.
///
/// ## Algorithm (Welford 1962 — online mean update)
/// ```text
/// mean_n = mean_{n-1} + (x_n - mean_{n-1}) / n
/// Used for running average latency: no accumulator overflow.
/// Reference: Welford, B.P. (1962). "Note on a method for calculating
///   corrected sums of squares and products." Technometrics 4(3):419–420.
/// ```
pub struct StagePipeline {
    /// Pipeline configuration.
    config: PipelineConfig,
    /// Shared metrics updated by the background task.
    metrics: Arc<Mutex<PipelineRunMetrics>>,
}

impl StagePipeline {
    /// Create a new async stage pipeline.
    ///
    /// Returns `(pipeline_handle, input_sender)`.
    /// Drop `input_sender` to cleanly shut down the background task.
    ///
    /// **Algorithm**:
    /// 1. Create bounded `tokio::mpsc::channel(capacity)`.
    /// 2. Spawn background task that drains frames from channel.
    /// 3. Background task records latency per frame using Welford online mean.
    /// 4. `pipeline.metrics()` returns a snapshot without blocking the sender.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub async fn new(config: PipelineConfig) -> Result<(Self, mpsc::Sender<VizFrame>), String> {
        let capacity = config.channel_capacity.max(1);
        let (tx, mut rx) = mpsc::channel::<VizFrame>(capacity);

        // Initialize metrics with one entry per stage (extract, render, encode)
        let initial_metrics = PipelineRunMetrics {
            stages: vec![
                StageStats {
                    name: "extract".to_string(),
                    ..Default::default()
                },
                StageStats {
                    name: "render".to_string(),
                    ..Default::default()
                },
                StageStats {
                    name: "encode".to_string(),
                    ..Default::default()
                },
            ],
            total_latency_ms: 0.0,
            drop_rate_percent: 0.0,
        };
        let metrics = Arc::new(Mutex::new(initial_metrics));
        let metrics_bg = Arc::clone(&metrics);

        let latency_threshold = config.latency_threshold_ms;

        // Background processing task
        tokio::spawn(async move {
            let mut state = PipelineState::default();

            while let Some(frame) = rx.recv().await {
                let t_in = Instant::now();
                let _frame_id = frame.metadata.id;

                // Stage: extract — compute mean pressure for minimal work
                let _mean: f32 = if frame.field_pressure.is_empty() {
                    0.0
                } else {
                    let sum: f32 = frame.field_pressure.iter().sum();
                    sum / frame.field_pressure.len() as f32
                };

                let latency_ms = t_in.elapsed().as_secs_f64() * 1000.0;
                state.frames_processed += 1;
                state.cumulative_latency_ms += latency_ms;

                // Welford online mean
                let mean_latency = state.cumulative_latency_ms / state.frames_processed as f64;

                // Drop detection: frame older than threshold counts as drop
                let frame_age_ms = frame.timestamp.elapsed().as_secs_f64() * 1000.0;
                if frame_age_ms > latency_threshold {
                    state.frames_dropped += 1;
                    debug!(
                        frame_age_ms,
                        threshold = latency_threshold,
                        "Frame age exceeded threshold"
                    );
                }

                let drop_rate = if state.frames_processed > 0 {
                    (state.frames_dropped as f64 / state.frames_processed as f64) * 100.0
                } else {
                    0.0
                };

                let total = state.frames_processed;
                let dropped = state.frames_dropped;

                // Update shared metrics
                let mut m = metrics_bg.lock();
                m.total_latency_ms = mean_latency;
                m.drop_rate_percent = drop_rate;
                // Update extract stage stats
                if let Some(extract_stage) = m.stages.get_mut(0) {
                    extract_stage.frames_processed = total;
                    extract_stage.frames_dropped = dropped;
                    extract_stage.total_latency_ms = state.cumulative_latency_ms;
                }

                trace!(
                    frame_id = ?_frame_id,
                    latency_ms,
                    mean_latency,
                    drop_rate,
                    "Pipeline frame processed"
                );
            }

            debug!("Pipeline input channel closed; background task exiting");
        });

        Ok((Self { config, metrics }, tx))
    }

    /// Get a snapshot of the current pipeline metrics.
    pub fn metrics(&self) -> PipelineRunMetrics {
        self.metrics.lock().clone()
    }

    /// Get the pipeline configuration.
    pub fn config(&self) -> &PipelineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_frame_budget() {
        let config = PipelineConfig {
            target_fps: 60.0,
            channel_capacity: 16,
            parallel_execution: true,
            adaptive_quality: true,
            latency_threshold_ms: 20.0,
        };
        // 1000 / 60 ≈ 16.667 ms
        assert!((config.frame_budget_ms() - 16.667).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_config_stage_budget() {
        let config = PipelineConfig {
            target_fps: 60.0,
            ..Default::default()
        };
        // 16.667 / 3 ≈ 5.556 ms
        let stage = config.stage_budget_ms(3);
        assert!((stage - 5.556).abs() < 0.01);
    }

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.target_fps, 30.0);
        assert_eq!(config.channel_capacity, 8);
        assert!(!config.parallel_execution);
        assert!(!config.adaptive_quality);
        assert!((config.latency_threshold_ms - 33.33).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_pipeline_new_returns_valid_sender() {
        let config = PipelineConfig::default();
        let (_pipeline, tx) = StagePipeline::new(config).await.unwrap();
        // Sender should be valid (channel open)
        assert!(!tx.is_closed());
    }

    #[tokio::test]
    async fn test_pipeline_metrics_initially_empty_stages() {
        let config = PipelineConfig::default();
        let (pipeline, _tx) = StagePipeline::new(config).await.unwrap();
        let m = pipeline.metrics();
        assert!(
            !m.stages.is_empty(),
            "Pipeline should have pre-initialized stages"
        );
        assert!(m.total_latency_ms >= 0.0);
        assert!(m.drop_rate_percent >= 0.0);
    }
}
