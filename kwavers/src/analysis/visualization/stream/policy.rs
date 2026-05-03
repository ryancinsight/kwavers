//! Buffer policy and stream statistics for visualization backpressure management.

use std::time::Instant;

/// Buffer policy for managing backpressure when renderer lags behind simulation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BufferPolicy {
    /// Ring buffer with maximum capacity, drops oldest frames when full.
    /// Parameter: maximum number of frames to retain.
    DropOldest(usize),
    /// Skip new frames when consumer is busy (current frame is being rendered).
    DropLatest,
    /// Block the producer until the consumer processes a frame.
    Block,
    /// Adaptively manage latency based on a target millisecond threshold.
    /// Drops old frames if latency exceeds target.
    AdaptiveLatency(f64),
}

/// Statistics for stream performance monitoring.
#[derive(Debug, Clone, Default)]
pub struct StreamStatistics {
    /// Total frames produced.
    pub frames_produced: u64,
    /// Total frames consumed.
    pub frames_consumed: u64,
    /// Frames dropped due to buffer policy.
    pub frames_dropped: u64,
    /// Average latency from production to consumption (milliseconds).
    pub avg_latency_ms: f64,
    /// Current number of frames in queue.
    pub queue_depth: usize,
    /// Timestamp of last drop event.
    pub last_drop_time: Option<Instant>,
}
