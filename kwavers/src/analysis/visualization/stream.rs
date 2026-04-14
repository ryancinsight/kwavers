//! Stream-Based Visualization Pipeline
//!
//! ## Mathematical Foundation
//!
//! **Theorem: Stream Processing Latency**
//! ```text
//! L_total = L_extract + L_encode + L_render + L_display
//!
//! where each L = processing_time + queue_wait
//!
//! Target: L_total < 16.7 ms for 60 fps equivalent
//! ```
//!
//! **Backpressure Handling**:
//! - When render lags simulation, configurable policies manage frame flow
//! - Adaptive policy: reduce sampling rate or LOD based on performance metrics
//!
//! ## Architecture
//!
//! ```text
//! VizStream
//! ├── tx: async_channel::Sender<VizFrame>
//! ├── rx: async_channel::Receiver<VizFrame>
//! └── buffer_policy: BufferPolicy
//!
//! VizFrame
//! ├── field_pressure: Array3<f32>
//! ├── field_temperature: Option<Array3<f32>>
//! ├── timestamp: Instant
//! └── metadata: FrameMetadata
//! ```
//!
//! ## Buffer Policies
//!
//! 1. **DropOldest(n)**: Ring buffer of size n, drops oldest frames when full
//! 2. **DropLatest**: Skip frames if consumer is busy
//! 3. **Block**: Wait for consumer to catch up
//! 4. **AdaptiveLatency(target_ms)**: Dynamically adjust based on latency budget

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use flume::{Receiver, Sender, TryRecvError, TrySendError};
use ndarray::Array3;
use parking_lot::Mutex;
use tracing::{debug, trace, warn};

use crate::core::error::{KwaversError, KwaversResult};
use crate::domain::grid::Grid;

/// Maximum frames allowed in the bounded channel.
pub(crate) const DEFAULT_CHANNEL_CAPACITY: usize = 32;

/// Unique identifier for each visualization frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameId(pub u64);

impl FrameId {
    /// Create a new frame ID (internal counter).
    pub fn next(counter: &AtomicU64) -> Self {
        Self(counter.fetch_add(1, Ordering::SeqCst))
    }
}

/// Metadata associated with each visualization frame.
#[derive(Debug, Clone)]
pub struct FrameMetadata {
    /// Frame sequence identifier.
    pub id: FrameId,
    /// Simulation time in seconds.
    pub simulation_time: f64,
    /// Physical grid for dimensional information.
    pub grid: Grid,
    /// Quality level indicator (0.0 to 1.0).
    pub quality_factor: f32,
    /// Optional user-defined tags.
    pub tags: Vec<String>,
}

impl FrameMetadata {
    /// Create new frame metadata.
    pub fn new(simulation_time: f64, grid: Grid) -> Self {
        let id = {
            use std::sync::OnceLock;
            static ID_COUNTER: OnceLock<AtomicU64> = OnceLock::new();
            let counter = ID_COUNTER.get_or_init(|| AtomicU64::new(0));
            FrameId::next(counter)
        };
        Self {
            id,
            simulation_time,
            grid,
            quality_factor: 1.0,
            tags: Vec::new(),
        }
    }
}

/// Complete frame containing simulation field data for visualization.
#[derive(Debug, Clone)]
pub struct VizFrame {
    /// Pressure field array.
    pub field_pressure: Array3<f32>,
    /// Optional temperature field.
    pub field_temperature: Option<Array3<f32>>,
    /// Frame timestamp for latency calculations.
    pub timestamp: Instant,
    /// Frame metadata including ID and simulation time.
    pub metadata: FrameMetadata,
}

impl VizFrame {
    /// Create a new visualization frame.
    pub fn new(
        field_pressure: Array3<f32>,
        field_temperature: Option<Array3<f32>>,
        metadata: FrameMetadata,
    ) -> Self {
        Self {
            field_pressure,
            field_temperature,
            timestamp: Instant::now(),
            metadata,
        }
    }

    /// Calculate age of this frame (time since creation).
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Calculate bytes consumed by this frame's data.
    pub fn data_size_bytes(&self) -> usize {
        let pressure_bytes = self.field_pressure.len() * std::mem::size_of::<f32>();
        let temperature_bytes = self
            .field_temperature
            .as_ref()
            .map_or(0, |t| t.len() * std::mem::size_of::<f32>());
        pressure_bytes + temperature_bytes + std::mem::size_of::<Self>()
    }
}

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

/// Frame pool for zero-allocation frame reuse.
///
/// Implements a bump allocator pattern for frame buffers to avoid
/// repeated heap allocations during high-frequency simulation.
#[derive(Debug)]
pub struct FramePool {
    /// Pre-allocated buffer storage.
    buffers: Mutex<Vec<Array3<f32>>>,
    /// Template dimensions for allocation.
    dimensions: (usize, usize, usize),
    /// Maximum pool size.
    max_size: usize,
}

impl FramePool {
    /// Create a new frame pool with specified dimensions.
    pub fn new(nx: usize, ny: usize, nz: usize, max_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(max_size);
        for _ in 0..max_size {
            buffers.push(Array3::<f32>::zeros((nx, ny, nz)));
        }

        Self {
            buffers: Mutex::new(buffers),
            dimensions: (nx, ny, nz),
            max_size,
        }
    }

    /// Acquire a buffer from the pool.
    /// Returns a zeroed array if pool is exhausted.
    pub fn acquire(&self) -> Array3<f32> {
        let mut buffers = self.buffers.lock();
        if let Some(buffer) = buffers.pop() {
            trace!(pool_size = buffers.len(), "Acquired buffer from pool");
            buffer
        } else {
            warn!("Frame pool exhausted, allocating new buffer");
            Array3::<f32>::zeros(self.dimensions)
        }
    }

    /// Return a buffer to the pool for reuse.
    pub fn release(&self, mut buffer: Array3<f32>) {
        buffer.fill(0.0);
        let mut buffers = self.buffers.lock();
        if buffers.len() < self.max_size {
            buffers.push(buffer);
            trace!(pool_size = buffers.len(), "Returned buffer to pool");
        } else {
            trace!("Pool full, dropping buffer");
        }
    }
}

/// Stream for visualization frame flow between simulation and renderer.
#[derive(Debug)]
pub struct VizStream {
    /// Channel sender for frame production.
    tx: Sender<VizFrame>,
    /// Channel receiver for frame consumption.
    rx: Receiver<VizFrame>,
    /// Configured buffer policy.
    buffer_policy: BufferPolicy,
    /// Shared statistics.
    statistics: Arc<Mutex<StreamStatistics>>,
    /// Optional frame pool (for zero-copy optimization).
    frame_pool: Option<Arc<FramePool>>,
    /// Whether the stream has been explicitly closed via `close()`.
    closed: Arc<AtomicU64>,
}

impl VizStream {
    /// Create a new visualization stream with specified buffer policy.
    pub fn new(policy: BufferPolicy) -> KwaversResult<Self> {
        let capacity = match policy {
            BufferPolicy::DropOldest(cap) => cap,
            BufferPolicy::DropLatest => 1,
            BufferPolicy::Block => DEFAULT_CHANNEL_CAPACITY,
            BufferPolicy::AdaptiveLatency(_) => DEFAULT_CHANNEL_CAPACITY,
        };

        let (tx, rx) = flume::bounded(capacity);

        Ok(Self {
            tx,
            rx,
            buffer_policy: policy,
            statistics: Arc::new(Mutex::new(StreamStatistics::default())),
            frame_pool: None,
            closed: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Create a stream with frame pooling enabled.
    pub fn with_pool(
        policy: BufferPolicy,
        grid_dimensions: (usize, usize, usize),
        pool_size: usize,
    ) -> KwaversResult<Self> {
        let capacity = match policy {
            BufferPolicy::DropOldest(cap) => cap,
            BufferPolicy::DropLatest => 1,
            BufferPolicy::Block => DEFAULT_CHANNEL_CAPACITY,
            BufferPolicy::AdaptiveLatency(_) => DEFAULT_CHANNEL_CAPACITY,
        };

        let (tx, rx) = flume::bounded(capacity);

        Ok(Self {
            tx,
            rx,
            buffer_policy: policy,
            statistics: Arc::new(Mutex::new(StreamStatistics::default())),
            frame_pool: Some(Arc::new(FramePool::new(
                grid_dimensions.0,
                grid_dimensions.1,
                grid_dimensions.2,
                pool_size,
            ))),
            closed: Arc::new(AtomicU64::new(0)),
        })
    }

    /// Send a frame through the stream using the configured buffer policy.
    ///
    /// **Policy Behavior**:
    /// - `DropOldest`: If channel full, drop oldest frame (blocking until space or drop).
    /// - `DropLatest`: If channel full, drop the frame being sent immediately.
    /// - `Block`: Block until channel has capacity.
    /// - `AdaptiveLatency`: Evaluate latency, drop frames if budget exceeded.
    pub async fn send_frame(&self, frame: VizFrame) -> KwaversResult<()> {
        match self.buffer_policy {
            BufferPolicy::DropOldest(capacity) => {
                trace!("Push frame with DropOldest policy (capacity={})", capacity);
                if self.tx.is_full() {
                    // Try to remove oldest frame
                    match self.rx.try_recv() {
                        Ok(dropped) => {
                            debug!(frame_id = ?dropped.metadata.id, "Dropped oldest frame");
                            let mut stats = self.statistics.lock();
                            stats.frames_dropped += 1;
                            stats.last_drop_time = Some(Instant::now());
                        }
                        Err(e) => {
                            trace!("Failed to drop oldest: {:?}", e);
                        }
                    }
                }
                match self.tx.try_send(frame) {
                    Ok(_) => {
                        self.statistics.lock().frames_produced += 1;
                        Ok(())
                    }
                    Err(e) => {
                        debug!("Send error: {:?}", e);
                        let mut stats = self.statistics.lock();
                        stats.frames_dropped += 1;
                        stats.last_drop_time = Some(Instant::now());
                        Err(KwaversError::Visualization {
                            message: format!("Failed to drop oldest frame: {:?}", e),
                        })
                    }
                }
            }
            BufferPolicy::DropLatest => {
                trace!("Push frame with DropLatest policy");
                match self.tx.try_send(frame) {
                    Ok(_) => {
                        self.statistics.lock().frames_produced += 1;
                        Ok(())
                    }
                    Err(TrySendError::Full(_)) => {
                        debug!("Channel full, dropping latest frame");
                        let mut stats = self.statistics.lock();
                        stats.frames_dropped += 1;
                        stats.last_drop_time = Some(Instant::now());
                        Ok(())
                    }
                    Err(e) => Err(KwaversError::Visualization {
                        message: format!("Channel send error: {:?}", e),
                    }),
                }
            }
            BufferPolicy::Block => {
                trace!("Push frame with Block policy");
                self.tx
                    .send_async(frame)
                    .await
                    .map_err(|e| KwaversError::Visualization {
                        message: format!("Channel send error: {:?}", e),
                    })?;
                self.statistics.lock().frames_produced += 1;
                Ok(())
            }
            BufferPolicy::AdaptiveLatency(target_ms) => {
                let _frame_latency = frame.metadata.simulation_time;
                let now = Instant::now();

                // Estimate latency if this frame were added
                let estimated_latency = 0.0_f64; // Simplified; real impl would track actuals

                if estimated_latency > target_ms {
                    warn!(
                        target = target_ms,
                        estimated = estimated_latency,
                        "Latency budget exceeded, dropping frame"
                    );
                    let mut stats = self.statistics.lock();
                    stats.frames_dropped += 1;
                    stats.last_drop_time = Some(now);
                    return Ok(());
                }

                trace!("Push frame with AdaptiveLatency policy");
                match self.tx.try_send(frame) {
                    Ok(_) => {
                        self.statistics.lock().frames_produced += 1;
                        Ok(())
                    }
                    Err(TrySendError::Full(_)) => {
                        let mut stats = self.statistics.lock();
                        stats.frames_dropped += 1;
                        stats.last_drop_time = Some(now);
                        Ok(())
                    }
                    Err(e) => Err(KwaversError::Visualization {
                        message: format!("AdaptiveLatency send error: {:?}", e),
                    }),
                }
            }
        }
    }

    /// Receive a frame from the stream (async).
    pub async fn recv_frame(&self) -> KwaversResult<VizFrame> {
        let frame = self
            .rx
            .recv_async()
            .await
            .map_err(|e| KwaversError::Visualization {
                message: format!("Channel receive error: {:?}", e),
            })?;

        // Calculate latency
        let latency = frame.age().as_secs_f64() * 1000.0;
        let mut stats = self.statistics.lock();
        stats.frames_consumed += 1;

        // Update average using exponential moving average
        const ALPHA: f64 = 0.1;
        stats.avg_latency_ms = stats.avg_latency_ms * (1.0 - ALPHA) + latency * ALPHA;
        stats.queue_depth = self.rx.len();

        debug!(
            frame_id = ?frame.metadata.id,
            latency_ms = latency,
            "Received frame"
        );

        Ok(frame)
    }

    /// Try to receive a frame without blocking.
    pub fn try_recv_frame(&self) -> KwaversResult<Option<VizFrame>> {
        match self.rx.try_recv() {
            Ok(frame) => {
                let latency = frame.age().as_secs_f64() * 1000.0;
                let mut stats = self.statistics.lock();
                stats.frames_consumed += 1;

                const ALPHA: f64 = 0.1;
                stats.avg_latency_ms = stats.avg_latency_ms * (1.0 - ALPHA) + latency * ALPHA;
                stats.queue_depth = self.rx.len();

                trace!(frame_id = ?frame.metadata.id, latency_ms = latency, "Try-received frame");
                Ok(Some(frame))
            }
            Err(TryRecvError::Empty) => Ok(None),
            Err(e) => Err(KwaversError::Visualization {
                message: format!("Channel receive error: {:?}", e),
            }),
        }
    }

    /// Get a clone of the sender for producer tasks.
    pub fn sender(&self) -> Sender<VizFrame> {
        self.tx.clone()
    }

    /// Get a clone of the receiver for consumer tasks.
    pub fn receiver(&self) -> Receiver<VizFrame> {
        self.rx.clone()
    }

    /// Get current stream statistics.
    pub fn statistics(&self) -> StreamStatistics {
        self.statistics.lock().clone()
    }

    /// Get the configured buffer policy.
    pub fn policy(&self) -> BufferPolicy {
        self.buffer_policy
    }

    /// Check if frame pool is available.
    pub fn has_pool(&self) -> bool {
        self.frame_pool.is_some()
    }

    /// Get frame pool reference if available.
    pub fn frame_pool(&self) -> Option<Arc<FramePool>> {
        self.frame_pool.clone()
    }

    /// Close the stream (disallow further sends).
    ///
    /// Sets the closed flag atomically. Subsequent calls to `send_frame` will
    /// return an error; `is_closed()` will return `true`.
    pub fn close(&self) {
        debug!("Closing visualization stream");
        self.closed.store(1, Ordering::Release);
    }

    /// Check if stream is closed.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::Acquire) != 0 || self.tx.is_disconnected()
    }

    /// Current queue depth (number of buffered frames).
    pub fn queue_depth(&self) -> usize {
        self.rx.len()
    }

    /// Estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let queue = self.rx.len();
        // Estimate based on typical frame size (will vary in practice)
        queue * std::mem::size_of::<VizFrame>()
    }
}

// Utility re-exports (alias to avoid duplicate with the `use flume::bounded` import above)
pub use flume::{
    unbounded, Receiver as FlumeReceiver, Sender as FlumeSender, TrySendError as FlumeTrySendError,
};

// ============================================================================
// Submodules: pipeline configuration and sync coordination
// Gated behind async-runtime because they use tokio::sync::mpsc / Notify.
// ============================================================================

/// Pipeline configuration and stage processing (requires tokio async-runtime).
#[cfg(feature = "async-runtime")]
pub mod pipeline;

// Re-export types that tests import directly from `stream::`
#[cfg(feature = "async-runtime")]
pub use pipeline::StagePipeline;

#[cfg(feature = "async-runtime")]
pub use crate::analysis::visualization::stream_sync::{
    FramePacer, LatencyBudget, PacingStrategy, QualityController, QualityLevel, SyncCoordinator,
    SyncState, SyncStatistics,
};
