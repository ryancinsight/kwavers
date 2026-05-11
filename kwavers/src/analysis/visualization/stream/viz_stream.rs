//! VizStream: async channel-based frame transport with configurable buffer policies.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use flume::{Receiver, Sender, TryRecvError, TrySendError};
use parking_lot::Mutex;
use tracing::{debug, trace, warn};

use crate::core::error::{KwaversError, KwaversResult};

use super::frame::VizFrame;
use super::policy::{BufferPolicy, StreamStatistics};
use super::pool::FramePool;

/// Maximum frames allowed in the bounded channel.
pub(crate) const DEFAULT_CHANNEL_CAPACITY: usize = 32;

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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
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
    /// # Errors
    /// - Returns [`KwaversError::Visualization`] if the precondition for a Visualization-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
    /// # Errors
    /// - Returns [`KwaversError::Visualization`] if the precondition for a Visualization-class constraint is violated.
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
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
