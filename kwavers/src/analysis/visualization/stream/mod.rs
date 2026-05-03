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

pub mod frame;
pub mod policy;
pub mod pool;
pub mod viz_stream;

pub use frame::{FrameId, FrameMetadata, VizFrame};
pub use policy::{BufferPolicy, StreamStatistics};
pub use pool::FramePool;
pub use viz_stream::{VizStream, DEFAULT_CHANNEL_CAPACITY};

// Utility re-exports
pub use flume::{
    unbounded, Receiver as FlumeReceiver, Sender as FlumeSender,
    TrySendError as FlumeTrySendError,
};

/// Pipeline configuration and stage processing (requires tokio async-runtime).
#[cfg(feature = "async-runtime")]
pub mod pipeline;

#[cfg(feature = "async-runtime")]
pub use pipeline::StagePipeline;

#[cfg(feature = "async-runtime")]
pub use crate::analysis::visualization::stream_sync::{
    FramePacer, LatencyBudget, PacingStrategy, QualityController, QualityLevel, SyncCoordinator,
    SyncState, SyncStatistics,
};
