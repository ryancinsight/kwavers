//! Synchronization Module - Simulation-to-Render Coordination
//!
//! ## Mathematical Foundation
//!
//! **Theorem: Frame Pacing and Synchronization**
//! ```text
//! Given:
//! - T_sim: simulation timestep
//! - T_target: target display interval (1/target_fps)
//! - L_pipeline: total pipeline latency
//!
//! Frame pacing constraint:
//! T_display[n+1] - T_display[n] ≥ T_target
//!
//! Synchronization condition:
//! T_display[n] = T_sim[m] + L_pipeline ≤ T_deadline[n]
//!
//! where T_deadline[n] = T_start + n × T_target
//! ```
//!
//! **Adaptive Quality Control**:
//! ```text
//! Q[n+1] = Q[n] × (1 - α) + Q_target × α
//!
//! where:
//! - Q_target = 1.0 if L_total ≤ T_target × 0.8
//! - Q_target = 0.7 if T_target × 0.8 < L_total ≤ T_target × 0.95
//! - Q_target = 0.5 if T_target × 0.95 < L_total ≤ T_target
//! - Q_target = 0.3 if L_total > T_target
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::{Mutex, RwLock};
use tokio::sync::Notify;
use tracing::{debug, info, instrument, trace, warn};

use crate::core::error::{KwaversError, KwaversResult};

pub mod budget;
pub mod coordinator;
pub mod pacer;
pub mod quality;
pub mod state;

pub use budget::*;
pub use coordinator::*;
pub use pacer::*;
pub use quality::*;
pub use state::*;
