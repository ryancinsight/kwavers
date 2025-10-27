//! Runtime infrastructure for async I/O, tracing, and zero-copy serialization
//!
//! This module provides production-ready infrastructure aligned with senior
//! Rust engineer persona requirements:
//!
//! - **Async I/O**: Tokio-based async runtime for I/O-bound operations
//! - **Structured Logging**: Tracing infrastructure with spans and events
//! - **Zero-Copy**: rkyv-based serialization for high-performance data exchange
//!
//! ## Sprint 138 Implementation
//!
//! These modules address critical infrastructure gaps identified in the
//! comprehensive gap analysis:
//!
//! 1. **tokio**: Async runtime for I/O operations (P0)
//! 2. **tracing**: Structured logging with spans (P0)
//! 3. **rkyv**: Zero-copy serialization (P1)
//! 4. **loom**: Concurrency testing (P1)
//!
//! ## Feature Flags
//!
//! - `async-runtime`: Enable tokio-based async I/O
//! - `structured-logging`: Enable tracing infrastructure
//! - `zero-copy`: Enable rkyv serialization

pub mod async_io;
pub mod tracing_config;
pub mod zero_copy;

// Re-export key types explicitly to avoid ambiguous glob imports
#[cfg(feature = "async-runtime")]
pub use async_io::{spawn_task, AsyncFileReader, AsyncFileWriter};

#[cfg(feature = "structured-logging")]
pub use tracing_config::{init_tracing, init_tracing_production, timed_span};

#[cfg(feature = "zero-copy")]
pub use zero_copy::{deserialize_grid, serialize_grid, SerializableGrid, SimulationData};
