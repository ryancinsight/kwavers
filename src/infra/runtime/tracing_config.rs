//! Structured logging infrastructure using tracing
//!
//! This module provides production-grade observability with the tracing crate,
//! following senior Rust engineer persona requirements.
//!
//! ## Design Principles
//!
//! - **Structured Events**: Use spans and events for hierarchical logging
//! - **Performance**: Zero-cost when disabled, minimal overhead when enabled
//! - **Flexibility**: RUST_LOG environment variable support
//! - **Production Ready**: Proper filtering and formatting
//!
//! ## References
//!
//! - Tracing Documentation: https://docs.rs/tracing
//! - Tokio Tracing Guide: https://tokio.rs/tokio/topics/tracing
//!
//! ## Example
//!
//! ```no_run
//! #[cfg(feature = "structured-logging")]
//! use kwavers::runtime::tracing_config::init_tracing;
//!
//! #[cfg(feature = "structured-logging")]
//! # fn example() {
//! // Initialize tracing with RUST_LOG support
//! init_tracing();
//!  
//! // Use tracing macros
//! tracing::info!("Starting simulation");
//! tracing::debug!(grid_size = 256, "Created computational grid");
//! # }
//! ```

#[cfg(feature = "structured-logging")]
pub use tracing_impl::*;

#[cfg(feature = "structured-logging")]
mod tracing_impl {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    /// Initialize tracing subscriber with RUST_LOG support
    ///
    /// Sets up structured logging with:
    /// - Environment-based filtering (RUST_LOG)
    /// - Pretty-printed output for development
    /// - Hierarchical span tracking
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "structured-logging")]
    /// # fn example() {
    /// use kwavers::runtime::tracing_config::init_tracing;
    ///
    /// // Set RUST_LOG=kwavers=debug before running
    /// init_tracing();
    /// # }
    /// ```
    pub fn init_tracing() {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("kwavers=info"));

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().pretty())
            .init();
    }

    /// Initialize tracing for production with compact formatting
    ///
    /// Configures logging optimized for production environments:
    /// - Compact output for efficiency
    /// - Warning level by default
    /// - Minimal overhead
    pub fn init_tracing_production() {
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("kwavers=warn"));

        tracing_subscriber::registry()
            .with(filter)
            .with(fmt::layer().compact())
            .init();
    }

    /// Create a span for performance measurement
    ///
    /// # Example
    ///
    /// ```no_run
    /// # #[cfg(feature = "structured-logging")]
    /// # fn example() {
    /// use kwavers::runtime::tracing_config::timed_span;
    ///
    /// let _span = timed_span("fdtd_step");
    /// // Simulation step here - time is automatically recorded
    /// # }
    /// ```
    pub fn timed_span(name: &'static str) -> tracing::span::EnteredSpan {
        tracing::info_span!("{}", name).entered()
    }

    /// Macro for instrumenting functions with tracing
    ///
    /// Re-export of tracing::instrument for convenience
    pub use tracing::instrument;

    #[cfg(test)]
    mod tests {
        use super::*;
        use tracing::{debug, info, warn};

        #[test]
        fn test_tracing_initialization() {
            // Initialize tracing for tests
            let _ = tracing_subscriber::fmt().with_test_writer().try_init();

            info!("Test logging initialized");
            debug!(value = 42, "Debug value");
            warn!("Warning message");
        }

        #[test]
        fn test_timed_span() {
            let _ = tracing_subscriber::fmt().with_test_writer().try_init();

            let _span = timed_span("test_operation");
            // Simulate work
            std::thread::sleep(std::time::Duration::from_millis(10));
            // Span automatically closed when dropped
        }
    }
}

#[cfg(not(feature = "structured-logging"))]
pub mod stub {
    //! Stub implementations when structured-logging feature is disabled

    /// Tracing not available - enable "structured-logging" feature
    pub fn init_tracing() {}

    /// Tracing not available - enable "structured-logging" feature
    pub fn init_tracing_production() {}

    /// Tracing not available - enable "structured-logging" feature
    #[derive(Debug)]
    pub struct EnteredSpan;

    /// Tracing not available - enable "structured-logging" feature
    pub fn timed_span(_name: &'static str) -> EnteredSpan {
        EnteredSpan
    }
}

#[cfg(not(feature = "structured-logging"))]
pub use stub::*;
