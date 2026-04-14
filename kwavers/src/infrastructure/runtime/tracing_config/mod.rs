//! Structured logging infrastructure using `tracing`.

#[cfg(feature = "structured-logging")]
mod init;
#[cfg(feature = "structured-logging")]
mod span;

#[cfg(feature = "structured-logging")]
pub use init::{init_tracing, init_tracing_production};
#[cfg(feature = "structured-logging")]
pub use span::timed_span;

#[cfg(all(test, feature = "structured-logging"))]
mod tests;
