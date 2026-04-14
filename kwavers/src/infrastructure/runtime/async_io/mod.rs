//! Async I/O helpers built on Tokio.

#[cfg(feature = "async-runtime")]
mod reader;
#[cfg(feature = "async-runtime")]
mod writer;

#[cfg(feature = "async-runtime")]
pub use reader::AsyncFileReader;
#[cfg(feature = "async-runtime")]
pub use writer::{spawn_task, AsyncFileWriter};

#[cfg(all(test, feature = "async-runtime"))]
mod tests;
