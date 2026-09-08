#![doc = include_str!("../README.md")]

pub mod arena;
pub mod constants;
pub mod error;
pub mod log;
/// Rejection assertions for the workspace's test modules.
///
/// Gated behind `test-util` so the helpers never enter a default build.
#[cfg(feature = "test-util")]
pub mod test_support;
pub mod time;
pub mod units;
pub mod utils;
