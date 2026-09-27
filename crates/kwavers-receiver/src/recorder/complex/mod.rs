//! Complex recorder with full event detection and field recording.

pub mod recorder;
mod trait_impl;

pub use recorder::Recorder;

#[cfg(test)]
mod tests;
