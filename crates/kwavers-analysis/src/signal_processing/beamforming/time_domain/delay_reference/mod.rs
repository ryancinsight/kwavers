//! Time-domain delay reference policy and utilities.

pub mod functions;
pub mod policy;
#[cfg(test)]
mod tests;

pub use functions::{alignment_shifts_s, relative_delays_s};
pub use policy::DelayReference;
