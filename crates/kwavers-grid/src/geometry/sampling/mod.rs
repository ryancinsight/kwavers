//! Allocation-bounded bridges from Tyche designs to physical domains.

mod counter;
mod design;
mod matrix;

pub(crate) use counter::sample_counter;
pub use design::DesignSamplingExt;
pub(crate) use matrix::collect_points;
