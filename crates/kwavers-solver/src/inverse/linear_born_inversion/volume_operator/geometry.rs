//! Utility functions shared by [`super::VolumeOperator`] submodules.

/// Euclidean distance between two 3-D points \[m\].
pub(super) fn distance(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    ((ax - bx).powi(2) + (ay - by).powi(2) + (az - bz).powi(2)).sqrt()
}
