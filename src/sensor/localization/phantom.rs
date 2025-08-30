//! Phantom targets for calibration

pub use super::calibration::CentroidPhantom;

/// Phantom target for calibration
#[derive(Debug, Clone)]
#[derive(Debug)]
pub struct PhantomTarget {
    /// Target position [x, y, z] in meters
    pub position: [f64; 3],
    /// Target reflectivity
    pub reflectivity: f64,
    /// Target size [m]
    pub size: f64,
}
