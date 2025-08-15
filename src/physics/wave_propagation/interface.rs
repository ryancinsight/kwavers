//! Interface properties and types for wave propagation

/// Interface type between media
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterfaceType {
    /// Planar interface
    Planar,
    /// Curved interface
    Curved,
    /// Rough interface with given RMS roughness
    Rough(f64),
    /// Layered interface (stratified media)
    Layered,
}

/// Interface properties
#[derive(Debug, Clone)]
pub struct InterfaceProperties {
    /// Interface type
    pub interface_type: InterfaceType,
    /// Surface roughness RMS [m]
    pub roughness: f64,
    /// Interface curvature radius [m] (for curved interfaces)
    pub curvature_radius: Option<f64>,
}