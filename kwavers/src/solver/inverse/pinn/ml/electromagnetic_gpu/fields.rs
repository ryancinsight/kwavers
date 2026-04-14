use ndarray::{Array4, Array5};

/// Electromagnetic field data
#[derive(Debug, Clone)]
pub struct EMFieldData {
    /// Electric field components [time, x, y, z, component]
    pub electric_field: Array5<f64>,
    /// Magnetic field components [time, x, y, z, component]
    pub magnetic_field: Array5<f64>,
    /// Current density [time, x, y, z, component]
    pub current_density: Array5<f64>,
    /// Charge density [time, x, y, z]
    pub charge_density: Array4<f64>,
    /// Time points (s)
    pub time_points: Vec<f64>,
    /// Spatial coordinates
    pub coordinates: [Vec<f64>; 3],
}
