//! Refraction calculations for wave propagation

/// Refraction calculator
pub struct RefractionCalculator {
    // Implementation details
}

/// Refraction angles
#[derive(Debug, Clone)]
pub struct RefractionAngles {
    /// Incident angle [radians]
    pub incident: f64,
    /// Refracted angle [radians]
    pub refracted: f64,
    /// Critical angle for total internal reflection [radians]
    pub critical: Option<f64>,
}
