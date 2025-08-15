//! Reflection calculations for wave propagation


/// Reflection calculator
pub struct ReflectionCalculator {
    // Implementation details
}

/// Reflection coefficients
#[derive(Debug, Clone)]
pub struct ReflectionCoefficients {
    /// Amplitude reflection coefficient
    pub amplitude: f64,
    /// Phase shift upon reflection [radians]
    pub phase: f64,
    /// Energy reflection coefficient (R = |r|²)
    pub energy: f64,
}