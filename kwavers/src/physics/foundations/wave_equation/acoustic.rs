//! Acoustic specific trait extensions

use super::core::WaveEquation;
use ndarray::ArrayD;

/// Acoustic wave equation trait (scalar pressure field)
///
/// Governs propagation of pressure waves in fluids:
///
/// ```text
/// ∂²p/∂t² = c²∇²p + f
/// ```
///
/// where:
/// - p(x,t) is acoustic pressure [Pa]
/// - c(x) is sound speed [m/s]
/// - f(x,t) is acoustic source [Pa/s²]
pub trait AcousticWaveEquation: WaveEquation {
    /// Get sound speed field c(x) [m/s]
    fn sound_speed(&self) -> ArrayD<f64>;

    /// Get density field ρ(x) [kg/m³]
    fn density(&self) -> ArrayD<f64>;

    /// Get absorption coefficient α(x) [Np/m]
    fn absorption(&self) -> ArrayD<f64>;

    /// Compute acoustic energy
    ///
    /// E = ∫ (½ρ|∂p/∂t|² + ½|∇p|²/(ρc²)) dV
    fn acoustic_energy(&self, pressure: &ArrayD<f64>, velocity: &ArrayD<f64>) -> f64;
}
