use super::super::types::{EMWaveType, Polarization};

/// Electromagnetic source trait
pub trait EMSource: Send + Sync {
    /// Get source polarization
    fn polarization(&self) -> Polarization;

    /// Get source wave type
    fn wave_type(&self) -> EMWaveType;

    /// Get source frequency spectrum
    fn frequency_spectrum(&self) -> Vec<f64>;

    /// Get peak electric field amplitude (V/m)
    fn peak_electric_field(&self) -> f64;

    /// Compute time-domain electric field at given time
    fn electric_field_at_time(&self, time: f64, position: &[f64]) -> [f64; 3];

    /// Compute frequency-domain electric field at given frequency
    fn electric_field_at_frequency(
        &self,
        frequency: f64,
        position: &[f64],
    ) -> num_complex::Complex<f64>;

    /// Check if source is active at given time
    fn is_active(&self, time: f64) -> bool;

    /// Get source directivity pattern (radiation pattern)
    fn directivity(&self, direction: &[f64]) -> f64;
}
