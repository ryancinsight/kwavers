use super::maxwell::ElectromagneticWaveEquation;
use leto::Array3;

/// Photoacoustic coupling trait for EM-acoustic interactions
///
/// Defines the physics of optical absorption → thermal expansion → acoustic wave generation
pub trait PhotoacousticCoupling: ElectromagneticWaveEquation {
    /// Optical absorption coefficient μ_a (m⁻¹)
    fn optical_absorption(&self, position: &[f64]) -> f64;

    /// Grüneisen parameter Γ (dimensionless)
    /// Γ = (β c²)/(C_p) where β is thermal expansion, c speed of sound, C_p specific heat
    fn gruneisen_parameter(&self, position: &[f64]) -> f64;

    /// Reduced scattering coefficient μ_s' (m⁻¹)
    fn reduced_scattering(&self, position: &[f64]) -> f64;

    /// Compute initial pressure from optical fluence Φ (J/m²)
    /// p₀ = Γ μ_a Φ
    fn initial_pressure_from_fluence(
        &self,
        fluence: &Array3<f64>,
        position: &[f64],
    ) -> Array3<f64> {
        let gamma = self.gruneisen_parameter(position);
        let mu_a = self.optical_absorption(position);
        let shape = fluence.shape();
        let mut result = Array3::zeros(shape);
        for (r, &phi) in result.iter_mut().zip(fluence.iter()) {
            *r = gamma * mu_a * phi;
        }
        result
    }
}
