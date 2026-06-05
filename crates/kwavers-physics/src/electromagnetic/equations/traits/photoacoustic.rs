use super::maxwell::ElectromagneticWaveEquation;

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
    fn reduced_scattering(&self, position: &[f64]) -> f64 {
        // Default: isotropic scattering approximation
        // Full implementation would depend on anisotropy factor g
        self.optical_absorption(position) * 10.0 // Typical μ_s' ≈ 10 * μ_a for tissue
    }

    /// Compute initial pressure from optical fluence Φ (J/m²)
    /// p₀ = Γ μ_a Φ
    fn initial_pressure_from_fluence(
        &self,
        fluence: &ndarray::ArrayD<f64>,
        position: &[f64],
    ) -> ndarray::ArrayD<f64> {
        let gamma = self.gruneisen_parameter(position);
        let mu_a = self.optical_absorption(position);
        fluence.mapv(|phi| gamma * mu_a * phi)
    }
}
