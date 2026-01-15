//! Photoacoustic Physics Implementations
//!
//! This module implements photoacoustic coupling physics, including
//! optical absorption, thermal expansion, and pressure wave generation.

use crate::core::error::KwaversResult;
use crate::domain::field::EMFields;
use crate::physics::electromagnetic::equations::{
    EMMaterialDistribution, ElectromagneticWaveEquation, PhotoacousticCoupling,
};
use ndarray::ArrayD;

/// Grüneisen parameter for thermoelastic coupling
#[derive(Debug, Clone)]
pub struct GruneisenParameter {
    /// Grüneisen parameter Γ (dimensionless)
    pub value: f64,
    /// Temperature dependence (dΓ/dT)
    pub temperature_coefficient: Option<f64>,
    /// Pressure dependence (dΓ/dP)
    pub pressure_coefficient: Option<f64>,
}

impl GruneisenParameter {
    /// Create a new Grüneisen parameter
    pub fn new(value: f64) -> Self {
        Self {
            value,
            temperature_coefficient: None,
            pressure_coefficient: None,
        }
    }

    /// Create with temperature dependence
    pub fn with_temperature_dependence(mut self, coeff: f64) -> Self {
        self.temperature_coefficient = Some(coeff);
        self
    }

    /// Create with pressure dependence
    pub fn with_pressure_dependence(mut self, coeff: f64) -> Self {
        self.pressure_coefficient = Some(coeff);
        self
    }

    /// Get Grüneisen parameter value (could depend on conditions)
    pub fn get_value(&self, _temperature: f64, _pressure: f64) -> f64 {
        // For now, return constant value
        // Full implementation would include temperature/pressure dependence
        self.value
    }
}

/// Optical absorption properties
#[derive(Debug, Clone)]
pub struct OpticalAbsorption {
    /// Absorption coefficient μ_a (m⁻¹)
    pub absorption_coefficient: f64,
    /// Reduced scattering coefficient μ_s' (m⁻¹)
    pub reduced_scattering: f64,
    /// Anisotropy factor g (dimensionless, -1 to 1)
    pub anisotropy_factor: f64,
    /// Optical wavelength (m)
    pub wavelength: f64,
}

impl OpticalAbsorption {
    /// Create optical absorption properties
    pub fn new(mu_a: f64, mu_s_prime: f64, g: f64, wavelength: f64) -> Self {
        Self {
            absorption_coefficient: mu_a,
            reduced_scattering: mu_s_prime,
            anisotropy_factor: g,
            wavelength,
        }
    }

    /// Get total attenuation coefficient μ_t = μ_a + μ_s'
    pub fn total_attenuation(&self) -> f64 {
        self.absorption_coefficient + self.reduced_scattering
    }

    /// Get optical penetration depth δ = 1/μ_t
    pub fn penetration_depth(&self) -> f64 {
        1.0 / self.total_attenuation()
    }

    /// Get albedo (scattering probability) a = μ_s' / μ_t
    pub fn albedo(&self) -> f64 {
        let mu_t = self.total_attenuation();
        if mu_t > 0.0 {
            self.reduced_scattering / mu_t
        } else {
            0.0
        }
    }
}

/// Tissue optical properties database
#[derive(Debug)]
pub struct TissueOpticalProperties;

impl TissueOpticalProperties {
    /// Get optical properties for a specific tissue type at wavelength
    pub fn get_properties(tissue_type: &str, wavelength: f64) -> Option<OpticalAbsorption> {
        match tissue_type {
            "blood" => Some(OpticalAbsorption::new(
                200.0, // μ_a ≈ 200 cm⁻¹ at 800 nm
                100.0, // μ_s' ≈ 100 cm⁻¹
                0.99,  // g ≈ 0.99 (highly forward scattering)
                wavelength,
            )),
            "muscle" => Some(OpticalAbsorption::new(
                5.0,  // μ_a ≈ 5 cm⁻¹
                50.0, // μ_s' ≈ 50 cm⁻¹
                0.9,  // g ≈ 0.9
                wavelength,
            )),
            "fat" => Some(OpticalAbsorption::new(
                2.0,  // μ_a ≈ 2 cm⁻¹
                30.0, // μ_s' ≈ 30 cm⁻¹
                0.8,  // g ≈ 0.8
                wavelength,
            )),
            "skin" => Some(OpticalAbsorption::new(
                20.0,  // μ_a ≈ 20 cm⁻¹
                150.0, // μ_s' ≈ 150 cm⁻¹
                0.8,   // g ≈ 0.8
                wavelength,
            )),
            _ => None,
        }
    }
}

/// Photoacoustic solver implementation
pub struct PhotoacousticSolver<T: ElectromagneticWaveEquation> {
    /// Electromagnetic wave solver
    pub em_solver: T,
    /// Grüneisen parameter for thermoelastic coupling
    pub gruneisen: GruneisenParameter,
    /// Optical absorption properties
    pub optical_properties: OpticalAbsorption,
    /// Initial acoustic pressure field
    pub initial_pressure: Option<ArrayD<f64>>,
}

impl<T: ElectromagneticWaveEquation> std::fmt::Debug for PhotoacousticSolver<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhotoacousticSolver")
            .field("gruneisen", &self.gruneisen)
            .field("optical_properties", &self.optical_properties)
            .field("has_initial_pressure", &self.initial_pressure.is_some())
            .finish()
    }
}

impl<T: ElectromagneticWaveEquation> PhotoacousticSolver<T> {
    /// Create a new photoacoustic solver
    pub fn new(
        em_solver: T,
        gruneisen: GruneisenParameter,
        optical_properties: OpticalAbsorption,
    ) -> Self {
        Self {
            em_solver,
            gruneisen,
            optical_properties,
            initial_pressure: None,
        }
    }

    /// Compute initial pressure distribution from optical fluence
    pub fn compute_initial_pressure(
        &mut self,
        fluence: &ArrayD<f64>,
    ) -> KwaversResult<ArrayD<f64>> {
        let gamma = self.gruneisen.get_value(310.0, 1e5); // Body temperature, atmospheric pressure
        let mu_a = self.optical_properties.absorption_coefficient;

        // p₀ = Γ μ_a Φ
        let pressure = fluence.mapv(|phi| gamma * mu_a * phi);

        self.initial_pressure = Some(pressure.clone());
        Ok(pressure)
    }

    /// Compute optical fluence using diffusion approximation
    pub fn compute_fluence_diffusion(
        &self,
        _source_position: &[f64],
        evaluation_points: &ArrayD<f64>,
    ) -> KwaversResult<ArrayD<f64>> {
        // Simplified diffusion approximation
        // Φ(r) ∝ exp(-μ_eff r)/r where μ_eff = √(3 μ_a μ_s')

        let mu_a = self.optical_properties.absorption_coefficient;
        let mu_s_prime = self.optical_properties.reduced_scattering;
        let _mu_eff = (3.0 * mu_a * (mu_a + mu_s_prime)).sqrt();

        // This is a placeholder - real implementation would need proper spatial computation
        let fluence = ArrayD::from_elem(evaluation_points.raw_dim(), 1.0);
        Ok(fluence)
    }

    /// Get acoustic energy deposited by photoacoustic effect
    pub fn acoustic_energy_deposited(&self) -> Option<f64> {
        self.initial_pressure.as_ref().map(|pressure| {
            // E = (1/(2ρc²)) ∫ p² dV (approximate acoustic energy)
            let rho = 1000.0; // kg/m³
            let c = 1500.0; // m/s

            0.5 / (rho * c * c) * pressure.iter().map(|&p| p * p).sum::<f64>()
        })
    }
}

impl<T: ElectromagneticWaveEquation> ElectromagneticWaveEquation for PhotoacousticSolver<T> {
    fn em_dimension(&self) -> crate::physics::electromagnetic::equations::EMDimension {
        self.em_solver.em_dimension()
    }

    fn material_properties(&self) -> &EMMaterialDistribution {
        self.em_solver.material_properties()
    }

    fn em_fields(&self) -> &EMFields {
        self.em_solver.em_fields()
    }

    fn step_maxwell(&mut self, dt: f64) -> Result<(), String> {
        self.em_solver.step_maxwell(dt)
    }

    fn apply_em_boundary_conditions(&mut self, fields: &mut EMFields) {
        self.em_solver.apply_em_boundary_conditions(fields)
    }

    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String> {
        self.em_solver.check_em_constraints(fields)
    }
}

impl<T: ElectromagneticWaveEquation> PhotoacousticCoupling for PhotoacousticSolver<T> {
    fn optical_absorption(&self, _position: &[f64]) -> f64 {
        self.optical_properties.absorption_coefficient
    }

    fn gruneisen_parameter(&self, _position: &[f64]) -> f64 {
        self.gruneisen.get_value(310.0, 1e5)
    }

    fn reduced_scattering(&self, _position: &[f64]) -> f64 {
        self.optical_properties.reduced_scattering
    }
}

/// Pulsed laser source for photoacoustic excitation
#[derive(Debug)]
pub struct PulsedLaser {
    /// Peak power (W)
    pub peak_power: f64,
    /// Pulse duration (s)
    pub pulse_duration: f64,
    /// Repetition rate (Hz)
    pub repetition_rate: f64,
    /// Wavelength (m)
    pub wavelength: f64,
    /// Beam profile
    pub beam_profile: BeamProfile,
}

#[derive(Debug, Clone)]
pub enum BeamProfile {
    Gaussian { beam_radius: f64 },
    FlatTop { beam_radius: f64 },
    Bessel { central_lobe_radius: f64 },
}

impl PulsedLaser {
    /// Create a new pulsed laser
    pub fn new(peak_power: f64, pulse_duration: f64, wavelength: f64) -> Self {
        Self {
            peak_power,
            pulse_duration,
            repetition_rate: 10.0, // 10 Hz default
            wavelength,
            beam_profile: BeamProfile::Gaussian { beam_radius: 1e-3 },
        }
    }

    /// Compute peak fluence (J/m²)
    pub fn peak_fluence(&self) -> f64 {
        match &self.beam_profile {
            BeamProfile::Gaussian { beam_radius } => {
                // For Gaussian beam: Φ₀ = (2E_pulse)/(π w₀²)
                let beam_area = std::f64::consts::PI * beam_radius * beam_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                2.0 * pulse_energy / beam_area
            }
            BeamProfile::FlatTop { beam_radius } => {
                let beam_area = std::f64::consts::PI * beam_radius * beam_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                pulse_energy / beam_area
            }
            BeamProfile::Bessel {
                central_lobe_radius,
            } => {
                // Simplified for central lobe
                let beam_area = std::f64::consts::PI * central_lobe_radius * central_lobe_radius;
                let pulse_energy = self.peak_power * self.pulse_duration;
                pulse_energy / beam_area
            }
        }
    }

    /// Compute average power (W)
    pub fn average_power(&self) -> f64 {
        self.peak_power * self.pulse_duration * self.repetition_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::grid::Grid;
    use crate::physics::electromagnetic::maxwell::FDTD;
    use crate::physics::electromagnetic::equations::EMMaterialDistribution;

    #[test]
    fn test_gruneisen_parameter() {
        let gamma = GruneisenParameter::new(0.5);
        assert_eq!(gamma.get_value(310.0, 1e5), 0.5);
    }

    #[test]
    fn test_optical_absorption() {
        let absorption = OpticalAbsorption::new(10.0, 50.0, 0.9, 800e-9);
        assert_eq!(absorption.total_attenuation(), 60.0);
        assert_eq!(absorption.albedo(), 50.0 / 60.0);
    }

    #[test]
    fn test_tissue_optical_properties() {
        let blood_props = TissueOpticalProperties::get_properties("blood", 800e-9);
        assert!(blood_props.is_some());

        let blood = blood_props.unwrap();
        assert!(blood.absorption_coefficient > 0.0);
        assert!(blood.reduced_scattering > 0.0);
    }

    #[test]
    fn test_pulsed_laser() {
        let laser = PulsedLaser::new(1e6, 10e-9, 800e-9); // 1 MW peak, 10 ns pulse

        let peak_fluence = laser.peak_fluence();
        assert!(peak_fluence > 0.0);

        let avg_power = laser.average_power();
        assert!(avg_power < laser.peak_power); // Average should be less than peak
    }

    #[test]
    fn test_photoacoustic_solver() {
        // Create mock EM solver
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        // Use canonical domain composition pattern
        let materials = EMMaterialDistribution::vacuum(&[10, 10, 10]);

        let em_solver = FDTD::new(grid, materials, 1e-12).unwrap();

        // Create photoacoustic solver
        let gruneisen = GruneisenParameter::new(0.5);
        let optical_props = OpticalAbsorption::new(10.0, 50.0, 0.9, 800e-9);

        let mut pa_solver = PhotoacousticSolver::new(em_solver, gruneisen, optical_props);

        // Test fluence to pressure conversion
        let fluence = ArrayD::from_elem(ndarray::IxDyn(&[5, 5, 5]), 100.0); // 100 J/m²
        let pressure = pa_solver.compute_initial_pressure(&fluence).unwrap();

        // Pressure should be positive and proportional to fluence
        assert!(pressure.iter().all(|&p| p > 0.0));
        assert!(pressure.iter().any(|&p| p > 10.0)); // Should be significant pressure
    }
}
