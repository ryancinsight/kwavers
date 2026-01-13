//! Electromagnetic physics traits and interfaces
//!
//! This module defines the trait-based interfaces for electromagnetic wave
//! physics, including Maxwell's equations, photoacoustic coupling, and
//! plasmonic effects.

use super::{materials::EMMaterialDistribution, types::*};
use crate::domain::field::EMFields;

/// Core electromagnetic wave equation trait
///
/// Defines the mathematical structure for solving Maxwell's equations.
/// Implementations can use FDTD, FEM, spectral methods, or analytical solutions.
pub trait ElectromagneticWaveEquation: Send + Sync {
    /// Get electromagnetic spatial dimension
    fn em_dimension(&self) -> EMDimension;

    /// Get material properties at current time
    fn material_properties(&self) -> &EMMaterialDistribution;

    /// Get current electromagnetic fields
    fn em_fields(&self) -> &EMFields;

    /// Compute wave impedance η = √(μ/ε) (Ω)
    fn wave_impedance(&self) -> ndarray::ArrayD<f64> {
        let props = self.material_properties();
        let eps0 = 8.854e-12; // Vacuum permittivity
        let mu0 = 4.0 * std::f64::consts::PI * 1e-7; // Vacuum permeability

        // η = √(μ/ε) where μ = μ_r * μ₀, ε = ε_r * ε₀
        let mu = &props.permeability * mu0;
        let eps = &props.permittivity * eps0;

        ndarray::Zip::from(&mu)
            .and(&eps)
            .map_collect(|&m, &e| (m / e).sqrt())
    }

    /// Compute skin depth δ = √(2/(ωμσ)) (m)
    fn skin_depth(&self, frequency: f64) -> ndarray::ArrayD<f64> {
        let props = self.material_properties();
        let mu0 = 4.0 * std::f64::consts::PI * 1e-7;
        let omega = 2.0 * std::f64::consts::PI * frequency;

        let mu = &props.permeability * mu0;
        let sigma = &props.conductivity;

        // δ = √(2/(ωμσ))
        ndarray::Zip::from(&mu).and(sigma).map_collect(|&m, &s| {
            if s > 0.0 {
                (2.0 / (omega * m * s)).sqrt()
            } else {
                f64::INFINITY // No skin effect in insulators
            }
        })
    }

    /// Solve Maxwell's equations for one time step
    fn step_maxwell(&mut self, dt: f64) -> Result<(), String>;

    /// Apply electromagnetic boundary conditions
    fn apply_em_boundary_conditions(&mut self, fields: &mut EMFields);

    /// Check electromagnetic physics constraints
    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String>;
}

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

/// Plasmonic enhancement trait for surface plasmon effects
///
/// Models enhanced electromagnetic fields near metallic nanostructures
pub trait PlasmonicEnhancement: ElectromagneticWaveEquation {
    /// Surface plasmon resonance frequency ω_res (rad/s)
    fn plasmon_resonance_frequency(
        &self,
        _nanoparticle_radius: f64,
        dielectric_constant: f64,
    ) -> f64 {
        // Drude model for spherical nanoparticles
        // ω_res² = ω_p² / (1 + 2ε_m / ε_d) where ε_m, ε_d are dielectric constants
        // This is a simplified approximation

        let omega_p = 1.2e16; // Plasma frequency for gold (approximate)
        let eps_m = -2.0; // Metal dielectric constant (approximate for visible)
        let eps_d = dielectric_constant;

        omega_p * (1.0 / (1.0 + 2.0 * eps_m / eps_d)).sqrt()
    }

    /// Local field enhancement factor |E_local|/|E_incident|
    fn field_enhancement_factor(
        &self,
        _position: &[f64],
        nanoparticle_geometry: &NanoparticleGeometry,
    ) -> f64 {
        // This would implement the full Mie theory or quasistatic approximation
        // For now, return a placeholder enhancement factor
        match nanoparticle_geometry {
            NanoparticleGeometry::Sphere { radius } => {
                // Simplified quasistatic enhancement for sphere
                3.0 * (2.0 * radius / (2.0 * radius + 1.0)) // Rough approximation
            }
            _ => 1.0, // No enhancement
        }
    }

    /// Near-field coupling between nanoparticles
    fn near_field_coupling(
        &self,
        particle1_pos: &[f64],
        particle2_pos: &[f64],
        wavelength: f64,
    ) -> f64 {
        // Dipole-dipole coupling approximation
        let distance = ((particle1_pos[0] - particle2_pos[0]).powi(2)
            + (particle1_pos[1] - particle2_pos[1]).powi(2)
            + (particle1_pos[2] - particle2_pos[2]).powi(2))
        .sqrt();

        if distance > 0.0 {
            let k = 2.0 * std::f64::consts::PI / wavelength;
            // Coupling strength ∝ 1/r³ exp(ikr)
            (1.0 / distance.powi(3)) * (k * distance).cos()
        } else {
            0.0
        }
    }

    /// Purcell factor for enhanced emission rates
    fn purcell_factor(&self, position: &[f64], wavelength: f64) -> f64 {
        // Purcell factor quantifies enhancement of spontaneous emission
        // F = (3/2π²) (λ/n)³ (Q/V) where Q is quality factor, V mode volume

        let enhancement = self
            .field_enhancement_factor(position, &NanoparticleGeometry::Sphere { radius: 15e-9 });
        let quality_factor = 10.0; // Typical plasmonic Q-factor
        let mode_volume = 15e-9_f64.powi(3) * 10.0;

        let lambda_over_n = wavelength / 1.5; // Effective wavelength in medium
        let base_factor = 3.0 / (2.0 * std::f64::consts::PI * std::f64::consts::PI)
            * (lambda_over_n / wavelength).powi(3)
            * (quality_factor / mode_volume);

        base_factor * enhancement * enhancement // |E|⁴ dependence
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    // Mock implementation for testing traits
    struct MockEMSolver {
        materials: EMMaterialDistribution,
        fields: EMFields,
    }

    impl MockEMSolver {
        fn new() -> Self {
            let vacuum_props =
                crate::domain::medium::properties::ElectromagneticPropertyData::vacuum();
            let materials = crate::physics::electromagnetic::equations::materials::EMMaterialUtils::create_uniform_distribution(&[10, 10], vacuum_props);
            let electric = ndarray::ArrayD::zeros(ndarray::IxDyn(&[10, 10, 2]));
            let magnetic = ndarray::ArrayD::zeros(ndarray::IxDyn(&[10, 10, 2]));
            let fields = EMFields::new(electric, magnetic);

            Self { materials, fields }
        }
    }

    impl ElectromagneticWaveEquation for MockEMSolver {
        fn em_dimension(&self) -> EMDimension {
            EMDimension::Two
        }
        fn material_properties(&self) -> &EMMaterialDistribution {
            &self.materials
        }
        fn em_fields(&self) -> &EMFields {
            &self.fields
        }

        fn step_maxwell(&mut self, _dt: f64) -> Result<(), String> {
            Ok(())
        }
        fn apply_em_boundary_conditions(&mut self, _fields: &mut EMFields) {}
        fn check_em_constraints(&self, _fields: &EMFields) -> Result<(), String> {
            Ok(())
        }
    }

    #[test]
    fn test_wave_impedance_calculation() {
        let solver = MockEMSolver::new();
        let impedance = solver.wave_impedance();

        // Vacuum impedance should be approximately 377 Ω
        let vacuum_impedance = impedance.iter().next().unwrap();
        assert!((vacuum_impedance - 377.0).abs() < 1.0);
    }

    #[test]
    fn test_skin_depth_insulator() {
        let solver = MockEMSolver::new();
        let skin_depth = solver.skin_depth(1e9); // 1 GHz

        // Insulator should have infinite skin depth
        assert!(skin_depth.iter().all(|&d| d.is_infinite()));
    }
}
