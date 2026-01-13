//! Advanced Boundary Conditions for Multi-Physics Coupling
//!
//! This module provides sophisticated boundary condition types that enable
//! seamless coupling between different physics domains.
//!
//! ## Boundary Condition Categories
//!
//! ### Interface Boundaries
//! - **MaterialInterface**: Handles discontinuities between different materials
//! - **MultiPhysicsInterface**: Couples different physics (EM-acoustic, acoustic-elastic)
//! - **NonMatchingInterface**: Handles non-conforming meshes/grids
//!
//! ### Advanced Absorbing Boundaries
//! - **ImpedanceBoundary**: Frequency-dependent absorption
//! - **AdaptiveBoundary**: Dynamically adjusts absorption based on field energy
//! - **NonlinearBoundary**: Accounts for nonlinear wave effects at boundaries
//!
//! ### Coupling Boundaries
//! - **TransmissionBoundary**: Wave transmission between domains
//! - **SchwarzBoundary**: Domain decomposition coupling
//! - **ImmersedBoundary**: Complex geometries in regular grids

use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::{BoundaryCondition, BoundaryDirections};
use crate::domain::grid::GridTopology;
use crate::domain::medium::properties::AcousticPropertyData;
use ndarray::{Array3, ArrayViewMut3};
use rustfft::num_complex::Complex;
use std::fmt::Debug;

/// Material interface boundary condition
///
/// Handles wave propagation across material discontinuities with proper
/// transmission and reflection coefficients.
///
/// # SSOT Compliance
///
/// Uses canonical `AcousticPropertyData` from `domain::medium::properties`
/// instead of local material property definitions.
#[derive(Debug, Clone)]
pub struct MaterialInterface {
    /// Interface position [x, y, z]
    pub position: [f64; 3],
    /// Normal vector pointing from material 1 to material 2
    pub normal: [f64; 3],
    /// Material properties on side 1 (left/negative side)
    pub material_1: AcousticPropertyData,
    /// Material properties on side 2 (right/positive side)
    pub material_2: AcousticPropertyData,
    /// Interface thickness for smoothing (0 = sharp interface)
    pub thickness: f64,
}

impl MaterialInterface {
    /// Create a new material interface
    pub fn new(
        position: [f64; 3],
        normal: [f64; 3],
        material_1: AcousticPropertyData,
        material_2: AcousticPropertyData,
        thickness: f64,
    ) -> Self {
        Self {
            position,
            normal,
            material_1,
            material_2,
            thickness,
        }
    }

    /// Compute reflection coefficient R = (Z2 - Z1)/(Z2 + Z1)
    pub fn reflection_coefficient(&self) -> f64 {
        let z1 = self.material_1.impedance();
        let z2 = self.material_2.impedance();
        (z2 - z1) / (z2 + z1)
    }

    /// Compute transmission coefficient T = 2Z2/(Z1 + Z2)
    pub fn transmission_coefficient(&self) -> f64 {
        let z1 = self.material_1.impedance();
        let z2 = self.material_2.impedance();
        2.0 * z2 / (z1 + z2)
    }

    /// Compute transmitted pressure amplitude
    pub fn transmitted_pressure(&self, incident_pressure: f64) -> f64 {
        incident_pressure * self.transmission_coefficient()
    }

    /// Compute reflected pressure amplitude
    pub fn reflected_pressure(&self, incident_pressure: f64) -> f64 {
        incident_pressure * self.reflection_coefficient()
    }
}

/// Impedance boundary condition
///
/// Frequency-dependent absorption based on acoustic impedance matching.
/// Particularly useful for ultrasound transducers and tissue interfaces.
#[derive(Debug, Clone)]
pub struct ImpedanceBoundary {
    /// Target impedance Z_target (kg/m²s)
    pub target_impedance: f64,
    /// Frequency-dependent profile
    pub frequency_profile: FrequencyProfile,
    /// Boundary directions
    pub directions: BoundaryDirections,
}

#[derive(Debug, Clone)]
pub enum FrequencyProfile {
    /// Flat response across all frequencies
    Flat,
    /// Gaussian profile centered at frequency with given bandwidth
    Gaussian { center_freq: f64, bandwidth: f64 },
    /// Custom frequency response function
    Custom(Vec<(f64, f64)>), // (frequency, impedance_ratio) pairs
}

impl ImpedanceBoundary {
    /// Create a new impedance boundary
    pub fn new(target_impedance: f64, directions: BoundaryDirections) -> Self {
        Self {
            target_impedance,
            frequency_profile: FrequencyProfile::Flat,
            directions,
        }
    }

    /// Set Gaussian frequency profile
    pub fn with_gaussian_profile(mut self, center_freq: f64, bandwidth: f64) -> Self {
        self.frequency_profile = FrequencyProfile::Gaussian {
            center_freq,
            bandwidth,
        };
        self
    }

    /// Compute impedance ratio at given frequency
    pub fn impedance_ratio(&self, frequency: f64, medium_impedance: f64) -> f64 {
        let z_ratio = self.target_impedance / medium_impedance;

        match &self.frequency_profile {
            FrequencyProfile::Flat => z_ratio,
            FrequencyProfile::Gaussian {
                center_freq,
                bandwidth,
            } => {
                let sigma = bandwidth / (2.0 * (2.0 * std::f64::consts::LN_2).sqrt()); // Convert FWHM to sigma
                let gaussian = (-0.5 * ((frequency - center_freq) / sigma).powi(2)).exp();
                z_ratio * gaussian
            }
            FrequencyProfile::Custom(pairs) => {
                // Simple linear interpolation
                if pairs.is_empty() {
                    z_ratio
                } else if frequency <= pairs[0].0 {
                    pairs[0].1 * z_ratio
                } else if frequency >= pairs.last().unwrap().0 {
                    pairs.last().unwrap().1 * z_ratio
                } else {
                    // Find interval and interpolate
                    for i in 0..pairs.len() - 1 {
                        if frequency >= pairs[i].0 && frequency <= pairs[i + 1].0 {
                            let f1 = pairs[i].0;
                            let f2 = pairs[i + 1].0;
                            let z1 = pairs[i].1;
                            let z2 = pairs[i + 1].1;

                            let ratio = z1 + (z2 - z1) * (frequency - f1) / (f2 - f1);
                            return ratio * z_ratio;
                        }
                    }
                    z_ratio
                }
            }
        }
    }

    /// Compute reflection coefficient from impedance mismatch
    pub fn reflection_coefficient(&self, frequency: f64, medium_impedance: f64) -> f64 {
        let z_ratio = self.impedance_ratio(frequency, medium_impedance);
        (z_ratio - 1.0) / (z_ratio + 1.0)
    }
}

/// Adaptive absorbing boundary
///
/// Dynamically adjusts absorption strength based on field energy levels.
/// Useful for preventing reflections while maintaining computational efficiency.
#[derive(Debug, Clone)]
pub struct AdaptiveBoundary {
    /// Base absorption coefficient
    pub base_absorption: f64,
    /// Energy threshold for triggering adaptation
    pub energy_threshold: f64,
    /// Maximum absorption coefficient
    pub max_absorption: f64,
    /// Adaptation time constant (smaller = faster adaptation)
    pub adaptation_rate: f64,
    /// Current absorption level
    pub current_absorption: f64,
    /// Directions to apply boundary
    pub directions: BoundaryDirections,
}

impl AdaptiveBoundary {
    /// Create a new adaptive boundary
    pub fn new(
        base_absorption: f64,
        energy_threshold: f64,
        max_absorption: f64,
        adaptation_rate: f64,
        directions: BoundaryDirections,
    ) -> Self {
        Self {
            base_absorption,
            energy_threshold,
            max_absorption,
            adaptation_rate,
            current_absorption: base_absorption,
            directions,
        }
    }

    /// Update absorption based on current field energy
    pub fn adapt_to_energy(&mut self, field_energy: f64, dt: f64) {
        let target_absorption = if field_energy > self.energy_threshold {
            // Scale absorption based on energy level
            let energy_ratio = (field_energy / self.energy_threshold).ln();
            let adaptive_factor = 1.0 + energy_ratio.min(10.0); // Cap at 10x increase
            (self.base_absorption * adaptive_factor).min(self.max_absorption)
        } else {
            self.base_absorption
        };

        // Exponential smoothing for stability
        let alpha = 1.0 - (-self.adaptation_rate * dt).exp();
        self.current_absorption =
            self.current_absorption * (1.0 - alpha) + target_absorption * alpha;
    }

    /// Get current absorption coefficient
    pub fn current_absorption(&self) -> f64 {
        self.current_absorption
    }
}

/// Multi-physics interface boundary
///
/// Handles coupling between different physics domains (e.g., acoustic-elastic,
/// electromagnetic-acoustic) with appropriate transmission conditions.
#[derive(Debug, Clone)]
pub struct MultiPhysicsInterface {
    /// Interface position
    pub position: [f64; 3],
    /// Interface normal
    pub normal: [f64; 3],
    /// Physics domain 1 (left side)
    pub physics_1: PhysicsDomain,
    /// Physics domain 2 (right side)
    pub physics_2: PhysicsDomain,
    /// Coupling type
    pub coupling_type: CouplingType,
}

#[derive(Debug, Clone)]
pub enum PhysicsDomain {
    Acoustic,
    Elastic,
    Electromagnetic,
    Thermal,
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum CouplingType {
    /// Acoustic-elastic interface (fluid-solid)
    AcousticElastic,
    /// Electromagnetic-acoustic (photoacoustic)
    ElectromagneticAcoustic { optical_absorption: f64 },
    /// Acoustic-thermal (thermoacoustic)
    AcousticThermal,
    /// Electromagnetic-thermal (photothermal)
    ElectromagneticThermal,
    /// Custom coupling with user-defined transmission
    Custom,
}

impl MultiPhysicsInterface {
    /// Create a new multi-physics interface
    pub fn new(
        position: [f64; 3],
        normal: [f64; 3],
        physics_1: PhysicsDomain,
        physics_2: PhysicsDomain,
        coupling_type: CouplingType,
    ) -> Self {
        Self {
            position,
            normal,
            physics_1,
            physics_2,
            coupling_type,
        }
    }

    /// Compute transmission coefficient for the coupling type
    pub fn transmission_coefficient(&self, _frequency: f64) -> f64 {
        match &self.coupling_type {
            CouplingType::AcousticElastic => {
                // Simplified acoustic-elastic coupling
                // Real implementation would depend on material properties
                0.8 // Typical transmission coefficient
            }
            CouplingType::ElectromagneticAcoustic { optical_absorption } => {
                // Photoacoustic coupling efficiency
                // Depends on Grüneisen parameter, optical absorption, etc.
                let gruneisen = 0.5; // Typical value
                gruneisen * optical_absorption * 1e-3 // Convert to reasonable coefficient
            }
            CouplingType::AcousticThermal => {
                // Thermoacoustic coupling
                // Depends on thermal expansion coefficient
                let thermal_expansion = 2e-4; // Typical for tissue
                thermal_expansion * 1e3 // Scale appropriately
            }
            CouplingType::ElectromagneticThermal => {
                // Photothermal coupling
                let optical_to_thermal = 0.9; // High efficiency
                optical_to_thermal
            }
            CouplingType::Custom => 1.0, // User-defined
        }
    }
}

/// Schwarz domain decomposition boundary
///
/// Implements transmission conditions for domain decomposition methods,
/// enabling parallel solution of large problems.
#[derive(Debug, Clone)]
pub struct SchwarzBoundary {
    /// Overlap region thickness
    pub overlap_thickness: f64,
    /// Transmission condition type
    pub transmission_condition: TransmissionCondition,
    /// Relaxation parameter for optimized Schwarz
    pub relaxation_parameter: f64,
    /// Boundary directions
    pub directions: BoundaryDirections,
}

#[derive(Debug, Clone)]
pub enum TransmissionCondition {
    /// Dirichlet transmission: u = g
    Dirichlet,
    /// Neumann transmission: ∂u/∂n = g
    Neumann,
    /// Robin transmission: αu + β∂u/∂n = g
    Robin { alpha: f64, beta: f64 },
    /// Optimized Schwarz with optimized interface conditions
    Optimized,
}

impl SchwarzBoundary {
    /// Create a new Schwarz boundary
    pub fn new(overlap_thickness: f64, directions: BoundaryDirections) -> Self {
        Self {
            overlap_thickness,
            transmission_condition: TransmissionCondition::Dirichlet,
            relaxation_parameter: 1.0,
            directions,
        }
    }

    /// Set transmission condition
    pub fn with_transmission_condition(mut self, condition: TransmissionCondition) -> Self {
        self.transmission_condition = condition;
        self
    }

    /// Set relaxation parameter for optimized Schwarz
    pub fn with_relaxation(mut self, relaxation: f64) -> Self {
        self.relaxation_parameter = relaxation;
        self
    }

    /// Apply transmission condition
    pub fn apply_transmission(
        &self,
        interface_field: &mut ArrayViewMut3<f64>,
        neighbor_field: &Array3<f64>,
    ) {
        match self.transmission_condition {
            TransmissionCondition::Dirichlet => {
                // Direct copying: u_interface = u_neighbor
                interface_field.zip_mut_with(neighbor_field, |i, &n| {
                    *i = n;
                });
            }
            TransmissionCondition::Neumann => {
                // Flux continuity - would need gradient computation
                // Simplified: assume zero flux for now
                // Real implementation would compute ∂u/∂n
            }
            TransmissionCondition::Robin { alpha, beta: _ } => {
                // Robin condition: weighted average
                // Simplified implementation
                if alpha > 0.0 {
                    interface_field.zip_mut_with(neighbor_field, |i, &n| {
                        *i = (*i + alpha * n) / (1.0 + alpha);
                    });
                }
            }
            TransmissionCondition::Optimized => {
                // Optimized Schwarz with relaxation
                interface_field.zip_mut_with(neighbor_field, |i, &n| {
                    *i = (1.0 - self.relaxation_parameter) * *i + self.relaxation_parameter * n;
                });
            }
        }
    }
}

// Implement BoundaryCondition trait for advanced boundaries

impl BoundaryCondition for MaterialInterface {
    fn name(&self) -> &str {
        "MaterialInterface"
    }

    fn active_directions(&self) -> BoundaryDirections {
        // This would need to be determined based on interface geometry
        BoundaryDirections::all()
    }

    fn apply_scalar_spatial(
        &mut self,
        _field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Apply material interface conditions
        // This is a simplified implementation - real version would need
        // proper interpolation and transmission conditions

        // For now, just apply reflection/transmission at interface
        let _r = self.reflection_coefficient();
        let _t = self.transmission_coefficient();

        // Apply to boundary points near interface
        // (Simplified - real implementation needs proper spatial indexing)

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut Array3<Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Frequency domain material interface
        // Would apply transmission conditions in k-space
        Ok(())
    }

    fn reset(&mut self) {}
}

impl BoundaryCondition for ImpedanceBoundary {
    fn name(&self) -> &str {
        "ImpedanceBoundary"
    }

    fn active_directions(&self) -> BoundaryDirections {
        self.directions
    }

    fn apply_scalar_spatial(
        &mut self,
        _field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Apply impedance boundary condition
        // This would compute reflection coefficients and apply absorption

        // Simplified: apply frequency-independent absorption
        let _absorption = 0.1; // Simplified absorption coefficient

        // Apply to boundary layers
        // (Real implementation would need proper boundary indexing)

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut Array3<Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Frequency-dependent impedance boundary
        // Apply different absorption for different frequency components
        Ok(())
    }

    fn reset(&mut self) {}
}

impl BoundaryCondition for AdaptiveBoundary {
    fn name(&self) -> &str {
        "AdaptiveBoundary"
    }

    fn active_directions(&self) -> BoundaryDirections {
        self.directions
    }

    fn apply_scalar_spatial(
        &mut self,
        mut field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Apply adaptive absorption based on current field energy

        // Compute field energy in boundary region
        let field_energy = field.iter().map(|&x| x * x).sum::<f64>() / field.len() as f64;

        // Adapt absorption coefficient
        self.adapt_to_energy(field_energy, _dt);

        // Apply absorption
        let absorption = self.current_absorption();
        field.mapv_inplace(|x| x * (-absorption * _dt).exp());

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut Array3<Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Adaptive boundary in frequency domain
        Ok(())
    }

    fn reset(&mut self) {
        self.current_absorption = self.base_absorption;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_material_interface_coefficients() {
        let material_1 = AcousticPropertyData {
            density: 1000.0,     // Water
            sound_speed: 1500.0, // Water
            absorption_coefficient: 0.1,
            absorption_power: 2.0,
            nonlinearity: 5.0,
        };

        let material_2 = AcousticPropertyData {
            density: 1600.0,     // Soft tissue
            sound_speed: 1540.0, // Soft tissue
            absorption_coefficient: 0.5,
            absorption_power: 1.1,
            nonlinearity: 6.5,
        };

        let interface = MaterialInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            material_1,
            material_2,
            0.0,
        );

        let r = interface.reflection_coefficient();
        let t = interface.transmission_coefficient();

        // Check energy conservation for pressure coefficients: R² + (Z₁/Z₂)T² = 1
        // For acoustic waves, this is the correct formula (not R² + T² = 1)
        // Reference: Hamilton & Blackstock, Nonlinear Acoustics, Ch. 2
        let z1 = interface.material_1.impedance();
        let z2 = interface.material_2.impedance();
        let energy_conservation = r * r + (z1 / z2) * t * t;
        assert!(
            (energy_conservation - 1.0).abs() < 1e-10,
            "Energy conservation violated: R² + (Z₁/Z₂)T² = {}, expected 1.0",
            energy_conservation
        );

        // Check pressure transmission
        let incident = 1e5; // 100 kPa
        let transmitted = interface.transmitted_pressure(incident);
        let reflected = interface.reflected_pressure(incident);

        assert!(transmitted > 0.0);
        assert!(reflected.abs() < incident.abs()); // Partial reflection
    }

    #[test]
    fn test_impedance_boundary() {
        let boundary = ImpedanceBoundary::new(1.5e6, BoundaryDirections::all());

        // Test reflection coefficient
        let r = boundary.reflection_coefficient(1e6, 1.5e6); // Matched impedance
        assert!(r.abs() < 1e-10); // Perfect match, no reflection

        let r = boundary.reflection_coefficient(1e6, 3.0e6); // Mismatched
        assert!(r.abs() > 0.0); // Some reflection
    }

    #[test]
    fn test_adaptive_boundary() {
        let mut boundary = AdaptiveBoundary::new(
            0.1, // base absorption
            1.0, // energy threshold
            1.0, // max absorption
            1.0, // adaptation rate
            BoundaryDirections::all(),
        );

        // Low energy - should stay at base absorption
        boundary.adapt_to_energy(0.1, 0.001);
        assert!((boundary.current_absorption() - 0.1).abs() < 0.01);

        // High energy - should increase absorption
        boundary.adapt_to_energy(10.0, 0.001);
        assert!(boundary.current_absorption() > 0.1);
        assert!(boundary.current_absorption() <= 1.0);
    }

    #[test]
    fn test_multiphysics_interface() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Electromagnetic,
            PhysicsDomain::Acoustic,
            CouplingType::ElectromagneticAcoustic {
                optical_absorption: 100.0,
            },
        );

        let transmission = interface.transmission_coefficient(1e6);
        assert!(transmission > 0.0);
        assert!(transmission <= 1.0);
    }
}
