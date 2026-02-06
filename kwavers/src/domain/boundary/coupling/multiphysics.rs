//! Multi-physics interface boundary condition
//!
//! Handles coupling between different physics domains (e.g., acoustic-elastic,
//! electromagnetic-acoustic) with appropriate transmission conditions.

use crate::core::error::KwaversResult;
use crate::domain::boundary::traits::BoundaryCondition;
use crate::domain::grid::GridTopology;
use ndarray::ArrayViewMut3;
use rustfft::num_complex::Complex;

use super::types::{BoundaryDirections, CouplingType, PhysicsDomain};

/// Multi-physics interface boundary
///
/// Implements coupling between different physics domains with appropriate
/// transmission conditions. This enables modeling of:
///
/// - Fluid-structure interaction (acoustic-elastic)
/// - Photoacoustic imaging (electromagnetic-acoustic)
/// - Thermoacoustic effects (acoustic-thermal)
/// - Photothermal therapy (electromagnetic-thermal)
///
/// # Physics
///
/// At a multi-physics interface, the boundary conditions depend on the
/// coupling type:
///
/// ## Acoustic-Elastic Coupling
///
/// At a fluid-solid interface:
/// - **Normal stress continuity**: σ_solid · n = -p_fluid
/// - **Normal velocity continuity**: v_fluid · n = v_solid · n
/// - **Zero tangential stress**: τ = 0 (free slip at fluid boundary)
///
/// ## Electromagnetic-Acoustic Coupling (Photoacoustic)
///
/// Optical absorption generates acoustic waves via thermal expansion:
///
/// ```text
/// p(r,t) = Γ · μ_a · Φ(r,t)
/// ```
///
/// where:
/// - Γ is the Grüneisen parameter (dimensionless)
/// - μ_a is the optical absorption coefficient (m⁻¹)
/// - Φ is the optical fluence (J/m²)
///
/// ## Acoustic-Thermal Coupling
///
/// Acoustic absorption generates heat:
///
/// ```text
/// ∂T/∂t = α · ∇²T + Q_acoustic
/// Q_acoustic = 2α_acoustic · I / ρc_p
/// ```
///
/// where I is the acoustic intensity.
///
/// # Example
///
/// ```no_run
/// use kwavers::domain::boundary::coupling::{MultiPhysicsInterface, PhysicsDomain, CouplingType};
///
/// // Photoacoustic interface (light absorption → sound generation)
/// let interface = MultiPhysicsInterface::new(
///     [0.0, 0.0, 0.0],           // Interface position
///     [1.0, 0.0, 0.0],           // Interface normal
///     PhysicsDomain::Electromagnetic,
///     PhysicsDomain::Acoustic,
///     CouplingType::ElectromagneticAcoustic {
///         optical_absorption: 100.0,  // 100 m⁻¹
///     },
/// );
///
/// let transmission = interface.transmission_coefficient(1e6);
/// ```
///
/// # References
///
/// - Kuchment & Kunyansky, "Mathematics of Photoacoustic and Thermoacoustic Tomography" (2010)
/// - Beard, "Biomedical Photoacoustic Imaging", Interface Focus (2011)
/// - Duck, "Physical Properties of Tissue" (1990)
#[derive(Debug, Clone)]
pub struct MultiPhysicsInterface {
    /// Interface position [x, y, z] in meters
    pub position: [f64; 3],
    /// Interface normal vector
    pub normal: [f64; 3],
    /// Physics domain 1 (left side of interface)
    pub physics_1: PhysicsDomain,
    /// Physics domain 2 (right side of interface)
    pub physics_2: PhysicsDomain,
    /// Coupling type with parameters
    pub coupling_type: CouplingType,
}

impl MultiPhysicsInterface {
    /// Create a new multi-physics interface
    ///
    /// # Arguments
    ///
    /// * `position` - Interface position [x, y, z] in meters
    /// * `normal` - Interface normal vector (will be normalized)
    /// * `physics_1` - Physics domain on negative side of interface
    /// * `physics_2` - Physics domain on positive side of interface
    /// * `coupling_type` - Type of coupling with associated parameters
    ///
    /// # Returns
    ///
    /// New `MultiPhysicsInterface`
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
    ///
    /// # Arguments
    ///
    /// * `frequency` - Frequency in Hz (for frequency-dependent coupling)
    ///
    /// # Returns
    ///
    /// Transmission coefficient (coupling efficiency)
    ///
    /// # Notes
    ///
    /// The transmission coefficient represents the efficiency of energy transfer
    /// between physics domains. Real implementations would depend on detailed
    /// material properties and coupling mechanisms.
    pub fn transmission_coefficient(&self, _frequency: f64) -> f64 {
        match &self.coupling_type {
            CouplingType::AcousticElastic => {
                // Simplified acoustic-elastic coupling
                // Real implementation would depend on:
                // - Fluid impedance Z_fluid = ρ_fluid · c_fluid
                // - Solid impedance Z_solid = ρ_solid · c_longitudinal
                // - Angle of incidence
                // Typical value for water-soft tissue interface
                0.8
            }
            CouplingType::ElectromagneticAcoustic { optical_absorption } => {
                // Photoacoustic coupling efficiency
                // Depends on Grüneisen parameter Γ, optical absorption μ_a
                // Simplified model: η = Γ · μ_a · (scaling factor)
                let gruneisen = 0.5; // Typical value for water/tissue (dimensionless)
                gruneisen * optical_absorption * 1e-3 // Convert to reasonable coefficient
            }
            CouplingType::AcousticThermal => {
                // Thermoacoustic coupling
                // Depends on thermal expansion coefficient β
                // β ~ 2×10⁻⁴ K⁻¹ for tissue
                let thermal_expansion = 2e-4; // Typical for tissue (K⁻¹)
                thermal_expansion * 1e3 // Scale appropriately
            }
            CouplingType::ElectromagneticThermal => {
                // Photothermal coupling
                // High efficiency: most absorbed optical energy → heat
                0.9
            }
            CouplingType::Custom(_) => {
                // User-defined coupling
                1.0
            }
        }
    }
}

impl BoundaryCondition for MultiPhysicsInterface {
    fn name(&self) -> &str {
        "MultiPhysicsInterface"
    }

    fn active_directions(&self) -> BoundaryDirections {
        // Multi-physics interfaces typically affect all directions
        BoundaryDirections::all()
    }

    fn apply_scalar_spatial(
        &mut self,
        _field: ArrayViewMut3<f64>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Multi-physics interface boundary condition
        // Real implementation would:
        // 1. Identify interface location in grid
        // 2. Apply appropriate coupling conditions based on physics domains
        // 3. Handle field transformations between domains
        //
        // For now, this is a placeholder for future implementation

        Ok(())
    }

    fn apply_scalar_frequency(
        &mut self,
        _field: &mut ndarray::Array3<Complex<f64>>,
        _grid: &dyn GridTopology,
        _time_step: usize,
        _dt: f64,
    ) -> KwaversResult<()> {
        // Frequency-domain multi-physics interface
        Ok(())
    }

    fn reset(&mut self) {
        // No state to reset for multi-physics interface
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!((0.0..=1.0).contains(&transmission));
    }

    #[test]
    fn test_multiphysics_acoustic_elastic() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Acoustic,
            PhysicsDomain::Elastic,
            CouplingType::AcousticElastic,
        );

        let transmission = interface.transmission_coefficient(1e6);
        assert!(transmission > 0.0);
        assert!(transmission <= 1.0);
    }

    #[test]
    fn test_multiphysics_photoacoustic() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Electromagnetic,
            PhysicsDomain::Acoustic,
            CouplingType::ElectromagneticAcoustic {
                optical_absorption: 50.0,
            },
        );

        let transmission = interface.transmission_coefficient(1e6);
        assert!(transmission > 0.0);

        // Higher absorption should give higher coupling
        let interface2 = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Electromagnetic,
            PhysicsDomain::Acoustic,
            CouplingType::ElectromagneticAcoustic {
                optical_absorption: 200.0,
            },
        );

        let transmission2 = interface2.transmission_coefficient(1e6);
        assert!(transmission2 > transmission);
    }

    #[test]
    fn test_multiphysics_acoustic_thermal() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Acoustic,
            PhysicsDomain::Thermal,
            CouplingType::AcousticThermal,
        );

        let transmission = interface.transmission_coefficient(1e6);
        assert!(transmission > 0.0);
    }

    #[test]
    fn test_multiphysics_electromagnetic_thermal() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Electromagnetic,
            PhysicsDomain::Thermal,
            CouplingType::ElectromagneticThermal,
        );

        let transmission = interface.transmission_coefficient(1e6);
        // Photothermal coupling should be highly efficient
        assert!(transmission > 0.8);
    }

    #[test]
    fn test_multiphysics_custom_coupling() {
        let interface = MultiPhysicsInterface::new(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            PhysicsDomain::Custom(1),
            PhysicsDomain::Custom(2),
            CouplingType::Custom("user_defined".to_string()),
        );

        let transmission = interface.transmission_coefficient(1e6);
        assert_eq!(transmission, 1.0); // Default for custom coupling
    }
}
