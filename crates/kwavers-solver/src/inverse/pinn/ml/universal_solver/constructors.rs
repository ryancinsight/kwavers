//! `UniversalPINNSolver` factory constructors.
//!
//! SRP: changes when domain registration logic or device setup changes.

use super::solver::UniversalPINNSolver;
use crate::inverse::pinn::ml::physics::PhysicsDomainRegistry;
use kwavers_core::constants::fundamental::{
    SOUND_SPEED_AIR, VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY,
};
use kwavers_core::constants::numerical::MHZ_TO_HZ;
use kwavers_core::constants::tissue_acoustics::DENSITY_AIR;
use kwavers_core::error::KwaversResult;
use std::collections::HashMap;

impl<B: coeus_ops::BackendOps<f32> + coeus_ops::CpuBackend + Default> UniversalPINNSolver<B>
where
    B::DeviceBuffer<f32>:
        coeus_core::CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    /// Create a new universal PINN solver
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn new() -> KwaversResult<Self> {
        Ok(Self {
            physics_registry: PhysicsDomainRegistry::new(),
            models: HashMap::new(),
            configs: HashMap::new(),
            stats: HashMap::new(),
        })
    }

    /// Create a universal solver with all available physics domains pre-registered.
    ///
    /// Registers acoustic wave, electromagnetic (electrostatic, magnetostatic,
    /// quasi-static), and thermal domains.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn with_all_domains() -> KwaversResult<Self> {
        let mut solver = Self::new()?;

        let acoustic_linear = super::super::acoustic_wave::AcousticWaveDomain::new(
            super::super::acoustic_wave::AcousticProblemType::Linear,
            SOUND_SPEED_AIR,
            DENSITY_AIR,
            None,
        );
        solver.register_physics_domain(acoustic_linear)?;

        let em_electrostatic = super::super::electromagnetic::ElectromagneticDomain::new(
            super::super::electromagnetic::EMProblemType::Electrostatic,
            VACUUM_PERMITTIVITY,
            VACUUM_PERMEABILITY,
            0.0,
            vec![1.0, 1.0],
        );
        solver.register_physics_domain(em_electrostatic)?;

        let em_magnetostatic = super::super::electromagnetic::ElectromagneticDomain::new(
            super::super::electromagnetic::EMProblemType::Magnetostatic,
            VACUUM_PERMITTIVITY,
            VACUUM_PERMEABILITY,
            0.0,
            vec![1.0, 1.0],
        );
        solver.register_physics_domain(em_magnetostatic)?;

        let em_quasi_static = super::super::electromagnetic::ElectromagneticDomain::new(
            super::super::electromagnetic::EMProblemType::QuasiStatic,
            VACUUM_PERMITTIVITY,
            VACUUM_PERMEABILITY,
            0.0,
            vec![1.0, 1.0],
        );
        solver.register_physics_domain(em_quasi_static)?;

        Ok(solver)
    }

    /// Create universal solver for cavitation-sonoluminescence-electromagnetic coupling.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by called functions.
    ///
    pub fn with_cavitation_sonoluminescence_coupling() -> KwaversResult<Self> {
        let mut solver = Self::new()?;

        let cavitation_config = super::super::cavitation_coupled::CavitationCouplingConfig {
            enable_coupling: true,
            coupling_strength: 0.1,
            bubble_params: Default::default(),
            bubbles_per_point: 1,
            multi_bubble_effects: false,
            nonlinear_acoustic: true,
            center_frequency: 2.5 * MHZ_TO_HZ,
            sound_speed: kwavers_core::constants::fundamental::SOUND_SPEED_TISSUE,
            domain_size: vec![0.1, 0.1, 0.1],
        };
        let cavitation_domain = super::super::cavitation_coupled::CavitationCoupledDomain::new(
            cavitation_config,
            super::super::cavitation_coupled::CavitationCouplingType::Strong,
            vec![1e-2, 1e-2],
        );
        solver.register_physics_domain(cavitation_domain)?;

        let sl_config = super::super::sonoluminescence_coupled::SonoluminescenceCouplingConfig {
            enable_coupling: true,
            coupling_efficiency: 0.001,
            emission_params: Default::default(),
            grid_shape: (32, 32, 32),
            grid_spacing: (3e-4, 3e-4, 3e-4),
            spectral_resolution: true,
            wavelength_range: (300e-9, 800e-9),
            n_wavelengths: 20,
        };
        let sonoluminescence_domain =
            super::super::sonoluminescence_coupled::SonoluminescenceCoupledDomain::new(
                sl_config,
                super::super::sonoluminescence_coupled::SonoluminescenceCouplingType::SpectralCoupling,
            );
        solver.register_physics_domain(sonoluminescence_domain)?;

        let em_wave_propagation = super::super::electromagnetic::ElectromagneticDomain::new(
            super::super::electromagnetic::EMProblemType::WavePropagation,
            VACUUM_PERMITTIVITY,
            VACUUM_PERMEABILITY,
            0.0,
            vec![1e-2, 1e-2],
        );
        solver.register_physics_domain(em_wave_propagation)?;

        Ok(solver)
    }
}
