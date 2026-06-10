use super::types::{EMProblemType, ElectromagneticBoundarySpec};
use crate::inverse::pinn::ml::adapters::electromagnetic::PinnEMSource;
use crate::inverse::pinn::ml::physics::BoundaryPosition;
use burn::tensor::backend::AutodiffBackend;
use kwavers_core::constants::fundamental::{VACUUM_PERMEABILITY, VACUUM_PERMITTIVITY};
use kwavers_core::error::{KwaversError, KwaversResult};

/// Electromagnetic physics domain implementation
#[derive(Debug)]
pub struct ElectromagneticDomain<B: AutodiffBackend> {
    /// Problem type
    pub problem_type: EMProblemType,
    /// Electric permittivity (F/m)
    pub permittivity: f64,
    /// Magnetic permeability (H/m)
    pub permeability: f64,
    /// Electrical conductivity (S/m)
    pub conductivity: f64,
    /// Speed of light in medium (m/s)
    pub c: f64,
    /// Current sources (adapted from domain layer)
    pub current_sources: Vec<PinnEMSource>,
    /// Boundary conditions
    pub boundary_specs: Vec<ElectromagneticBoundarySpec>,
    /// Domain dimensions [Lx, Ly]
    pub domain_size: Vec<f64>,
    /// Backend marker
    pub(crate) _backend: std::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> Default for ElectromagneticDomain<B> {
    fn default() -> Self {
        let permittivity = VACUUM_PERMITTIVITY;
        let permeability = VACUUM_PERMEABILITY;
        let c = 1.0 / (permittivity * permeability).sqrt();

        Self {
            problem_type: EMProblemType::Electrostatic,
            permittivity,
            permeability,
            conductivity: 0.0, // Perfect dielectric
            c,
            current_sources: Vec::new(),
            boundary_specs: Vec::new(),
            domain_size: vec![1.0, 1.0],
            _backend: std::marker::PhantomData,
        }
    }
}

impl<B: AutodiffBackend> ElectromagneticDomain<B> {
    /// Create a new electromagnetic domain
    pub fn new(
        problem_type: EMProblemType,
        permittivity: f64,
        permeability: f64,
        conductivity: f64,
        domain_size: Vec<f64>,
    ) -> Self {
        let c = 1.0 / (permittivity * permeability).sqrt();

        Self {
            problem_type,
            permittivity,
            permeability,
            conductivity,
            c,
            current_sources: Vec::new(),
            boundary_specs: Vec::new(),
            domain_size,
            _backend: std::marker::PhantomData,
        }
    }

    /// Add a current source (adapted from domain layer)
    pub fn add_current_source(mut self, source: PinnEMSource) -> Self {
        self.current_sources.push(source);
        self
    }

    /// Add a perfect electric conductor boundary
    pub fn add_pec_boundary(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs
            .push(ElectromagneticBoundarySpec::PerfectElectricConductor { position });
        self
    }

    /// Add a perfect magnetic conductor boundary
    pub fn add_pmc_boundary(mut self, position: BoundaryPosition) -> Self {
        self.boundary_specs
            .push(ElectromagneticBoundarySpec::PerfectMagneticConductor { position });
        self
    }

    /// Set problem type
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn with_problem_type(mut self, problem_type: EMProblemType) -> Self {
        self.problem_type = problem_type;
        self
    }

    /// Validate domain configuration
    /// # Errors
    /// - Returns [`KwaversError::Validation`] if the precondition for a Validation-class constraint is violated.
    ///
    pub fn validate(&self) -> KwaversResult<()> {
        if self.permittivity <= 0.0 {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::InvalidValue {
                    parameter: "permittivity".to_string(),
                    value: self.permittivity,
                    reason: "Permittivity must be positive".to_string(),
                },
            ));
        }

        if self.permeability <= 0.0 {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::InvalidValue {
                    parameter: "permeability".to_string(),
                    value: self.permeability,
                    reason: "Permeability must be positive".to_string(),
                },
            ));
        }

        if self.conductivity < 0.0 {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::InvalidValue {
                    parameter: "conductivity".to_string(),
                    value: self.conductivity,
                    reason: "Conductivity cannot be negative".to_string(),
                },
            ));
        }

        if self.c <= 0.0 || !self.c.is_finite() {
            return Err(KwaversError::Validation(
                kwavers_core::error::ValidationError::InvalidValue {
                    parameter: "speed_of_light".to_string(),
                    value: self.c,
                    reason: "Speed of light must be positive and finite".to_string(),
                },
            ));
        }

        Ok(())
    }
}
