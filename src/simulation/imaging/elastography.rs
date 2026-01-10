//! Orchestrator for Shear Wave Elastography (SWE)
//!
//! Provides a high-level API for simulating and reconstructing shear wave elastography
//! experiments, integrating wave propagation and inversion algorithms.

use crate::domain::core::error::KwaversResult;
use crate::domain::grid::Grid;
use crate::domain::imaging::ultrasound::elastography::{ElasticityMap, InversionMethod};
use crate::domain::medium::Medium;

// Physics imports
use crate::physics::acoustics::imaging::modalities::elastography::displacement::DisplacementField;
use crate::physics::acoustics::imaging::modalities::elastography::elastic_wave_solver::{
    ElasticBodyForceConfig, ElasticWaveConfig, ElasticWaveField, ElasticWaveSolver,
};
use crate::physics::acoustics::imaging::modalities::elastography::inversion::ShearWaveInversion;
use crate::physics::acoustics::imaging::modalities::elastography::radiation_force::AcousticRadiationForce;

/// High-level orchestrator for shear wave elastography simulations
#[derive(Debug)]
pub struct ShearWaveElastography {
    /// Computational grid
    grid: Grid,
    /// Medium properties
    medium: Box<dyn Medium>,
    /// Elastic wave solver
    solver: ElasticWaveSolver,
    /// Inversion algorithm
    inversion: ShearWaveInversion,
}

impl ShearWaveElastography {
    /// Create new shear wave elastography orchestrator
    ///
    /// # Arguments
    ///
    /// * `grid` - Computational grid
    /// * `medium` - Tissue medium properties
    /// * `method` - Inversion algorithm to use for reconstruction
    /// * `config` - Configuration for elastic wave simulation
    pub fn new<M: Medium + Clone + 'static>(
        grid: &Grid,
        medium: &M,
        method: InversionMethod,
        config: ElasticWaveConfig,
    ) -> KwaversResult<Self> {
        let solver = ElasticWaveSolver::new(grid, medium, config)?;
        let inversion = ShearWaveInversion::new(method);

        Ok(Self {
            grid: grid.clone(),
            medium: Box::new(medium.clone()),
            solver,
            inversion,
        })
    }

    /// Generate shear wave propagation from an ARFI push pulse
    ///
    /// # Arguments
    ///
    /// * `push_location` - Focal point [x, y, z] in meters for the push pulse
    ///
    /// # Returns
    ///
    /// Vector of displacement fields at different time points
    pub fn generate_shear_wave(
        &self,
        push_location: [f64; 3],
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        let arf = AcousticRadiationForce::new(&self.grid, self.medium.as_ref())?;

        // Correctness-first (ARFI-as-forcing):
        // ARFI excitation is modeled as a body-force source term in the momentum equation:
        //
        //   ρ ∂v/∂t = ∇·σ + f
        //
        // rather than an ad-hoc “initial displacement” assignment.
        let body_force: ElasticBodyForceConfig = arf.push_pulse_body_force(push_location)?;

        // Use the solver's one-off override API to avoid requiring `Clone` on `dyn Medium`.
        // The override is applied only for this call and does not mutate the solver's configuration.
        self.solver
            .propagate_waves_with_body_force_only_override(Some(&body_force))
    }

    /// Reconstruct elasticity from tracked displacement data
    ///
    /// # Arguments
    ///
    /// * `displacement_history` - History of displacement fields from simulation
    ///
    /// # Returns
    ///
    /// Reconstructed elasticity map
    pub fn reconstruct_elasticity(
        &self,
        displacement_history: &[ElasticWaveField],
    ) -> KwaversResult<ElasticityMap> {
        // For simplicity, we use the displacement at the final time point for inversion
        // in this high-level API. More advanced methods would use the full history.
        let final_field = displacement_history.last().ok_or_else(|| {
            crate::domain::core::error::KwaversError::Validation(
                crate::domain::core::error::ValidationError::InvalidValue {
                    parameter: "displacement_history".to_string(),
                    value: 0.0,
                    reason: "History cannot be empty".to_string(),
                },
            )
        })?;

        let displacement = DisplacementField {
            ux: final_field.ux.clone(),
            uy: final_field.uy.clone(),
            uz: final_field.uz.clone(),
        };

        self.inversion.reconstruct(&displacement, &self.grid)
    }
}
