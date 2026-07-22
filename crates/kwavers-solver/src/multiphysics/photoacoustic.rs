//! Photoacoustic multi-physics solver
//!
//! This module implements the coupled photoacoustic solver that integrates
//! electromagnetic (optical) and acoustic physics.

use kwavers_core::constants::fundamental::{DENSITY_WATER_NOMINAL, SOUND_SPEED_WATER_SIM};
use kwavers_core::constants::numerical::FOUR_PI;
use kwavers_core::constants::thermodynamic::BODY_TEMPERATURE_C;
use aequitas::systems::si::{quantities::Length, units::PerMeter};
use hyperion::{
    TransportError,
    quantity::PathLength,
    transport::DiffusionCoefficients,
};
use kwavers_core::error::{KwaversError, KwaversResult, ValidationError};
use kwavers_field::EMFields;
use kwavers_physics::electromagnetic::equations::{
    EMMaterialDistribution, ElectromagneticWaveEquation, PhotoacousticCoupling,
};
use kwavers_physics::electromagnetic::photoacoustic::GrueneisenModel;
use leto::{ArrayD, VecStorage};

/// Photoacoustic solver implementation
pub struct PhotoacousticSolver<T: ElectromagneticWaveEquation> {
    /// Electromagnetic wave solver
    pub em_solver: T,
    /// Grüneisen parameter for thermoelastic coupling
    pub gruneisen: GrueneisenModel,
    /// Optical absorption properties
    pub optical_properties: DiffusionCoefficients<f64>,
    /// Initial acoustic pressure field
    pub initial_pressure: Option<ArrayD<f64, VecStorage<f64>>>,
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
        gruneisen: GrueneisenModel,
        optical_properties: DiffusionCoefficients<f64>,
    ) -> Self {
        Self {
            em_solver,
            gruneisen,
            optical_properties,
            initial_pressure: None,
        }
    }

    /// Compute initial pressure distribution from optical fluence
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn compute_initial_pressure(
        &mut self,
        fluence: &ArrayD<f64, VecStorage<f64>>,
    ) -> KwaversResult<ArrayD<f64, VecStorage<f64>>> {
        let gamma = self.gruneisen.evaluate(BODY_TEMPERATURE_C);
        let mu_a = self.optical_properties.absorption().in_unit::<PerMeter>();

        // p₀ = Γ μ_a Φ
        let mapped: Vec<f64> = fluence.iter().map(|&phi| gamma * mu_a * phi).collect();
        let pressure = ArrayD::from_shape_vec(fluence.shape(), mapped)
            .expect("invariant: mapped data preserves fluence shape");

        self.initial_pressure = Some(pressure.clone());
        Ok(pressure)
    }

    /// Compute optical fluence using the diffusion approximation to the RTE.
    ///
    /// For a point source at `source_position`, the steady-state fluence is:
    ///
    /// ```text
    /// Φ(r) = S₀ / (4π D |r − rₛ|) · exp(−μ_eff |r − rₛ|)
    /// ```
    ///
    /// where `D = 1 / (3(μ_a + μ_s'))` and `μ_eff = √(3 μ_a (μ_a + μ_s'))`.
    ///
    /// The `evaluation_points` array serves as a **shape template**: the output
    /// has the same dimensionality.  It is interpreted as a 3-D grid whose
    /// physical extent is taken from `source_position` (assumed to be the centre
    /// of the domain with unit spacing unless overridden by a future API change).
    ///
    /// # Arguments
    ///
    /// * `source_position` - Source coordinates `[x, y, z]` in metres.
    /// * `evaluation_points` - Shape template for the output fluence array.
    ///
    /// # Returns
    ///
    /// Fluence array of the same shape as `evaluation_points`.
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    pub fn compute_fluence_diffusion(
        &self,
        source_position: &[f64],
        evaluation_points: &ArrayD<f64, VecStorage<f64>>,
    ) -> KwaversResult<ArrayD<f64, VecStorage<f64>>> {
        let effective_attenuation = self
            .optical_properties
            .effective_attenuation()
            .map_err(map_transport_error)?;
        let d_coeff = self
            .optical_properties
            .diffusion_coefficient()
            .map_err(map_transport_error)?
            .into_quantity()
            .into_base();
        let inv_4pi_d = 1.0 / (FOUR_PI * d_coeff);

        let shape = evaluation_points.shape();
        let ndim = shape.len();

        // Interpret the first 3 (or fewer) dimensions as spatial (nx, ny, nz)
        let nx = if ndim >= 1 { shape[0] } else { 1 };
        let ny = if ndim >= 2 { shape[1] } else { 1 };
        let nz = if ndim >= 3 { shape[2] } else { 1 };

        // Default grid spacing: 1 mm (override via future API)
        let dx = 1e-3_f64;
        let dy = 1e-3_f64;
        let dz = 1e-3_f64;

        // Source position in grid units, clamped to domain
        let sx = if !source_position.is_empty() {
            source_position[0]
        } else {
            (nx as f64 * dx) / 2.0
        };
        let sy = if (source_position.len()) > 1 {
            source_position[1]
        } else {
            (ny as f64 * dy) / 2.0
        };
        let sz = if (source_position.len()) > 2 {
            source_position[2]
        } else {
            (nz as f64 * dz) / 2.0
        };

        // Compute fluence at every grid point
        let total = nx * ny * nz;
        let extra: usize = shape.iter().skip(3).product();
        let extra = extra.max(1);

        let mut data = vec![0.0_f64; total * extra];

        for i in 0..nx {
            let rx = (i as f64).mul_add(dx, -sx);
            for j in 0..ny {
                let ry = (j as f64).mul_add(dy, -sy);
                for k in 0..nz {
                    let rz = (k as f64).mul_add(dz, -sz);
                    let r = (rx * rx + ry * ry + rz * rz).sqrt();

                    // Regularise near singularity (r → 0): clamp to dx/2
                    let r_safe = r.max(dx * 0.5);

                    let path = PathLength::new(Length::from_base(r_safe))
                        .map_err(map_transport_error)?;
                    let transmission = effective_attenuation
                        .optical_depth(path)
                        .map_err(map_transport_error)?
                        .transmission()
                        .into_quantity()
                        .into_base();
                    let phi = inv_4pi_d * transmission / r_safe;

                    let linear = (i * ny + j) * nz + k;
                    // Replicate across any trailing dimensions
                    for e in 0..extra {
                        data[linear * extra + e] = phi;
                    }
                }
            }
        }

        let result = ArrayD::from_shape_vec(evaluation_points.shape(), data).map_err(|e| {
            kwavers_core::error::KwaversError::DimensionMismatch(format!(
                "Fluence shape mismatch: expected {:?}, got flat vec of length {}: {}",
                evaluation_points.shape(),
                total * extra,
                e
            ))
        })?;

        Ok(result)
    }

    /// Get acoustic energy deposited by photoacoustic effect
    pub fn acoustic_energy_deposited(&self) -> Option<f64> {
        self.initial_pressure.as_ref().map(|pressure| {
            // E = (1/(2ρc²)) ∫ p² dV (approximate acoustic energy)
            let rho = DENSITY_WATER_NOMINAL; // kg/m³
            let c = SOUND_SPEED_WATER_SIM; // m/s

            0.5 / (rho * c * c) * pressure.iter().map(|&p| p * p).sum::<f64>()
        })
    }
}

impl<T: ElectromagneticWaveEquation> ElectromagneticWaveEquation for PhotoacousticSolver<T> {
    fn em_dimension(&self) -> kwavers_physics::electromagnetic::equations::EMDimension {
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
        self.em_solver.apply_em_boundary_conditions(fields);
    }

    fn check_em_constraints(&self, fields: &EMFields) -> Result<(), String> {
        self.em_solver.check_em_constraints(fields)
    }
}

impl<T: ElectromagneticWaveEquation> PhotoacousticCoupling for PhotoacousticSolver<T> {
    fn optical_absorption(&self, _position: &[f64]) -> f64 {
        self.optical_properties.absorption().in_unit::<PerMeter>()
    }

    fn gruneisen_parameter(&self, _position: &[f64]) -> f64 {
        self.gruneisen.evaluate(BODY_TEMPERATURE_C)
    }

    fn reduced_scattering(&self, _position: &[f64]) -> f64 {
        self.optical_properties
            .reduced_scattering()
            .in_unit::<PerMeter>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::forward::fdtd::ElectromagneticFdtdSolver;
    use kwavers_grid::Grid;
    use kwavers_physics::electromagnetic::equations::EMMaterialDistribution;

    #[test]
    fn test_photoacoustic_solver() {
        // Create mock EM solver
        let grid = Grid::new(10, 10, 10, 1e-3, 1e-3, 1e-3).unwrap();
        // Use canonical domain composition pattern
        let materials = EMMaterialDistribution::vacuum(&[10, 10, 10]);

        // Use standard FDTD solver from solver module
        let em_solver = ElectromagneticFdtdSolver::new(grid, materials, 1e-12, 4).unwrap();

        // Create photoacoustic solver
        let gruneisen = GrueneisenModel::constant(0.5);
        let optical_props = kwavers_medium::properties::OpticalPropertyData::new(
            10.0, 500.0, 0.9, 1.4,
        )
        .unwrap()
        .diffusion_coefficients()
        .unwrap();

        let mut pa_solver = PhotoacousticSolver::new(em_solver, gruneisen, optical_props);

        // Test fluence to pressure conversion
        let fluence = ArrayD::from_elem(&[5, 5, 5], 100.0).unwrap(); // 100 J/m²
        let pressure = pa_solver.compute_initial_pressure(&fluence).unwrap();

        // Pressure should be positive and proportional to fluence
        assert!(pressure.iter().all(|&p| p > 0.0));
        assert!(pressure.iter().any(|&p| p > 10.0)); // Should be significant pressure
    }
}

fn map_transport_error(error: TransportError<f64>) -> KwaversError {
    KwaversError::Validation(ValidationError::ConstraintViolation {
        message: format!("Hyperion optical transport rejected solver input: {error}"),
    })
}
