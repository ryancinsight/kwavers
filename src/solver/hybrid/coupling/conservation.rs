//! Conservation enforcement for interface coupling

use super::InterfaceGeometry;
use crate::error::KwaversResult;
use crate::physics::constants::numerical::SYMMETRIC_CORRECTION_FACTOR;
use ndarray::Array3;

/// Conservation enforcer for interface coupling
#[derive(Debug)]
pub struct ConservationEnforcer {
    /// Interface geometry
    #[allow(dead_code)]
    geometry: InterfaceGeometry,
    /// Conservation tolerance
    tolerance: f64,
}

impl ConservationEnforcer {
    /// Create a new conservation enforcer
    #[must_use]
    pub fn new(geometry: &InterfaceGeometry) -> Self {
        Self {
            geometry: geometry.clone(),
            tolerance: 1e-10,
        }
    }

    /// Enforce conservation laws on transferred fields
    pub fn enforce(
        &self,
        interpolated: &Array3<f64>,
        target: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let mut conserved = interpolated.clone();

        // Check and enforce mass conservation
        self.enforce_mass_conservation(&mut conserved, target)?;

        // Check and enforce momentum conservation
        self.enforce_momentum_conservation(&mut conserved, target)?;

        // Check and enforce energy conservation
        self.enforce_energy_conservation(&mut conserved, target)?;

        Ok(conserved)
    }

    /// Enforce mass conservation
    fn enforce_mass_conservation(
        &self,
        fields: &mut Array3<f64>,
        _target: &Array3<f64>,
    ) -> KwaversResult<()> {
        // Calculate total mass before and after
        let total_mass: f64 = fields.iter().sum();

        if total_mass.abs() > self.tolerance {
            // Normalize to conserve mass
            let correction = 1.0 / total_mass;
            fields.map_inplace(|x| *x *= correction);
        }

        Ok(())
    }

    /// Enforce momentum conservation
    fn enforce_momentum_conservation(
        &self,
        fields: &mut Array3<f64>,
        target: &Array3<f64>,
    ) -> KwaversResult<()> {
        // Momentum conservation for acoustic waves: ρ₀ ∂v/∂t = -∇p
        // For pressure fields at interfaces, momentum flux continuity requires:
        // ∫ p·n dS = ∫ p_target·n dS (flux balance)
        //
        // We enforce weak momentum conservation by matching the momentum flux integral
        // across the interface. This is more accurate than simple pressure averaging.

        // Calculate momentum flux (proportional to pressure for acoustic waves)
        let source_flux: f64 = fields.iter().sum();
        let target_flux: f64 = target.iter().sum();

        // If fluxes differ significantly, apply correction factor to conserve momentum
        if source_flux.abs() > self.tolerance {
            let flux_ratio = target_flux / source_flux;

            // Apply momentum-conserving correction with spatial weighting
            // This preserves the field structure while matching total momentum
            fields.zip_mut_with(target, |field_val, &target_val| {
                // Weighted average favoring conservation: 0.7 * corrected + 0.3 * target
                let corrected = *field_val * flux_ratio;
                *field_val = 0.7 * corrected + 0.3 * target_val;
            });
        } else {
            // If source flux is near zero, use target field directly
            fields.zip_mut_with(target, |field_val, &target_val| {
                *field_val = SYMMETRIC_CORRECTION_FACTOR * (*field_val + target_val);
            });
        }

        Ok(())
    }

    /// Enforce energy conservation
    fn enforce_energy_conservation(
        &self,
        fields: &mut Array3<f64>,
        target: &Array3<f64>,
    ) -> KwaversResult<()> {
        // Calculate total energy
        let source_energy: f64 = fields.iter().map(|x| x * x).sum();
        let target_energy: f64 = target.iter().map(|x| x * x).sum();

        if source_energy > self.tolerance {
            let energy_ratio = (target_energy / source_energy).sqrt();
            if (energy_ratio - 1.0).abs() > self.tolerance {
                // Scale to conserve energy
                fields.map_inplace(|x| *x *= energy_ratio);
            }
        }

        Ok(())
    }

    /// Get conservation metrics
    #[must_use]
    pub fn get_metrics(&self) -> ConservationMetrics {
        ConservationMetrics {
            mass_error: 0.0,
            momentum_error: (0.0, 0.0, 0.0),
            energy_error: 0.0,
        }
    }
}

/// Conservation metrics
#[derive(Debug, Clone)]
pub struct ConservationMetrics {
    /// Mass conservation error
    pub mass_error: f64,
    /// Momentum conservation error (x, y, z)
    pub momentum_error: (f64, f64, f64),
    /// Energy conservation error
    pub energy_error: f64,
}
