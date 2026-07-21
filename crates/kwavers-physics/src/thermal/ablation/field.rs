use aequitas::systems::si::quantities::{ThermodynamicTemperature, Time};
use asclepius::DamageIntegral;
use kwavers_core::constants::thermodynamic::KELVIN_OFFSET_C;
use kwavers_core::error::{KwaversError, KwaversResult};
use leto::Array3;

use super::kinetics::AblationKinetics;

/// Tissue ablation field solver
#[derive(Debug)]
pub struct AblationField {
    /// Accumulated thermal damage field
    damage: Array3<f64>,
    /// Reusable proposed damage preserves failure atomicity.
    proposed_damage: Array3<f64>,
    /// Tissue viability field (0-1)
    viability: Array3<f64>,
    /// Ablation extent (boolean field)
    ablated: Array3<bool>,
    /// Kinetics model
    kinetics: AblationKinetics,
}

impl AblationField {
    /// Create new ablation field
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(shape: [usize; 3], kinetics: AblationKinetics) -> Self {
        Self {
            damage: Array3::zeros(shape),
            proposed_damage: Array3::zeros(shape),
            viability: Array3::ones(shape),
            ablated: Array3::from_elem(shape, false),
            kinetics,
        }
    }

    /// Update ablation field with new temperature field
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn update(&mut self, temperature: &Array3<f64>, dt: f64) -> KwaversResult<()> {
        if temperature.shape() != self.damage.shape() {
            return Err(KwaversError::DimensionMismatch(format!(
                "ablation temperature shape {:?} does not match damage shape {:?}",
                temperature.shape(),
                self.damage.shape()
            )));
        }
        let (nx, ny, nz) = (
            self.damage.shape()[0],
            self.damage.shape()[1],
            self.damage.shape()[2],
        );
        let step = Time::from_base(dt);

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let temperature = ThermodynamicTemperature::from_base(
                        temperature[[i, j, k]] + KELVIN_OFFSET_C,
                    );
                    let increment = self.kinetics.damage_increment(temperature, step)?;
                    let current =
                        DamageIntegral::new(self.damage[[i, j, k]]).map_err(|source| {
                            KwaversError::InvalidInput(format!(
                                "invalid accumulated ablation damage: {source}"
                            ))
                        })?;
                    self.proposed_damage[[i, j, k]] =
                        if temperature >= self.kinetics.ablation_threshold() {
                            DamageIntegral::new(current.get() + increment.get())
                                .map_err(|source| {
                                    KwaversError::InvalidInput(format!(
                                        "invalid accumulated ablation damage: {source}"
                                    ))
                                })?
                                .get()
                        } else {
                            current.get()
                        };
                }
            }
        }

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let damage = DamageIntegral::new(self.proposed_damage[[i, j, k]])
                        .expect("invariant: proposed ablation damage was validated");
                    self.damage[[i, j, k]] = damage.get();
                    self.viability[[i, j, k]] = AblationKinetics::viability(damage).get();
                    self.ablated[[i, j, k]] = self.kinetics.is_ablated(damage);
                }
            }
        }

        Ok(())
    }

    /// Get damage field
    #[must_use]
    pub fn damage(&self) -> &Array3<f64> {
        &self.damage
    }

    /// Get viability field
    #[must_use]
    pub fn viability(&self) -> &Array3<f64> {
        &self.viability
    }

    /// Get ablation field
    #[must_use]
    pub fn ablated(&self) -> &Array3<bool> {
        &self.ablated
    }

    /// Total ablated volume (count of ablated voxels)
    #[must_use]
    pub fn ablated_volume(&self) -> usize {
        self.ablated.iter().filter(|&&x| x).count()
    }

    /// Reset ablation field
    pub fn reset(&mut self) {
        self.damage.fill(0.0);
        self.proposed_damage.fill(0.0);
        self.viability.fill(1.0);
        self.ablated.fill(false);
    }
}
