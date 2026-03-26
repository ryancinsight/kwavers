use crate::core::error::KwaversResult;
use ndarray::Array3;

use super::kinetics::AblationKinetics;

/// Tissue ablation field solver
#[derive(Debug)]
pub struct AblationField {
    /// Accumulated thermal damage field
    damage: Array3<f64>,
    /// Tissue viability field (0-1)
    viability: Array3<f64>,
    /// Ablation extent (boolean field)
    ablated: Array3<bool>,
    /// Kinetics model
    kinetics: AblationKinetics,
}

impl AblationField {
    /// Create new ablation field
    pub fn new(shape: (usize, usize, usize), kinetics: AblationKinetics) -> Self {
        let (nx, ny, nz) = shape;
        Self {
            damage: Array3::zeros((nx, ny, nz)),
            viability: Array3::ones((nx, ny, nz)),
            ablated: Array3::from_elem((nx, ny, nz), false),
            kinetics,
        }
    }

    /// Update ablation field with new temperature field
    pub fn update(&mut self, temperature: &Array3<f64>, dt: f64) -> KwaversResult<()> {
        let (nx, ny, nz) = (
            self.damage.dim().0,
            self.damage.dim().1,
            self.damage.dim().2,
        );

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let t = temperature[[i, j, k]];

                    // Convert to absolute temperature
                    let t_abs = t + 273.15;

                    // Skip if below ablation threshold
                    if t < self.kinetics.ablation_threshold {
                        continue;
                    }

                    // Calculate damage rate
                    let damage_rate = self.kinetics.damage_rate(t_abs);

                    // Accumulate damage
                    let current_damage = self.damage[[i, j, k]];
                    let new_damage =
                        AblationKinetics::accumulated_damage(current_damage, damage_rate, dt);
                    self.damage[[i, j, k]] = new_damage;

                    // Update viability
                    self.viability[[i, j, k]] = self.kinetics.viability(new_damage);

                    // Check ablation
                    self.ablated[[i, j, k]] = self.kinetics.is_ablated(new_damage);
                }
            }
        }

        Ok(())
    }

    /// Get damage field
    pub fn damage(&self) -> &Array3<f64> {
        &self.damage
    }

    /// Get viability field
    pub fn viability(&self) -> &Array3<f64> {
        &self.viability
    }

    /// Get ablation field
    pub fn ablated(&self) -> &Array3<bool> {
        &self.ablated
    }

    /// Total ablated volume (count of ablated voxels)
    pub fn ablated_volume(&self) -> usize {
        self.ablated.iter().filter(|&&x| x).count()
    }

    /// Reset ablation field
    pub fn reset(&mut self) {
        self.damage.fill(0.0);
        self.viability.fill(1.0);
        self.ablated.fill(false);
    }
}
