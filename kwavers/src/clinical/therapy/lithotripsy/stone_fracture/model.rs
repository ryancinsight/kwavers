//! Stone fracture mechanics model for lithotripsy simulation.
//!
//! ## Mathematical Foundation
//!
//! Implements cumulative damage mechanics:
//!
//! ```text
//! If σ_tensile > σ_threshold:
//!     ΔD = k · (σ_tensile / σ_threshold - 1)
//! ```
//!
//! - `D`: Damage parameter (0 = intact, 1 = failed)
//! - `k`: Damage rate parameter (0.01 per event)

use super::material::StoneMaterial;
use ndarray::Array3;

/// Stone fracture mechanics model.
#[derive(Debug, Clone)]
pub struct StoneFractureModel {
    material: StoneMaterial,
    damage_field: Array3<f64>,
    fragment_sizes: Vec<f64>,
}

impl StoneFractureModel {
    /// Create new fracture model for given material and grid dimensions.
    pub fn new(material: StoneMaterial, dimensions: (usize, usize, usize)) -> Self {
        Self {
            material,
            damage_field: Array3::zeros(dimensions),
            fragment_sizes: Vec::new(),
        }
    }

    /// Get material properties.
    #[inline]
    pub fn material(&self) -> &StoneMaterial {
        &self.material
    }

    /// Apply stress loading to update damage field.
    ///
    /// Tensile stress (negative pressure in shock wave convention) exceeding
    /// the material tensile strength accumulates damage proportional to overstress.
    pub fn apply_stress_loading(
        &mut self,
        stress_field: &Array3<f64>,
        _dt: f64,
        _strain_rate: f64,
    ) {
        let threshold = self.material.tensile_strength();
        const DAMAGE_RATE: f64 = 0.01;

        for ((i, j, k), &stress) in stress_field.indexed_iter() {
            if stress < -threshold {
                let overstress_ratio = (-stress / threshold) - 1.0;
                self.damage_field[[i, j, k]] += DAMAGE_RATE * (1.0 + overstress_ratio);
                if self.damage_field[[i, j, k]] > 1.0 {
                    self.damage_field[[i, j, k]] = 1.0;
                }
            }
        }
    }

    /// Get current damage field (0 = intact, 1 = fully fractured).
    #[inline]
    pub fn damage_field(&self) -> &Array3<f64> {
        &self.damage_field
    }

    /// Get distribution of fragment sizes (meters).
    #[inline]
    pub fn fragment_sizes(&self) -> &[f64] {
        &self.fragment_sizes
    }
}
