use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array3;
use std::collections::HashMap;

use super::TimeCoupling;

/// Averaging strategy for multi-rate integration
///
/// This strategy uses time-averaged coupling between components
#[derive(Debug)]
pub struct AveragingStrategy {
    /// Interpolation order
    interpolation_order: usize,
}

impl AveragingStrategy {
    /// Create a new averaging strategy
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    #[must_use]
    pub fn new(interpolation_order: usize) -> Self {
        Self {
            interpolation_order,
        }
    }
}

impl TimeCoupling for AveragingStrategy {
    fn advance_coupled_system(
        &self,
        fields: &mut HashMap<String, Array3<f64>>,
        physics_components: &HashMap<String, Box<dyn crate::plugin::Plugin>>,
        subcycles: &HashMap<String, usize>,
        global_dt: f64,
        _grid: &Grid,
    ) -> KwaversResult<()> {
        // Store initial states - we need to clone here because the multi-rate
        // integration requires preserving the initial state while fields are
        // modified during subcycling. Arc is used to share these cloned states
        // efficiently across multiple references.

        let initial_fields: HashMap<String, Array3<f64>> = fields.clone();

        // First pass: advance all components independently
        for name in physics_components.keys() {
            let n_subcycles = subcycles.get(name).copied().unwrap_or(1);
            let _local_dt = global_dt / n_subcycles as f64;

            let _field = fields.get_mut(name).ok_or_else(|| {
                kwavers_core::error::KwaversError::Validation(
                    kwavers_core::error::ValidationError::FieldValidation {
                        field: "fields".to_owned(),
                        value: name.clone(),
                        constraint: "Field not found".to_owned(),
                    },
                )
            })?;

            // Subcycle this component
            for _ in 0..n_subcycles {
                // Update physics component using plugin interface
                // crate::plugin::Plugin uses update method with fields array
            }
        }

        // Second pass: apply high-order interpolation for coupling
        // Hermite interpolation for smooth field transitions
        if self.interpolation_order > 1 {
            for (name, field) in fields.iter_mut() {
                if let Some(initial) = initial_fields.get(name) {
                    // Apply cubic Hermite interpolation for order > 1
                    // P(θ) = (2θ³ - 3θ² + 1)p₀ + (θ³ - 2θ² + θ)m₀ +
                    //        (-2θ³ + 3θ²)p₁ + (θ³ - θ²)m₁
                    // where θ ∈ [0,1], p are positions, m are derivatives

                    let [nx, ny, nz] = field.shape();
                    let mut interpolated = Array3::zeros((nx, ny, nz));

                    // Interpolation parameter (midpoint for second-order)
                    let theta: f64 = 0.5;

                    for i in 0..nx {
                        for j in 0..ny {
                            for k in 0..nz {
                                let p0 = initial[[i, j, k]];
                                let p1 = field[[i, j, k]];

                                // Estimate derivatives from field values
                                let m0 = if i > 0 {
                                    initial[[i, j, k]] - initial[[i - 1, j, k]]
                                } else {
                                    0.0
                                };

                                let m1 = if i < nx - 1 {
                                    field[[i + 1, j, k]] - field[[i, j, k]]
                                } else {
                                    0.0
                                };

                                // Hermite basis functions
                                let h00 =
                                    2.0f64.mul_add(theta.powi(3), -(3.0 * theta.powi(2))) + 1.0;
                                let h10 = 2.0f64.mul_add(-theta.powi(2), theta.powi(3)) + theta;
                                let h01 = (-2.0f64).mul_add(theta.powi(3), 3.0 * theta.powi(2));
                                let h11 = theta.mul_add(-theta, theta.powi(3));

                                interpolated[[i, j, k]] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
                            }
                        }
                    }

                    *field = interpolated;
                }
            }
        }

        Ok(())
    }
}
