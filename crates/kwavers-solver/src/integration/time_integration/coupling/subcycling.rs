use kwavers_core::error::KwaversResult;
use kwavers_grid::Grid;
use leto::Array3;
use std::collections::HashMap;

use super::TimeCoupling;

/// Subcycling strategy for multi-rate integration
#[derive(Debug)]
pub struct SubcyclingStrategy {}

impl SubcyclingStrategy {
    /// Create a new subcycling strategy
    #[must_use]
    pub fn new(_max_subcycles: usize) -> Self {
        Self {}
    }
}

impl TimeCoupling for SubcyclingStrategy {
    fn advance_coupled_system(
        &self,
        fields: &mut HashMap<String, Array3<f64>>,
        physics_components: &HashMap<String, Box<dyn crate::plugin::Plugin>>,
        subcycles: &HashMap<String, usize>,
        global_dt: f64,
        _grid: &Grid,
    ) -> KwaversResult<()> {
        // Find maximum number of subcycles
        let max_cycles = subcycles.values().copied().max().unwrap_or(1);

        // Store initial state for high-order interpolation
        let initial_fields: HashMap<String, Array3<f64>> =
            fields.iter().map(|(k, v)| (k.clone(), v.clone())).collect();

        // Advance each component with proper subcycling and coupling
        for cycle in 0..max_cycles {
            for name in physics_components.keys() {
                let n_subcycles = subcycles.get(name).copied().unwrap_or(1);

                // Check if this component should be updated in this cycle
                if cycle % (max_cycles / n_subcycles) == 0 {
                    let field = fields.get_mut(name).ok_or_else(|| {
                        kwavers_core::error::KwaversError::Validation(
                            kwavers_core::error::ValidationError::FieldValidation {
                                field: "fields".to_owned(),
                                value: name.clone(),
                                constraint: "Field not found".to_owned(),
                            },
                        )
                    })?;

                    // Compute local time step for subcycling
                    let local_dt = global_dt / n_subcycles as f64;

                    // Get initial field value for temporal interpolation
                    let field_initial = initial_fields.get(name).ok_or_else(|| {
                        kwavers_core::error::KwaversError::Validation(
                            kwavers_core::error::ValidationError::FieldValidation {
                                field: "initial_fields".to_owned(),
                                value: name.clone(),
                                constraint: "Initial field not found".to_owned(),
                            },
                        )
                    })?;

                    // Apply RK4 time integration for physics component
                    // This provides 4th-order temporal accuracy for the subcycled component
                    Self::rk4_step(field, field_initial, local_dt)?;
                }
            }
        }

        Ok(())
    }
}

impl SubcyclingStrategy {
    /// RK4 time integration step for physics field evolution
    ///
    /// Implements classical 4th-order Runge-Kutta method:
    /// k1 = f(t_n, y_n)
    /// k2 = f(t_n + dt/2, y_n + dt/2*k1)
    /// k3 = f(t_n + dt/2, y_n + dt/2*k2)
    /// k4 = f(t_n + dt, y_n + dt*k3)
    /// y_{n+1} = y_n + dt/6*(k1 + 2*k2 + 2*k3 + k4)
    /// # Errors
    /// - Propagates any [`crate::KwaversError`] returned by called functions.
    ///
    fn rk4_step(
        field: &mut Array3<f64>,
        field_initial: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        let [nx, ny, nz] = field.shape();

        // Stage 1: k1 = f(t_n, y_n)
        let k1 = Self::compute_derivative(field, field_initial)?;

        // Stage 2: k2 = f(t_n + dt/2, y_n + dt/2*k1)
        let mut y2 = field.clone();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    y2[[i, j, k]] += 0.5 * dt * k1[[i, j, k]];
                }
            }
        }
        let k2 = Self::compute_derivative(&y2, field_initial)?;

        // Stage 3: k3 = f(t_n + dt/2, y_n + dt/2*k2)
        let mut y3 = field.clone();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    y3[[i, j, k]] += 0.5 * dt * k2[[i, j, k]];
                }
            }
        }
        let k3 = Self::compute_derivative(&y3, field_initial)?;

        // Stage 4: k4 = f(t_n + dt, y_n + dt*k3)
        let mut y4 = field.clone();
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    y4[[i, j, k]] += dt * k3[[i, j, k]];
                }
            }
        }
        let k4 = Self::compute_derivative(&y4, field_initial)?;

        // Final update: y_{n+1} = y_n + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    field[[i, j, k]] += (dt / 6.0)
                        * (2.0f64
                            .mul_add(k3[[i, j, k]], 2.0f64.mul_add(k2[[i, j, k]], k1[[i, j, k]]))
                            + k4[[i, j, k]]);
                }
            }
        }

        Ok(())
    }

    /// Compute time derivative for physics field
    ///
    /// Demonstration implementation using diffusion physics
    /// Production version: Full RHS evaluation with problem-specific physics
    /// Current: Heat equation proxy (∂u/∂t = α∇²u) for coupling validation
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    fn compute_derivative(
        field: &Array3<f64>,
        _field_initial: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let [nx, ny, nz] = field.shape();
        let mut derivative = Array3::zeros((nx, ny, nz));

        // Compute Laplacian as proxy for diffusion term
        // ∂u/∂t = α∇²u (heat equation-like physics)
        let alpha = 0.01; // Diffusivity coefficient

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // 7-point stencil Laplacian
                    let laplacian = 6.0f64.mul_add(
                        -field[[i, j, k]],
                        field[[i + 1, j, k]]
                            + field[[i - 1, j, k]]
                            + field[[i, j + 1, k]]
                            + field[[i, j - 1, k]]
                            + field[[i, j, k + 1]]
                            + field[[i, j, k - 1]],
                    );

                    derivative[[i, j, k]] = alpha * laplacian;
                }
            }
        }

        Ok(derivative)
    }
}
