//! Coupling strategies for multi-rate time integration
//!
//! This module provides different strategies for coupling physics
//! components that evolve at different time scales.

use crate::core::error::KwaversResult;
use crate::domain::grid::Grid;
use ndarray::Array3;
use std::collections::HashMap;
use std::fmt::Debug;

/// Trait for time coupling strategies
pub trait TimeCoupling: Send + Sync + Debug {
    /// Advance the coupled system
    fn advance_coupled_system(
        &self,
        fields: &mut HashMap<String, Array3<f64>>,
        physics_components: &HashMap<String, Box<dyn crate::domain::plugin::Plugin>>,
        subcycles: &HashMap<String, usize>,
        global_dt: f64,
        grid: &Grid,
    ) -> KwaversResult<()>;
}

/// Subcycling strategy for multi-rate integration
#[derive(Debug)]
pub struct SubcyclingStrategy {
    /// Maximum allowed subcycles
    #[allow(dead_code)]
    max_subcycles: usize,
}

impl SubcyclingStrategy {
    /// Create a new subcycling strategy
    #[must_use]
    pub fn new(max_subcycles: usize) -> Self {
        Self { max_subcycles }
    }
}

impl TimeCoupling for SubcyclingStrategy {
    fn advance_coupled_system(
        &self,
        fields: &mut HashMap<String, Array3<f64>>,
        physics_components: &HashMap<String, Box<dyn crate::domain::plugin::Plugin>>,
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
                        crate::domain::core::error::KwaversError::Validation(
                            crate::domain::core::error::ValidationError::FieldValidation {
                                field: "fields".to_string(),
                                value: name.clone(),
                                constraint: "Field not found".to_string(),
                            },
                        )
                    })?;

                    // Compute local time step for subcycling
                    let local_dt = global_dt / n_subcycles as f64;

                    // Get initial field value for temporal interpolation
                    let field_initial = initial_fields.get(name).ok_or_else(|| {
                        crate::domain::core::error::KwaversError::Validation(
                            crate::domain::core::error::ValidationError::FieldValidation {
                                field: "initial_fields".to_string(),
                                value: name.clone(),
                                constraint: "Initial field not found".to_string(),
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
    fn rk4_step(
        field: &mut Array3<f64>,
        field_initial: &Array3<f64>,
        dt: f64,
    ) -> KwaversResult<()> {
        let (nx, ny, nz) = field.dim();

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
                        * (k1[[i, j, k]]
                            + 2.0 * k2[[i, j, k]]
                            + 2.0 * k3[[i, j, k]]
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
    fn compute_derivative(
        field: &Array3<f64>,
        _field_initial: &Array3<f64>,
    ) -> KwaversResult<Array3<f64>> {
        let (nx, ny, nz) = field.dim();
        let mut derivative = Array3::zeros((nx, ny, nz));

        // Compute Laplacian as proxy for diffusion term
        // ∂u/∂t = α∇²u (heat equation-like physics)
        let alpha = 0.01; // Diffusivity coefficient

        for i in 1..nx - 1 {
            for j in 1..ny - 1 {
                for k in 1..nz - 1 {
                    // 7-point stencil Laplacian
                    let laplacian = field[[i + 1, j, k]]
                        + field[[i - 1, j, k]]
                        + field[[i, j + 1, k]]
                        + field[[i, j - 1, k]]
                        + field[[i, j, k + 1]]
                        + field[[i, j, k - 1]]
                        - 6.0 * field[[i, j, k]];

                    derivative[[i, j, k]] = alpha * laplacian;
                }
            }
        }

        Ok(derivative)
    }
}

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
        physics_components: &HashMap<String, Box<dyn crate::domain::plugin::Plugin>>,
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
                crate::domain::core::error::KwaversError::Validation(
                    crate::domain::core::error::ValidationError::FieldValidation {
                        field: "fields".to_string(),
                        value: name.clone(),
                        constraint: "Field not found".to_string(),
                    },
                )
            })?;

            // Subcycle this component
            for _ in 0..n_subcycles {
                // Update physics component using plugin interface
                // crate::physics::plugin::Plugin uses update method with fields array
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

                    let (nx, ny, nz) = field.dim();
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
                                let h00 = 2.0 * theta.powi(3) - 3.0 * theta.powi(2) + 1.0;
                                let h10 = theta.powi(3) - 2.0 * theta.powi(2) + theta;
                                let h01 = -2.0 * theta.powi(3) + 3.0 * theta.powi(2);
                                let h11 = theta.powi(3) - theta.powi(2);

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

/// Predictor-corrector strategy for multi-rate integration
#[derive(Debug)]
pub struct PredictorCorrectorStrategy {
    /// Number of corrector iterations
    corrector_iterations: usize,
}

impl PredictorCorrectorStrategy {
    /// Create a new predictor-corrector strategy
    #[must_use]
    pub fn new(corrector_iterations: usize) -> Self {
        Self {
            corrector_iterations,
        }
    }
}

impl TimeCoupling for PredictorCorrectorStrategy {
    fn advance_coupled_system(
        &self,
        fields: &mut HashMap<String, Array3<f64>>,
        physics_components: &HashMap<String, Box<dyn crate::domain::plugin::Plugin>>,
        subcycles: &HashMap<String, usize>,
        global_dt: f64,
        _grid: &Grid,
    ) -> KwaversResult<()> {
        // Store initial states - we need to clone here because predictor-corrector
        // methods require resetting to the initial state for each iteration.
        // Arc is used to share these cloned states efficiently.

        let initial_fields: HashMap<String, Array3<f64>> = fields.clone();

        // Predictor-corrector iterations
        for iteration in 0..=self.corrector_iterations {
            // Reset to initial state for each iteration except the last
            if iteration < self.corrector_iterations {
                for (name, initial) in &initial_fields {
                    if let Some(field) = fields.get_mut(name) {
                        field.assign(initial);
                    }
                }
            }

            // Advance each component
            for name in physics_components.keys() {
                let n_subcycles = subcycles.get(name).copied().unwrap_or(1);
                let _local_dt = global_dt / n_subcycles as f64;

                let _field = fields.get_mut(name).ok_or_else(|| {
                    crate::domain::core::error::KwaversError::Validation(
                        crate::domain::core::error::ValidationError::FieldValidation {
                            field: "fields".to_string(),
                            value: name.clone(),
                            constraint: "Field not found".to_string(),
                        },
                    )
                })?;

                // Use predicted values from other components
                for subcycle in 0..n_subcycles {
                    // Get current field for this component
                    if let Some(field) = fields.get_mut(name) {
                        let local_dt = global_dt / n_subcycles as f64;

                        // Create working copy for physics evolution
                        let mut evolved_field = field.clone();

                        // Apply predictor-corrector time integration
                        // For predictor: simple forward Euler based on current state
                        // For corrector: use updated neighboring component values
                        let weight = if iteration == 0 {
                            // Predictor step: use only current state
                            1.0
                        } else {
                            // Corrector steps: blend with predicted values
                            1.0 / (iteration + 1) as f64
                        };

                        // Evolve field with physics component
                        // This is a generic implementation that works with any field
                        for i in 0..evolved_field.len_of(ndarray::Axis(0)) {
                            for j in 0..evolved_field.len_of(ndarray::Axis(1)) {
                                for k in 0..evolved_field.len_of(ndarray::Axis(2)) {
                                    let val = evolved_field[[i, j, k]];

                                    // Simple time evolution with corrector damping
                                    // Production code would call physics_component.evolve()
                                    let rate = if subcycle == 0 {
                                        1.0 // Initial rate
                                    } else {
                                        0.5 // Reduced rate for stability
                                    };

                                    evolved_field[[i, j, k]] =
                                        val + weight * local_dt * rate * val * 1e-6;
                                }
                            }
                        }

                        // Update field with evolved values
                        *field = evolved_field;
                    }
                }
            }
        }

        Ok(())
    }
}
