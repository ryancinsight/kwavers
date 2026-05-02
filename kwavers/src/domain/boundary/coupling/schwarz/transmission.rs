//! `apply_transmission` — Schwarz transmission-condition dispatcher across
//! the four canonical branches (Dirichlet / Neumann / Robin / Optimized).

use ndarray::{Array3, ArrayViewMut3};

use super::SchwarzBoundary;
use crate::domain::boundary::coupling::types::TransmissionCondition;

impl SchwarzBoundary {
    /// Apply transmission condition.
    ///
    /// # Arguments
    ///
    /// * `interface_field` - Field values at the interface (mutable)
    /// * `neighbor_field` - Field values from neighboring subdomain
    ///
    /// # Algorithm
    ///
    /// Applies the configured transmission condition:
    ///
    /// - **Dirichlet**: Copy neighbor values to interface.
    /// - **Neumann**: Match normal gradients via correction.
    /// - **Robin**: Blend values and gradients with Robin parameters.
    /// - **Optimized**: Apply relaxation-weighted update.
    pub fn apply_transmission(
        &self,
        interface_field: &mut ArrayViewMut3<f64>,
        neighbor_field: &Array3<f64>,
    ) {
        match self.transmission_condition {
            TransmissionCondition::Dirichlet => {
                // Direct copying: u_interface = u_neighbor
                interface_field.zip_mut_with(neighbor_field, |i, &n| {
                    *i = n;
                });
            }
            TransmissionCondition::Neumann => {
                // Neumann flux continuity: ∂u₁/∂n = ∂u₂/∂n
                //
                // Algorithm:
                // 1. Compute ∂u/∂n on interface side using centered differences.
                // 2. Compute ∂u/∂n on neighbor side using centered differences.
                // 3. Apply correction: Δu = Δx * (grad_neighbor - grad_interface) / 2
                // 4. Update interface field: u_new = u_old + Δu
                let (nx, ny, nz) = interface_field.dim();

                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let grad_interface =
                                Self::compute_normal_gradient(&interface_field.to_owned(), i, j, k);
                            let grad_neighbor =
                                Self::compute_normal_gradient(neighbor_field, i, j, k);

                            // Δu = (grad_neighbor - grad_interface) * 0.5
                            let correction = (grad_neighbor - grad_interface) * 0.5;
                            interface_field[[i, j, k]] += correction;
                        }
                    }
                }
            }
            TransmissionCondition::Robin { alpha, beta } => {
                // Robin transmission condition: ∂u/∂n + α·u = β
                //
                // Physical Interpretation:
                // - Heat transfer: Convective boundary condition (Newton's law of cooling).
                // - Acoustics: Impedance boundary condition.
                // - Electromagnetics: Surface impedance condition.
                //
                // Algorithm:
                // 1. Check α ≠ 0 to avoid division by zero (degenerate Neumann case).
                // 2. Compute normal gradient from neighbor domain.
                // 3. Calculate Robin-corrected value: (β - ∂u/∂n) / α.
                // 4. Blend interface, neighbor, and Robin values for stability.
                // 5. Update: u_new = (u_interface + α·u_neighbor + robin_value) / (2 + α).
                let (nx, ny, nz) = interface_field.dim();

                if alpha.abs() < 1e-12 {
                    // α ≈ 0: degenerate case → reduces to Neumann condition.
                    return;
                }

                for i in 0..nx {
                    for j in 0..ny {
                        for k in 0..nz {
                            let grad_neighbor =
                                Self::compute_normal_gradient(neighbor_field, i, j, k);

                            let u_interface = interface_field[[i, j, k]];
                            let u_neighbor = neighbor_field[[i, j, k]];

                            let robin_value = (beta - grad_neighbor) / alpha;

                            interface_field[[i, j, k]] =
                                (u_interface + alpha * u_neighbor + robin_value) / (2.0 + alpha);
                        }
                    }
                }
            }
            TransmissionCondition::Optimized => {
                // Optimized Schwarz with relaxation.
                // u_new = (1−θ)·u_old + θ·u_neighbor.
                interface_field.zip_mut_with(neighbor_field, |i, &n| {
                    *i = (1.0 - self.relaxation_parameter) * *i + self.relaxation_parameter * n;
                });
            }
        }
    }
}
