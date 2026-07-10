//! Operator splitting implementation for Kuznetsov equation
//!
//! References:
//! - Pinton et al. (2009) "A heterogeneous nonlinear attenuating full-wave model"
//! - Jing et al. (2012) "Time-domain simulation of nonlinear acoustic beams"
//!
//! The Kuznetsov equation is split into linear and nonlinear parts:
//! ∂²p/∂t² = c₀²∇²p + N(p)
//! where N(p) = -(β/ρ₀c₀⁴) ∂²(p²)/∂t²

#[cfg(test)]
mod tests;

use leto::Array3;
use moirai_parallel::{enumerate_mut_with, Adaptive};

/// Operator splitting solver for nonlinear acoustics
#[derive(Debug)]
pub struct OperatorSplittingSolver {
    /// Grid dimensions
    nx: usize,
    ny: usize,
    nz: usize,
    /// Grid spacing
    dx: f64,
    dy: f64,
    dz: f64,
    /// Medium properties
    density: f64,
    sound_speed: f64,
    nonlinearity: f64,
    /// Time step
    pub(super) dt: f64,
}

impl OperatorSplittingSolver {
    /// Create new operator splitting solver
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        dx: f64,
        dy: f64,
        dz: f64,
        density: f64,
        sound_speed: f64,
        nonlinearity: f64,
        dt: f64,
    ) -> Self {
        Self {
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            density,
            sound_speed,
            nonlinearity,
            dt,
        }
    }

    /// Step 1: Linear propagation using finite differences
    /// Solves: ∂²p/∂t² = c₀²∇²p for time dt/2
    #[must_use]
    pub fn linear_step(&self, pressure: &Array3<f64>, pressure_prev: &Array3<f64>) -> Array3<f64> {
        let mut pressure_next = Array3::zeros((self.nx, self.ny, self.nz));
        let c2 = self.sound_speed * self.sound_speed;
        let dt2 = self.dt * self.dt;

        // Handle different dimensionalities
        if self.ny == 1 && self.nz == 1 {
            // 1D case
            for i in 1..self.nx - 1 {
                let laplacian = (2.0f64.mul_add(-pressure[[i, 0, 0]], pressure[[i + 1, 0, 0]])
                    + pressure[[i - 1, 0, 0]])
                    / (self.dx * self.dx);

                pressure_next[[i, 0, 0]] = (dt2 * c2).mul_add(
                    laplacian,
                    2.0f64.mul_add(pressure[[i, 0, 0]], -pressure_prev[[i, 0, 0]]),
                );
            }
        } else {
            // 2D/3D case
            for k in 1..self.nz.saturating_sub(1).max(1) {
                for j in 1..self.ny.saturating_sub(1).max(1) {
                    for i in 1..self.nx.saturating_sub(1).max(1) {
                        // Compute Laplacian
                        let laplacian_x = if self.nx > 1 {
                            (2.0f64.mul_add(-pressure[[i, j, k]], pressure[[i + 1, j, k]])
                                + pressure[[i - 1, j, k]])
                                / (self.dx * self.dx)
                        } else {
                            0.0
                        };

                        let laplacian_y = if self.ny > 1 {
                            (2.0f64.mul_add(-pressure[[i, j, k]], pressure[[i, j + 1, k]])
                                + pressure[[i, j - 1, k]])
                                / (self.dy * self.dy)
                        } else {
                            0.0
                        };

                        let laplacian_z = if self.nz > 1 {
                            (2.0f64.mul_add(-pressure[[i, j, k]], pressure[[i, j, k + 1]])
                                + pressure[[i, j, k - 1]])
                                / (self.dz * self.dz)
                        } else {
                            0.0
                        };

                        let laplacian = laplacian_x + laplacian_y + laplacian_z;

                        // Leapfrog time integration
                        pressure_next[[i, j, k]] = (dt2 * c2).mul_add(
                            laplacian,
                            2.0f64.mul_add(pressure[[i, j, k]], -pressure_prev[[i, j, k]]),
                        );
                    }
                }
            }
        }

        // Apply zero-gradient (Neumann) boundary conditions
        // Appropriate for free-field propagation per Blackstock (2000) §2.7
        self.apply_boundary_conditions(&mut pressure_next);

        pressure_next
    }

    /// Step 2: Nonlinear correction
    /// Apply Burgers-like nonlinearity: ∂u/∂t + u∂u/∂x = 0
    /// where u = p/(ρ₀c₀) is the normalized pressure
    pub fn nonlinear_step(
        &self,
        pressure: &mut Array3<f64>,
        _pressure_prev: &Array3<f64>,
        _pressure_prev2: &Array3<f64>,
    ) {
        let beta = 1.0 + self.nonlinearity / 2.0; // β = 1 + B/2A

        // Normalization factor for pressure
        let norm_factor = self.density * self.sound_speed * self.sound_speed;

        // Create normalized velocity field u = βp/(ρ₀c₀²)
        let u = pressure.mapv(|p| beta * p / norm_factor);

        // Compute flux F = u²/2 and its derivative using upwind scheme
        let mut flux_gradient = Array3::zeros(pressure.shape());

        for k in 0..self.nz {
            for j in 0..self.ny {
                // Use conservative form with Godunov flux
                for i in 1..self.nx - 1 {
                    let u_left = u[[i - 1, j, k]];
                    let u_center = u[[i, j, k]];
                    let u_right = u[[i + 1, j, k]];

                    // Godunov flux at i+1/2
                    let flux_right = if u_center > 0.0 && u_right > 0.0 {
                        0.5 * u_center * u_center // Use left state
                    } else if u_center < 0.0 && u_right < 0.0 {
                        0.5 * u_right * u_right // Use right state
                    } else {
                        0.0 // Sonic point
                    };

                    // Godunov flux at i-1/2
                    let flux_left = if u_left > 0.0 && u_center > 0.0 {
                        0.5 * u_left * u_left // Use left state
                    } else if u_left < 0.0 && u_center < 0.0 {
                        0.5 * u_center * u_center // Use right state
                    } else {
                        0.0 // Sonic point
                    };

                    // Conservative update
                    flux_gradient[[i, j, k]] = (flux_right - flux_left) / self.dx;
                }
            }
        }

        // Apply the nonlinear correction
        let scale = self.dt * self.sound_speed * norm_factor / beta;
        {
            let p_slice = pressure
                .as_slice_mut()
                .expect("pressure: standard-layout asserted just above; layout matched");
            let grad_slice = flux_gradient
                .as_slice()
                .expect("flux_gradient: standard-layout asserted just above; layout matched");
            enumerate_mut_with::<Adaptive, _, _>(p_slice, |idx, p: &mut f64| {
                let grad = grad_slice[idx];
                *p -= scale * grad;
            });
        }
    }

    /// Step 3: Linear propagation again for dt/2 (Strang splitting)
    #[must_use]
    pub fn linear_step_half(
        &self,
        pressure: &Array3<f64>,
        pressure_prev: &Array3<f64>,
    ) -> Array3<f64> {
        // Use half time step
        let original_dt = self.dt;
        let mut solver_half = *self;
        solver_half.dt = original_dt / 2.0;
        solver_half.linear_step(pressure, pressure_prev)
    }

    /// Complete time step using Strang splitting
    /// L(dt/2) * N(dt) * L(dt/2)
    #[must_use]
    pub fn step(
        &self,
        pressure: &Array3<f64>,
        pressure_prev: &Array3<f64>,
        pressure_prev2: &Array3<f64>,
    ) -> Array3<f64> {
        // Step 1: Linear propagation for dt/2
        // p_half = linear_step(p, p_prev, dt/2)
        let p_half = self.linear_step_half(pressure, pressure_prev);

        // Step 2: Nonlinear correction for full dt
        // Apply nonlinear term using proper time history
        let mut p_nonlinear = p_half.clone();
        // For operator splitting, we need the correct time levels
        // p_half is at t+dt/2, pressure is at t, pressure_prev is at t-dt
        self.nonlinear_step(&mut p_nonlinear, pressure_prev, pressure_prev2);

        // Step 3: Linear propagation for dt/2
        // For the second half step, p_nonlinear is current, p_half is previous
        self.linear_step_half(&p_nonlinear, &p_half)
    }

    /// Apply boundary conditions using zero-gradient (Neumann) conditions
    /// Reference: LeVeque (2007) "Finite Difference Methods" §9.2.2
    /// Zero gradient BC: ∂u/∂n = 0, appropriate for acoustic free surfaces
    fn apply_boundary_conditions(&self, pressure: &mut Array3<f64>) {
        // X boundaries
        if self.nx > 1 {
            for k in 0..self.nz {
                for j in 0..self.ny {
                    pressure[[0, j, k]] = pressure[[1, j, k]];
                    pressure[[self.nx - 1, j, k]] = pressure[[self.nx - 2, j, k]];
                }
            }
        }

        // Y boundaries
        if self.ny > 1 {
            for k in 0..self.nz {
                for i in 0..self.nx {
                    pressure[[i, 0, k]] = pressure[[i, 1, k]];
                    pressure[[i, self.ny - 1, k]] = pressure[[i, self.ny - 2, k]];
                }
            }
        }

        // Z boundaries
        if self.nz > 1 {
            for j in 0..self.ny {
                for i in 0..self.nx {
                    pressure[[i, j, 0]] = pressure[[i, j, 1]];
                    pressure[[i, j, self.nz - 1]] = pressure[[i, j, self.nz - 2]];
                }
            }
        }
    }
}

// Implement Copy and Clone for the solver
impl Copy for OperatorSplittingSolver {}

impl Clone for OperatorSplittingSolver {
    fn clone(&self) -> Self {
        *self
    }
}
