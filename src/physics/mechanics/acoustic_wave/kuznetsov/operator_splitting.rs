//! Operator splitting implementation for Kuznetsov equation
//!
//! References:
//! - Pinton et al. (2009) "A heterogeneous nonlinear attenuating full-wave model"
//! - Jing et al. (2012) "Time-domain simulation of nonlinear acoustic beams"
//!
//! The Kuznetsov equation is split into linear and nonlinear parts:
//! ∂²p/∂t² = c₀²∇²p + N(p)
//! where N(p) = -(β/ρ₀c₀⁴) ∂²(p²)/∂t²

use ndarray::{Array3, Zip};

/// Operator splitting solver for nonlinear acoustics
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
    dt: f64,
}

impl OperatorSplittingSolver {
    /// Create new operator splitting solver
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
    pub fn linear_step(&self, pressure: &Array3<f64>, pressure_prev: &Array3<f64>) -> Array3<f64> {
        let mut pressure_next = Array3::zeros((self.nx, self.ny, self.nz));
        let c2 = self.sound_speed * self.sound_speed;
        let dt2 = self.dt * self.dt;

        // Handle different dimensionalities
        if self.ny == 1 && self.nz == 1 {
            // 1D case
            for i in 1..self.nx - 1 {
                let laplacian = (pressure[[i + 1, 0, 0]] - 2.0 * pressure[[i, 0, 0]]
                    + pressure[[i - 1, 0, 0]])
                    / (self.dx * self.dx);

                pressure_next[[i, 0, 0]] =
                    2.0 * pressure[[i, 0, 0]] - pressure_prev[[i, 0, 0]] + dt2 * c2 * laplacian;
            }
        } else {
            // 2D/3D case
            for k in 1..self.nz.saturating_sub(1).max(1) {
                for j in 1..self.ny.saturating_sub(1).max(1) {
                    for i in 1..self.nx.saturating_sub(1).max(1) {
                        // Compute Laplacian
                        let laplacian_x = if self.nx > 1 {
                            (pressure[[i + 1, j, k]] - 2.0 * pressure[[i, j, k]]
                                + pressure[[i - 1, j, k]])
                                / (self.dx * self.dx)
                        } else {
                            0.0
                        };

                        let laplacian_y = if self.ny > 1 {
                            (pressure[[i, j + 1, k]] - 2.0 * pressure[[i, j, k]]
                                + pressure[[i, j - 1, k]])
                                / (self.dy * self.dy)
                        } else {
                            0.0
                        };

                        let laplacian_z = if self.nz > 1 {
                            (pressure[[i, j, k + 1]] - 2.0 * pressure[[i, j, k]]
                                + pressure[[i, j, k - 1]])
                                / (self.dz * self.dz)
                        } else {
                            0.0
                        };

                        let laplacian = laplacian_x + laplacian_y + laplacian_z;

                        // Leapfrog time integration
                        pressure_next[[i, j, k]] = 2.0 * pressure[[i, j, k]]
                            - pressure_prev[[i, j, k]]
                            + dt2 * c2 * laplacian;
                    }
                }
            }
        }

        // Apply boundary conditions (zero gradient for now)
        self.apply_boundary_conditions(&mut pressure_next);

        pressure_next
    }

    /// Step 2: Nonlinear correction
    /// Applies the nonlinear term N(p) = -(β/ρ₀c₀⁴) ∂²(p²)/∂t²
    /// This generates harmonics through the p² term
    pub fn nonlinear_step(
        &self,
        pressure: &mut Array3<f64>,
        pressure_prev: &Array3<f64>,
        pressure_prev2: &Array3<f64>,
    ) {
        let beta = 1.0 + self.nonlinearity / 2.0; // β = 1 + B/2A
                                                  // Positive coefficient for shock steepening in compression phase
        let coeff = beta / (self.density * self.sound_speed.powi(4));
        let dt2 = self.dt * self.dt;

        // Compute p² at different time levels
        let p2_curr = pressure.mapv(|p| p * p);
        let p2_prev = pressure_prev.mapv(|p| p * p);
        let p2_prev2 = pressure_prev2.mapv(|p| p * p);

        // Apply nonlinear correction
        // ∂²(p²)/∂t² ≈ (p²[n] - 2*p²[n-1] + p²[n-2]) / dt²
        Zip::from(pressure)
            .and(&p2_curr)
            .and(&p2_prev)
            .and(&p2_prev2)
            .for_each(|p, &p2_c, &p2_p, &p2_p2| {
                let d2p2_dt2 = (p2_c - 2.0 * p2_p + p2_p2) / dt2;
                // The nonlinear term acts as an acceleration term
                // We multiply by dt² for the discrete update
                let nonlinear_term = coeff * d2p2_dt2 * dt2;

                // Apply correction (additive splitting)
                // This steepens the wave fronts
                *p += nonlinear_term;
            });
    }

    /// Step 3: Linear propagation again for dt/2 (Strang splitting)
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

    /// Apply boundary conditions (zero gradient for now)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_harmonic_generation() {
        // Test that nonlinear propagation generates harmonics
        let nx = 128;
        let ny = 1;
        let nz = 1;
        let dx = 1e-4;
        let dy = dx;
        let dz = dx;

        let frequency = 1e6; // 1 MHz
        let wavelength = 1500.0 / frequency;
        let k = 2.0 * PI / wavelength;

        let solver = OperatorSplittingSolver::new(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            1000.0,              // density
            1500.0,              // sound speed
            3.5,                 // B/A for water
            dx / (1500.0 * 4.0), // dt from CFL
        );

        // Initialize with sinusoidal wave
        let mut pressure = Array3::zeros((nx, ny, nz));
        let mut pressure_prev = Array3::zeros((nx, ny, nz));
        let mut pressure_prev2 = Array3::zeros((nx, ny, nz));

        for i in 0..nx {
            let x = i as f64 * dx;
            let amplitude = 1e6; // 1 MPa
            pressure[[i, 0, 0]] = amplitude * (k * x).sin();
            pressure_prev[[i, 0, 0]] = amplitude * (k * x - 2.0 * PI * frequency * solver.dt).sin();
        }

        // Propagate for multiple steps
        let initial_max = pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let initial_energy: f64 = pressure.iter().map(|p| p * p).sum();

        for step in 0..100 {
            let pressure_next = solver.step(&pressure, &pressure_prev, &pressure_prev2);

            // Debug: Check for NaN or extreme values
            if pressure_next.iter().any(|p| p.is_nan() || p.abs() > 1e10) {
                panic!(
                    "Numerical instability at step {}: NaN or overflow detected",
                    step
                );
            }

            pressure_prev2.assign(&pressure_prev);
            pressure_prev.assign(&pressure);
            pressure.assign(&pressure_next);

            // Track energy conservation
            if step % 20 == 0 || step == 99 {
                let current_max = pressure.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
                let current_energy: f64 = pressure.iter().map(|p| p * p).sum();
                println!(
                    "Step {}: max={:.2e}, energy={:.2e} (initial: max={:.2e}, energy={:.2e})",
                    step, current_max, current_energy, initial_max, initial_energy
                );
            }
        }

        // Perform FFT to check for harmonics
        use rustfft::{num_complex::Complex, FftPlanner};
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nx);

        let mut spectrum: Vec<Complex<f64>> = pressure
            .slice(ndarray::s![.., 0, 0])
            .iter()
            .map(|&p| Complex::new(p, 0.0))
            .collect();

        fft.process(&mut spectrum);

        // Find fundamental and second harmonic peaks
        // For spatial FFT: wavenumber k = 2π/λ = 2πf/c
        // FFT bin = k * L / (2π) = k * nx * dx / (2π)
        let wavelength = 1500.0 / frequency;
        let k_fundamental = 2.0 * PI / wavelength;
        let fundamental_idx = (k_fundamental * nx as f64 * dx / (2.0 * PI)).round() as usize;
        let second_harmonic_idx = 2 * fundamental_idx;

        // Make sure indices are within bounds
        let fundamental_idx = fundamental_idx.min(nx / 2 - 1);
        let second_harmonic_idx = second_harmonic_idx.min(nx / 2 - 1);

        let fundamental_amp = spectrum[fundamental_idx].norm();
        let second_harmonic_amp = if second_harmonic_idx < spectrum.len() {
            spectrum[second_harmonic_idx].norm()
        } else {
            0.0
        };

        // Second harmonic should be generated (non-zero)
        assert!(
            second_harmonic_amp > fundamental_amp * 0.01,
            "Second harmonic not generated: fundamental={:.2e}, second={:.2e}",
            fundamental_amp,
            second_harmonic_amp
        );
    }

    #[test]
    fn test_shock_steepening() {
        // Test that high-amplitude waves steepen (shock formation)
        let nx = 256;
        let ny = 1;
        let nz = 1;
        let dx = 5e-5;
        let dy = dx;
        let dz = dx;

        let solver = OperatorSplittingSolver::new(
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            1000.0,              // density
            1500.0,              // sound speed
            3.5,                 // B/A for water
            dx / (1500.0 * 4.0), // dt from CFL
        );

        // Initialize with high-amplitude sine wave
        let mut pressure = Array3::zeros((nx, ny, nz));
        let mut pressure_prev = Array3::zeros((nx, ny, nz));
        let mut pressure_prev2 = Array3::zeros((nx, ny, nz));

        let wavelength = 1.5e-3; // 1.5 mm
        let k = 2.0 * PI / wavelength;
        let amplitude = 5e6; // 5 MPa (high amplitude for shock formation)

        for i in 0..nx {
            let x = i as f64 * dx;
            pressure[[i, 0, 0]] = amplitude * (k * x).sin();
            pressure_prev[[i, 0, 0]] = pressure[[i, 0, 0]];
        }

        // Measure initial gradient
        let mut initial_max_gradient: f64 = 0.0;
        for i in 1..nx - 1 {
            let gradient = (pressure[[i + 1, 0, 0]] - pressure[[i - 1, 0, 0]]).abs() / (2.0 * dx);
            initial_max_gradient = initial_max_gradient.max(gradient);
        }

        // Propagate to allow shock formation
        for _ in 0..200 {
            let pressure_next = solver.step(&pressure, &pressure_prev, &pressure_prev2);
            pressure_prev2.assign(&pressure_prev);
            pressure_prev.assign(&pressure);
            pressure.assign(&pressure_next);
        }

        // Measure final gradient (should be steepened)
        let mut final_max_gradient: f64 = 0.0;
        for i in 1..nx - 1 {
            let gradient = (pressure[[i + 1, 0, 0]] - pressure[[i - 1, 0, 0]]).abs() / (2.0 * dx);
            final_max_gradient = final_max_gradient.max(gradient);
        }

        // Gradient should increase (steepening)
        assert!(
            final_max_gradient > initial_max_gradient * 1.5,
            "No shock steepening: initial gradient={:.2e}, final={:.2e}",
            initial_max_gradient,
            final_max_gradient
        );
    }
}
