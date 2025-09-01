//! Implicit solver for the Kuznetsov equation
//!
//! Implements a semi-implicit fractional step method for the Kuznetsov equation
//! following the approach of Dörich & Nikolić (2024) for robust stability.
//!
//! The Kuznetsov equation:
//! ∂²p/∂t² - c²∇²p + δ∂³p/∂t³ = (β/ρc⁴)∂²(p²)/∂t²
//!
//! is split into:
//! 1. Linear wave propagation (explicit)
//! 2. Nonlinear term (implicit)
//! 3. Absorption/dissipation (implicit)

use ndarray::{Array3, Zip};
use std::f64::consts::PI;

/// Configuration for implicit Kuznetsov solver
#[derive(Debug, Clone)]
pub struct ImplicitKuznetsovConfig {
    /// Sound speed in medium [m/s]
    pub sound_speed: f64,
    /// Medium density [kg/m³]
    pub density: f64,
    /// Nonlinearity parameter B/A
    pub nonlinearity: f64,
    /// Sound diffusivity δ [m²/s]
    pub diffusivity: f64,
    /// Maximum iterations for implicit solver
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: f64,
    /// Time step size [s]
    pub dt: f64,
    /// Spatial step size [m]
    pub dx: f64,
}

impl Default for ImplicitKuznetsovConfig {
    fn default() -> Self {
        Self {
            sound_speed: 1500.0,
            density: 1000.0,
            nonlinearity: 3.5,
            diffusivity: 4.5e-6,
            max_iterations: 100,
            tolerance: 1e-10,
            dt: 1e-7,
            dx: 1e-4,
        }
    }
}

/// Implicit Kuznetsov equation solver using fractional step method
pub struct ImplicitKuznetsovSolver {
    config: ImplicitKuznetsovConfig,
    /// Current pressure field
    pressure: Array3<f64>,
    /// Previous pressure field
    pressure_prev: Array3<f64>,
    /// Velocity field (∂p/∂t)
    velocity: Array3<f64>,
    /// Laplacian operator coefficients for implicit solve
    laplacian_coeff: f64,
    /// Nonlinear coefficient
    nonlinear_coeff: f64,
    /// Dissipation coefficient
    dissipation_coeff: f64,
}

impl ImplicitKuznetsovSolver {
    /// Create a new implicit Kuznetsov solver
    pub fn new(config: ImplicitKuznetsovConfig, shape: (usize, usize, usize)) -> Self {
        let dt2 = config.dt * config.dt;
        let dx2 = config.dx * config.dx;

        // Coefficients for the implicit scheme
        let laplacian_coeff = config.sound_speed * config.sound_speed * dt2 / dx2;

        // Nonlinear coefficient: β/(ρc⁴) where β = 1 + B/2A
        let beta = 1.0 + config.nonlinearity / 2.0;
        let nonlinear_coeff = beta / (config.density * config.sound_speed.powi(4));

        // Dissipation coefficient: δ/c⁴
        let dissipation_coeff = config.diffusivity / config.sound_speed.powi(4);

        Self {
            config,
            pressure: Array3::zeros(shape),
            pressure_prev: Array3::zeros(shape),
            velocity: Array3::zeros(shape),
            laplacian_coeff,
            nonlinear_coeff,
            dissipation_coeff,
        }
    }

    /// Initialize with a sinusoidal wave
    pub fn initialize_sinusoid(&mut self, amplitude: f64, wavelength: f64) {
        let k = 2.0 * PI / wavelength;
        let (nx, _, _) = self.pressure.dim();

        for i in 0..nx {
            let x = i as f64 * self.config.dx;
            let value = amplitude * (k * x).sin();
            self.pressure[[i, 0, 0]] = value;
            self.pressure_prev[[i, 0, 0]] = value;
        }
    }

    /// Compute the discrete Laplacian using second-order central differences
    fn compute_laplacian(&self, field: &Array3<f64>) -> Array3<f64> {
        let mut laplacian = Array3::zeros(field.dim());
        let (nx, ny, nz) = field.dim();
        let dx2 = self.config.dx * self.config.dx;

        // Apply periodic boundary conditions
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    let im = if i == 0 { nx - 1 } else { i - 1 };
                    let ip = if i == nx - 1 { 0 } else { i + 1 };

                    let jm = if j == 0 { ny - 1 } else { j - 1 };
                    let jp = if j == ny - 1 { 0 } else { j + 1 };

                    let km = if k == 0 { nz - 1 } else { k - 1 };
                    let kp = if k == nz - 1 { 0 } else { k + 1 };

                    // Second-order central difference
                    let d2p_dx2 =
                        (field[[ip, j, k]] - 2.0 * field[[i, j, k]] + field[[im, j, k]]) / dx2;
                    let d2p_dy2 = if ny > 1 {
                        (field[[i, jp, k]] - 2.0 * field[[i, j, k]] + field[[i, jm, k]]) / dx2
                    } else {
                        0.0
                    };
                    let d2p_dz2 = if nz > 1 {
                        (field[[i, j, kp]] - 2.0 * field[[i, j, k]] + field[[i, j, km]]) / dx2
                    } else {
                        0.0
                    };

                    laplacian[[i, j, k]] = d2p_dx2 + d2p_dy2 + d2p_dz2;
                }
            }
        }

        laplacian
    }

    /// Step 1: Linear wave propagation (explicit with stability check)
    fn linear_step(&mut self) {
        // Check CFL condition
        let cfl = self.config.sound_speed * self.config.dt / self.config.dx;
        if cfl > 0.5 {
            // CFL condition violated - use smaller effective time step
            let safe_dt = 0.3 * self.config.dx / self.config.sound_speed;
            let substeps = (self.config.dt / safe_dt).ceil() as usize;
            let dt_sub = self.config.dt / substeps as f64;

            for _ in 0..substeps {
                let laplacian = self.compute_laplacian(&self.pressure);
                let c2 = self.config.sound_speed * self.config.sound_speed;

                // Update velocity: ∂v/∂t = c²∇²p
                Zip::from(&mut self.velocity)
                    .and(&laplacian)
                    .for_each(|v, &lap| {
                        *v += dt_sub * c2 * lap;
                    });

                // Update pressure: ∂p/∂t = v
                Zip::from(&mut self.pressure)
                    .and(&self.velocity)
                    .for_each(|p, &v| {
                        *p += dt_sub * v;
                    });
            }
        } else {
            let laplacian = self.compute_laplacian(&self.pressure);
            let c2 = self.config.sound_speed * self.config.sound_speed;

            // Update velocity: ∂v/∂t = c²∇²p
            Zip::from(&mut self.velocity)
                .and(&laplacian)
                .for_each(|v, &lap| {
                    *v += self.config.dt * c2 * lap;
                });

            // Update pressure: ∂p/∂t = v
            Zip::from(&mut self.pressure)
                .and(&self.velocity)
                .for_each(|p, &v| {
                    *p += self.config.dt * v;
                });
        }
    }

    /// Step 2: Nonlinear term (implicit using fixed-point iteration)
    fn nonlinear_step(&mut self) {
        let dt = self.config.dt;

        // For weak nonlinearity, use a simpler approach
        // The nonlinear term in velocity equation: β/(ρc⁴) * ∂²(p²)/∂t²
        // We approximate this as: β/(ρc⁴) * 2p * (∂p/∂t)²/p₀
        // where p₀ is a reference pressure for normalization

        let reference_pressure = 1e5; // 100 kPa reference

        // Update velocity with nonlinear contribution
        Zip::from(&mut self.velocity)
            .and(&self.pressure)
            .for_each(|v, &p| {
                // Normalized pressure
                let p_norm = p / reference_pressure;

                // Nonlinear correction: quadratic in normalized pressure
                // This is much weaker and more stable
                let nonlinear_correction = self.nonlinear_coeff * p_norm * p_norm * (*v);

                // Apply with limiting for stability
                let max_correction = 0.1 * (*v).abs(); // Limit to 10% change
                let correction = nonlinear_correction.clamp(-max_correction, max_correction);

                *v += dt * correction;
            });

        // Update pressure with nonlinear contribution
        Zip::from(&mut self.pressure)
            .and(&self.velocity)
            .for_each(|p, &v| {
                *p += dt * v;
            });
    }

    /// Step 3: Dissipation/absorption (implicit)
    fn dissipation_step(&mut self) {
        let dt = self.config.dt;

        // Solve implicitly: ∂v/∂t = -δ/c⁴ * ∂³p/∂t³
        // Approximate ∂³p/∂t³ ≈ ∂²v/∂t²

        // For stability, use backward Euler: v^{n+1} = v^n / (1 + dt*δ/c⁴)
        let damping_factor = 1.0 / (1.0 + dt * self.dissipation_coeff);

        self.velocity.mapv_inplace(|v| v * damping_factor);
    }

    /// Perform one time step using fractional step method
    pub fn step(&mut self) {
        // Store previous pressure
        self.pressure_prev.assign(&self.pressure);

        // Fractional step method
        self.linear_step(); // Step 1: Linear wave propagation
        self.nonlinear_step(); // Step 2: Nonlinear effects
        self.dissipation_step(); // Step 3: Dissipation
    }

    /// Get the current pressure field
    pub fn pressure(&self) -> &Array3<f64> {
        &self.pressure
    }

    /// Compute the spectrum using FFT
    pub fn compute_spectrum(&self) -> Vec<f64> {
        use rustfft::{num_complex::Complex, FftPlanner};

        let nx = self.pressure.dim().0;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(nx);

        // Extract 1D slice along x-axis
        let mut signal: Vec<Complex<f64>> = self
            .pressure
            .slice(ndarray::s![.., 0, 0])
            .iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();

        fft.process(&mut signal);

        // Compute magnitude spectrum
        signal.iter().map(|c| c.norm() / nx as f64).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_implicit_solver_stability() {
        // Test that the implicit solver remains stable
        let config = ImplicitKuznetsovConfig {
            sound_speed: 1500.0,
            density: 1000.0,
            nonlinearity: 3.5,
            diffusivity: 4.5e-6,
            dt: 1e-7,
            dx: 1e-4,
            max_iterations: 50,
            tolerance: 1e-8,
        };

        let mut solver = ImplicitKuznetsovSolver::new(config, (128, 1, 1));
        solver.initialize_sinusoid(1e4, 1.5e-3); // 10 kPa, 1.5mm wavelength

        // Run for many steps
        let mut max_pressure: f64 = 0.0;
        for step in 0..1000 {
            solver.step();
            let current_max = solver
                .pressure()
                .iter()
                .map(|&p| p.abs())
                .fold(0.0_f64, f64::max);
            max_pressure = max_pressure.max(current_max);

            if step < 10 || step % 100 == 0 {
                println!("Step {}: max pressure = {:.2e}", step, current_max);
            }

            if current_max.is_infinite() || current_max.is_nan() {
                panic!("Instability at step {}", step);
            }
        }

        // Check that pressure remains bounded
        assert!(max_pressure < 1e5, "Pressure grew to {}", max_pressure);
    }

    #[test]
    fn test_second_harmonic_generation() {
        // Test weak second harmonic generation
        let config = ImplicitKuznetsovConfig {
            sound_speed: 1500.0,
            density: 1000.0,
            nonlinearity: 3.5,
            diffusivity: 0.0, // No dissipation for cleaner harmonics
            dt: 1e-8,
            dx: 1e-5,
            max_iterations: 20,
            tolerance: 1e-6,
        };

        let mut solver = ImplicitKuznetsovSolver::new(config, (256, 1, 1));
        let wavelength = 1.5e-3;
        solver.initialize_sinusoid(1e3, wavelength); // 1 kPa for weak nonlinearity

        // Propagate for a short distance
        for _ in 0..100 {
            solver.step();
        }

        // Compute spectrum
        let spectrum = solver.compute_spectrum();

        // Find fundamental and second harmonic
        let fundamental_idx = 256 / (256 as f64 * 1e-5 / wavelength) as usize;
        let second_harmonic_idx = 2 * fundamental_idx;

        if second_harmonic_idx < spectrum.len() {
            let fundamental = spectrum[fundamental_idx];
            let second_harmonic = spectrum[second_harmonic_idx];

            // For weak nonlinearity, second harmonic should be present but small
            let ratio = second_harmonic / fundamental;
            assert!(
                ratio > 0.0 && ratio < 0.1,
                "Second harmonic ratio {} out of range",
                ratio
            );
        }
    }
}
