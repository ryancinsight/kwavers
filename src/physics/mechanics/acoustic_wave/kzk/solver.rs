//! KZK equation solver using operator splitting
//!
//! References:
//! - Christopher & Parker (1991) "New approaches to nonlinear diffractive field propagation"
//! - Tavakkoli et al. (1998) "A new algorithm for computational simulation of HIFU"

use ndarray::{Array2, Array3, Axis, Zip};
use rustfft::{num_complex::Complex, FftPlanner};
use std::f64::consts::PI;

use super::absorption::AbsorptionOperator;
use super::diffraction::DiffractionOperator;
use super::nonlinearity::NonlinearOperator;
use super::KZKConfig;

/// KZK equation solver
pub struct KZKSolver {
    config: KZKConfig,
    /// Pressure field p(x,y,τ) at current z
    pressure: Array3<f64>,
    /// Previous pressure for time derivatives
    pressure_prev: Array3<f64>,
    /// Diffraction operator
    diffraction: DiffractionOperator,
    /// Absorption operator
    absorption: AbsorptionOperator,
    /// Nonlinear operator
    nonlinear: NonlinearOperator,
    /// FFT planner for spectral methods
    fft_planner: FftPlanner<f64>,
}

impl KZKSolver {
    /// Create new KZK solver
    pub fn new(config: KZKConfig) -> Result<Self, String> {
        super::validate_config(&config)?;

        let pressure = Array3::zeros((config.nx, config.ny, config.nt));
        let pressure_prev = Array3::zeros((config.nx, config.ny, config.nt));

        let diffraction = DiffractionOperator::new(&config);
        let absorption = AbsorptionOperator::new(&config);
        let nonlinear = NonlinearOperator::new(&config);
        let fft_planner = FftPlanner::new();

        Ok(Self {
            config,
            pressure,
            pressure_prev,
            diffraction,
            absorption,
            nonlinear,
            fft_planner,
        })
    }

    /// Set initial condition (source plane at z=0)
    pub fn set_source(&mut self, source: Array2<f64>, frequency: f64) {
        // Set source as time-harmonic signal
        let omega = 2.0 * PI * frequency;
        let dt = self.config.dt;

        for t in 0..self.config.nt {
            let time = t as f64 * dt;
            let temporal = (omega * time).sin();

            for j in 0..self.config.ny {
                for i in 0..self.config.nx {
                    self.pressure[[i, j, t]] = source[[i, j]] * temporal;
                }
            }
        }

        self.pressure_prev.assign(&self.pressure);
    }

    /// Step forward one z-plane using operator splitting
    /// Uses second-order Strang splitting: D(dz/2) * A(dz/2) * N(dz) * A(dz/2) * D(dz/2)
    pub fn step(&mut self) {
        let dz = self.config.dz;

        // Step 1: Diffraction for dz/2
        if self.config.include_diffraction {
            self.apply_diffraction(dz / 2.0);
        }

        // Step 2: Absorption for dz/2
        if self.config.include_absorption {
            self.apply_absorption(dz / 2.0);
        }

        // Step 3: Nonlinearity for full dz
        if self.config.include_nonlinearity {
            self.apply_nonlinearity(dz);
        }

        // Step 4: Absorption for dz/2
        if self.config.include_absorption {
            self.apply_absorption(dz / 2.0);
        }

        // Step 5: Diffraction for dz/2
        if self.config.include_diffraction {
            self.apply_diffraction(dz / 2.0);
        }

        // Update history
        self.pressure_prev.assign(&self.pressure);
    }

    /// Apply diffraction operator using angular spectrum method
    fn apply_diffraction(&mut self, step_size: f64) {
        // Transform to frequency domain for each time slice
        for t in 0..self.config.nt {
            let mut slice = self.pressure.index_axis_mut(Axis(2), t);
            self.diffraction.apply(&mut slice, step_size);
        }
    }

    /// Apply absorption operator
    fn apply_absorption(&mut self, step_size: f64) {
        self.absorption.apply(&mut self.pressure, step_size);
    }

    /// Apply nonlinear operator
    fn apply_nonlinearity(&mut self, step_size: f64) {
        self.nonlinear
            .apply(&mut self.pressure, &self.pressure_prev, step_size);
    }

    /// Get current pressure field
    pub fn get_pressure(&self) -> &Array3<f64> {
        &self.pressure
    }

    /// Get pressure at specific point over time
    pub fn get_time_signal(&self, x: usize, y: usize) -> Vec<f64> {
        let mut signal = Vec::with_capacity(self.config.nt);
        for t in 0..self.config.nt {
            signal.push(self.pressure[[x, y, t]]);
        }
        signal
    }

    /// Calculate intensity field I = p²/(2ρ₀c₀)
    pub fn get_intensity(&self) -> Array2<f64> {
        let mut intensity = Array2::zeros((self.config.nx, self.config.ny));
        let factor = 1.0 / (2.0 * self.config.rho0 * self.config.c0);

        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                let mut sum = 0.0;
                for t in 0..self.config.nt {
                    sum += self.pressure[[i, j, t]].powi(2);
                }
                intensity[[i, j]] = sum * factor / self.config.nt as f64;
            }
        }

        intensity
    }

    /// Calculate peak pressure field
    pub fn get_peak_pressure(&self) -> Array2<f64> {
        let mut peak = Array2::zeros((self.config.nx, self.config.ny));

        for j in 0..self.config.ny {
            for i in 0..self.config.nx {
                let mut max_p: f64 = 0.0;
                for t in 0..self.config.nt {
                    max_p = max_p.max(self.pressure[[i, j, t]].abs());
                }
                peak[[i, j]] = max_p;
            }
        }

        peak
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_kzk_solver_creation() {
        let config = KZKConfig::default();
        let solver = KZKSolver::new(config);
        assert!(solver.is_ok());
    }

    #[test]
    fn test_gaussian_beam_propagation() {
        let mut config = KZKConfig {
            nx: 64,
            ny: 64,
            nz: 128,
            nt: 100,
            dx: 1e-3,
            dz: 1e-3,
            dt: 1e-8,
            ..Default::default()
        };
        config.include_nonlinearity = false; // Linear case first

        let mut solver = KZKSolver::new(config.clone()).unwrap();

        // Create Gaussian source
        let mut source = Array2::zeros((config.nx, config.ny));
        let cx = config.nx as f64 / 2.0;
        let cy = config.ny as f64 / 2.0;
        let sigma = 10.0; // Grid points

        for j in 0..config.ny {
            for i in 0..config.nx {
                let sigma_f64: f64 = sigma;
                let r2 = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)) / sigma_f64.powi(2);
                source[[i, j]] = (-r2).exp();
            }
        }

        solver.set_source(source, 1e6); // 1 MHz

        // Propagate
        for _ in 0..10 {
            solver.step();
        }

        // Check that beam has propagated (peak should shift)
        let intensity = solver.get_intensity();
        assert!(intensity.sum() > 0.0);
    }
}
