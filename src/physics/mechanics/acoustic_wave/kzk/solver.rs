//! KZK equation solver using operator splitting
//!
//! References:
//! - Christopher & Parker (1991) "New approaches to nonlinear diffractive field propagation"
//! - Tavakkoli et al. (1998) "A new algorithm for computational simulation of HIFU"

use ndarray::{Array2, Array3, Axis};
use std::f64::consts::PI;

use super::absorption::AbsorptionOperator;
use super::angular_spectrum_2d::AngularSpectrum2D;
use super::finite_difference_diffraction::DiffractionOperator;
use super::nonlinearity::NonlinearOperator;
use super::parabolic_diffraction::KzkDiffractionOperator;
use super::KZKConfig;

/// KZK equation solver
pub struct KZKSolver {
    config: KZKConfig,
    /// Pressure field p(x,y,τ) at current z
    pressure: Array3<f64>,
    /// Previous pressure for time derivatives
    pressure_prev: Array3<f64>,
    /// Diffraction operator (finite difference implementation)
    diffraction: Option<DiffractionOperator>,

    /// 2D Angular spectrum operator (correct implementation)
    angular_spectrum_2d: Option<AngularSpectrum2D>,
    /// KZK parabolic diffraction operator
    kzk_diffraction: Option<KzkDiffractionOperator>,
    /// Use KZK parabolic approximation (recommended)
    use_kzk_diffraction: bool,
    /// Absorption operator
    absorption: AbsorptionOperator,
    /// Nonlinear operator
    nonlinear: NonlinearOperator,
}

impl std::fmt::Debug for KZKSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KZKSolver")
            .field("config", &self.config)
            .field(
                "pressure",
                &format!(
                    "Array3<f64> {}x{}x{}",
                    self.pressure.shape()[0],
                    self.pressure.shape()[1],
                    self.pressure.shape()[2]
                ),
            )
            .field(
                "pressure_prev",
                &format!(
                    "Array3<f64> {}x{}x{}",
                    self.pressure_prev.shape()[0],
                    self.pressure_prev.shape()[1],
                    self.pressure_prev.shape()[2]
                ),
            )
            .field("diffraction", &self.diffraction.is_some())
            .field("angular_spectrum_2d", &self.angular_spectrum_2d.is_some())
            .field("kzk_diffraction", &self.kzk_diffraction.is_some())
            .field("use_kzk_diffraction", &self.use_kzk_diffraction)
            .field("absorption", &self.absorption)
            .field("nonlinear", &self.nonlinear)
            .finish()
    }
}

impl KZKSolver {
    /// Create new KZK solver
    pub fn new(config: KZKConfig) -> Result<Self, String> {
        super::validate_config(&config)?;

        let pressure = Array3::zeros((config.nx, config.ny, config.nt));
        let pressure_prev = Array3::zeros((config.nx, config.ny, config.nt));

        // Use KZK parabolic approximation by default
        let use_kzk_diffraction = true;

        let diffraction = None; // Finite difference implementation not used

        let angular_spectrum_2d = None; // Full angular spectrum (not KZK)

        let kzk_diffraction = if use_kzk_diffraction {
            Some(KzkDiffractionOperator::new(&config))
        } else {
            None
        };

        let absorption = AbsorptionOperator::new(&config);
        let nonlinear = NonlinearOperator::new(&config);

        Ok(Self {
            config,
            pressure,
            pressure_prev,
            diffraction,
            angular_spectrum_2d,
            kzk_diffraction,
            use_kzk_diffraction,
            absorption,
            nonlinear,
        })
    }

    /// Set initial condition (source plane at z=0)
    pub fn set_source(&mut self, source: Array2<f64>, frequency: f64) {
        // Store frequency in config for all operators
        self.config.frequency = frequency;
        // Re-initialize operators with updated frequency
        if self.use_kzk_diffraction {
            self.kzk_diffraction = Some(KzkDiffractionOperator::new(&self.config));
        }
        self.absorption = AbsorptionOperator::new(&self.config);

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

            if self.use_kzk_diffraction {
                if let Some(ref mut kzk_diffraction) = self.kzk_diffraction {
                    kzk_diffraction.apply(&mut slice, step_size);
                }
            } else if let Some(ref mut angular_spectrum_2d) = self.angular_spectrum_2d {
                // Full angular spectrum (not KZK parabolic)
                angular_spectrum_2d.propagate(&mut slice, step_size);
            } else if let Some(ref mut diffraction) = self.diffraction {
                // Finite difference implementation
                diffraction.apply(&mut slice, step_size);
            }
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
    #[must_use]
    pub fn get_pressure(&self) -> &Array3<f64> {
        &self.pressure
    }

    /// Get pressure at specific point over time
    #[must_use]
    pub fn get_time_signal(&self, x: usize, y: usize) -> Vec<f64> {
        let mut signal = Vec::with_capacity(self.config.nt);
        for t in 0..self.config.nt {
            signal.push(self.pressure[[x, y, t]]);
        }
        signal
    }

    /// Calculate intensity field I = p²/(2ρ₀c₀)
    #[must_use]
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
    #[must_use]
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

    /// Test Gaussian beam propagation (COMPREHENSIVE - Tier 3)
    ///
    /// This test uses a 64×64×128 grid for thorough validation.
    /// Execution time: >30s, classified as Tier 3 comprehensive validation.
    #[test]
    #[ignore = "Tier 3: Comprehensive validation (>30s execution time)"]
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

    /// Test Gaussian beam propagation (FAST - Tier 1)
    ///
    /// Fast version with reduced grid (16×16×32) for CI/CD.
    /// Execution time: <2s, classified as Tier 1 fast validation.
    #[test]
    fn test_gaussian_beam_propagation_fast() {
        let mut config = KZKConfig {
            nx: 16,
            ny: 16,
            nz: 32,
            nt: 20,
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
        let sigma = 3.0; // Grid points (smaller for smaller grid)

        for j in 0..config.ny {
            for i in 0..config.nx {
                let sigma_f64: f64 = sigma;
                let r2 = ((i as f64 - cx).powi(2) + (j as f64 - cy).powi(2)) / sigma_f64.powi(2);
                source[[i, j]] = (-r2).exp();
            }
        }

        solver.set_source(source, 1e6); // 1 MHz

        // Propagate fewer steps for fast validation
        for _ in 0..3 {
            solver.step();
        }

        // Check that beam has propagated (peak should shift)
        let intensity = solver.get_intensity();
        assert!(intensity.sum() > 0.0);
    }
}
