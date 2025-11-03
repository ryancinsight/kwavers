//! FDTD Reference Solution Generator for PINN Validation
//!
//! Provides finite-difference time-domain (FDTD) reference solutions
//! for the 1D wave equation to validate PINN predictions.
//!
//! ## Implementation
//!
//! Uses central difference scheme:
//! - Spatial: ∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
//! - Temporal: ∂²u/∂t² ≈ (u[n+1] - 2u[n] + u[n-1]) / dt²
//!
//! ## References
//!
//! - Courant-Friedrichs-Lewy (CFL) condition: c×dt/dx ≤ 1
//! - Standard FDTD textbook implementations

use crate::error::{KwaversError, KwaversResult};
use ndarray::{Array1, Array2};

/// FDTD solver configuration for 1D wave equation
#[derive(Debug, Clone)]
pub struct FDTDConfig {
    /// Wave speed (m/s)
    pub wave_speed: f64,
    /// Spatial step size (m)
    pub dx: f64,
    /// Temporal step size (s)
    pub dt: f64,
    /// Number of spatial points
    pub nx: usize,
    /// Number of time steps
    pub nt: usize,
    /// Initial condition type
    pub initial_condition: InitialCondition,
}

/// Initial condition types for 1D wave equation
#[derive(Debug, Clone, Copy)]
pub enum InitialCondition {
    /// Gaussian pulse at center
    GaussianPulse { width: f64, amplitude: f64 },
    /// Sine wave
    SineWave { frequency: f64, amplitude: f64 },
    /// Custom (user-provided)
    Custom,
}

impl Default for FDTDConfig {
    fn default() -> Self {
        Self {
            wave_speed: 1500.0,
            dx: 0.01,
            dt: 0.000005,
            nx: 100,
            nt: 100,
            initial_condition: InitialCondition::GaussianPulse {
                width: 0.05,
                amplitude: 1.0,
            },
        }
    }
}

impl FDTDConfig {
    /// Validate configuration for numerical stability
    ///
    /// Checks CFL condition: c×dt/dx ≤ 1
    pub fn validate(&self) -> KwaversResult<()> {
        if self.wave_speed <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Wave speed must be positive".to_string(),
            ));
        }

        if self.dx <= 0.0 || self.dt <= 0.0 {
            return Err(KwaversError::InvalidInput(
                "Spatial and temporal steps must be positive".to_string(),
            ));
        }

        if self.nx < 3 || self.nt < 3 {
            return Err(KwaversError::InvalidInput(
                "Grid must have at least 3 points in each dimension".to_string(),
            ));
        }

        // Check CFL condition
        let cfl = self.wave_speed * self.dt / self.dx;
        if cfl > 1.0 {
            return Err(KwaversError::InvalidInput(format!(
                "CFL condition violated: c×dt/dx = {:.3} > 1.0. Reduce dt or increase dx.",
                cfl
            )));
        }

        Ok(())
    }

    /// Get CFL number
    #[must_use]
    pub fn cfl_number(&self) -> f64 {
        self.wave_speed * self.dt / self.dx
    }
}

/// 1D FDTD solver for wave equation
#[derive(Debug)]
pub struct FDTD1DWaveSolver {
    config: FDTDConfig,
    /// Current field values
    u_current: Array1<f64>,
    /// Previous field values
    u_previous: Array1<f64>,
    /// Current time step
    current_step: usize,
}

impl FDTD1DWaveSolver {
    /// Create new FDTD solver
    ///
    /// # Arguments
    ///
    /// * `config` - FDTD configuration
    ///
    /// # Returns
    ///
    /// New FDTD solver instance
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "pinn")]
    /// # {
    /// use kwavers::ml::pinn::fdtd_reference::{FDTD1DWaveSolver, FDTDConfig};
    ///
    /// let config = FDTDConfig::default();
    /// let solver = FDTD1DWaveSolver::new(config)?;
    /// # Ok::<(), kwavers::error::KwaversError>(())
    /// # }
    /// ```
    pub fn new(config: FDTDConfig) -> KwaversResult<Self> {
        config.validate()?;

        let nx = config.nx;
        let mut u_current = Array1::zeros(nx);
        let u_previous = Array1::zeros(nx);

        // Initialize with specified initial condition
        match config.initial_condition {
            InitialCondition::GaussianPulse { width, amplitude } => {
                let x_center = (nx / 2) as f64 * config.dx;
                for i in 0..nx {
                    let x = i as f64 * config.dx;
                    let dist = x - x_center;
                    u_current[i] = amplitude * (-dist.powi(2) / (2.0 * width.powi(2))).exp();
                }
            }
            InitialCondition::SineWave { frequency, amplitude } => {
                for i in 0..nx {
                    let x = i as f64 * config.dx;
                    u_current[i] = amplitude * (2.0 * std::f64::consts::PI * frequency * x).sin();
                }
            }
            InitialCondition::Custom => {
                // Custom initialization handled externally
            }
        }

        Ok(Self {
            config,
            u_current,
            u_previous,
            current_step: 0,
        })
    }

    /// Step forward one time step
    ///
    /// Uses central difference scheme with CFL-stable parameters
    pub fn step(&mut self) -> KwaversResult<()> {
        let nx = self.config.nx;
        let c = self.config.wave_speed;
        let dx = self.config.dx;
        let dt = self.config.dt;

        let alpha = (c * dt / dx).powi(2);

        // Create new field array
        let mut u_next = Array1::zeros(nx);

        // Interior points (central difference)
        for i in 1..nx - 1 {
            u_next[i] = 2.0 * self.u_current[i] - self.u_previous[i]
                + alpha * (self.u_current[i + 1] - 2.0 * self.u_current[i] + self.u_current[i - 1]);
        }

        // Boundary conditions (Dirichlet: u = 0 at boundaries)
        u_next[0] = 0.0;
        u_next[nx - 1] = 0.0;

        // Update fields
        self.u_previous = self.u_current.clone();
        self.u_current = u_next;
        self.current_step += 1;

        Ok(())
    }

    /// Solve for all time steps and return field history
    ///
    /// # Returns
    ///
    /// `Array2<f64>` of shape (nx, nt) containing field values at all times
    ///
    /// # Example
    ///
    /// ```
    /// # #[cfg(feature = "pinn")]
    /// # {
    /// use kwavers::ml::pinn::fdtd_reference::{FDTD1DWaveSolver, FDTDConfig};
    ///
    /// let config = FDTDConfig::default();
    /// let mut solver = FDTD1DWaveSolver::new(config)?;
    /// let solution = solver.solve()?;
    /// assert_eq!(solution.dim(), (100, 100));
    /// # Ok::<(), kwavers::error::KwaversError>(())
    /// # }
    /// ```
    pub fn solve(&mut self) -> KwaversResult<Array2<f64>> {
        let nx = self.config.nx;
        let nt = self.config.nt;

        let mut solution = Array2::zeros((nx, nt));

        // Store initial condition
        for i in 0..nx {
            solution[[i, 0]] = self.u_current[i];
        }

        // Solve for all time steps
        for t in 1..nt {
            self.step()?;
            for i in 0..nx {
                solution[[i, t]] = self.u_current[i];
            }
        }

        Ok(solution)
    }

    /// Get current field values
    #[must_use]
    pub fn current_field(&self) -> &Array1<f64> {
        &self.u_current
    }

    /// Get current time step
    #[must_use]
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Get configuration
    #[must_use]
    pub fn config(&self) -> &FDTDConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fdtd_config_validation() {
        let config = FDTDConfig::default();
        assert!(config.validate().is_ok());

        // Invalid wave speed
        let mut bad_config = config.clone();
        bad_config.wave_speed = -1.0;
        assert!(bad_config.validate().is_err());

        // Invalid dx
        let mut bad_config = config.clone();
        bad_config.dx = 0.0;
        assert!(bad_config.validate().is_err());

        // CFL violation
        let mut bad_config = config.clone();
        bad_config.dt = 1.0; // Very large dt
        assert!(bad_config.validate().is_err());
    }

    #[test]
    fn test_fdtd_config_cfl() {
        let config = FDTDConfig::default();
        let cfl = config.cfl_number();
        assert!(cfl <= 1.0);
        assert!(cfl > 0.0);
    }

    #[test]
    fn test_fdtd_solver_creation() {
        let config = FDTDConfig::default();
        let solver = FDTD1DWaveSolver::new(config);
        assert!(solver.is_ok());

        let solver = solver.unwrap();
        assert_eq!(solver.current_step(), 0);
        assert_eq!(solver.current_field().len(), 100);
    }

    #[test]
    fn test_fdtd_step() {
        let config = FDTDConfig::default();
        let mut solver = FDTD1DWaveSolver::new(config).unwrap();

        let initial_field = solver.current_field().clone();
        solver.step().unwrap();

        assert_eq!(solver.current_step(), 1);
        assert_ne!(solver.current_field(), &initial_field);
    }

    #[test]
    fn test_fdtd_solve() {
        let config = FDTDConfig {
            nx: 50,
            nt: 50,
            ..Default::default()
        };
        let mut solver = FDTD1DWaveSolver::new(config).unwrap();

        let solution = solver.solve();
        assert!(solution.is_ok());

        let solution = solution.unwrap();
        assert_eq!(solution.dim(), (50, 50));

        // Check all values are finite
        for &val in solution.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_boundary_conditions() {
        let config = FDTDConfig::default();
        let mut solver = FDTD1DWaveSolver::new(config).unwrap();

        for _ in 0..10 {
            solver.step().unwrap();
            // Check boundaries remain zero (Dirichlet BC)
            assert_eq!(solver.current_field()[0], 0.0);
            assert_eq!(solver.current_field()[99], 0.0);
        }
    }

    #[test]
    fn test_gaussian_initial_condition() {
        let config = FDTDConfig {
            initial_condition: InitialCondition::GaussianPulse {
                width: 0.05,
                amplitude: 2.0,
            },
            ..Default::default()
        };
        let solver = FDTD1DWaveSolver::new(config).unwrap();

        // Check peak is around center
        let field = solver.current_field();
        let max_val = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(max_val > 1.0); // Amplitude should be ~2.0
        assert!(max_val < 2.5);
    }

    #[test]
    fn test_sine_initial_condition() {
        let config = FDTDConfig {
            initial_condition: InitialCondition::SineWave {
                frequency: 10.0,
                amplitude: 1.0,
            },
            ..Default::default()
        };
        let solver = FDTD1DWaveSolver::new(config).unwrap();

        let field = solver.current_field();
        // Check field oscillates
        let max_val = field.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_val = field.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        assert!(max_val > 0.5);
        assert!(min_val < -0.5);
    }
}
