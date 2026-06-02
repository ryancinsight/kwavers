//! FDTD1DWaveSolver: time-stepping, full solve, and field accessors.

use super::{config::FDTDConfig, InitialCondition};
use kwavers_core::constants::numerical::TWO_PI;
use kwavers_core::error::KwaversResult;
use ndarray::{Array1, Array2};

/// 1D FDTD solver for wave equation.
///
/// Central difference scheme:
/// - Spatial: `∂²u/∂x² ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²`
/// - Temporal: `∂²u/∂t² ≈ (u[n+1] - 2u[n] + u[n-1]) / dt²`
#[derive(Debug)]
pub struct FDTD1DWaveSolver {
    pub(super) config: FDTDConfig,
    /// Current field values
    pub(super) u_current: Array1<f64>,
    /// Previous field values
    pub(super) u_previous: Array1<f64>,
    /// Current time step
    pub(super) current_step: usize,
}

impl FDTD1DWaveSolver {
    /// Create new FDTD solver with validated configuration.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by `FDTDConfig::validate`.
    ///
    pub fn new(config: FDTDConfig) -> KwaversResult<Self> {
        config.validate()?;

        let nx = config.nx;
        let mut u_current = Array1::zeros(nx);
        let u_previous = Array1::zeros(nx);

        match config.initial_condition {
            InitialCondition::GaussianPulse { width, amplitude } => {
                let x_center = (nx / 2) as f64 * config.dx;
                for i in 0..nx {
                    let x = i as f64 * config.dx;
                    let dist = x - x_center;
                    u_current[i] = amplitude * (-dist.powi(2) / (2.0 * width.powi(2))).exp();
                }
            }
            InitialCondition::SineWave {
                frequency,
                amplitude,
            } => {
                for i in 0..nx {
                    let x = i as f64 * config.dx;
                    u_current[i] = amplitude * (TWO_PI * frequency * x).sin();
                }
            }
            InitialCondition::Custom => {}
        }

        Ok(Self {
            config,
            u_current,
            u_previous,
            current_step: 0,
        })
    }

    /// Advance one time step (central difference, Dirichlet BC at boundaries).
    /// # Errors
    /// - Returns [`Err`] if an internal constraint is violated.
    ///
    pub fn step(&mut self) -> KwaversResult<()> {
        let nx = self.config.nx;
        let c = self.config.wave_speed;
        let dx = self.config.dx;
        let dt = self.config.dt;

        let alpha = (c * dt / dx).powi(2);
        let mut u_next = Array1::zeros(nx);

        for i in 1..nx - 1 {
            u_next[i] = 2.0f64.mul_add(self.u_current[i], -self.u_previous[i])
                + alpha * (self.u_current[i + 1] - 2.0 * self.u_current[i] + self.u_current[i - 1]);
        }

        // Dirichlet boundary conditions
        u_next[0] = 0.0;
        u_next[nx - 1] = 0.0;

        self.u_previous = self.u_current.clone();
        self.u_current = u_next;
        self.current_step += 1;

        Ok(())
    }

    /// Solve for all time steps, returning field history of shape `(nx, nt)`.
    /// # Errors
    /// - Propagates any [`KwaversError`] returned by `step`.
    ///
    pub fn solve(&mut self) -> KwaversResult<Array2<f64>> {
        let nx = self.config.nx;
        let nt = self.config.nt;
        let mut solution = Array2::zeros((nx, nt));

        for i in 0..nx {
            solution[[i, 0]] = self.u_current[i];
        }

        for t in 1..nt {
            self.step()?;
            for i in 0..nx {
                solution[[i, t]] = self.u_current[i];
            }
        }

        Ok(solution)
    }

    /// Current field values.
    #[must_use]
    pub fn current_field(&self) -> &Array1<f64> {
        &self.u_current
    }

    /// Current time step index.
    #[must_use]
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Solver configuration.
    #[must_use]
    pub fn config(&self) -> &FDTDConfig {
        &self.config
    }
}
