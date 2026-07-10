//! Point-force-driven propagation for elastic FWI (ADR 033).
//!
//! Drives the velocity-Verlet integrator while injecting per-point body forces
//! and returns the full per-step displacement/velocity history. The elastic
//! shear-wave FWI uses this single primitive for **both** the forward run (a
//! localized shear-wave source) and the adjoint run (the time-reversed receiver
//! residual injected at the receiver points). Sharing one injection operator
//! keeps the forward and adjoint source paths identical, which is what makes the
//! reconstructed gradient a faithful descent direction (ADR 033 §Verification).

use super::super::super::integration::TimeIntegrator;
use super::super::super::scratch::ElasticStepScratch;
use super::super::super::types::ElasticWaveField;
use super::definition::ElasticWaveSolver;
use kwavers_core::error::{KwaversResult, NumericalError};

/// A time-varying body force localized at a single grid point.
///
/// `fx`/`fy`/`fz` are per-step force densities \[N/m³]; each must have length
/// ≥ `n_steps` when passed to [`ElasticWaveSolver::propagate_point_forces`].
#[derive(Debug, Clone)]
pub struct ElasticPointForce {
    /// Grid index `(i, j, k)` of the force-application point.
    pub index: (usize, usize, usize),
    /// Per-step force density along x \[N/m³].
    pub fx: Vec<f64>,
    /// Per-step force density along y \[N/m³].
    pub fy: Vec<f64>,
    /// Per-step force density along z \[N/m³].
    pub fz: Vec<f64>,
}

impl ElasticPointForce {
    /// A point force whose components are all zero for `n_steps`.
    #[must_use]
    pub fn zeros(index: (usize, usize, usize), n_steps: usize) -> Self {
        Self {
            index,
            fx: vec![0.0; n_steps],
            fy: vec![0.0; n_steps],
            fz: vec![0.0; n_steps],
        }
    }
}

impl ElasticWaveSolver {
    /// CFL-stable time step for the current medium \[s].
    ///
    /// Constructs the same [`TimeIntegrator`] used by the propagation methods and
    /// returns its stable step for the configured `cfl_factor`.
    #[must_use]
    pub fn recommended_timestep(&self, cfl_factor: f64) -> f64 {
        TimeIntegrator::new(&self.grid, &self.lambda, &self.mu, &self.density, &self.pml)
            .calculate_stable_timestep(cfl_factor)
    }

    /// Propagate `n_steps` velocity-Verlet steps from rest, injecting the given
    /// per-point body forces, and return the full per-step history. `history[n]`
    /// is the field state after step `n` (so the returned vector has length
    /// `n_steps`).
    ///
    /// Each force is applied as a pre-step velocity increment `Δv = (f/ρ)·dt` at
    /// its grid point (k-Wave Additive convention; the increment participates in
    /// the same step's velocity-Verlet displacement update). Points with `ρ ≤ 0`
    /// are skipped.
    ///
    /// # Errors
    /// Returns [`NumericalError::InvalidOperation`] for a non-positive `dt`, or
    /// when any force time series is shorter than `n_steps`.
    pub fn propagate_point_forces(
        &self,
        n_steps: usize,
        dt: f64,
        forces: &[ElasticPointForce],
    ) -> KwaversResult<Vec<ElasticWaveField>> {
        if dt <= 0.0 {
            return Err(NumericalError::InvalidOperation("dt must be positive".to_owned()).into());
        }
        for f in forces {
            if f.fx.len() < n_steps || f.fy.len() < n_steps || f.fz.len() < n_steps {
                return Err(NumericalError::InvalidOperation(
                    "point-force time series shorter than n_steps".to_owned(),
                )
                .into());
            }
        }

        let (nx, ny, nz) = self.grid.dimensions();
        let integrator =
            TimeIntegrator::new(&self.grid, &self.lambda, &self.mu, &self.density, &self.pml);
        let mut scratch = ElasticStepScratch::new(nx, ny, nz);
        let mut field = ElasticWaveField::new(nx, ny, nz);
        let mut history = Vec::with_capacity(n_steps);

        for step in 0..n_steps {
            for f in forces {
                let (i, j, k) = f.index;
                let rho = self.density[[i, j, k]];
                if rho > 0.0 {
                    let scale = dt / rho;
                    field.vx[[i, j, k]] += scale * f.fx[step];
                    field.vy[[i, j, k]] += scale * f.fy[step];
                    field.vz[[i, j, k]] += scale * f.fz[step];
                }
            }
            integrator.step(&mut field, dt, None, &mut scratch)?;
            field.time += dt;
            history.push(field.clone());
        }
        Ok(history)
    }
}
