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
    /// CFL-stable time step for the current medium \\[s\].
    ///
    /// Constructs the same [`TimeIntegrator`] used by the propagation methods and
    /// returns its stable step for the configured `cfl_factor`.
    #[must_use]
    pub fn recommended_timestep(&self, cfl_factor: f64) -> f64 {
        TimeIntegrator::new(&self.grid, &self.lambda, &self.mu, &self.density, &self.pml)
            .calculate_stable_timestep(cfl_factor)
    }

    /// Propagate `n_steps` velocity-Verlet steps from rest, injecting the given
    /// per-point body forces, and return the full per-step history. `history\[n\]`
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
        let mut history = Vec::with_capacity(n_steps);
        self.propagate_point_forces_observing(n_steps, dt, forces, |_, field| {
            history.push(field.clone());
        })?;
        Ok(history)
    }

    /// Propagate point forces and record displacement only at `receivers`.
    ///
    /// This is the sensor-only counterpart to [`Self::propagate_point_forces`].
    /// It retains `O(receivers × n_steps)` displacement traces rather than
    /// cloning six full wave-field arrays at every time step. Objective-only
    /// inverse evaluations use this path; adjoint kernels continue to request
    /// the full displacement history they require.
    ///
    /// # Errors
    /// Returns [`NumericalError::InvalidOperation`] for invalid propagation
    /// inputs or a receiver index outside the solver grid.
    #[cfg(feature = "clinical-imaging")]
    pub(crate) fn propagate_point_forces_recording(
        &self,
        n_steps: usize,
        dt: f64,
        forces: &[ElasticPointForce],
        receivers: &[(usize, usize, usize)],
    ) -> KwaversResult<Vec<Vec<[f64; 3]>>> {
        let (nx, ny, nz) = self.grid.dimensions();
        for &(i, j, k) in receivers {
            if i >= nx || j >= ny || k >= nz {
                return Err(NumericalError::InvalidOperation(format!(
                    "receiver index ({i}, {j}, {k}) is outside grid ({nx}, {ny}, {nz})"
                ))
                .into());
            }
        }

        let mut traces = vec![Vec::with_capacity(n_steps); receivers.len()];
        self.propagate_point_forces_observing(n_steps, dt, forces, |_, field| {
            for (trace, &(i, j, k)) in traces.iter_mut().zip(receivers) {
                trace.push([
                    field.ux[[i, j, k]],
                    field.uy[[i, j, k]],
                    field.uz[[i, j, k]],
                ]);
            }
        })?;
        Ok(traces)
    }

    /// Propagate `n_steps` steps and observe each completed step.
    ///
    /// The observer receives a zero-based step index and the post-step field;
    /// `field.time` already includes that step's `dt`. Reverse-time consumers
    /// rely on this ordering to align forward and adjoint states.
    pub(crate) fn propagate_point_forces_observing<F>(
        &self,
        n_steps: usize,
        dt: f64,
        forces: &[ElasticPointForce],
        mut observe: F,
    ) -> KwaversResult<()>
    where
        F: FnMut(usize, &ElasticWaveField),
    {
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
            observe(step, &field);
        }
        Ok(())
    }
}

#[cfg(all(test, feature = "clinical-imaging"))]
mod tests {
    use super::*;
    use crate::forward::elastic::swe::ElasticWaveConfig;
    use kwavers_grid::Grid;
    use kwavers_medium::homogeneous::HomogeneousMedium;

    #[test]
    fn sensor_only_recording_matches_full_history_sampling() {
        let grid = Grid::new(12, 12, 1, 1.0e-3, 1.0e-3, 1.0e-3).expect("grid");
        let medium = HomogeneousMedium::elastic_homogeneous(1000.0, 3.464_101_6, 2.0, &grid)
            .expect("medium");
        let config = ElasticWaveConfig {
            pml_thickness: 2,
            ..ElasticWaveConfig::default()
        };
        let solver = ElasticWaveSolver::new(&grid, &medium, config).expect("solver");
        let n_steps = 8;
        let mut force = ElasticPointForce::zeros((4, 4, 0), n_steps);
        force.fy[0] = 1.0e6;
        let receivers = [(5, 4, 0), (6, 4, 0)];
        let dt = solver.recommended_timestep(0.3);

        let history = solver
            .propagate_point_forces(n_steps, dt, &[force.clone()])
            .expect("history");
        let recorded = solver
            .propagate_point_forces_recording(n_steps, dt, &[force], &receivers)
            .expect("recording");
        let expected = receivers
            .iter()
            .map(|&(i, j, k)| {
                history
                    .iter()
                    .map(|field| {
                        [
                            field.ux[[i, j, k]],
                            field.uy[[i, j, k]],
                            field.uz[[i, j, k]],
                        ]
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        assert_eq!(recorded, expected);
    }
}
